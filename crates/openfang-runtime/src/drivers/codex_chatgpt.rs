//! ChatGPT-backed Codex driver for OAuth sessions.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use crate::oauth_providers::{openai_codex_obtain_api_key, openai_codex_refresh_token};
use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use openfang_types::message::{
    ContentBlock, Message, MessageContent, Role, StopReason, TokenUsage,
};
use openfang_types::tool::ToolCall;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use tracing::{debug, warn};
use zeroize::Zeroizing;

const CODEX_ORIGINATOR: &str = "codex_cli_rs";
const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

#[derive(Clone, Debug, Default)]
struct PendingToolCall {
    name: String,
    arguments: String,
}

pub struct CodexChatGPTDriver {
    access_token: RwLock<Zeroizing<String>>,
    account_id: RwLock<Option<String>>,
    base_url: String,
    client: reqwest::Client,
}

impl CodexChatGPTDriver {
    pub fn new(access_token: String, base_url: String) -> Self {
        let account_id = parse_chatgpt_account_id_from_jwt(&access_token)
            .or_else(|| {
                std::env::var("CODEX_OAUTH_ID_TOKEN")
                    .ok()
                    .and_then(|jwt| parse_chatgpt_account_id_from_jwt(&jwt))
            })
            .or_else(|| {
                std::env::var("CODEX_OAUTH_ACCESS_TOKEN")
                    .ok()
                    .and_then(|jwt| parse_chatgpt_account_id_from_jwt(&jwt))
            });

        Self {
            access_token: RwLock::new(Zeroizing::new(access_token)),
            account_id: RwLock::new(account_id),
            base_url: if base_url.trim().is_empty() {
                DEFAULT_CODEX_BASE_URL.to_string()
            } else {
                base_url
            },
            client: reqwest::Client::builder()
                .user_agent(crate::USER_AGENT)
                .build()
                .unwrap_or_default(),
        }
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }

    fn current_access_token(&self) -> String {
        self.access_token
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_str()
            .to_string()
    }

    fn current_account_id(&self) -> Option<String> {
        self.account_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn build_request_body(&self, request: &CompletionRequest) -> Value {
        let instructions = request
            .system
            .clone()
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());
        let input = build_message_items(&request.messages);

        let mut body = serde_json::json!({
            "model": request.model,
            "instructions": instructions,
            "input": input,
            "stream": true,
            "store": false,
            "parallel_tool_calls": !request.tools.is_empty(),
        });

        // The ChatGPT Codex backend currently rejects `previous_response_id`
        // even though the official OpenAI Responses API supports it, so we
        // keep sending the full conversation history here.

        if let Some(thinking) = request.thinking.as_ref() {
            if let Some(effort) = thinking.reasoning_effort {
                body["reasoning"] = serde_json::json!({
                    "effort": effort.to_string(),
                });
            }
        }

        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|tool| {
                    let mut parameters = openfang_types::tool::normalize_schema_for_provider(
                        &tool.input_schema,
                        "openai",
                    );
                    enforce_strict_object_schema(&mut parameters);
                    serde_json::json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                        "strict": true,
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
            body["tool_choice"] = Value::String("auto".to_string());
        }

        body
    }

    async fn send_request(&self, body: &Value) -> Result<reqwest::Response, LlmError> {
        let mut refresh_attempted = false;
        let max_rate_limit_retries = 3;

        for attempt in 0..=max_rate_limit_retries {
            let url = self.responses_url();
            let token = self.current_access_token();

            let mut builder = self
                .client
                .post(&url)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .header("originator", CODEX_ORIGINATOR)
                .header("authorization", format!("Bearer {token}"))
                .json(body);

            if let Some(account_id) = self.current_account_id() {
                builder = builder.header("ChatGPT-Account-Id", account_id);
            }

            let resp = builder
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            let status = resp.status().as_u16();

            if status == 429 {
                if attempt < max_rate_limit_retries {
                    let retry_ms = (attempt + 1) as u64 * 2000;
                    warn!(status, retry_ms, "Codex ChatGPT rate limited, retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(LlmError::RateLimited {
                    retry_after_ms: 5000,
                });
            }

            if status == 401 && !refresh_attempted && self.refresh_access_token().await? {
                refresh_attempted = true;
                continue;
            }

            return Ok(resp);
        }

        Err(LlmError::Api {
            status: 0,
            message: "Max retries exceeded".to_string(),
        })
    }

    async fn refresh_access_token(&self) -> Result<bool, LlmError> {
        let Some(refresh_token) = std::env::var("CODEX_OAUTH_REFRESH_TOKEN")
            .ok()
            .filter(|value| !value.trim().is_empty())
        else {
            return Ok(false);
        };

        let mut tokens = openai_codex_refresh_token(&refresh_token)
            .await
            .map_err(LlmError::AuthenticationFailed)?;

        if let Some(id_token) = tokens.id_token.clone() {
            match openai_codex_obtain_api_key(&id_token).await {
                Ok(api_key) => tokens.api_key = Some(api_key),
                Err(error) => debug!("Codex API key exchange skipped during refresh: {error}"),
            }
        }

        persist_refreshed_codex_tokens(&tokens)
            .map_err(|e| LlmError::Http(format!("Failed to persist refreshed Codex token: {e}")))?;

        {
            let mut access = self.access_token.write().unwrap_or_else(|e| e.into_inner());
            *access = Zeroizing::new(tokens.access_token.clone());
        }

        let account_id = tokens
            .id_token
            .as_deref()
            .and_then(parse_chatgpt_account_id_from_jwt)
            .or_else(|| parse_chatgpt_account_id_from_jwt(&tokens.access_token));
        let mut current_account_id = self.account_id.write().unwrap_or_else(|e| e.into_inner());
        *current_account_id = account_id;

        Ok(true)
    }

    async fn execute(
        &self,
        request: CompletionRequest,
        tx: Option<tokio::sync::mpsc::Sender<StreamEvent>>,
    ) -> Result<CompletionResponse, LlmError> {
        let body = self.build_request_body(&request);
        let resp = self.send_request(&body).await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status,
                message: body,
            });
        }

        let mut buffer = String::new();
        let mut text_content = String::new();
        let mut fallback_text = String::new();
        let mut reasoning_content = String::new();
        let mut reasoning_summary = String::new();
        let mut usage = TokenUsage::default();
        let mut response_id: Option<String> = None;
        let mut pending_calls: HashMap<String, PendingToolCall> = HashMap::new();
        let mut item_to_call: HashMap<String, String> = HashMap::new();

        let mut byte_stream = resp.bytes_stream();
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = chunk_result.map_err(|e| LlmError::Http(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim_end().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }

                let json: Value = match serde_json::from_str(data) {
                    Ok(value) => value,
                    Err(_) => continue,
                };

                match json.get("type").and_then(Value::as_str).unwrap_or_default() {
                    "response.created" => {
                        if let Some(id) = json
                            .pointer("/response/id")
                            .and_then(Value::as_str)
                            .filter(|value| !value.is_empty())
                        {
                            response_id = Some(id.to_string());
                        }
                    }
                    "response.output_text.delta" => {
                        if let Some(delta) = json.get("delta").and_then(Value::as_str) {
                            if !delta.is_empty() {
                                text_content.push_str(delta);
                                if let Some(tx) = tx.as_ref() {
                                    let _ = tx
                                        .send(StreamEvent::TextDelta {
                                            text: delta.to_string(),
                                        })
                                        .await;
                                }
                            }
                        }
                    }
                    "response.reasoning_text.delta" => {
                        if let Some(delta) = json.get("delta").and_then(Value::as_str) {
                            if !delta.is_empty() {
                                reasoning_content.push_str(delta);
                                if let Some(tx) = tx.as_ref() {
                                    let _ = tx
                                        .send(StreamEvent::ThinkingDelta {
                                            text: delta.to_string(),
                                        })
                                        .await;
                                }
                            }
                        }
                    }
                    "response.reasoning_summary_text.delta" => {
                        if let Some(delta) = json.get("delta").and_then(Value::as_str) {
                            if !delta.is_empty() {
                                reasoning_summary.push_str(delta);
                                if reasoning_content.is_empty() {
                                    if let Some(tx) = tx.as_ref() {
                                        let _ = tx
                                            .send(StreamEvent::ThinkingDelta {
                                                text: delta.to_string(),
                                            })
                                            .await;
                                    }
                                }
                            }
                        }
                    }
                    "response.output_item.added" => {
                        if let Some(item) = json.get("item") {
                            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                                let call_id = item
                                    .get("call_id")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                let name = item
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                let arguments = item
                                    .get("arguments")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                let item_id =
                                    item.get("id").and_then(Value::as_str).map(str::to_string);

                                if !call_id.is_empty() {
                                    if let Some(item_id) = item_id {
                                        item_to_call.insert(item_id, call_id.clone());
                                    }
                                    let entry = pending_calls.entry(call_id.clone()).or_default();
                                    entry.name = name.clone();
                                    entry.arguments = arguments;

                                    if let Some(tx) = tx.as_ref() {
                                        let _ = tx
                                            .send(StreamEvent::ToolUseStart { id: call_id, name })
                                            .await;
                                    }
                                }
                            }
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        let item_id = json
                            .get("item_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let delta = json
                            .get("delta")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if let Some(call_id) = item_to_call.get(item_id).cloned() {
                            let entry = pending_calls.entry(call_id).or_default();
                            entry.arguments.push_str(delta);
                            if let Some(tx) = tx.as_ref() {
                                let _ = tx
                                    .send(StreamEvent::ToolInputDelta {
                                        text: delta.to_string(),
                                    })
                                    .await;
                            }
                        }
                    }
                    "response.output_item.done" => {
                        if let Some(item) = json.get("item") {
                            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                                "function_call" => {
                                    let call_id = item
                                        .get("call_id")
                                        .and_then(Value::as_str)
                                        .unwrap_or_default()
                                        .to_string();
                                    if !call_id.is_empty() {
                                        let entry =
                                            pending_calls.entry(call_id.clone()).or_default();
                                        if let Some(name) = item.get("name").and_then(Value::as_str)
                                        {
                                            entry.name = name.to_string();
                                        }
                                        if let Some(arguments) =
                                            item.get("arguments").and_then(Value::as_str)
                                        {
                                            entry.arguments = arguments.to_string();
                                        }
                                        if let Some(tx) = tx.as_ref() {
                                            let input = serde_json::from_str(&entry.arguments)
                                                .unwrap_or_else(|_| serde_json::json!({}));
                                            let _ = tx
                                                .send(StreamEvent::ToolUseEnd {
                                                    id: call_id,
                                                    name: entry.name.clone(),
                                                    input,
                                                })
                                                .await;
                                        }
                                    }
                                }
                                "message" => {
                                    if text_content.is_empty() {
                                        if let Some(text) = extract_output_text(item) {
                                            fallback_text.push_str(&text);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(response) = json.get("response") {
                            if let Some(id) = response
                                .get("id")
                                .and_then(Value::as_str)
                                .filter(|value| !value.is_empty())
                            {
                                response_id = Some(id.to_string());
                            }
                            if let Some(error) =
                                response.get("error").filter(|value| !value.is_null())
                            {
                                return Err(LlmError::Api {
                                    status: 500,
                                    message: error.to_string(),
                                });
                            }
                            if let Some(resp_usage) = response.get("usage") {
                                if let Some(tokens) =
                                    resp_usage.get("input_tokens").and_then(Value::as_u64)
                                {
                                    usage.input_tokens = tokens;
                                }
                                if let Some(tokens) = resp_usage
                                    .get("input_tokens_details")
                                    .and_then(|value| value.get("cached_tokens"))
                                    .and_then(Value::as_u64)
                                {
                                    usage.cached_input_tokens = tokens;
                                }
                                if let Some(tokens) =
                                    resp_usage.get("output_tokens").and_then(Value::as_u64)
                                {
                                    usage.output_tokens = tokens;
                                }
                            }
                        }
                    }
                    "response.failed" => {
                        let message = json
                            .pointer("/response/error/message")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| json.to_string());
                        return Err(LlmError::Api {
                            status: 500,
                            message,
                        });
                    }
                    _ => {}
                }
            }
        }

        let final_text = if text_content.is_empty() {
            fallback_text
        } else {
            text_content
        };

        let mut content = Vec::new();
        let mut tool_calls = Vec::new();

        let final_reasoning = if !reasoning_content.trim().is_empty() {
            reasoning_content.trim().to_string()
        } else {
            reasoning_summary.trim().to_string()
        };
        if !final_reasoning.is_empty() {
            content.push(ContentBlock::Thinking {
                thinking: final_reasoning,
            });
        }

        if !final_text.is_empty() {
            content.push(ContentBlock::Text {
                text: final_text,
                provider_metadata: None,
            });
        }

        for (call_id, pending) in pending_calls {
            if call_id.is_empty() || pending.name.is_empty() {
                continue;
            }
            let input =
                serde_json::from_str(&pending.arguments).unwrap_or_else(|_| serde_json::json!({}));
            content.push(ContentBlock::ToolUse {
                id: call_id.clone(),
                name: pending.name.clone(),
                input: input.clone(),
                provider_metadata: None,
            });
            tool_calls.push(ToolCall {
                id: call_id,
                name: pending.name,
                input,
            });
        }

        if !content.is_empty() && usage.input_tokens == 0 && usage.output_tokens == 0 {
            usage.output_tokens = 1;
        }

        let stop_reason = if tool_calls.is_empty() {
            StopReason::EndTurn
        } else {
            StopReason::ToolUse
        };

        if let Some(tx) = tx.as_ref() {
            let _ = tx
                .send(StreamEvent::ContentComplete {
                    stop_reason: stop_reason.clone(),
                    usage: usage.clone(),
                })
                .await;
        }

        Ok(CompletionResponse {
            content,
            stop_reason,
            tool_calls,
            usage,
            response_id,
        })
    }
}

#[async_trait]
impl LlmDriver for CodexChatGPTDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.execute(request, None).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        self.execute(request, Some(tx)).await
    }
}

fn build_message_items(messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::new();

    for message in messages {
        match (&message.role, &message.content) {
            (Role::System, _) => {}
            (Role::User, MessageContent::Text(text)) => {
                items.push(message_item("user", vec![input_text_part(text)]));
            }
            (Role::Assistant, MessageContent::Text(text)) => {
                items.push(message_item("assistant", vec![output_text_part(text)]));
            }
            (Role::User, MessageContent::Blocks(blocks)) => {
                let mut content = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => content.push(input_text_part(text)),
                        ContentBlock::Image { media_type, data } => {
                            content.push(serde_json::json!({
                                "type": "input_image",
                                "image_url": format!("data:{media_type};base64,{data}"),
                            }));
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            items.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": if content.is_empty() { "(empty)" } else { content.as_str() },
                            }));
                        }
                        _ => {}
                    }
                }
                if !content.is_empty() {
                    items.push(message_item("user", content));
                }
            }
            (Role::Assistant, MessageContent::Blocks(blocks)) => {
                let mut content = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => content.push(output_text_part(text)),
                        ContentBlock::ToolUse {
                            id, name, input, ..
                        } => {
                            let arguments =
                                serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                            items.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": arguments,
                            }));
                        }
                        _ => {}
                    }
                }
                if !content.is_empty() {
                    items.push(message_item("assistant", content));
                }
            }
        }
    }

    items
}

fn message_item(role: &str, content: Vec<Value>) -> Value {
    serde_json::json!({
        "role": role,
        "content": content,
    })
}

fn input_text_part(text: &str) -> Value {
    serde_json::json!({
        "type": "input_text",
        "text": text,
    })
}

fn output_text_part(text: &str) -> Value {
    serde_json::json!({
        "type": "output_text",
        "text": text,
    })
}

fn extract_output_text(item: &Value) -> Option<String> {
    let content = item.get("content")?.as_array()?;
    let mut text = String::new();
    for part in content {
        if part.get("type").and_then(Value::as_str) == Some("output_text") {
            if let Some(value) = part.get("text").and_then(Value::as_str) {
                text.push_str(value);
            }
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn parse_chatgpt_account_id_from_jwt(jwt: &str) -> Option<String> {
    let mut parts = jwt.split('.');
    let (_header, payload, _sig) = match (parts.next(), parts.next(), parts.next()) {
        (Some(h), Some(p), Some(s)) if !h.is_empty() && !p.is_empty() && !s.is_empty() => (h, p, s),
        _ => return None,
    };

    let padded = format!("{}{}", payload, "=".repeat((4 - payload.len() % 4) % 4));
    let payload_bytes = base64::engine::general_purpose::URL_SAFE
        .decode(padded)
        .ok()?;
    let claims: Value = serde_json::from_slice(&payload_bytes).ok()?;
    claims
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn enforce_strict_object_schema(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            if map.get("type").and_then(Value::as_str) == Some("object") {
                map.entry("additionalProperties".to_string())
                    .or_insert(Value::Bool(false));
                let existing_required: std::collections::HashSet<String> = map
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();

                if let Some(properties) = map.get_mut("properties").and_then(Value::as_object_mut) {
                    let required_keys: Vec<String> = properties.keys().cloned().collect();
                    for (key, value) in properties.iter_mut() {
                        if !existing_required.contains(key) {
                            make_schema_nullable(value);
                        }
                        enforce_strict_object_schema(value);
                    }
                    map.insert(
                        "required".to_string(),
                        Value::Array(
                            required_keys
                                .into_iter()
                                .map(Value::String)
                                .collect::<Vec<_>>(),
                        ),
                    );
                }
            }

            if let Some(items) = map.get_mut("items") {
                enforce_strict_object_schema(items);
            }

            if let Some(any_of) = map.get_mut("anyOf").and_then(Value::as_array_mut) {
                for value in any_of {
                    enforce_strict_object_schema(value);
                }
            }

            if let Some(one_of) = map.get_mut("oneOf").and_then(Value::as_array_mut) {
                for value in one_of {
                    enforce_strict_object_schema(value);
                }
            }

            if let Some(all_of) = map.get_mut("allOf").and_then(Value::as_array_mut) {
                for value in all_of {
                    enforce_strict_object_schema(value);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                enforce_strict_object_schema(value);
            }
        }
        _ => {}
    }
}

fn make_schema_nullable(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        if obj.contains_key("anyOf") {
            if let Some(any_of) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
                let already_nullable = any_of
                    .iter()
                    .any(|value| value.get("type").and_then(Value::as_str) == Some("null"));
                if !already_nullable {
                    any_of.push(serde_json::json!({"type": "null"}));
                }
                return;
            }
        }
    }

    let original = std::mem::take(schema);
    *schema = serde_json::json!({
        "anyOf": [
            original,
            { "type": "null" }
        ]
    });
}

fn persist_refreshed_codex_tokens(
    tokens: &crate::oauth_providers::OAuthTokenSet,
) -> Result<(), std::io::Error> {
    let secrets_path = openfang_home().join("secrets.env");

    if let Some(api_key) = tokens
        .api_key
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        write_secret_env(&secrets_path, "CODEX_API_KEY", api_key)?;
        std::env::set_var("CODEX_API_KEY", api_key);
    }

    write_secret_env(
        &secrets_path,
        "CODEX_OAUTH_ACCESS_TOKEN",
        &tokens.access_token,
    )?;
    std::env::set_var("CODEX_OAUTH_ACCESS_TOKEN", &tokens.access_token);

    if let Some(id_token) = tokens.id_token.as_deref() {
        write_secret_env(&secrets_path, "CODEX_OAUTH_ID_TOKEN", id_token)?;
        std::env::set_var("CODEX_OAUTH_ID_TOKEN", id_token);
    }

    if let Some(refresh_token) = tokens.refresh_token.as_deref() {
        write_secret_env(&secrets_path, "CODEX_OAUTH_REFRESH_TOKEN", refresh_token)?;
        std::env::set_var("CODEX_OAUTH_REFRESH_TOKEN", refresh_token);
    }

    Ok(())
}

fn openfang_home() -> PathBuf {
    if let Ok(home) = std::env::var("OPENFANG_HOME") {
        return PathBuf::from(home);
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(home) = std::env::var("USERPROFILE") {
            return PathBuf::from(home).join(".openfang");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".openfang");
        }
    }
    std::env::temp_dir().join(".openfang")
}

fn write_secret_env(path: &Path, key: &str, value: &str) -> Result<(), std::io::Error> {
    let mut lines: Vec<String> = if path.exists() {
        std::fs::read_to_string(path)?
            .lines()
            .map(|line| line.to_string())
            .collect()
    } else {
        Vec::new()
    };

    lines.retain(|line| !line.starts_with(&format!("{key}=")));
    lines.push(format!("{key}={value}"));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(path, lines.join("\n") + "\n")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request(messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            model: "gpt-5.4".to_string(),
            messages,
            tools: vec![],
            max_tokens: 1024,
            temperature: 0.0,
            system: Some("You are helpful.".to_string()),
            thinking: None,
            continuity_key: None,
            previous_response_id: None,
        }
    }

    #[test]
    fn codex_ignores_previous_response_id_for_chatgpt_backend() {
        let driver = CodexChatGPTDriver::new("token".to_string(), "".to_string());
        let mut request = base_request(vec![
            Message::user("old question"),
            Message::assistant("old answer"),
            Message::user("new question"),
        ]);
        request.previous_response_id = Some("resp_prev".to_string());

        let body = driver.build_request_body(&request);
        assert!(body.get("previous_response_id").is_none());

        let input = body
            .get("input")
            .and_then(Value::as_array)
            .expect("input array");
        assert_eq!(input.len(), 3);
        assert_eq!(input[0].get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(
            input[1].get("role").and_then(Value::as_str),
            Some("assistant")
        );
        assert_eq!(input[2].get("role").and_then(Value::as_str), Some("user"));
    }

    #[test]
    fn codex_still_sends_full_history_without_previous_response_id() {
        let driver = CodexChatGPTDriver::new("token".to_string(), "".to_string());
        let mut request = base_request(vec![Message::user("first turn")]);
        request.previous_response_id = Some("resp_prev".to_string());

        let body = driver.build_request_body(&request);
        assert!(body.get("previous_response_id").is_none());

        let input = body
            .get("input")
            .and_then(Value::as_array)
            .expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0].get("role").and_then(Value::as_str), Some("user"));
    }
}
