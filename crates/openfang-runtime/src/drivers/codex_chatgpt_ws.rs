//! WebSocket-backed ChatGPT Codex driver.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use crate::oauth_providers::{openai_codex_obtain_api_key, openai_codex_refresh_token};
use async_trait::async_trait;
use base64::Engine;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use http::HeaderValue;
use openfang_types::message::{
    ContentBlock, Message, MessageContent, Role, StopReason, TokenUsage,
};
use openfang_types::tool::ToolCall;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, RwLock};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::sync::mpsc::error::TrySendError;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::tungstenite::Error as WsError;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tracing::{debug, warn};
use uuid::Uuid;
use zeroize::Zeroizing;

const CODEX_ORIGINATOR: &str = "codex_cli_rs";
const CODEX_WS_BETA: &str = "responses_websockets=2026-02-06";
const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const KEEPALIVE_PING_TIMEOUT_REASON: &str = "keepalive ping timeout";

type CodexWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

static WS_SESSIONS: LazyLock<DashMap<String, Arc<CodexWsSession>>> =
    LazyLock::new(DashMap::new);

#[derive(Clone, Debug, Default)]
struct PendingToolCall {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestSignature {
    model: String,
    instructions: String,
    tools_json: String,
    reasoning_json: Option<String>,
    parallel_tool_calls: bool,
}

impl RequestSignature {
    fn from_request(request: &CompletionRequest, instructions: &str) -> Self {
        let tools = build_tool_payloads(request);
        let reasoning = request
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.reasoning_effort)
            .map(|effort| serde_json::json!({ "effort": effort.to_string() }))
            .map(|value| value.to_string());

        Self {
            model: request.model.clone(),
            instructions: instructions.to_string(),
            tools_json: serde_json::to_string(&tools).unwrap_or_else(|_| "[]".to_string()),
            reasoning_json: reasoning,
            parallel_tool_calls: false,
        }
    }
}

#[derive(Debug, Default)]
struct CodexWsTransportState {
    socket: Option<CodexWsStream>,
    last_response_id: Option<String>,
    last_signature: Option<RequestSignature>,
}

#[derive(Debug)]
struct CodexWsSession {
    continuity_key: String,
    session_id: String,
    prompt_cache_key: String,
    installation_id: String,
    state: Mutex<CodexWsTransportState>,
}

impl CodexWsSession {
    fn new(continuity_key: String) -> Self {
        Self {
            continuity_key,
            session_id: Uuid::new_v4().to_string(),
            prompt_cache_key: Uuid::new_v4().to_string(),
            installation_id: Uuid::new_v4().to_string(),
            state: Mutex::new(CodexWsTransportState::default()),
        }
    }
}

#[derive(Debug)]
struct ParsedResponse {
    content: Vec<ContentBlock>,
    tool_calls: Vec<ToolCall>,
    usage: TokenUsage,
    response_id: Option<String>,
    stop_reason: StopReason,
}

pub struct CodexChatGPTWsDriver {
    access_token: RwLock<Zeroizing<String>>,
    account_id: RwLock<Option<String>>,
    base_url: String,
}

impl CodexChatGPTWsDriver {
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
        }
    }

    fn is_keepalive_timeout_error(error: &LlmError) -> bool {
        match error {
            LlmError::Http(message) => message
                .to_ascii_lowercase()
                .contains(KEEPALIVE_PING_TIMEOUT_REASON),
            _ => false,
        }
    }

    fn is_missing_tool_output_error(error: &LlmError) -> bool {
        match error {
            LlmError::Api { message, .. } => message
                .to_ascii_lowercase()
                .contains("no tool output found for function call"),
            LlmError::Http(message) => message
                .to_ascii_lowercase()
                .contains("no tool output found for function call"),
            _ => false,
        }
    }

    fn websocket_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if let Some(rest) = base.strip_prefix("https://") {
            format!("wss://{rest}/responses")
        } else if let Some(rest) = base.strip_prefix("http://") {
            format!("ws://{rest}/responses")
        } else if base.starts_with("wss://") || base.starts_with("ws://") {
            format!("{base}/responses")
        } else {
            format!("wss://{base}/responses")
        }
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

    async fn connect_socket(&self, session: &CodexWsSession) -> Result<CodexWsStream, LlmError> {
        let mut refreshed = false;

        loop {
            let mut request = self
                .websocket_url()
                .into_client_request()
                .map_err(|e| LlmError::Http(format!("Failed to build Codex websocket request: {e}")))?;

            request.headers_mut().insert(
                "originator",
                HeaderValue::from_static(CODEX_ORIGINATOR),
            );
            request.headers_mut().insert(
                "OpenAI-Beta",
                HeaderValue::from_static(CODEX_WS_BETA),
            );
            request.headers_mut().insert(
                "x-client-request-id",
                HeaderValue::from_str(&session.session_id)
                    .map_err(|e| LlmError::Http(format!("Invalid websocket session id: {e}")))?,
            );
            request.headers_mut().insert(
                "session_id",
                HeaderValue::from_str(&session.session_id)
                    .map_err(|e| LlmError::Http(format!("Invalid websocket session id: {e}")))?,
            );
            request.headers_mut().insert(
                http::header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", self.current_access_token()))
                    .map_err(|e| LlmError::Http(format!("Invalid auth header: {e}")))?,
            );

            if let Some(account_id) = self.current_account_id() {
                request.headers_mut().insert(
                    "ChatGPT-Account-Id",
                    HeaderValue::from_str(&account_id)
                        .map_err(|e| LlmError::Http(format!("Invalid account header: {e}")))?,
                );
            }

            match connect_async(request).await {
                Ok((socket, _)) => return Ok(socket),
                Err(WsError::Http(response)) if response.status().as_u16() == 401 && !refreshed => {
                    if self.refresh_access_token().await? {
                        refreshed = true;
                        continue;
                    }
                    return Err(LlmError::AuthenticationFailed(
                        "Codex websocket authentication failed".to_string(),
                    ));
                }
                Err(WsError::Http(response)) => {
                    return Err(LlmError::Api {
                        status: response.status().as_u16(),
                        message: format!(
                            "Codex websocket handshake failed: {}",
                            response.status()
                        ),
                    });
                }
                Err(error) => {
                    return Err(LlmError::Http(format!(
                        "Codex websocket connection failed: {error}"
                    )));
                }
            }
        }
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

    fn session_for_key(&self, continuity_key: &str) -> Arc<CodexWsSession> {
        WS_SESSIONS
            .entry(continuity_key.to_string())
            .or_insert_with(|| Arc::new(CodexWsSession::new(continuity_key.to_string())))
            .clone()
    }

    fn reset_session_for_key(&self, continuity_key: &str) -> Arc<CodexWsSession> {
        let session = Arc::new(CodexWsSession::new(continuity_key.to_string()));
        WS_SESSIONS.insert(continuity_key.to_string(), session.clone());
        session
    }

    fn build_request_payload(
        &self,
        request: &CompletionRequest,
        session: &CodexWsSession,
        use_previous_response_id: bool,
    ) -> (Value, RequestSignature) {
        let instructions = request
            .system
            .clone()
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());
        let signature = RequestSignature::from_request(request, &instructions);
        let tools = build_tool_payloads(request);

        let messages = if use_previous_response_id {
            incremental_messages(&request.messages).unwrap_or_else(|| request.messages.clone())
        } else {
            request.messages.clone()
        };

        let mut body = serde_json::json!({
            "type": "response.create",
            "model": request.model,
            "instructions": instructions,
            "input": build_message_items(&messages),
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "store": false,
            "stream": true,
            "include": [],
            "prompt_cache_key": session.prompt_cache_key,
            "client_metadata": {
                "x-codex-installation-id": session.installation_id,
            },
        });

        if use_previous_response_id {
            if let Some(previous_response_id) = request.previous_response_id.as_ref() {
                body["previous_response_id"] = Value::String(previous_response_id.clone());
            }
        }

        if let Some(thinking) = request.thinking.as_ref() {
            if let Some(effort) = thinking.reasoning_effort {
                body["reasoning"] = serde_json::json!({
                    "effort": effort.to_string(),
                });
            }
        }

        (body, signature)
    }

    async fn read_response(
        &self,
        socket: &mut CodexWsStream,
        tx: Option<&tokio::sync::mpsc::Sender<StreamEvent>>,
    ) -> Result<ParsedResponse, LlmError> {
        let mut text_content = String::new();
        let mut fallback_text = String::new();
        let mut reasoning_content = String::new();
        let mut reasoning_summary = String::new();
        let mut usage = TokenUsage::default();
        let mut response_id: Option<String> = None;
        let mut pending_calls: HashMap<String, PendingToolCall> = HashMap::new();
        let mut item_to_call: HashMap<String, String> = HashMap::new();

        loop {
            let Some(message) = socket.next().await else {
                return Err(LlmError::Http(
                    "Codex websocket disconnected before completion".to_string(),
                ));
            };

            let message = message
                .map_err(|e| LlmError::Http(format!("Codex websocket receive failed: {e}")))?;

            let text = match message {
                WsMessage::Text(text) => text.to_string(),
                WsMessage::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                WsMessage::Ping(payload) => {
                    socket
                        .send(WsMessage::Pong(payload))
                        .await
                        .map_err(|e| {
                            LlmError::Http(format!(
                                "Codex websocket failed to answer ping with pong: {e}"
                            ))
                        })?;
                    continue;
                }
                WsMessage::Pong(_) => continue,
                WsMessage::Close(frame) => {
                    return Err(LlmError::Http(format!(
                        "Codex websocket closed: {}",
                        frame
                            .as_ref()
                            .map(|value| value.reason.to_string())
                            .unwrap_or_else(|| "no reason".to_string())
                    )));
                }
                _ => continue,
            };

            let json: Value = serde_json::from_str(&text)
                .map_err(|e| LlmError::Parse(format!("Invalid websocket event: {e}")))?;

            match json.get("type").and_then(Value::as_str).unwrap_or_default() {
                "codex.rate_limits" => {}
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
                            try_emit_stream_event(
                                tx,
                                StreamEvent::TextDelta {
                                    text: delta.to_string(),
                                },
                            );
                        }
                    }
                }
                "response.reasoning_text.delta" => {
                    if let Some(delta) = json.get("delta").and_then(Value::as_str) {
                        if !delta.is_empty() {
                            reasoning_content.push_str(delta);
                            try_emit_stream_event(
                                tx,
                                StreamEvent::ThinkingDelta {
                                    text: delta.to_string(),
                                },
                            );
                        }
                    }
                }
                "response.reasoning_summary_text.delta" => {
                    if let Some(delta) = json.get("delta").and_then(Value::as_str) {
                        if !delta.is_empty() {
                            reasoning_summary.push_str(delta);
                            if reasoning_content.is_empty() {
                                try_emit_stream_event(
                                    tx,
                                    StreamEvent::ThinkingDelta {
                                        text: delta.to_string(),
                                    },
                                );
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

                                try_emit_stream_event(
                                    tx,
                                    StreamEvent::ToolUseStart { id: call_id, name },
                                );
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
                        try_emit_stream_event(
                            tx,
                            StreamEvent::ToolInputDelta {
                                text: delta.to_string(),
                            },
                        );
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
                                    let entry = pending_calls.entry(call_id.clone()).or_default();
                                    if let Some(name) = item.get("name").and_then(Value::as_str) {
                                        entry.name = name.to_string();
                                    }
                                    if let Some(arguments) =
                                        item.get("arguments").and_then(Value::as_str)
                                    {
                                        entry.arguments = arguments.to_string();
                                    }
                                    let input = serde_json::from_str(&entry.arguments)
                                        .unwrap_or_else(|_| serde_json::json!({}));
                                    try_emit_stream_event(
                                        tx,
                                        StreamEvent::ToolUseEnd {
                                            id: call_id,
                                            name: entry.name.clone(),
                                            input,
                                        },
                                    );
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
                        if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
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

                    break;
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
                "error" => {
                    let status = json.get("status").and_then(Value::as_u64).unwrap_or(500) as u16;
                    let message = json
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| json.to_string());
                    return Err(LlmError::Api { status, message });
                }
                _ => {}
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

        try_emit_stream_event(
            tx,
            StreamEvent::ContentComplete {
                stop_reason: stop_reason.clone(),
                usage: usage.clone(),
            },
        );

        Ok(ParsedResponse {
            content,
            tool_calls,
            usage,
            response_id,
            stop_reason,
        })
    }

    async fn execute_inner(
        &self,
        request: CompletionRequest,
        tx: Option<tokio::sync::mpsc::Sender<StreamEvent>>,
        allow_recovery_retry: bool,
    ) -> Result<CompletionResponse, LlmError> {
        let continuity_key = request
            .continuity_key
            .clone()
            .unwrap_or_else(|| format!("ephemeral:{}", Uuid::new_v4()));
        let session = self.session_for_key(&continuity_key);
        let mut transport = session.state.lock().await;

        if transport.socket.is_none() {
            transport.socket = Some(self.connect_socket(&session).await?);
        }

        let instructions = request
            .system
            .clone()
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());
        let signature = RequestSignature::from_request(&request, &instructions);
        let use_previous_response_id = request.previous_response_id.is_some()
            && transport.socket.is_some()
            && transport.last_response_id == request.previous_response_id
            && transport.last_signature.as_ref() == Some(&signature)
            && incremental_messages(&request.messages).is_some();

        let (body, signature) = self.build_request_payload(&request, &session, use_previous_response_id);
        let payload = serde_json::to_string(&body)
            .map_err(|e| LlmError::Parse(format!("Failed to encode Codex websocket request: {e}")))?;

        let result = {
            let socket = transport
                .socket
                .as_mut()
                .ok_or_else(|| LlmError::Http("Codex websocket session not connected".to_string()))?;
            socket
                .send(WsMessage::Text(payload.into()))
                .await
                .map_err(|e| LlmError::Http(format!("Codex websocket send failed: {e}")))?;
            self.read_response(socket, tx.as_ref()).await
        };

        match result {
            Ok(parsed) => {
                transport.last_response_id = parsed.response_id.clone();
                transport.last_signature = Some(signature);
                Ok(CompletionResponse {
                    content: parsed.content,
                    stop_reason: parsed.stop_reason,
                    tool_calls: parsed.tool_calls,
                    usage: parsed.usage,
                    response_id: parsed.response_id,
                })
            }
            Err(error) => {
                transport.socket = None;
                if Self::is_keepalive_timeout_error(&error) {
                    warn!(
                        continuity_key = %session.continuity_key,
                        "Codex websocket session expired due to keepalive timeout"
                    );
                }
                if matches!(
                    &error,
                    LlmError::Api { message, .. } if message.contains("previous_response_not_found")
                ) {
                    warn!(
                        continuity_key = %session.continuity_key,
                        "Codex websocket continuity state was rejected by server"
                    );
                    transport.last_response_id = None;
                    transport.last_signature = None;
                }
                if Self::is_missing_tool_output_error(&error) {
                    warn!(
                        continuity_key = %session.continuity_key,
                        "Codex websocket continuity state was rejected because tool output linkage was lost"
                    );
                    transport.last_response_id = None;
                    transport.last_signature = None;
                }
                let should_retry_clean = allow_recovery_retry
                    && Self::is_missing_tool_output_error(&error);
                drop(transport);
                if should_retry_clean {
                    self.reset_session_for_key(&continuity_key);
                    let mut retry_request = request.clone();
                    retry_request.previous_response_id = None;
                    return Box::pin(self.execute_inner(retry_request, tx, false)).await;
                }
                Err(error)
            }
        }
    }
}

#[async_trait]
impl LlmDriver for CodexChatGPTWsDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.execute_inner(request, None, true).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        self.execute_inner(request, Some(tx), true).await
    }
}

fn try_emit_stream_event(
    tx: Option<&tokio::sync::mpsc::Sender<StreamEvent>>,
    event: StreamEvent,
) {
    let Some(tx) = tx else {
        return;
    };

    if let Err(error) = tx.try_send(event) {
        match error {
            TrySendError::Full(_) => {
                debug!("Dropping Codex stream event because receiver channel is full");
            }
            TrySendError::Closed(_) => {
                debug!("Dropping Codex stream event because receiver channel is closed");
            }
        }
    }
}

fn incremental_messages(messages: &[Message]) -> Option<Vec<Message>> {
    let last_assistant_index = messages
        .iter()
        .rposition(|message| matches!(message.role, Role::Assistant))?;
    let tail = messages.get(last_assistant_index + 1..)?;
    if tail.is_empty() {
        None
    } else {
        Some(tail.to_vec())
    }
}

fn incremental_can_use_previous_response_id(messages: &[Message]) -> bool {
    let Some(last_assistant_index) = messages
        .iter()
        .rposition(|message| matches!(message.role, Role::Assistant))
    else {
        return false;
    };

    let Some(tail) = messages.get(last_assistant_index + 1..) else {
        return false;
    };

    !tail.iter().any(message_contains_tool_result)
}

fn message_contains_tool_result(message: &Message) -> bool {
    match &message.content {
        MessageContent::Blocks(blocks) => blocks.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolResult { .. }
            )
        }),
        _ => false,
    }
}

fn build_tool_payloads(request: &CompletionRequest) -> Vec<Value> {
    request
        .tools
        .iter()
        .map(|tool| {
            let mut parameters =
                openfang_types::tool::normalize_schema_for_provider(&tool.input_schema, "openai");
            enforce_strict_object_schema(&mut parameters);
            serde_json::json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": true,
            })
        })
        .collect()
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
                        ContentBlock::ToolUse { id, name, input, .. } => {
                            let arguments =
                                serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
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
        "type": "message",
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
                        Value::Array(required_keys.into_iter().map(Value::String).collect()),
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
            continuity_key: Some("codex:test".to_string()),
            previous_response_id: Some("resp_prev".to_string()),
        }
    }

    #[test]
    fn incremental_messages_uses_tail_after_last_assistant() {
        let delta = incremental_messages(&[
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ])
        .expect("delta");

        assert_eq!(delta.len(), 1);
        assert!(matches!(delta[0].role, Role::User));
    }

    #[test]
    fn websocket_payload_includes_previous_response_id_only_for_incremental_turns() {
        let driver = CodexChatGPTWsDriver::new("token".to_string(), "".to_string());
        let session = CodexWsSession::new("codex:test".to_string());
        let request = base_request(vec![
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ]);

        let (body, _) = driver.build_request_payload(&request, &session, true);
        assert_eq!(
            body.get("previous_response_id").and_then(Value::as_str),
            Some("resp_prev")
        );
        let input = body
            .get("input")
            .and_then(Value::as_array)
            .expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0].get("role").and_then(Value::as_str), Some("user"));
    }

    #[test]
    fn websocket_payload_falls_back_to_full_history_when_incremental_disabled() {
        let driver = CodexChatGPTWsDriver::new("token".to_string(), "".to_string());
        let session = CodexWsSession::new("codex:test".to_string());
        let request = base_request(vec![
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ]);

        let (body, _) = driver.build_request_payload(&request, &session, false);
        assert!(body.get("previous_response_id").is_none());
        let input = body
            .get("input")
            .and_then(Value::as_array)
            .expect("input array");
        assert_eq!(input.len(), 3);
    }

    #[test]
    fn keepalive_timeout_error_is_detected() {
        let error = LlmError::Http("Codex websocket closed: keepalive ping timeout".to_string());
        assert!(CodexChatGPTWsDriver::is_keepalive_timeout_error(&error));
    }
}
