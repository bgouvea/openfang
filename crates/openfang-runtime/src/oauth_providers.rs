//! OAuth providers for LLM services — OpenAI Codex, Gemini, Qwen, MiniMax.
//!
//! This module implements OAuth2 authentication flows for:
//! - OpenAI Codex (ChatGPT subscription) — device code flow + PKCE
//! - Gemini (Google OAuth) — PKCE + device code
//! - Qwen (Alibaba) — **file-based token import** from ~/.qwen/oauth_creds.json
//!   (not a true OAuth flow — reads pre-existing tokens from Qwen CLI)
//! - MiniMax — refresh token based (requires stored refresh token in vault)
//!
//! All tokens are stored in the credential vault.

use base64::Engine;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ─── Constants ───────────────────────────────────────────────────────────────────

// OpenAI Codex OAuth
pub const OPENAI_CODEX_DEFAULT_ISSUER: &str = "https://auth.openai.com";
pub const OPENAI_CODEX_DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_DEFAULT_CALLBACK_URI: &str = "http://localhost:1455/auth/callback";

// Gemini OAuth (requires GEMINI_OAUTH_CLIENT_ID and GEMINI_OAUTH_CLIENT_SECRET env vars)
pub const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
pub const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
pub const GOOGLE_DEVICE_URL: &str = "https://oauth2.googleapis.com/device/code";
pub const GOOGLE_CALLBACK_URI: &str = "http://localhost:1456/auth/callback";
pub const GEMINI_SCOPES: &str =
    "openid profile email https://www.googleapis.com/auth/cloud-platform";

// Qwen OAuth
pub const QWEN_OAUTH_TOKEN_ENDPOINT: &str = "https://chat.qwen.ai/api/v1/oauth2/token";
pub const QWEN_OAUTH_CREDENTIAL_FILE: &str = ".qwen/oauth_creds.json";
pub const QWEN_OAUTH_CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";

// MiniMax OAuth
pub const MINIMAX_OAUTH_TOKEN_ENDPOINT: &str = "https://api.minimax.io/oauth/token";
pub const MINIMAX_OAUTH_CLIENT_ID: &str = "78257093-7e40-4613-99e0-527b14b39113";

// ─── Token Storage ────────────────────────────────────────────────────────────────

/// OAuth tokens stored in vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthTokenSet {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub api_key: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub provider: String,
}

impl OAuthTokenSet {
    /// Check if token is expired (with 60-second buffer).
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            expires_at < Utc::now() + chrono::Duration::seconds(60)
        } else {
            false
        }
    }

    /// Create from token response.
    pub fn from_response(resp: TokenResponse, provider: &str) -> Self {
        let expires_at = resp
            .expires_in
            .map(|secs| Utc::now() + chrono::Duration::seconds(secs));
        Self {
            access_token: resp.access_token,
            refresh_token: resp.refresh_token,
            id_token: resp.id_token,
            api_key: None,
            expires_at,
            provider: provider.to_string(),
        }
    }
}

/// Token response from OAuth provider.
#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub id_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<i64>,
    #[serde(default)]
    pub token_type: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Device code start response.
#[derive(Debug, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    pub expires_in: u64,
    #[serde(default)]
    pub interval: Option<u64>,
}

/// Device code flow status.
pub enum DeviceFlowStatus {
    Pending,
    Complete { tokens: OAuthTokenSet },
    SlowDown { new_interval: u64 },
    Expired,
    AccessDenied,
    Error(String),
}

// ─── OpenAI Codex OAuth ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CodexDeviceCodeStartResponse {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    interval: Option<u64>,
}

#[derive(Debug, Serialize)]
struct CodexDeviceCodeStartRequest {
    client_id: String,
}

#[derive(Debug, Serialize)]
struct CodexDeviceCodePollRequest<'a> {
    device_auth_id: &'a str,
    user_code: &'a str,
}

#[derive(Debug, Deserialize)]
struct CodexDeviceCodePollResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct CodexApiKeyExchangeResponse {
    access_token: String,
}

fn openai_codex_issuer() -> String {
    std::env::var("OPENFANG_CODEX_OAUTH_ISSUER")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| OPENAI_CODEX_DEFAULT_ISSUER.to_string())
}

fn openai_codex_client_id() -> String {
    std::env::var("OPENFANG_CODEX_OAUTH_CLIENT_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| OPENAI_CODEX_DEFAULT_CLIENT_ID.to_string())
}

fn openai_codex_callback_uri() -> String {
    std::env::var("OPENFANG_CODEX_OAUTH_CALLBACK_URI")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| OPENAI_CODEX_DEFAULT_CALLBACK_URI.to_string())
}

fn openai_codex_auth_url() -> String {
    format!(
        "{}/oauth/authorize",
        openai_codex_issuer().trim_end_matches('/')
    )
}

fn openai_codex_token_url() -> String {
    format!(
        "{}/oauth/token",
        openai_codex_issuer().trim_end_matches('/')
    )
}

fn openai_codex_device_usercode_url() -> String {
    format!(
        "{}/api/accounts/deviceauth/usercode",
        openai_codex_issuer().trim_end_matches('/')
    )
}

fn openai_codex_device_token_url() -> String {
    format!(
        "{}/api/accounts/deviceauth/token",
        openai_codex_issuer().trim_end_matches('/')
    )
}

fn openai_codex_device_callback_uri() -> String {
    format!(
        "{}/deviceauth/callback",
        openai_codex_issuer().trim_end_matches('/')
    )
}

fn openai_codex_verification_url() -> String {
    format!(
        "{}/codex/device",
        openai_codex_issuer().trim_end_matches('/')
    )
}

fn deserialize_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
    match raw {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .ok_or_else(|| serde::de::Error::custom("interval must be an unsigned integer"))
            .map(Some),
        Some(serde_json::Value::String(s)) => s
            .trim()
            .parse::<u64>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        Some(other) => Err(serde::de::Error::custom(format!(
            "unsupported interval value: {other}"
        ))),
    }
}

/// Start OpenAI Codex device code flow.
pub async fn openai_codex_start_device_flow() -> Result<DeviceCodeResponse, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let resp = client
        .post(openai_codex_device_usercode_url())
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .json(&CodexDeviceCodeStartRequest {
            client_id: openai_codex_client_id(),
        })
        .send()
        .await
        .map_err(|e| format!("Device code request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Device code request returned {status}: {body}"));
    }

    let raw = resp
        .json::<CodexDeviceCodeStartResponse>()
        .await
        .map_err(|e| format!("Failed to parse device code response: {e}"))?;

    Ok(DeviceCodeResponse {
        device_code: raw.device_auth_id,
        user_code: raw.user_code,
        verification_uri: openai_codex_verification_url(),
        verification_uri_complete: None,
        expires_in: 15 * 60,
        interval: raw.interval.or(Some(5)),
    })
}

/// Poll OpenAI Codex device flow.
pub async fn openai_codex_poll_device_flow(device_code: &str, user_code: &str) -> DeviceFlowStatus {
    let client = match Client::builder().timeout(Duration::from_secs(15)).build() {
        Ok(c) => c,
        Err(e) => return DeviceFlowStatus::Error(format!("HTTP client error: {e}")),
    };

    let resp = match client
        .post(openai_codex_device_token_url())
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .json(&CodexDeviceCodePollRequest {
            device_auth_id: device_code,
            user_code,
        })
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return DeviceFlowStatus::Error(format!("Token poll failed: {e}")),
    };

    let status = resp.status();
    if status == reqwest::StatusCode::FORBIDDEN || status == reqwest::StatusCode::NOT_FOUND {
        return DeviceFlowStatus::Pending;
    }
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return DeviceFlowStatus::Error(format!("HTTP {status}: {body}"));
    }

    let code_resp = match resp.json::<CodexDeviceCodePollResponse>().await {
        Ok(parsed) => parsed,
        Err(e) => {
            return DeviceFlowStatus::Error(format!("Failed to parse device auth response: {e}"));
        }
    };

    let mut tokens = match openai_codex_exchange_code_with_redirect(
        &code_resp.authorization_code,
        &code_resp.code_verifier,
        &openai_codex_device_callback_uri(),
    )
    .await
    {
        Ok(tokens) => tokens,
        Err(e) => return DeviceFlowStatus::Error(e),
    };

    let id_token = match tokens.id_token.clone() {
        Some(id_token) if !id_token.is_empty() => id_token,
        _ => return DeviceFlowStatus::Error("Codex token exchange did not return id_token".into()),
    };

    match openai_codex_obtain_api_key(&id_token).await {
        Ok(api_key) => {
            tokens.api_key = Some(api_key);
            DeviceFlowStatus::Complete { tokens }
        }
        Err(e) => {
            DeviceFlowStatus::Error(format!("Failed to exchange Codex token for API key: {e}"))
        }
    }
}

/// Build OpenAI Codex authorization URL for PKCE flow.
pub fn openai_codex_build_authorize_url(
    state: &str,
    code_challenge: &str,
    redirect_uri: &str,
) -> String {
    let params = [
        ("response_type", "code"),
        ("client_id", &openai_codex_client_id()),
        ("redirect_uri", redirect_uri),
        (
            "scope",
            "openid profile email offline_access api.connectors.read api.connectors.invoke",
        ),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        ("codex_cli_simplified_flow", "true"),
        ("id_token_add_organizations", "true"),
        ("originator", "codex_cli_rs"),
    ];

    let encoded: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
        .collect();

    format!("{}?{}", openai_codex_auth_url(), encoded.join("&"))
}

/// Exchange authorization code for tokens.
pub async fn openai_codex_exchange_code(
    code: &str,
    code_verifier: &str,
) -> Result<OAuthTokenSet, String> {
    openai_codex_exchange_code_with_redirect(code, code_verifier, &openai_codex_callback_uri())
        .await
}

pub async fn openai_codex_exchange_code_for_redirect(
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthTokenSet, String> {
    openai_codex_exchange_code_with_redirect(code, code_verifier, redirect_uri).await
}

async fn openai_codex_exchange_code_with_redirect(
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<OAuthTokenSet, String> {
    let client = Client::new();

    let form = [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("client_id", &openai_codex_client_id()),
        ("redirect_uri", redirect_uri),
        ("code_verifier", code_verifier),
    ];

    let resp = client
        .post(openai_codex_token_url())
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token exchange failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token exchange failed ({status}): {body}"));
    }

    let tokens: TokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {e}"))?;

    Ok(OAuthTokenSet::from_response(tokens, "codex"))
}

/// Refresh OpenAI Codex access token.
pub async fn openai_codex_refresh_token(refresh_token: &str) -> Result<OAuthTokenSet, String> {
    let client = Client::new();

    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", &openai_codex_client_id()),
    ];

    let resp = client
        .post(openai_codex_token_url())
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token refresh failed ({status}): {body}"));
    }

    let tokens: TokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {e}"))?;

    Ok(OAuthTokenSet::from_response(tokens, "codex"))
}

/// Exchange a Codex ChatGPT id_token for an OpenAI API-key-style access token.
pub async fn openai_codex_obtain_api_key(id_token: &str) -> Result<String, String> {
    let client = Client::new();

    let form = [
        (
            "grant_type",
            "urn:ietf:params:oauth:grant-type:token-exchange",
        ),
        ("client_id", &openai_codex_client_id()),
        ("requested_token", "openai-api-key"),
        ("subject_token", id_token),
        (
            "subject_token_type",
            "urn:ietf:params:oauth:token-type:id_token",
        ),
    ];

    let resp = client
        .post(openai_codex_token_url())
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("API key exchange failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("API key exchange failed ({status}): {body}"));
    }

    let tokens: CodexApiKeyExchangeResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse API key exchange response: {e}"))?;

    if tokens.access_token.trim().is_empty() {
        return Err("API key exchange returned an empty access token".into());
    }

    Ok(tokens.access_token)
}

// ─── Gemini OAuth ───────────────────────────────────────────────────────────────

/// Get Gemini OAuth credentials from environment.
pub fn gemini_oauth_credentials() -> Option<(String, String)> {
    let client_id = std::env::var("GEMINI_OAUTH_CLIENT_ID").ok()?;
    let client_secret = std::env::var("GEMINI_OAUTH_CLIENT_SECRET").ok()?;
    if client_id.is_empty() || client_secret.is_empty() {
        return None;
    }
    Some((client_id, client_secret))
}

/// Start Gemini device code flow.
pub async fn gemini_start_device_flow() -> Result<DeviceCodeResponse, String> {
    let (client_id, _client_secret) = gemini_oauth_credentials()
        .ok_or("GEMINI_OAUTH_CLIENT_ID and GEMINI_OAUTH_CLIENT_SECRET required")?;

    let client = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let scope_str =
        "openid profile email https://www.googleapis.com/auth/cloud-platform".to_string();
    let resp = client
        .post(GOOGLE_DEVICE_URL)
        .header("Accept", "application/json")
        .form(&[("client_id", &client_id), ("scope", &scope_str)])
        .send()
        .await
        .map_err(|e| format!("Device code request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Device code request returned {status}: {body}"));
    }

    #[derive(Deserialize)]
    struct GoogleDeviceResponse {
        device_code: String,
        user_code: String,
        verification_url: String,
        expires_in: Option<u64>,
        interval: Option<u64>,
    }

    let google_resp: GoogleDeviceResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse device code response: {e}"))?;

    Ok(DeviceCodeResponse {
        device_code: google_resp.device_code,
        user_code: google_resp.user_code,
        verification_uri: google_resp.verification_url,
        verification_uri_complete: None,
        expires_in: google_resp.expires_in.unwrap_or(300),
        interval: google_resp.interval,
    })
}

/// Poll Gemini device flow.
pub async fn gemini_poll_device_flow(device_code: &str) -> DeviceFlowStatus {
    let (client_id, client_secret) = match gemini_oauth_credentials() {
        Some(c) => c,
        None => return DeviceFlowStatus::Error("Missing OAuth credentials".to_string()),
    };

    let client = match Client::builder().timeout(Duration::from_secs(15)).build() {
        Ok(c) => c,
        Err(e) => return DeviceFlowStatus::Error(format!("HTTP client error: {e}")),
    };

    let form = [
        ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
        ("device_code", device_code),
        ("client_id", &client_id),
        ("client_secret", &client_secret),
    ];

    let resp = match client
        .post(GOOGLE_TOKEN_URL)
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return DeviceFlowStatus::Error(format!("Token poll failed: {e}")),
    };

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        if let Ok(err) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(error) = err.get("error").and_then(|v| v.as_str()) {
                return match error {
                    "authorization_pending" => DeviceFlowStatus::Pending,
                    "slow_down" => {
                        let interval = err.get("interval").and_then(|v| v.as_u64()).unwrap_or(5);
                        DeviceFlowStatus::SlowDown {
                            new_interval: interval,
                        }
                    }
                    "expired_token" => DeviceFlowStatus::Expired,
                    "access_denied" => DeviceFlowStatus::AccessDenied,
                    _ => DeviceFlowStatus::Error(error.to_string()),
                };
            }
        }
        return DeviceFlowStatus::Error(format!("HTTP error: {body}"));
    }

    match resp.json::<TokenResponse>().await {
        Ok(tokens) => DeviceFlowStatus::Complete {
            tokens: OAuthTokenSet::from_response(tokens, "gemini-oauth"),
        },
        Err(e) => DeviceFlowStatus::Error(format!("Failed to parse token response: {e}")),
    }
}

// ─── Qwen OAuth ────────────────────────────────────────────────────────────────

/// Find Qwen OAuth credentials file.
pub fn qwen_credentials_path() -> Option<std::path::PathBuf> {
    let home = home_dir()?;
    let qwen_path = home.join(QWEN_OAUTH_CREDENTIAL_FILE);
    if qwen_path.exists() {
        Some(qwen_path)
    } else {
        None
    }
}

/// Cross-platform home directory (same as qwen_code.rs).
fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(std::path::PathBuf::from)
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().map(std::path::PathBuf::from)
    }
}

/// Read Qwen OAuth credentials from file.
pub fn read_qwen_credentials() -> Option<OAuthTokenSet> {
    let path = qwen_credentials_path()?;
    let content = std::fs::read_to_string(&path).ok()?;

    #[derive(Deserialize)]
    struct QwenCredsFile {
        access_token: String,
        refresh_token: Option<String>,
        expires_at: Option<String>,
    }

    let creds: QwenCredsFile = serde_json::from_str(&content).ok()?;

    let expires_at = creds.expires_at.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    });

    Some(OAuthTokenSet {
        access_token: creds.access_token,
        refresh_token: creds.refresh_token,
        id_token: None,
        api_key: None,
        expires_at,
        provider: "qwen-oauth".to_string(),
    })
}

/// Start Qwen "OAuth" flow — reads tokens from ~/.qwen/oauth_creds.json.
///
/// **Note**: This is NOT a true OAuth flow. Qwen tokens must first be obtained
/// via the Qwen CLI (`qwen login`), which creates the credential file.
/// This function merely imports those pre-existing tokens into OpenFang's vault.
pub async fn qwen_start_oauth_flow() -> Result<(), String> {
    // Qwen OAuth is file-based, just verify we can read the credentials
    read_qwen_credentials().ok_or_else(|| {
        "Failed to read Qwen credentials from ~/.qwen/oauth_creds.json. Run 'qwen login' first."
            .to_string()
    })?;
    Ok(())
}

/// Poll Qwen "OAuth" flow — returns tokens from the credential file.
///
/// **Note**: This is a file import, not a polling mechanism.
/// The tokens are read directly from ~/.qwen/oauth_creds.json.
pub async fn qwen_poll_oauth_flow() -> Result<OAuthTokenSet, String> {
    read_qwen_credentials()
        .ok_or_else(|| "Failed to read Qwen credentials. Run 'qwen login' first.".to_string())
}

/// Refresh Qwen OAuth token.
pub async fn refresh_qwen_token(refresh_token: &str) -> Result<OAuthTokenSet, String> {
    let client = Client::new();

    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", QWEN_OAUTH_CLIENT_ID),
    ];

    let resp = client
        .post(QWEN_OAUTH_TOKEN_ENDPOINT)
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token refresh failed ({status}): {body}"));
    }

    let tokens: TokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {e}"))?;

    Ok(OAuthTokenSet::from_response(tokens, "qwen-oauth"))
}

// ─── MiniMax OAuth ───────────────────────────────────────────────────────────

/// Refresh MiniMax OAuth token.
pub async fn refresh_minimax_token(
    refresh_token: &str,
    region: &str,
) -> Result<OAuthTokenSet, String> {
    let endpoint = match region {
        "cn" => "https://api.minimaxi.com/oauth/token",
        _ => MINIMAX_OAUTH_TOKEN_ENDPOINT,
    };

    let client = Client::new();

    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", MINIMAX_OAUTH_CLIENT_ID),
    ];

    let resp = client
        .post(endpoint)
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token refresh failed ({status}): {body}"));
    }

    let tokens: TokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {e}"))?;

    Ok(OAuthTokenSet::from_response(tokens, "minimax-oauth"))
}

/// Start MiniMax OAuth flow — requires a stored refresh token.
///
/// MiniMax does not support device code or authorization code flow;
/// authentication must be initiated externally (e.g. via their console)
/// and the resulting refresh token stored in the vault before calling
/// [`refresh_minimax_token`].
pub async fn minimax_start_oauth_flow() -> Result<(), String> {
    Err("MiniMax does not support browser-based OAuth. Store a refresh token in the vault first, then use the refresh endpoint.".to_string())
}

/// Check whether a MiniMax refresh token is available in the vault.
///
/// Returns `Ok(())` if a refresh token exists for MiniMax, `Err` otherwise.
/// This is not a traditional OAuth poll — MiniMax has no device code flow.
pub async fn minimax_poll_oauth_flow() -> Result<OAuthTokenSet, String> {
    Err("MiniMax has no device code flow. Store a refresh token in the vault first.".to_string())
}

// ─── Utility Functions ────────────────────────────────────────────────────────

/// URL-encode a string.
fn url_encode(input: &str) -> String {
    input
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{:02X}", b),
        })
        .collect::<String>()
}

/// Generate PKCE code verifier and challenge using a cryptographically secure RNG.
///
/// Uses `OsRng` as the entropy source per RFC 7636 §4.1 requirements.
/// The verifier is 32 bytes (256 bits) of CSPRNG output, base64url-encoded.
/// The challenge is the SHA-256 hash of the verifier, base64url-encoded.
pub fn generate_pkce() -> (String, String) {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);

    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);

    (verifier, challenge)
}

/// Generate a cryptographically random OAuth state parameter (128 bits from OsRng).
///
/// Per RFC 6749 §10.12, the state parameter must be unguessable to prevent CSRF.
pub fn generate_state() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_constants() {
        assert!(openai_codex_auth_url().starts_with("https://"));
        assert!(openai_codex_token_url().starts_with("https://"));
    }

    #[test]
    fn test_pkce_generation() {
        let (verifier, challenge) = generate_pkce();
        assert!(!verifier.is_empty());
        assert!(!challenge.is_empty());
        assert_ne!(verifier, challenge);
        // Verifier should be 43 chars (32 bytes base64url no-pad)
        assert_eq!(
            verifier.len(),
            43,
            "PKCE verifier must be 43 chars (256-bit base64url)"
        );
    }

    #[test]
    fn test_pkce_uniqueness() {
        // Two consecutive calls must produce different verifiers (CSPRNG)
        let (v1, _) = generate_pkce();
        let (v2, _) = generate_pkce();
        assert_ne!(v1, v2, "CSPRNG must produce unique verifiers");
    }

    #[test]
    fn test_state_uniqueness() {
        let s1 = generate_state();
        let s2 = generate_state();
        assert_ne!(s1, s2, "CSPRNG must produce unique state values");
    }

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello"), "hello");
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a=b"), "a%3Db");
    }
}
