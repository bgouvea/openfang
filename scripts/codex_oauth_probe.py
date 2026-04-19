#!/usr/bin/env python3
"""
Standalone Codex OAuth probe.

This script reproduces the official Codex browser OAuth flow as closely as
possible, but outside OpenFang:

- localhost callback on http://localhost:1455/auth/callback
- PKCE + state
- /oauth/authorize
- /oauth/token authorization_code exchange
- /oauth/token token-exchange to obtain an API-key-style token
- optional smoke test against the OpenAI API

No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import websocket


DEFAULT_ISSUER = "https://auth.openai.com"
DEFAULT_CLIENT_ID = ""
DEFAULT_PORT = 1455
DEFAULT_ORIGINATOR = "codex_cli_rs"
DEFAULT_SMOKE_MODEL = "gpt-4.1-mini"


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def generate_pkce() -> tuple[str, str]:
    verifier = b64url(secrets.token_bytes(32))
    challenge = b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def generate_state() -> str:
    return b64url(secrets.token_bytes(16))


def form_post(url: str, payload: Dict[str, str], timeout: int = 30) -> Dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def json_post(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=merged_headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def decode_jwt_claims(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
    except Exception:
        return {}


def find_open_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


@dataclass
class CallbackResult:
    code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None


class CallbackState:
    def __init__(self, expected_state: str) -> None:
        self.expected_state = expected_state
        self.event = threading.Event()
        self.result = CallbackResult()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    callback_state: CallbackState

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _write_html(self, status: int, title: str, message: str) -> None:
        body = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:680px;margin:48px auto;"
            "padding:0 20px;line-height:1.5;color:#111}.ok{color:#166534}.err{color:#991b1b}"
            "</style></head><body>"
            f"<h1>{title}</h1><p>{message}</p>"
            "<p>You can return to the terminal and close this tab.</p>"
            "</body></html>"
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self._write_html(404, "Not Found", "<span class='err'>Unknown path.</span>")
            return

        params = urllib.parse.parse_qs(parsed.query)
        state = params.get("state", [""])[0]
        code = params.get("code", [""])[0]
        error = params.get("error", [""])[0]
        error_description = params.get("error_description", [""])[0]

        if state != self.callback_state.expected_state:
            self.callback_state.result = CallbackResult(
                error="state_mismatch",
                error_description="OAuth state mismatch",
                state=state,
            )
            self.callback_state.event.set()
            self._write_html(
                400,
                "Codex Login Failed",
                "<span class='err'>OAuth state mismatch.</span>",
            )
            return

        self.callback_state.result = CallbackResult(
            code=code or None,
            state=state or None,
            error=error or None,
            error_description=error_description or None,
        )
        self.callback_state.event.set()

        if error:
            self._write_html(
                400,
                "Codex Login Failed",
                f"<span class='err'>{urllib.parse.unquote(error_description or error)}</span>",
            )
            return

        if not code:
            self._write_html(
                400,
                "Codex Login Failed",
                "<span class='err'>Missing authorization code.</span>",
            )
            return

        self._write_html(
            200,
            "Codex Login Complete",
            "<span class='ok'>Authorization code received successfully.</span>",
        )


def run_callback_server(expected_state: str, port: int) -> tuple[HTTPServer, CallbackState, threading.Thread]:
    callback_state = CallbackState(expected_state)
    handler_cls = type(
        "OAuthCallbackHandlerBound",
        (OAuthCallbackHandler,),
        {"callback_state": callback_state},
    )
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, callback_state, thread


def build_authorize_url(
    issuer: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    originator: str,
) -> str:
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access api.connectors.read api.connectors.invoke",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": originator,
        }
    )
    return issuer.rstrip("/") + "/oauth/authorize?" + query


def exchange_code_for_tokens(
    issuer: str,
    client_id: str,
    redirect_uri: str,
    code: str,
    code_verifier: str,
) -> Dict[str, Any]:
    return form_post(
        issuer.rstrip("/") + "/oauth/token",
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        },
    )


def obtain_api_key(issuer: str, client_id: str, id_token: str) -> Dict[str, Any]:
    return form_post(
        issuer.rstrip("/") + "/oauth/token",
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": client_id,
            "requested_token": "openai-api-key",
            "subject_token": id_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        },
    )


def smoke_test_models(api_key: str) -> Dict[str, Any]:
    return http_get_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def smoke_test_inference(bearer_token: str, model: str) -> Dict[str, Any]:
    return http_post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
            "temperature": 0,
        },
        headers={"Authorization": f"Bearer {bearer_token}"},
        timeout=60,
    )


def websocket_url_for_codex(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/responses"
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + "/responses"
    if base.startswith("wss://") or base.startswith("ws://"):
        return base + "/responses"
    return "wss://" + base + "/responses"


def websocket_probe(
    bearer_token: str,
    model: str,
    base_url: str,
    previous_response_id: Optional[str] = None,
    same_socket: bool = True,
) -> Dict[str, Any]:
    session_id = secrets.token_hex(16)
    installation_id = secrets.token_hex(16)
    prompt_cache_key = session_id
    headers = [
        f"Authorization: Bearer {bearer_token}",
        "originator: codex_cli_rs",
        "OpenAI-Beta: responses_websockets=2026-02-06",
        f"x-client-request-id: {session_id}",
        f"session_id: {session_id}",
    ]
    url = websocket_url_for_codex(base_url)

    def connect() -> websocket.WebSocket:
        return websocket.create_connection(url, header=headers, timeout=30)

    def send_and_drain(ws: websocket.WebSocket, payload: Dict[str, Any]) -> Dict[str, Any]:
        ws.send(json.dumps(payload))
        out: Dict[str, Any] = {"events": [], "response_id": None}
        while True:
            raw = ws.recv()
            data = json.loads(raw)
            out["events"].append(data.get("type"))
            if data.get("type") == "response.created":
                out["response_id"] = data.get("response", {}).get("id")
            if data.get("type") in ("response.completed", "response.failed", "error"):
                if data.get("type") == "response.completed":
                    out["response_id"] = data.get("response", {}).get("id") or out["response_id"]
                out["last"] = data
                return out

    first_payload = {
        "type": "response.create",
        "model": model,
        "instructions": "You are a helpful assistant.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with the single word ALPHA."}],
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
        "include": [],
        "prompt_cache_key": prompt_cache_key,
        "client_metadata": {"x-codex-installation-id": installation_id},
    }

    second_payload = {
        "type": "response.create",
        "model": model,
        "instructions": "You are a helpful assistant.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with the single word BETA."}],
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "stream": True,
        "include": [],
        "prompt_cache_key": prompt_cache_key,
        "client_metadata": {"x-codex-installation-id": installation_id},
    }

    if previous_response_id:
        second_payload["previous_response_id"] = previous_response_id

    first_ws = connect()
    try:
        first = send_and_drain(first_ws, first_payload)
        response_id = previous_response_id or first.get("response_id")
        if response_id:
            second_payload["previous_response_id"] = response_id
        if same_socket:
            second = send_and_drain(first_ws, second_payload)
        else:
            second_ws = connect()
            try:
                second = send_and_drain(second_ws, second_payload)
            finally:
                second_ws.close()
        return {"first": first, "second": second}
    finally:
        first_ws.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Codex OAuth probe")
    parser.add_argument("--issuer", default=DEFAULT_ISSUER)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--originator", default=DEFAULT_ORIGINATOR)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--test-inference", action="store_true")
    parser.add_argument("--test-websocket", action="store_true")
    parser.add_argument("--test-websocket-new-socket", action="store_true")
    parser.add_argument("--smoke-model", default=DEFAULT_SMOKE_MODEL)
    parser.add_argument("--chatgpt-base-url", default="https://chatgpt.com/backend-api/codex")
    parser.add_argument("--output-auth-json", default="")
    args = parser.parse_args()

    port = find_open_port(args.port)
    redirect_uri = f"http://localhost:{port}/auth/callback"
    state = generate_state()
    code_verifier, code_challenge = generate_pkce()
    authorize_url = build_authorize_url(
        issuer=args.issuer,
        client_id=args.client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        state=state,
        originator=args.originator,
    )

    server, callback_state, thread = run_callback_server(state, port)

    print(f"callback_uri={redirect_uri}")
    print(f"authorize_url={authorize_url}")
    print(f"originator={args.originator}")
    print("waiting_for_callback=true")

    if not args.no_open_browser:
        webbrowser.open(authorize_url)

    if not callback_state.event.wait(args.timeout):
        server.shutdown()
        thread.join(timeout=2)
        print("result=timeout")
        return 1

    server.shutdown()
    thread.join(timeout=2)

    if callback_state.result.error:
        print("result=oauth_error")
        print(f"oauth_error={callback_state.result.error}")
        if callback_state.result.error_description:
            print(f"oauth_error_description={callback_state.result.error_description}")
        return 1

    if not callback_state.result.code:
        print("result=missing_code")
        return 1

    try:
        tokens = exchange_code_for_tokens(
            issuer=args.issuer,
            client_id=args.client_id,
            redirect_uri=redirect_uri,
            code=callback_state.result.code,
            code_verifier=code_verifier,
        )
        id_token = tokens.get("id_token") or ""
        access_token = tokens.get("access_token") or ""
        refresh_token = tokens.get("refresh_token") or ""
        claims = decode_jwt_claims(id_token)
        org_id = claims.get("organization_id")
        account_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")

        print("result=token_exchange_ok")
        print(f"id_token_has_org={bool(org_id)}")
        print(f"id_token_org={org_id or ''}")
        print(f"chatgpt_account_id={account_id or ''}")

        api_key = ""
        api_key_exchange_error = ""
        try:
            api_key_payload = obtain_api_key(args.issuer, args.client_id, id_token)
            api_key = api_key_payload.get("access_token") or ""
            if api_key:
                print("api_key_exchange=ok")
            else:
                print("api_key_exchange=empty")
        except urllib.error.HTTPError as exc:
            api_key_exchange_error = exc.read().decode(errors="replace")[:1200]
            print(f"api_key_exchange=http_{exc.code}")
            print(f"api_key_exchange_body={api_key_exchange_error}")

        if args.output_auth_json:
            output_path = Path(args.output_auth_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "OPENAI_API_KEY": api_key or None,
                        "tokens": {
                            "id_token": id_token,
                            "access_token": access_token,
                            "refresh_token": refresh_token,
                        },
                        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=2,
                )
            )
            print(f"auth_json_written={output_path}")

        if args.list_models:
            try:
                models_payload = smoke_test_models(api_key or access_token)
                count = len(models_payload.get("data", []))
                print(f"list_models=ok count={count}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:1200]
                print(f"list_models=http_{exc.code}")
                print(f"list_models_body={body}")

        if args.test_inference:
            bearer_label = "api_key" if api_key else "access_token"
            bearer_token = api_key or access_token
            try:
                inference = smoke_test_inference(bearer_token, args.smoke_model)
                choice = ((inference.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                print(f"inference_bearer={bearer_label}")
                print(f"inference_model={args.smoke_model}")
                print(f"inference_preview={choice[:120]}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:1200]
                print(f"inference_bearer={bearer_label}")
                print(f"inference_model={args.smoke_model}")
                print(f"inference_http_status={exc.code}")
                print(f"inference_http_body={body}")

        if args.test_websocket or args.test_websocket_new_socket:
            bearer_token = access_token
            try:
                ws_result = websocket_probe(
                    bearer_token=bearer_token,
                    model=args.smoke_model,
                    base_url=args.chatgpt_base_url,
                    same_socket=not args.test_websocket_new_socket,
                )
                print(
                    "websocket_same_socket="
                    + ("false" if args.test_websocket_new_socket else "true")
                )
                print("websocket_probe_result=" + json.dumps(ws_result))
            except Exception as exc:
                print(f"websocket_probe_error={type(exc).__name__}: {exc}")

        return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"http_error_status={exc.code}")
        print(f"http_error_body={body[:1200]}")
        return 1
    except Exception as exc:
        print(f"unexpected_error={type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
