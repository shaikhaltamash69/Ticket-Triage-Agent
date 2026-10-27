"""LLM client with native tool calling (Groq / OpenAI / Anthropic / Ollama)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            # This is a self-contained take-home project: its .env is the
            # configured source of truth, so an old inherited shell value must
            # not silently select a different provider key or model.
            os.environ[_k.strip()] = _v.strip()


class ChatResult:
    """Normalized chat response. tool_calls use OpenAI-style ids."""

    def __init__(self, content: str | None = None, tool_calls: list[dict] | None = None):
        self.content = content or ""
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _gemini_schema(schema: Any) -> dict[str, Any]:
    """Convert the small OpenAI JSON-Schema subset used by our tools to Gemini.

    Gemini FunctionDeclaration accepts OpenAPI-style schema types as uppercase
    strings and does not accept JSON-Schema union type arrays (e.g. string|null).
    The agent already treats optional values as optional, so null is removed from
    those unions while preserving the field description and enum.
    """
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}
    result: dict[str, Any] = {}
    raw_type = schema.get("type", "object")
    if isinstance(raw_type, list):
        raw_type = next((value for value in raw_type if value != "null"), "string")
    type_map = {
        "object": "OBJECT", "array": "ARRAY", "string": "STRING",
        "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN",
    }
    result["type"] = type_map.get(str(raw_type).lower(), "STRING")
    if schema.get("description"):
        result["description"] = schema["description"]
    if isinstance(schema.get("enum"), list):
        result["enum"] = [str(value) for value in schema["enum"] if value is not None]
    if result["type"] == "OBJECT":
        result["properties"] = {
            str(key): _gemini_schema(value)
            for key, value in (schema.get("properties") or {}).items()
        }
        if schema.get("required"):
            result["required"] = list(schema["required"])
    elif result["type"] == "ARRAY":
        result["items"] = _gemini_schema(schema.get("items") or {"type": "string"})
    return result


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b",
            "anthropic": "claude-sonnet-4-5",
            "openai": "gpt-4o-mini",
            "groq": "qwen/qwen3.8-27b",
            "gemini": "gemini-3.8-flash",
        }.get(self.provider, "llama3.1:8b"))
        self.calls = 0

    def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = "auto",
    ) -> ChatResult:
        """Native tool calling when tools= is provided. Always returns ChatResult."""
        self.calls += 1
        if self.provider == "groq":
            return self._chat_groq(messages, system, max_tokens, temperature, tools, tool_choice)
        if self.provider == "openai":
            return self._chat_openai(messages, system, max_tokens, temperature, tools, tool_choice)
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, system, max_tokens, temperature, tools, tool_choice)
        if self.provider == "ollama":
            return self._chat_ollama(messages, system, max_tokens, temperature, tools)
        if self.provider == "gemini":
            return self._chat_gemini(messages, system, max_tokens, temperature, tools, tool_choice)
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _chat_gemini(self, messages, system, max_tokens, temperature, tools, tool_choice) -> ChatResult:
        """Gemini REST adapter using its native functionCall/functionResponse parts.

        The agent's internal message format stays OpenAI-like, but this method
        faithfully translates every assistant function call and every matching
        function result to Gemini's native protocol.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env")

        call_names: dict[str, str] = {}
        contents: list[dict[str, Any]] = []
        pending_responses: list[dict[str, Any]] = []

        def flush_responses() -> None:
            if pending_responses:
                contents.append({"role": "user", "parts": list(pending_responses)})
                pending_responses.clear()

        for message in messages:
            role = message.get("role")
            if role == "tool":
                call_id = message.get("tool_call_id", "")
                try:
                    result: Any = json.loads(message.get("content") or "null")
                except json.JSONDecodeError:
                    result = message.get("content") or ""
                pending_responses.append({
                    "functionResponse": {
                        "name": call_names.get(call_id, "unknown_tool"),
                        "response": {"result": result},
                    }
                })
                continue

            flush_responses()
            if role == "assistant":
                parts: list[dict[str, Any]] = []
                if message.get("content"):
                    parts.append({"text": message["content"]})
                for tc in message.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    call_id = tc.get("id", "")
                    name = fn.get("name") or tc.get("name") or ""
                    call_names[call_id] = name
                    parts.append({
                        "functionCall": {
                            "name": name,
                            "args": _parse_args(fn.get("arguments", tc.get("arguments", {}))),
                        }
                    })
                if parts:
                    contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": message.get("content") or ""}]})
        flush_responses()

        declarations = []
        for tool in tools or []:
            fn = tool.get("function", tool)
            declarations.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": _gemini_schema(fn.get("parameters", {"type": "object", "properties": {}})),
            })
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]
            mode = "AUTO"
            allowed = None
            if isinstance(tool_choice, dict):
                forced = ((tool_choice.get("function") or {}).get("name"))
                if forced:
                    mode, allowed = "ANY", [forced]
            config: dict[str, Any] = {"mode": mode}
            if allowed:
                config["allowedFunctionNames"] = allowed
            body["toolConfig"] = {"functionCallingConfig": config}

        encoded_model = urllib.parse.quote(self.model, safe="")
        encoded_key = urllib.parse.urlencode({"key": api_key})
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:generateContent?{encoded_key}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            # Fail fast enough for run.py to record a provider failure without
            # holding a review-ticket run indefinitely.
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Gemini API {exc.code}: {detail}") from exc

        candidate = (payload.get("candidates") or [{}])[0]
        parts = ((candidate.get("content") or {}).get("parts") or [])
        text_parts, calls = [], []
        for index, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            if "functionCall" in part:
                fc = part["functionCall"] or {}
                calls.append({
                    "id": fc.get("id") or f"gemini_call_{self.calls}_{index}",
                    "name": fc.get("name"),
                    "arguments": dict(fc.get("args") or {}),
                })
        return ChatResult(content="".join(text_parts), tool_calls=calls)

    def _chat_groq(self, messages, system, max_tokens, temperature, tools, tool_choice) -> ChatResult:
        import re
        import time
        from groq import Groq, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

        RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
        if not getattr(self, "_groq", None):
            self._groq = Groq(api_key=os.environ.get("GROQ_API_KEY"), timeout=90, max_retries=0)
        msgs = ([{"role": "system", "content": system}] if system else []) + messages

        def _parse_wait(err_msg: str, default_wait: float) -> float:
            m = re.search(r"try again in (?:(\d+)m)?(?:([\d.]+)s)?", err_msg)
            if m and (m.group(1) or m.group(2)):
                mins = float(m.group(1)) if m.group(1) else 0.0
                secs = float(m.group(2)) if m.group(2) else 0.0
                return min(90.0, max(5.0, mins * 60.0 + secs + 2.0))
            return min(60.0, default_wait)

        for attempt in range(20):
            try:
                time.sleep(1.5)
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": msgs,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice or "auto"
                    if getattr(self, "_allow_parallel", True):
                        kwargs["parallel_tool_calls"] = True
                try:
                    resp = self._groq.chat.completions.create(**kwargs)
                except Exception as pe:
                    if "parallel_tool_calls" in str(pe):
                        self._allow_parallel = False
                        kwargs.pop("parallel_tool_calls", None)
                        resp = self._groq.chat.completions.create(**kwargs)
                    else:
                        raise
                msg = resp.choices[0].message
                tcs = []
                for tc in (msg.tool_calls or []):
                    tcs.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": _parse_args(tc.function.arguments),
                    })
                return ChatResult(content=msg.content or "", tool_calls=tcs)
            except RETRYABLE as e:
                err_str = str(e)
                if ("tokens per day" in err_str.lower() or "tpd" in err_str.lower()) and "120b" not in self.model:
                    print(f"    [TPD Daily Quota on {self.model}] Switching to openai/gpt-oss-120b...", flush=True)
                    self.model = "openai/gpt-oss-120b"
                    time.sleep(2)
                    continue
                wait = _parse_wait(err_str, min(60.0, 4.0 * (2 ** attempt)))
                print(f"    [{type(e).__name__}] retry {attempt+1}/20 in {wait:.1f}s: {err_str[:120]}", flush=True)
                time.sleep(wait)
            except Exception as e:
                err_str = str(e)
                retryable = any(x in err_str.lower() for x in ("rate_limit", "429", "timeout", "connection", "503"))
                if retryable and attempt < 19:
                    wait = _parse_wait(err_str, min(60.0, 4.0 * (2 ** attempt)))
                    print(f"    [RetryableException] retry {attempt+1}/20 in {wait:.1f}s: {err_str[:120]}", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("LLM unavailable after 20 retries")

    def _chat_openai(self, messages, system, max_tokens, temperature, tools, tool_choice) -> ChatResult:
        from openai import OpenAI
        client = OpenAI()
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tcs = []
        for tc in (msg.tool_calls or []):
            tcs.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": _parse_args(tc.function.arguments),
            })
        return ChatResult(content=msg.content or "", tool_calls=tcs)

    def _chat_anthropic(self, messages, system, max_tokens, temperature, tools, tool_choice) -> ChatResult:
        import anthropic
        client = anthropic.Anthropic()
        anth_tools = []
        for t in tools or []:
            fn = t.get("function", t)
            anth_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        anth_msgs = []
        for m in messages:
            if m.get("role") == "tool":
                anth_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id"),
                        "content": m.get("content") or "",
                    }],
                })
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name") or tc.get("name"),
                        "input": _parse_args(fn.get("arguments", tc.get("arguments", {}))),
                    })
                anth_msgs.append({"role": "assistant", "content": blocks})
            else:
                anth_msgs.append({"role": m["role"], "content": m.get("content") or ""})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system or "",
            "messages": anth_msgs,
        }
        if anth_tools:
            kwargs["tools"] = anth_tools
        resp = client.messages.create(**kwargs)
        content_text = ""
        tcs = []
        for b in resp.content:
            if getattr(b, "type", "") == "text":
                content_text += b.text
            elif getattr(b, "type", "") == "tool_use":
                tcs.append({"id": b.id, "name": b.name, "arguments": dict(b.input or {})})
        return ChatResult(content=content_text, tool_calls=tcs)

    def _chat_ollama(self, messages, system, max_tokens, temperature, tools) -> ChatResult:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        body_obj: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            body_obj["tools"] = tools
        body = json.dumps(body_obj).encode()
        req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        msg = data.get("message") or {}
        tcs = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or tc
            tcs.append({
                "id": tc.get("id") or f"call_{fn.get('name', 'tool')}_{len(tcs)}",
                "name": fn.get("name"),
                "arguments": _parse_args(fn.get("arguments", {})),
            })
        return ChatResult(content=msg.get("content") or "", tool_calls=tcs)
