"""Connectors for the hosted LLMs, written with the standard library only.

No openai / anthropic / google SDKs: each provider is one HTTPS POST with
urllib. That keeps requirements.txt unchanged and makes every request
visible in one place.

Keys come from Streamlit secrets, then environment variables, then .env:
  OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY (or CLAUDE_API_KEY)
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The models compared. Edit the ids here to swap models; the app and the
# evaluation runner both read this table.
MODELS = {
    "gpt-5.4-mini": {"provider": "openai", "id": "gpt-5.4-mini",
                     "label": "GPT-5.4 mini (OpenAI)", "size": "small hosted"},
    "gemini-3.8-flash": {"provider": "gemini", "id": "gemini-3.8-flash",
                         "label": "Gemini 3.8 Flash (Google)", "size": "small hosted"},
    "claude-haiku-5-5": {"provider": "anthropic", "id": "claude-haiku-5-5",
                         "label": "Claude Haiku 5.5 (Anthropic)", "size": "small hosted"},
    "gemma-4-26b-a4b-it": {"provider": "gemini", "id": "gemma-4-26b-a4b-it",
                           "label": "Gemma 4 26B-A4B (open weights)", "size": "open, ~4B active"},
    "physics-slm": {"provider": "slm", "id": "physics-slm",
                    "label": "PhysicsSLM (from scratch)", "size": "7.6K params"},
}
# Larger models, selectable in the app for a small-vs-large comparison.
EXTRA_MODELS = {
    "gpt-5.5": {"provider": "openai", "id": "gpt-5.5",
                "label": "GPT-5.5 (OpenAI)", "size": "large hosted"},
    "claude-sonnet-5-5": {"provider": "anthropic", "id": "claude-sonnet-5-5",
                          "label": "Claude Sonnet 5.5 (Anthropic)", "size": "large hosted"},
    "gemini-3.1-pro-preview": {"provider": "gemini", "id": "gemini-3.1-pro-preview",
                               "label": "Gemini 3.1 Pro (Google)", "size": "large hosted"},
}
ALL_MODELS = {**MODELS, **EXTRA_MODELS}

KEY_NAMES = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
}


@dataclass
class Reply:
    text: str
    model: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    dropped_params: tuple = ()   # parameters the API refused (e.g. temperature)
    error: str = ""


def _read_dotenv():
    env = {}
    path = HERE / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_KEY_CACHE = {}


def get_key(provider):
    """Look a key up once and remember it (so worker threads never touch
    st.secrets)."""
    if provider not in _KEY_CACHE:
        _KEY_CACHE[provider] = _find_key(provider)
    return _KEY_CACHE[provider]


def _find_key(provider):
    names = KEY_NAMES[provider]
    try:  # Streamlit Cloud secrets
        import streamlit as st
        for n in names:
            if n in st.secrets:
                return str(st.secrets[n])
    except Exception:  # noqa: BLE001 — no streamlit or no secrets file
        pass
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    dot = _read_dotenv()
    for n in names:
        if dot.get(n):
            return dot[n]
    return ""


def has_key(provider):
    return provider == "slm" or bool(get_key(provider))


class APIError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status, self.body = status, body


def _post(url, headers, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise APIError(e.code, e.read().decode("utf-8", "replace")) from None


# Parameters that some models refuse (reasoning models often reject
# temperature). On a 400 that names one, it is dropped and the call is
# retried, and the reply records which parameters were dropped.
OPTIONAL = {
    "openai": ["temperature", "reasoning_effort", "top_p"],
    "gemini": ["temperature", "thinkingConfig", "systemInstruction", "topP"],
    "anthropic": ["temperature", "top_p"],
}


ALIASES = {
    "systemInstruction": ("systeminstruction", "system_instruction", "developer instruction"),
    "thinkingConfig": ("thinkingconfig", "thinking_config", "thinking_level", "thinkinglevel", "thinking"),
    "reasoning_effort": ("reasoning_effort", "reasoning.effort", "reasoning effort"),
    "temperature": ("temperature",),
    "top_p": ("top_p", "top-p"),
    "topP": ("topp", "top_p", "top-p"),
}


def _mentions(body, name):
    low = body.lower()
    return any(a in low for a in ALIASES.get(name, (name.lower(),)))


def _drop(payload, name):
    if name in payload:
        payload.pop(name)
        return True
    gc = payload.get("generationConfig", {})
    if name in gc:
        gc.pop(name)
        return True
    return False


def _openai(model_id, system, user, temperature, max_tokens, key, top_p=None):
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": "low",
    }
    if top_p is not None:
        payload["top_p"] = top_p
    return payload, ("https://api.openai.com/v1/chat/completions",
                     {"Authorization": f"Bearer {key}"})


def _openai_parse(data):
    text = data["choices"][0]["message"].get("content") or ""
    u = data.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _gemini(model_id, system, user, temperature, max_tokens, key, top_p=None):
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature,
                             "maxOutputTokens": max_tokens,
                             "thinkingConfig": {"thinkingLevel": "low"}},
    }
    if top_p is not None:
        payload["generationConfig"]["topP"] = top_p
    if model_id.startswith("gemma"):
        # Gemma has no system role and no thinking config on this API:
        # the system prompt goes at the top of the user turn instead.
        payload.pop("systemInstruction")
        payload["generationConfig"].pop("thinkingConfig")
        payload["contents"][0]["parts"][0]["text"] = system + "\n\n" + user
    return payload, (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        {"x-goog-api-key": key})


def _gemini_parse(data):
    cands = data.get("candidates") or []
    if not cands:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
        return f"[No answer returned: {reason}]", 0, 0
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = data.get("usageMetadata", {})
    return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def _anthropic(model_id, system, user, temperature, max_tokens, key, top_p=None):
    payload = {
        "model": model_id,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    return payload, ("https://api.anthropic.com/v1/messages",
                     {"x-api-key": key, "anthropic-version": "2023-06-01"})


def _anthropic_parse(data):
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    u = data.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


BUILDERS = {"openai": (_openai, _openai_parse),
            "gemini": (_gemini, _gemini_parse),
            "anthropic": (_anthropic, _anthropic_parse)}


def chat(model_key, system, user, temperature=0.2, max_tokens=1500, retries=2, top_p=None):
    """One system + user turn. Returns a Reply; never raises for API errors
    (the error is in Reply.error) so a batch run keeps going."""
    spec = ALL_MODELS[model_key]
    provider = spec["provider"]
    if provider == "slm":
        raise ValueError("PhysicsSLM is local; use slm_adapter.answer().")
    key = get_key(provider)
    if not key:
        return Reply("", model_key, 0.0, error=f"No API key for {provider}.")
    build, parse = BUILDERS[provider]
    payload, (url, headers) = build(spec["id"], system, user, temperature, max_tokens, key, top_p)
    dropped = []
    start = time.time()
    attempt = 0
    while True:
        try:
            data = _post(url, headers, payload)
            text, tin, tout = parse(data)
            return Reply(text.strip(), model_key, time.time() - start, tin, tout,
                         tuple(dropped))
        except APIError as e:
            if e.status == 400:
                for name in OPTIONAL[provider]:
                    if _mentions(e.body, name) and _drop(payload, name):
                        if name == "systemInstruction":  # fold system into user
                            part = payload["contents"][0]["parts"][0]
                            part["text"] = system + "\n\n" + part["text"]
                        dropped.append(name)
                        break
                else:
                    return Reply("", model_key, time.time() - start, error=str(e))
                continue
            if e.status in (429, 500, 502, 503, 529) and attempt < retries:
                attempt += 1
                time.sleep(20 * attempt)
                continue
            return Reply("", model_key, time.time() - start, error=str(e))
        except Exception as e:  # noqa: BLE001 — network errors
            if attempt < retries:
                attempt += 1
                time.sleep(5)
                continue
            return Reply("", model_key, time.time() - start,
                         error=f"{type(e).__name__}: {e}")


_ANSWER = re.compile(r"(?i:answer)\**\s*(?:(?i:is)\s*)?[:\-]?\s*\**\s*\(?\s*([A-E])\b")


def parse_choice(text):
    """The letter from the last 'Answer: X' line, or '' if none."""
    found = _ANSWER.findall(text or "")
    return found[-1].upper() if found else ""
