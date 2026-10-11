"""Unified chat client for the model-migration study.

Providers (all OpenAI-compatible chat completions):
  openrouter : https://openrouter.ai/api/v1          (paid credit; cost read from the response)
  gemini     : Google AI Studio OpenAI-compatible endpoint (free tier; cost 0)
  groq       : https://api.groq.com/openai/v1         (free tier; cost 0)

Guarantees
  * Keys come from the environment (e.g. loaded from a local .env file); they are never printed or written.
  * Every response is cached on disk (runs/cache/), keyed by provider, model, messages, parameters and a
    repeat salt, so a rerun after a crash never pays twice and repeat k uses its own key.
  * Spend is appended to runs/cost_ledger.json (total_usd, per model) and a job stops itself when its cap
    (MM_JOB_COST_CAP_USD) would be exceeded.
  * Daily free-tier exhaustion raises QuotaExhausted so a job can exit cleanly and resume later.
  * The response's serving provider (OpenRouter) and model fingerprint are recorded for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

import requests

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CACHE = ROOT / "runs" / "cache"
LEDGER = ROOT / "runs" / "cost_ledger.json"
CALLS = ROOT / "runs" / "calls.jsonl"
_LOCK = threading.Lock()
_JOB_SPENT = 0.0
_LAST_CALL: dict[str, list[float]] = {}

BASES = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}
FREE = {"gemini", "groq"}
DEFAULT_RPM = {"openrouter": 120, "gemini": 8, "groq": 25}


class BudgetExceeded(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    pass


class TransientAPIError(RuntimeError):
    """Provider kept failing (5xx/429/network); the episode must not be recorded as a model failure."""


def job_cap() -> float:
    try:
        return float(os.environ.get("MM_JOB_COST_CAP_USD", "0"))
    except ValueError:
        return 0.0


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for i in range(30):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, p)


def _record(model_label: str, cost: float, usage: dict, stage: str) -> None:
    global _JOB_SPENT
    with _LOCK:
        _JOB_SPENT += cost
        led = _read_json(LEDGER, {"total_usd": 0.0, "by_model": {}, "calls": 0})
        led["total_usd"] = round(led.get("total_usd", 0.0) + cost, 6)
        bm = led.setdefault("by_model", {}).setdefault(model_label, {"usd": 0.0, "calls": 0, "in": 0, "out": 0})
        bm["usd"] = round(bm["usd"] + cost, 6)
        bm["calls"] += 1
        bm["in"] += int(usage.get("prompt_tokens") or 0)
        bm["out"] += int(usage.get("completion_tokens") or 0)
        led["calls"] = led.get("calls", 0) + 1
        _write_json(LEDGER, led)


def _throttle(provider: str, rpm: int) -> None:
    """Simple sliding-window limiter shared by all threads in this process."""
    while True:
        with _LOCK:
            now = time.time()
            win = [t for t in _LAST_CALL.get(provider, []) if now - t < 60]
            if len(win) < rpm:
                win.append(now)
                _LAST_CALL[provider] = win
                return
            wait = 60 - (now - win[0]) + 0.05
        time.sleep(max(wait, 0.2))


def chat(spec: dict, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1500,
         salt: str = "", stage: str = "agent", use_cache: bool = True, tools: list | None = None,
         tool_choice: str | None = None) -> dict:
    """spec: one entry of config/models.yaml (label, provider, model, reasoning, rpm, price_in, price_out, extra).

    tools: optional OpenAI-format function tools; the reply's tool calls are returned as
    tool_calls = [{id, name, arguments(str)}] and the raw assistant message as `message` (to append to history).
    Returns {text, tool_calls, message, usage, cost, cached, provider_served, latency_s, finish_reason}."""
    provider = spec["provider"]
    url, key_name = BASES[provider]
    body = {"model": spec["model"], "messages": messages, "max_tokens": max_tokens}
    if temperature is not None and not spec.get("no_temperature"):
        body["temperature"] = temperature
    if spec.get("reasoning"):
        if provider == "openrouter":
            body["reasoning"] = {"effort": spec["reasoning"]} if spec["reasoning"] != "off" else {"enabled": False}
        else:
            body["reasoning_effort"] = spec["reasoning"] if spec["reasoning"] != "off" else "none"
    if provider == "openrouter":
        body["usage"] = {"include": True}
        if spec.get("provider_order"):
            body["provider"] = {"order": spec["provider_order"], "allow_fallbacks": bool(spec.get("allow_fallbacks", True))}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
    body.update(spec.get("extra", {}))
    key = _sha({"p": provider, "b": body, "salt": salt})
    cfile = CACHE / key[:2] / f"{key}.json"
    if use_cache and cfile.exists():
        rec = _read_json(cfile, None)
        if rec:
            rec["cached"] = True
            rec["cost"] = 0.0
            return rec

    cap = job_cap()
    if provider not in FREE:
        est = (spec.get("price_in", 1.0) * 6000 + spec.get("price_out", 4.0) * max_tokens) / 1e6
        if _JOB_SPENT + est > cap + 1e-9:
            raise BudgetExceeded(f"job cap {cap:.2f} USD reached (spent {_JOB_SPENT:.4f})")

    api_key = os.environ.get(key_name, "").strip()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "User-Agent": "mm-research/0.1"}
    if provider == "openrouter":
        headers.update({"HTTP-Referer": "https://localhost/model-migration", "X-Title": "model-migration study"})
    rpm = int(spec.get("rpm") or DEFAULT_RPM[provider])
    last_err = None
    for attempt in range(8):
        _throttle(provider, rpm)
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=240)
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}"
            time.sleep(min(60, 2 ** attempt + random.random()))
            continue
        dt = time.time() - t0
        if r.status_code == 200:
            d = r.json()
            if "choices" not in d or not d["choices"]:
                last_err = f"no choices: {str(d)[:200]}"
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            usage = d.get("usage") or {}
            if provider in FREE:
                cost = 0.0
            elif usage.get("cost") is not None:
                cost = float(usage["cost"])
            else:
                cost = (spec.get("price_in", 0) * (usage.get("prompt_tokens") or 0)
                        + spec.get("price_out", 0) * (usage.get("completion_tokens") or 0)) / 1e6
            tcs = [{"id": tc.get("id") or f"call_{i}", "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments") or "{}"}
                   for i, tc in enumerate(msg.get("tool_calls") or [])]
            amsg = {"role": "assistant", "content": text or ""}
            if msg.get("tool_calls"):
                # pass tool calls back verbatim: keeps provider extras such as Gemini thought signatures
                amsg["tool_calls"] = [dict(tc, type=tc.get("type") or "function", id=tc.get("id") or f"call_{i}")
                                      for i, tc in enumerate(msg["tool_calls"])]
            if msg.get("reasoning_details"):
                amsg["reasoning_details"] = msg["reasoning_details"]   # OpenRouter: preserve reasoning across turns
            rec = {"text": text, "tool_calls": tcs, "message": amsg, "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens",
                                                                         "total_tokens", "cost")},
                   "reasoning_tokens": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")),
                   "cost": cost, "cached": False, "provider_served": d.get("provider"),
                   "model_served": d.get("model"), "fingerprint": d.get("system_fingerprint"),
                   "finish_reason": ch.get("finish_reason"), "latency_s": round(dt, 2), "salt": salt}
            _record(spec["label"], cost, usage, stage)
            if use_cache:
                _write_json(cfile, rec)
            with _LOCK:
                CALLS.parent.mkdir(parents=True, exist_ok=True)
                with CALLS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.time(), "label": spec["label"], "stage": stage, "cost": cost,
                                        "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
                                        "served": d.get("provider"), "lat": round(dt, 2)}) + "\n")
            return rec
        txt = r.text[:400].replace(api_key, "***") if api_key else r.text[:400]
        last_err = f"HTTP {r.status_code}: {txt}"
        if r.status_code == 429:
            low = txt.lower()
            if provider in FREE and ("per day" in low or "perday" in low or "daily" in low or "requests per day" in low
                                     or ("exceeded your current quota" in low and attempt >= 2)):
                raise QuotaExhausted(last_err)
            ra = r.headers.get("retry-after")
            time.sleep(float(ra) if ra and ra.replace('.', '', 1).isdigit() else min(90, 5 * 2 ** attempt))
            continue
        if r.status_code in (402,):
            raise BudgetExceeded("OpenRouter credit exhausted (HTTP 402)")
        if r.status_code >= 500 or r.status_code in (408, 409):
            time.sleep(min(60, 2 ** attempt + random.random()))
            continue
        raise TransientAPIError(last_err)   # request/routing error: never recorded as a model failure
    if provider in FREE and "HTTP 429" in (last_err or ""):
        raise QuotaExhausted(last_err)
    raise TransientAPIError(f"gave up after retries: {last_err}")


def job_spent() -> float:
    return _JOB_SPENT


def load_models(path: Path | None = None) -> dict:
    import yaml
    p = path or (ROOT / "config" / "models.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {m["label"]: m for m in data["models"]}
