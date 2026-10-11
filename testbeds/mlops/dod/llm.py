"""LLM client with a response cache, a cost ledger and a hard spending cap enforced in code.

Providers: "ollama" (local, free) and "openai" (paid; key from the environment, never logged).
Every response is cached under runs/llm_cache/<hash>.json keyed by (provider, model, messages, params), so the
study replays without API keys. Paid calls are refused when the job's cap (DOD_JOB_COST_CAP_USD, set by the
runner from config/budget.json) would be exceeded; spend is appended to runs/cost_ledger.json.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests

from dod.core import ROOT, stable_hash

CACHE = ROOT / "runs" / "llm_cache"
LEDGER = ROOT / "runs" / "cost_ledger.json"
PRICES = ROOT / "config" / "prices.json"     # {"model": {"in": usd_per_mtok, "cached_in": ..., "out": ...}}
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _prices(model: str) -> dict:
    table = json.loads(PRICES.read_text()) if PRICES.exists() else {}
    if model not in table:
        raise BudgetExceeded(f"no price configured for {model}; refusing paid call")
    return table[model]


def _ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"total_usd": 0.0, "by_model": {}, "by_job": {}}


class _FileLock:
    """Cross-process lock via exclusive creation of a lock file (workers of one job share the ledger)."""
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(600):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if i > 200 and time.time() - self.path.stat().st_mtime > 60:   # stale lock
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                time.sleep(0.05)
        raise RuntimeError("ledger lock timeout")

    def __exit__(self, *exc):
        os.close(self.fd)
        try:
            self.path.unlink()
        except OSError:
            pass


def _charge(model: str, usd: float, usage: dict) -> None:
    with _lock, _FileLock(LEDGER.with_suffix(".lock")):
        L = _ledger()
        L["total_usd"] = L.get("total_usd", 0.0) + usd
        m = L["by_model"].setdefault(model, {"usd": 0.0, "calls": 0, "in": 0, "out": 0, "cached_in": 0,
                                             "reasoning": 0})
        m["usd"] += usd
        m["calls"] += 1
        m["in"] += usage.get("in", 0)
        m["out"] += usage.get("out", 0)
        m["cached_in"] += usage.get("cached_in", 0)
        m["reasoning"] = m.get("reasoning", 0) + usage.get("reasoning", 0)
        job = os.environ.get("DOD_JOB_ID", "interactive")
        L["by_job"][job] = L["by_job"].get(job, 0.0) + usd
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER.with_suffix(".tmp")
        tmp.write_text(json.dumps(L, indent=1))
        os.replace(tmp, LEDGER)


def job_spent() -> float:
    return _ledger().get("by_job", {}).get(os.environ.get("DOD_JOB_ID", "interactive"), 0.0)


def chat(provider: str, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 800,
         seed: int = 0, json_mode: bool = True, max_input_usd: float | None = None) -> dict:
    """Returns {"text", "usage": {"in","out","cached_in"}, "usd", "cached": bool}."""
    key = stable_hash({"p": provider, "m": model, "msg": messages, "t": temperature, "mt": max_tokens, "s": seed,
                       "j": json_mode}, 20)
    path = CACHE / provider / f"{key}.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        r["cached"] = True
        return r
    if os.environ.get("DOD_REPLAY_ONLY") == "1":
        raise RuntimeError(f"replay-only mode and no cached response for {key}")
    t0 = time.time()
    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": 16384}}
        if json_mode:
            body["format"] = "json"
        resp = requests.post(f"{host.rstrip('/')}/api/chat", json=body, timeout=600)
        resp.raise_for_status()
        d = resp.json()
        text = d["message"]["content"]
        usage = {"in": d.get("prompt_eval_count", 0), "out": d.get("eval_count", 0), "cached_in": 0}
        usd = 0.0
    elif provider == "openai":
        price = _prices(model)
        cap = float(os.environ.get("DOD_JOB_COST_CAP_USD", "0"))
        est_in = sum(len(m["content"]) for m in messages) / 3.5
        est = (est_in * price["in"] + (max_tokens + 2000) * price["out"]) / 1e6
        if job_spent() + est > cap:
            raise BudgetExceeded(f"job cap {cap:.2f} USD would be exceeded (spent {job_spent():.4f})")
        reasoning = model.startswith(("o", "gpt-5"))
        # reasoning models spend completion tokens on hidden reasoning: low effort, and room for the answer
        body = {"model": model, "messages": messages, "seed": seed,
                "max_completion_tokens": max_tokens + (2000 if reasoning else 0)}
        if reasoning:
            body["reasoning_effort"] = os.environ.get("DOD_REASONING_EFFORT", "low")
        elif temperature is not None:
            body["temperature"] = temperature
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        hdr = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}
        for attempt in range(5):
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=hdr, timeout=300)
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(5 * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            raise RuntimeError(f"openai error {resp.status_code}: {resp.text[:300]}")
        d = resp.json()
        text = d["choices"][0]["message"]["content"] or ""
        u = d.get("usage", {})
        cached_in = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        usage = {"in": u.get("prompt_tokens", 0) - cached_in, "out": u.get("completion_tokens", 0),
                 "cached_in": cached_in,
                 "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}
        usd = (usage["in"] * price["in"] + cached_in * price.get("cached_in", price["in"])
               + usage["out"] * price["out"]) / 1e6
        _charge(model, usd, usage)
    else:
        raise ValueError(provider)
    r = {"text": text, "usage": usage, "usd": usd, "wall_s": round(time.time() - t0, 2), "provider": provider,
         "model": model}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r), encoding="utf-8")
    r["cached"] = False
    return r
