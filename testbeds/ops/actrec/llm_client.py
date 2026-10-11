"""OpenAI Responses-API client with a response cache, a cost ledger and a hard spending cap.

Why the Responses endpoint: current reasoning models do not accept function tools on chat completions at their
default reasoning setting. The Responses endpoint lets every model run at its own default reasoning, which is
what a deployed agent would use. Requests are stateless (store=false); for reasoning models the encrypted
reasoning items are requested and passed back with the transcript, as the provider documents for multi-turn
tool use without server-side storage.

* Cache: every request (model, instructions, input items, tools, parameters, sample index) is hashed; a hit
  returns the stored output at zero cost. Re-running or replaying an episode therefore costs nothing, and the
  released cache lets anyone replay every model decision without an API key. Transport-negotiated fields
  (store, include, parallel_tool_calls) are not part of the hash, so a field dropped after a provider refusal
  does not break replay.
* Ledger: one JSON line per paid call (model, tokens, US$) in runs/cost_ledger.jsonl, shared by all
  processes through a lock file.
* Cap: before each paid call the worst case (estimated input + max output tokens) is added to the amount
  already spent; if it would exceed MAX_SPEND_USD (project) or MAX_SPEND_PER_RUN_USD (this grid), the call
  is refused with BudgetExceeded. Nothing is ever charged beyond the caps set in .env.
Prices are US$ per 1M tokens (input, cached input, output); verify them against the provider's price page
before a paid run. Unknown models are refused. Reasoning tokens are billed as output tokens and are included
in output_tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import httpx

from .config import ENV, RUNS, register_secret

PRICES = {   # (input, cached input, output) US$ / 1M tokens; verified 2026-10-01 on the provider's model pages
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
}
REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
CACHE = RUNS / "cache" / "llm"
LEDGER = RUNS / "cost_ledger.jsonl"
LOCK = RUNS / ".cost.lock"
URL = "https://api.openai.com/v1/responses"
OPTIONAL = ("parallel_tool_calls", "include", "store")      # dropped (and remembered) if the provider refuses them
HASHED = ("model", "instructions", "input", "tools", "tool_choice", "max_output_tokens", "reasoning")


class BudgetExceeded(RuntimeError):
    pass


class _Lock:
    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except (FileExistsError, PermissionError):
                # Windows raises PermissionError (not FileExistsError) while another process is deleting the lock
                # file; both mean "held by someone else, try again" (amendment 1, docs/PROTOCOL.md)
                if time.time() - t0 > 30:            # stale lock from a killed process
                    try:
                        os.remove(LOCK)
                    except OSError:
                        pass
                    t0 = time.time()
                time.sleep(0.05)

    def __exit__(self, *a):
        os.close(self.fd)
        try:
            os.remove(LOCK)
        except OSError:
            pass


def spent(run: str | None = None) -> float:
    if not LEDGER.exists():
        return 0.0
    tot = 0.0
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if run is None or r.get("run") == run:
            tot += float(r.get("usd", 0))
    return tot


def tokens(usage: dict) -> tuple[int, int, int, int]:
    """(input, cached input, output, reasoning) from a Responses usage block (chat-completions names accepted)."""
    inp = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    det = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached = int(det.get("cached_tokens", 0) or 0)
    out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    odet = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    reasoning = int(odet.get("reasoning_tokens", 0) or 0)
    return inp, cached, out, reasoning


def cost(model: str, usage: dict) -> float:
    pin, pcached, pout = PRICES[model]
    inp, cached, out, _ = tokens(usage)
    return ((inp - cached) * pin + cached * pcached + out * pout) / 1e6


def _input_chars(body: dict) -> int:
    """Characters that become input tokens (encrypted reasoning blobs are counted at a quarter of their length,
    still an over-estimate of the tokens they stand for)."""
    n = len(body.get("instructions") or "") + len(json.dumps(body.get("tools") or []))
    for it in body.get("input") or []:
        enc = it.get("encrypted_content") if isinstance(it, dict) else None
        n += len(json.dumps(it)) - (len(enc) * 3 // 4 if enc else 0)
    return n


class LLM:
    def __init__(self, model: str, grid: str, max_output_tokens: int = 6000, reasoning_effort: str | None = None):
        if model not in PRICES:
            raise ValueError(f"no price on file for {model}; add it to PRICES after checking the provider's page")
        self.model, self.grid = model, grid
        self.max_out = max_output_tokens
        self.effort = reasoning_effort
        self.reasoning = model.startswith(REASONING_PREFIXES)
        self.dropped: set[str] = set()
        self.key = ENV.get("OPENAI_API_KEY", "")
        register_secret(self.key)
        self.cap_total = float(ENV.get("MAX_SPEND_USD", "0") or 0)
        self.cap_run = float(ENV.get("MAX_SPEND_PER_RUN_USD", "0") or 0)

    def _body(self, instructions: str, items: list, tools: list) -> dict:
        b = {"model": self.model, "instructions": instructions, "input": items, "tools": tools, "tool_choice": "auto",
             "parallel_tool_calls": False, "max_output_tokens": self.max_out, "store": False}
        if self.reasoning:
            b["include"] = ["reasoning.encrypted_content"]
            if self.effort:
                b["reasoning"] = {"effort": self.effort}
        return {k: v for k, v in b.items() if k not in self.dropped}

    def respond(self, instructions: str, items: list, tools: list, sample: int = 0,
                episode: str = "") -> tuple[list, dict]:
        """Returns (output items, meta). meta: cached, usage, usd, latency_s, status, incomplete, resolved_model."""
        body = self._body(instructions, items, tools)
        h = hashlib.sha256(json.dumps({"b": {k: body.get(k) for k in HASHED}, "sample": sample},
                                      sort_keys=True).encode()).hexdigest()
        cpath = CACHE / self.model / f"{h}.json"
        if cpath.exists():
            d = json.loads(cpath.read_text(encoding="utf-8"))
            return d["output"], {"cached": True, "usage": d.get("usage", {}), "usd": 0.0, "latency_s": 0.0, "hash": h,
                                 "status": d.get("status"), "incomplete": d.get("incomplete"),
                                 "resolved_model": d.get("resolved_model")}
        if not self.key:
            raise BudgetExceeded("no OPENAI_API_KEY and no cached response (replay mode)")
        pin, _, pout = PRICES[self.model]
        worst = (_input_chars(body) / 3.0 * pin + self.max_out * pout) / 1e6
        with _Lock():
            tot, run = spent(), spent(self.grid)
            if tot + worst > self.cap_total or run + worst > self.cap_run:
                raise BudgetExceeded(f"cap reached: spent total ${tot:.4f} / cap ${self.cap_total}, "
                                     f"this run ${run:.4f} / cap ${self.cap_run}, next call worst case ${worst:.4f}")
        t0 = time.time()
        resp = self._post(body)
        lat = time.time() - t0
        usage = resp.get("usage") or {}
        usd = cost(self.model, usage)
        output = resp.get("output") or []
        status = resp.get("status")
        incomplete = (resp.get("incomplete_details") or {}).get("reason")
        with _Lock():
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "run": self.grid, "episode": episode, "model": self.model,
                                    "resolved_model": resp.get("model"), "usage": usage, "status": status,
                                    "usd": round(usd, 6)}) + "\n")
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({"request_hash": h, "model": self.model, "resolved_model": resp.get("model"),
                                     "output": output, "usage": usage, "status": status, "incomplete": incomplete}),
                         encoding="utf-8")
        return output, {"cached": False, "usage": usage, "usd": usd, "latency_s": round(lat, 3), "hash": h,
                        "status": status, "incomplete": incomplete, "resolved_model": resp.get("model")}

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        delay = 2.0
        for _attempt in range(8):
            try:
                r = httpx.post(URL, headers=headers, json=body, timeout=300)
            except httpx.TransportError:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 200:
                d = r.json()
                if d.get("status") == "failed":
                    raise RuntimeError(f"OpenAI response failed: {json.dumps(d.get('error'))[:300]}")
                return d
            if r.status_code == 400:
                hit = [k for k in OPTIONAL if k in body and k in r.text]
                if hit:                                    # remember, so later turns of this episode skip it
                    self.dropped.update(hit)
                    body = {k: v for k, v in body.items() if k not in hit}
                    continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:600]}")
        raise RuntimeError("OpenAI: retries exhausted")
