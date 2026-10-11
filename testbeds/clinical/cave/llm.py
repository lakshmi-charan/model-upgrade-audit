"""Shim: routes the clinical (MedAgentBench/CAVE) agent's chat() calls to the study's unified client.

The original CAVE client is replaced so every testbed shares one cache, cost ledger and budget cap.
The active model spec and repeat salt are set by code/run_clinical.py via set_active()."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from mm import llm as _mm  # noqa: E402

_ACTIVE = {"spec": None, "salt": "", "temperature": 0.0}


def set_active(spec, salt="", temperature=0.0):
    _ACTIVE.update(spec=spec, salt=salt, temperature=temperature)


def chat(model, messages, system=None, temperature=0.0, max_tokens=1024, stage="misc", salt=None):
    spec = _ACTIVE["spec"]
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    r = _mm.chat(spec, msgs, temperature=_ACTIVE["temperature"], max_tokens=max_tokens,
                 salt=f"{_ACTIVE['salt']}|{salt or ''}", stage=stage)
    return {"text": r["text"], "cost": r["cost"], "usage": r["usage"], "meta": {
        k: r.get(k) for k in ("provider_served", "model_served", "fingerprint", "finish_reason", "latency_s",
                              "cached", "reasoning_tokens")}}
