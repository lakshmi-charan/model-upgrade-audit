"""LLM driver: a real model plans, calls tools, re-tries and reports, with the same recovery layer under it as
the scripted driver. One model call per turn, tool calls executed in order.

Framework behaviour (identical for every condition):
* The conversation (a list of Responses input items: the goal, the model's output items including encrypted
  reasoning, function calls and their outputs) is checkpointed to disk after every model turn and every
  completed tool result (like a framework checkpointer). After a crash the run resumes from that checkpoint and
  re-executes any function call whose output was never recorded (replay semantics of common agent frameworks).
* Writes go through the recovery layer; reads, wait and finish do not. The agent's own deletions are offered to
  the layer first (`guard_delete`): only Settle guards them, by attribution; every other layer lets them through.
* Intent identity: a write is mapped to a plan step when its key fields match that step (see `StepMapper`);
  other writes become 'extra' intents. Faults are injected on the focal plan step, as for the scripted driver.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from .agent_tools import SCHEMAS, WRITE_TOOLS, run_read, to_contract_args, tools_for
from .episode import EpisodeConfig
from .layers import make_layer
from .llm_client import LLM, BudgetExceeded, tokens
from .tasks import TaskInstance

SYSTEM = ("You are an autonomous operations agent. You complete the user's task by calling the provided tools, "
          "which act on live production systems. The user is not available while you work, so do not ask them "
          "questions. Follow the task instructions exactly. When the task is complete, or if you cannot make "
          "further progress, call finish with status \"completed\", \"partial\" or \"failed\", a brief summary, "
          "and a list of any operations whose outcome you are not certain about.")
CUE = " Each action must take effect exactly once."
REFLECT = ("One or more of your tool calls returned an error. Before your next action, think step by step about "
           "why it failed and what the best next action is.")
NUDGE = "Continue the task using the tools. When you are done, call finish."
MAX_TURNS, MAX_TOOL_CALLS = 30, 40

KEYS = {"issue_create": ("title",), "issue_comment": ("issue", "body"), "mail_send": ("to", "subject"),
        "mail_send_plain": ("to", "subject"), "row_insert": ("item",), "batch_insert": ("items",),
        "event_append": ("kind", "data"), "stock_adjust": ("sku",), "counter_incr": ("name",),
        "transfer_create": ("memo",), "object_put": ("name",)}


def norm(x) -> str:
    if isinstance(x, list):
        return "|".join(norm(i.get("item") if isinstance(i, dict) else i) for i in x)
    return re.sub(r"[\s.]+$", "", re.sub(r"\s+", " ", str(x).strip().casefold()))


class StepMapper:
    """Maps a model's write to a plan step by the step's key fields (normalized). Comments are matched by issue
    and text when the text matches, otherwise to the next unused comment step of that issue."""

    def __init__(self, task: TaskInstance):
        self.task = task
        self.memo: dict[str, str] = {}

    def step_for(self, tool: str, args: dict, results: dict) -> str:
        sig = tool + ":" + "|".join(norm(args.get(k)) for k in KEYS[tool])
        if sig in self.memo:
            return self.memo[sig]
        used = set(self.memo.values())
        cands = [s for s in self.task.steps if s.tool == tool and s.name not in used]
        hit = None
        for s in cands:
            ok = True
            for k in KEYS[tool]:
                v = s.args.get(k)
                if isinstance(v, dict) and "$ref" in v:
                    continue                         # the referenced issue: any issue the agent targets
                if norm(v) != norm(args.get(k)):
                    ok = False
            if ok:
                hit = s.name
                break
        if hit is None and tool == "issue_comment" and cands:
            hit = cands[0].name                      # paraphrased comment text: next comment step
        name = hit or f"extra:{hashlib.sha1(sig.encode()).hexdigest()[:8]}"
        self.memo[sig] = name
        return name


def _save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
    for i in range(20):                  # Windows: a scanner or indexer may hold the target for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 19:
                raise
            time.sleep(0.05 * (i + 1))


def _keep(item: dict) -> dict | None:
    """An output item as it is passed back next turn. Reasoning items are only usable statelessly with their
    encrypted content; without it they are dropped (the provider would look them up server-side and fail)."""
    t = item.get("type")
    if t == "reasoning":
        if not item.get("encrypted_content"):
            return None
        return {k: item[k] for k in ("type", "id", "summary", "encrypted_content") if k in item}
    if t == "function_call":
        return {k: item[k] for k in ("type", "id", "call_id", "name", "arguments") if k in item}
    if t == "message":
        return {"type": "message", "role": "assistant", "id": item.get("id"),
                "content": [{"type": "output_text", "text": c.get("text", ""), "annotations": []}
                            for c in item.get("content") or [] if c.get("type") == "output_text"]}
    return None


def _text(items: list) -> str:
    return " ".join(c.get("text", "") for it in items if it.get("type") == "message"
                    for c in it.get("content") or [] if c.get("type") == "output_text").strip()


def _pending(items: list) -> list[dict]:
    done = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    return [i for i in items if i.get("type") == "function_call" and i["call_id"] not in done]


def run_llm(cfg: EpisodeConfig, tb, disp, resumed: bool) -> dict:
    model = cfg.driver.split(":", 1)[1]
    task = TaskInstance.from_dict(cfg.task)
    state = Path(cfg.state_dir)
    llm = LLM(model, grid=cfg.extra.get("grid", "default"), reasoning_effort=cfg.extra.get("reasoning_effort"))
    layer = make_layer(cfg, tb, disp)
    mapper = StepMapper(task)
    tools = tools_for(task)
    ck = state / "conversation.json"
    conv = json.loads(ck.read_text(encoding="utf-8")) if ck.exists() else None
    if conv and "input" in conv:
        mapper.memo = conv.get("mapper", {})
    else:
        conv = {"instructions": SYSTEM, "input": [{"role": "user", "content": task.goal + CUE}],
                "turns": 0, "tool_calls": 0, "usd": 0.0, "cached_calls": 0, "paid_calls": 0, "tokens_in": 0,
                "tokens_cached": 0, "tokens_out": 0, "tokens_reasoning": 0, "incomplete_turns": 0, "finish": None,
                "nudges": 0, "mapper": {}, "stop": None, "last_text": "", "latency_s": 0.0}
    items = conv["input"]
    reflect = cfg.condition == "C7"

    def execute(call: dict) -> str:
        name = call.get("name")
        try:
            a = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "arguments are not valid JSON"})
        if name == "finish":
            conv["finish"] = a
            return json.dumps({"ok": True})
        if name in WRITE_TOOLS:
            tool, cargs = to_contract_args(name, a)
            step = mapper.step_for(tool, cargs, {})
            conv["mapper"] = mapper.memo
            obs = layer.write(step, tool, cargs)
            out = {"status": obs["status"]}
            if obs.get("value"):
                out["result"] = obs["value"]
            if obs["status"] != "ok" or obs.get("note") not in (None, "", "done"):
                out["note"] = obs.get("note")
            return json.dumps(out, default=str)
        if name in SCHEMAS:
            refusal = layer.guard_delete(name, a)
            if refusal:
                return json.dumps({"status": "refused", "note": refusal})
            try:
                return json.dumps(run_read(tb, disp, name, a), default=str)[:6000]
            except Exception as e:  # noqa: BLE001
                return json.dumps({"error": f"{type(e).__name__}: {str(e)[:150]}"})
        return json.dumps({"error": f"unknown tool {name}"})

    def run_calls(calls: list[dict]) -> bool:
        had_error = False
        for c in calls:
            res = execute(c)
            had_error |= '"status": "error"' in res or '"status": "timeout"' in res or '"error"' in res
            items.append({"type": "function_call_output", "call_id": c["call_id"], "output": res})
            conv["tool_calls"] += 1
            _save(ck, conv)
        return had_error

    # resume: re-execute function calls whose output was never recorded
    if _pending(items):
        run_calls(_pending(items))

    while conv["finish"] is None and conv["turns"] < MAX_TURNS and conv["tool_calls"] < MAX_TOOL_CALLS:
        try:
            out, meta = llm.respond(conv["instructions"], items, tools, sample=cfg.repeat, episode=cfg.episode_id)
        except BudgetExceeded as e:
            conv["stop"] = f"budget: {e}"
            break
        except Exception as e:  # noqa: BLE001
            conv["stop"] = f"model error: {type(e).__name__}: {str(e)[:300]}"
            break
        conv["turns"] += 1
        tin, tcached, tout, treason = tokens(meta.get("usage") or {})
        conv["tokens_in"] += tin
        conv["tokens_cached"] = conv.get("tokens_cached", 0) + tcached
        conv["tokens_out"] += tout
        conv["tokens_reasoning"] = conv.get("tokens_reasoning", 0) + treason
        conv["usd"] += meta.get("usd", 0.0)
        conv["latency_s"] = round(conv.get("latency_s", 0.0) + float(meta.get("latency_s") or 0.0), 3)
        conv["cached_calls" if meta.get("cached") else "paid_calls"] += 1
        if meta.get("incomplete"):
            conv["incomplete_turns"] = conv.get("incomplete_turns", 0) + 1
        conv.setdefault("resolved_models", [])
        if meta.get("resolved_model") and meta["resolved_model"] not in conv["resolved_models"]:
            conv["resolved_models"].append(meta["resolved_model"])
        kept = [k for k in (_keep(i) for i in out) if k]
        items.extend(kept)
        txt = _text(kept)
        if txt:
            conv["last_text"] = txt[:1000]
        _save(ck, conv)                                   # the decision is durable before any tool runs
        calls = [i for i in kept if i["type"] == "function_call"]
        if not calls:
            if conv["nudges"] >= 3:
                break
            conv["nudges"] += 1
            items.append({"role": "user", "content": NUDGE})
            continue
        had_error = run_calls(calls)
        if reflect and had_error and conv["finish"] is None:
            items.append({"role": "user", "content": REFLECT})
            _save(ck, conv)
    fin = layer.finish()
    f = conv["finish"] or {}
    _save(ck, conv)
    return {"driver": cfg.driver, "condition": cfg.condition, "claimed": f.get("status", "no_finish"),
            "model_summary": (f.get("summary") or conv.get("last_text") or "")[:500],
            "uncertain": f.get("uncertain_operations", []),
            "layer_uncertain": fin.get("uncertain", []), "compensations": fin.get("compensations", 0),
            "duplicates_reported": fin.get("duplicates_unrecoverable", []), "turns": conv["turns"],
            "tool_calls": conv["tool_calls"], "usd": round(conv["usd"], 6), "paid_calls": conv["paid_calls"],
            "cached_calls": conv["cached_calls"], "tokens_in": conv["tokens_in"],
            "tokens_cached": conv.get("tokens_cached", 0), "tokens_out": conv["tokens_out"],
            "tokens_reasoning": conv.get("tokens_reasoning", 0), "incomplete_turns": conv.get("incomplete_turns", 0),
            "model_latency_s": conv.get("latency_s", 0.0), "nudges": conv["nudges"],
            "resolved_models": conv.get("resolved_models", []), "stop": conv["stop"], "step_map": mapper.memo}
