"""[model-migration port, new file] Chat-completions agent driver for the ops testbed.

Reproduces driver_llm.run_llm's agent behaviour with an OpenAI-compatible chat model called through
`mm.llm.chat` (passed in as `chat`):
  * same SYSTEM prompt, same task goal + CUE as the first user message, same NUDGE after a turn without
    tool calls (at most 3), same MAX_TURNS / MAX_TOOL_CALLS budget, same tool set (agent_tools.tools_for)
    converted to chat-completions function tools, same finish tool;
  * the same execute(): writes are mapped to plan steps by driver_llm.StepMapper and go through the recovery
    layer (cfg.condition; this project uses C0 = plain agent, i.e. layers.Layer with no recovery logic);
    reads, the agent's own corrections, wait and finish go through agent_tools.run_read; tool results are the
    same JSON strings the source returned;
  * every tool call of a turn is executed in order and answered with a {"role": "tool"} message.
Differences from the source, all forced by the chat API or by running in-process: no Responses reasoning
items (the provider handles reasoning internally); no crash/resume checkpointing (crash worlds are not used);
a write whose arguments cannot be executed (missing field, wrong type) returns an error to the model instead of
killing the agent process (the source's child process would have died with no report).
Fatal budget exceptions (`fatal`) propagate to the caller; any other exception from `chat` stops the episode
with stop = "model error: ...".
"""
from __future__ import annotations

import copy
import json
import time

from .agent_tools import SCHEMAS, WRITE_TOOLS, run_read, to_contract_args, tools_for
from .driver_llm import CUE, MAX_TOOL_CALLS, MAX_TURNS, NUDGE, SYSTEM, StepMapper
from .layers import make_layer
from .tasks import TaskInstance

MAX_TOKENS = 6000          # the source's max_output_tokens
READ_TOOLS = {"list_issues", "get_issue", "search_issues", "search_sent_email", "find_records", "list_events",
              "get_stock", "get_counter", "find_transfers", "object_exists"}
CORRECTION_TOOLS = {"delete_issue", "delete_comment", "delete_record", "delete_event", "reverse_transfer"}


def _strip(schema, key: str):
    if isinstance(schema, dict):
        return {k: _strip(v, key) for k, v in schema.items() if k != key}
    if isinstance(schema, list):
        return [_strip(v, key) for v in schema]
    return schema


def chat_tools(task: TaskInstance, provider: str = "") -> list[dict]:
    """The source's Responses-format tools (flat) as chat-completions function tools."""
    out = []
    for t in tools_for(task):
        params = copy.deepcopy(t["parameters"])
        if provider == "gemini":            # Google's OpenAI-compatible endpoint rejects additionalProperties
            params = _strip(params, "additionalProperties")
        out.append({"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                     "parameters": params}})
    return out


def _kind(name: str) -> str:
    if name in WRITE_TOOLS:
        return "write"
    if name in READ_TOOLS:
        return "read"
    if name in CORRECTION_TOOLS:
        return "correction"
    if name in ("wait", "finish"):
        return name
    return "unknown"


def run_chat(cfg, tb, disp, spec: dict, chat, *, temperature: float, salt: str, fatal: tuple = (),
             stage: str = "ops") -> dict:
    task = TaskInstance.from_dict(cfg.task)
    layer = make_layer(cfg, tb, disp)
    mapper = StepMapper(task)
    tools = chat_tools(task, spec.get("provider", ""))
    allowed = {t["function"]["name"] for t in tools}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.goal + CUE}]
    st = {"turns": 0, "tool_calls": 0, "nudges": 0, "finish": None, "stop": None, "last_text": "", "cost": 0.0,
          "tokens_in": 0, "tokens_out": 0, "tokens_reasoning": 0, "cached_calls": 0, "paid_calls": 0,
          "length_stops": 0, "malformed": 0, "latency_s": 0.0}
    actions: list[dict] = []
    served = set()

    def execute(call: dict, turn: int) -> str:
        name = call.get("name")
        rec = {"i": len(actions), "turn": turn, "call_id": call.get("id"), "tool": name, "kind": _kind(name),
               "t": round(time.time(), 3)}
        actions.append(rec)
        raw = call.get("arguments") or "{}"
        try:
            a = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(a, dict):
                raise json.JSONDecodeError("not an object", str(raw), 0)
        except json.JSONDecodeError:
            rec.update(args_raw=str(raw)[:2000], malformed="invalid JSON arguments")
            st["malformed"] += 1
            out = json.dumps({"error": "arguments are not valid JSON"})
            rec["output"] = out
            return out
        rec["args"] = a
        if name not in allowed and name in SCHEMAS:
            # a known tool that this task does not offer: the source would still run it (it checks SCHEMAS),
            # and so do we; recorded so the analysis can see it
            rec["offered"] = False
        if name == "finish":
            st["finish"] = a
            out = json.dumps({"ok": True})
        elif name in WRITE_TOOLS:
            try:
                tool, cargs = to_contract_args(name, a)
                step = mapper.step_for(tool, cargs, {})
                rec.update(contract_tool=tool, step=step)
                obs = layer.write(step, tool, cargs)
                o = {"status": obs["status"]}
                if obs.get("value"):
                    o["result"] = obs["value"]
                if obs["status"] != "ok" or obs.get("note") not in (None, "", "done"):
                    o["note"] = obs.get("note")
                rec["obs_status"] = obs["status"]
                out = json.dumps(o, default=str)
            except Exception as e:  # noqa: BLE001 - invalid arguments (missing field, wrong type)
                st["malformed"] += 1
                rec.update(malformed=f"{type(e).__name__}: {str(e)[:150]}", obs_status="invalid_args")
                out = json.dumps({"error": f"invalid arguments: {type(e).__name__}: {str(e)[:150]}"})
        elif name in SCHEMAS:
            refusal = layer.guard_delete(name, a)
            if refusal:
                out = json.dumps({"status": "refused", "note": refusal})
            else:
                try:
                    out = json.dumps(run_read(tb, disp, name, a), default=str)[:6000]
                except Exception as e:  # noqa: BLE001
                    out = json.dumps({"error": f"{type(e).__name__}: {str(e)[:150]}"})
        else:
            st["malformed"] += 1
            rec["malformed"] = "unknown tool"
            out = json.dumps({"error": f"unknown tool {name}"})
        rec["output"] = out[:800]
        return out

    while st["finish"] is None and st["turns"] < MAX_TURNS and st["tool_calls"] < MAX_TOOL_CALLS:
        try:
            r = chat(spec, messages, temperature=temperature, max_tokens=MAX_TOKENS, salt=salt, stage=stage,
                     tools=tools)
        except fatal:
            raise
        except Exception as e:  # noqa: BLE001
            st["stop"] = f"model error: {type(e).__name__}: {str(e)[:300]}"
            break
        st["turns"] += 1
        u = r.get("usage") or {}
        st["tokens_in"] += int(u.get("prompt_tokens") or 0)
        st["tokens_out"] += int(u.get("completion_tokens") or 0)
        st["tokens_reasoning"] += int(r.get("reasoning_tokens") or 0)
        st["cost"] += float(r.get("cost") or 0.0)
        st["latency_s"] += float(r.get("latency_s") or 0.0)
        st["cached_calls" if r.get("cached") else "paid_calls"] += 1
        if r.get("finish_reason") == "length":
            st["length_stops"] += 1
        if r.get("provider_served") or r.get("model_served"):
            served.add(f"{r.get('provider_served')}|{r.get('model_served')}")
        msg = r.get("message") or {"role": "assistant", "content": r.get("text") or ""}
        messages.append(msg)
        if r.get("text"):
            st["last_text"] = str(r["text"])[:1000]
        calls = r.get("tool_calls") or []
        if not calls:
            if st["nudges"] >= 3:
                st["stop"] = "no_tool_calls"
                break
            st["nudges"] += 1
            messages.append({"role": "user", "content": NUDGE})
            continue
        for c in calls:
            res = execute(c, st["turns"])
            messages.append({"role": "tool", "tool_call_id": c.get("id"), "content": res})
            st["tool_calls"] += 1
    if st["finish"] is not None:
        st["stop"] = "finished"
    elif st["stop"] is None:
        st["stop"] = "turn_limit" if st["turns"] >= MAX_TURNS else "tool_call_limit"
    fin = layer.finish()
    f = st["finish"] or {}
    return {"driver": cfg.driver, "condition": cfg.condition, "claimed": f.get("status", "no_finish"),
            "model_summary": (f.get("summary") or st["last_text"] or "")[:500],
            "uncertain": f.get("uncertain_operations", []), "layer_uncertain": fin.get("uncertain", []),
            "turns": st["turns"], "tool_calls": st["tool_calls"], "nudges": st["nudges"], "stop": st["stop"],
            "agent_cost": round(st["cost"], 6), "paid_calls": st["paid_calls"], "cached_calls": st["cached_calls"],
            "tokens_in": st["tokens_in"], "tokens_out": st["tokens_out"], "tokens_reasoning": st["tokens_reasoning"],
            "length_stops": st["length_stops"], "malformed_calls": st["malformed"],
            "model_latency_s": round(st["latency_s"], 2), "served": sorted(served),
            "step_map": mapper.memo, "actions": actions, "messages": messages}
