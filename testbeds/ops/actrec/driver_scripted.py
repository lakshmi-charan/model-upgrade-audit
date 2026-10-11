"""Scripted driver: a deterministic agent with no model. It executes the task's plan in order, keeps a durable
step checkpoint (like a framework checkpointer), re-issues a failed step up to `cfg.retries` times (0 for C1),
looks up a missing reference by content the way an agent would, and reports honestly what it believes.
The recovery layer under it is the only thing that differs between conditions.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import Toolbox
from .durable import Checkpoint
from .episode import EpisodeConfig
from .layers import make_layer, value_from_found
from .tasks import TaskInstance, resolve_args


def _lookup_refs(args: dict, task: TaskInstance, tb: Toolbox) -> dict | None:
    """Resolve {"$ref": "step.field"} by reading the service (content match), first match wins."""
    out = {}
    by_name = {s.name: s for s in task.steps}
    for k, v in args.items():
        if isinstance(v, dict) and "$ref" in v:
            step, _fld = v["$ref"].split(".")
            src = by_name[step]
            found = tb.by_content(src.tool, src.args)
            if not found:
                return None
            out[k] = list(value_from_found(src.tool, found, src.args).values())[0]
        else:
            out[k] = v
    return out


def run_scripted(cfg: EpisodeConfig, tb: Toolbox, disp, resumed: bool) -> dict:
    task = TaskInstance.from_dict(cfg.task)
    layer = make_layer(cfg, tb, disp)
    cp = Checkpoint(Path(cfg.state_dir) / "checkpoint.jsonl")
    retries = 0 if cfg.condition == "C1" else cfg.retries
    status: dict[str, str] = {}
    notes: dict[str, list] = {}
    restarts = 0
    i = 0
    while i < len(task.steps):
        st = task.steps[i]
        if st.name in cp.done:
            status[st.name] = "ok"
            i += 1
            continue
        args = resolve_args(st.args, cp.done)
        if args is None:
            args = _lookup_refs(st.args, task, tb)
            if args is None:
                status[st.name] = "missing_ref"
                i += 1
                continue
        tries = 0
        restart = False
        while True:
            obs = layer.write(st.name, st.tool, args)
            notes.setdefault(st.name, []).append(f"{obs['status']}: {obs.get('note', '')}"[:160])
            if obs["status"] == "ok":
                cp.add(st.name, obs.get("value"))
                status[st.name] = "ok"
                break
            if obs["status"] == "restart" and restarts == 0:
                restarts += 1
                cp.reset()
                status.clear()
                restart = True
                break
            if obs["status"] in ("error", "timeout") and tries < retries:
                tries += 1
                continue
            status[st.name] = obs["status"]
            break
        i = 0 if restart else i + 1
    fin = layer.finish()
    all_ok = all(status.get(s.name) == "ok" for s in task.steps)
    clean = not fin.get("uncertain") and not fin.get("duplicates_unrecoverable")
    claimed = "completed" if all_ok and clean else ("failed" if not any(
        v == "ok" for v in status.values()) else "partial")
    return {"driver": "scripted", "condition": cfg.condition, "step_status": status, "claimed": claimed,
            "uncertain": fin.get("uncertain", []), "compensations": fin.get("compensations", 0),
            "duplicates_reported": fin.get("duplicates_unrecoverable", []), "restarts": restarts, "notes": notes}
