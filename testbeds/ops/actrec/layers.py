"""Recovery conditions as harness layers between the driver (agent) and the dispatcher.

Every layer exposes write(step, tool, args) -> observation dict and finish() -> report.
An observation has status ok | error | timeout | blocked | unknown | restart, an optional value (effect id)
and a note shown to the agent. All layers use the same dispatcher, the same toolbox and the same budget of
re-issues; they differ only in what they do between the agent's request and the service.

C0 plain              the agent decides (scripted driver: re-issue once on failure)
C1 never retry        stop at the first failure (driver gets 0 re-issues)
C2 sdk-retry          transparent re-sends of the identical request (x3, backoff)
C3 keys               native keys where the contract has one (in-memory key per intent), otherwise plain
C5 verify-before-retry  content postcondition check before any re-send   [Mansoor et al. 2026]
C6 guard              auto-keys; on a re-issue of an uncertain intent wait the documented lag and verify by
                      content; block unverifiable repeats; report uncertainty   [Li 2026, LIMBO]
Gw guard + wait       C6 but waits the same assumed in-flight bound as Settle before verifying (matched latency)
C8 rollback           log completed writes; on a failure compensate them in reverse and restart once [RAC/SagaLLM]
S  Settle             durable intent journal, op-id tag in the contract field, resolve by tag, bounded
                      reconciliation horizon, evidence-gated compensation, abstain when not attributable
S-j / S-t / S-h / S-c Settle without journal / with content instead of tags / without horizon / without compensation
O  outcome oracle     knows the hidden outcome of the focal write (upper bound for client-side policies)

Follow-up study (docs/07_followup_plan.md; frozen separately, v3):
Sp    tag wait-then-count (Proposition 3 applied to every write): durable journal + tag; after an ambiguous
      outcome wait out the in-flight bound, count by tag, re-issue only if none; no final sweep, no compensation
S2    Settle as designed: as S, but abstains on every ambiguous content-only write (reports it, never confirms it
      by content) and always waits out the in-flight bound before the final sweep (catches delayed redelivery)
S-np  Settle with plan-free intent identity for LLM agents: intent = tool + normalized intent fields of the call
C5a / Gwa  verify-before-retry / guard+wait whose content checks only accept the agent's own records (author filter)
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from .contracts import SPECS, Toolbox
from .durable import Journal
from .episode import INTENDED_EXEC, EpisodeConfig
from .services.base import CallResult

MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------------------------- helpers
def exists_refusal(res: CallResult) -> bool:
    """A native-key refusal means 'already done': S3 412 on conditional create, TigerBeetle EXISTS."""
    if res.kind == "error" and res.raw_exc_type and "ClientError" in res.raw_exc_type and "412" in (res.raw_exc_msg or ""):
        return True
    if res.kind == "error" and "PreconditionFailed" in (res.exc_msg or ""):
        return True
    if res.kind == "ok" and isinstance(res.value, dict) and res.value.get("statuses"):
        return res.value["statuses"][-1] in ("CREATED", "EXISTS")
    return False


def effect_value(tool: str, res_value) -> dict:
    v = res_value or {}
    if tool == "transfer_create" and isinstance(v, dict) and "transfer_id" in v:
        return {"transfer_id": v["transfer_id"]}
    return v if isinstance(v, dict) else {"id": v}


def value_from_found(tool: str, found: list, args: dict) -> dict:
    if not found:
        return {}
    f = sorted(found, key=_sort_key)[0]
    return {"issue_create": {"number": f}, "issue_comment": {"comment_id": f}, "row_insert": {"id": f},
            "event_append": {"entry_id": f}, "transfer_create": {"transfer_id": str(f)},
            "object_put": {"name": args.get("name")}}.get(tool, {"id": f})


def _sort_key(x):
    s = str(x)
    if "-" in s and s.split("-")[0].isdigit():          # redis stream ids "ms-seq"
        a, b = s.split("-", 1)
        return (int(a), int(b) if b.isdigit() else 0)
    return (int(s), 0) if s.isdigit() else (0, 0)


def obs_from(res: CallResult, tool: str, note: str = "") -> dict:
    if res.kind == "ok" or exists_refusal(res):
        return {"status": "ok", "value": effect_value(tool, res.value), "note": note or "done"}
    return {"status": res.kind, "value": None,
            "note": note or (f"{res.exc_type}: {res.exc_msg}" if res.kind == "error" else res.exc_msg)}


# ---------------------------------------------------------------------------------------------- base
class Layer:
    name = "C0"

    def __init__(self, cfg: EpisodeConfig, tb: Toolbox, disp):
        self.cfg, self.tb, self.disp = cfg, tb, disp
        self.uncertain: dict[str, dict] = {}

    def send(self, step, tool, args, key=None, tag=None, batch_only=None) -> CallResult:
        if tool == "transfer_create" and key is None:
            # the agent's own transfer_id if it supplied one; otherwise a client without key discipline
            key = args.get("transfer_id") or str(uuid.uuid4())
        return self.disp.dispatch(step, tool, args, key=key, tag=tag, batch_only=batch_only)

    def write(self, step: str, tool: str, args: dict) -> dict:
        res = self.send(step, tool, args)
        if res.kind != "ok":
            self.uncertain[step] = {"tool": tool}
        else:
            self.uncertain.pop(step, None)
        return obs_from(res, tool)

    def finish(self) -> dict:
        return {"uncertain": sorted(self.uncertain), "compensations": 0, "duplicates_unrecoverable": []}

    def guard_delete(self, name: str, a: dict) -> str | None:
        """An LLM agent's own deletion/reversal request; None lets it through (only Settle guards it)."""
        return None


class NeverRetry(Layer):
    name = "C1"


class Reflect(Layer):
    """C7: plain layer; the LLM driver adds a reflection request after any tool error (Shinn et al. 2023)."""
    name = "C7"


class SdkRetry(Layer):
    name = "C2"

    def write(self, step, tool, args):
        key = (args.get("transfer_id") or str(uuid.uuid4())) if tool == "transfer_create" else None  # identical request => identical id
        res = self.send(step, tool, args, key=key)
        n = 0
        while res.kind in ("error", "timeout") and not exists_refusal(res) and n < 3:
            time.sleep(0.5 * 2 ** n)
            n += 1
            res = self.send(step, tool, args, key=key)
        if res.kind != "ok" and not exists_refusal(res):
            self.uncertain[step] = {"tool": tool}
        else:
            self.uncertain.pop(step, None)
        return obs_from(res, tool)


class Keys(Layer):
    name = "C3"

    def __init__(self, *a):
        super().__init__(*a)
        self.keys: dict[str, str] = {}          # in memory: lost on a crash

    def write(self, step, tool, args):
        key = self.keys.setdefault(step, args.get("transfer_id") or str(uuid.uuid4())) if SPECS[tool].cls == "K" else None
        res = self.send(step, tool, args, key=key)
        if res.kind != "ok" and not exists_refusal(res):
            self.uncertain[step] = {"tool": tool}
        else:
            self.uncertain.pop(step, None)
        return obs_from(res, tool)


class VerifyBeforeRetry(Layer):
    """Content postcondition check before re-sending; for aggregates, before/after comparison."""
    name = "C5"
    own_only = False                     # C5a: only the agent's own records count (author filter)

    def _done(self, tool, args, pre):
        if SPECS[tool].cls == "U":
            return self.tb.aggregate(tool, args) >= pre + self.tb.unit(tool, args), []
        found = self.tb.by_content(tool, args, own_only=True) if self.own_only else self.tb.by_content(tool, args)
        if tool == "batch_insert":
            return all(found), []
        return bool(found), found

    def write(self, step, tool, args):
        pre = self.tb.aggregate(tool, args) if SPECS[tool].cls == "U" else None
        res = self.send(step, tool, args)
        sends = 1
        while res.kind in ("error", "timeout") and not exists_refusal(res):
            done, found = self._done(tool, args, pre)
            if done:
                return {"status": "ok", "value": value_from_found(tool, found, args), "note": "verified: already done"}
            if sends >= 2:
                break
            res = self.send(step, tool, args)
            sends += 1
        if res.kind != "ok" and not exists_refusal(res):
            self.uncertain[step] = {"tool": tool}
        return obs_from(res, tool)


class Guard(Layer):
    """Reimplementation of the client-side guard of Li (2026) from its description."""
    name = "C6"
    wait_bound = False
    own_only = False

    def __init__(self, *a):
        super().__init__(*a)
        self.keys: dict[str, str] = {}

    def write(self, step, tool, args):
        spec = SPECS[tool]
        key = self.keys.setdefault(step, args.get("transfer_id") or str(uuid.uuid4())) if spec.cls == "K" else None
        u = self.uncertain.get(step)
        if u is not None and spec.cls != "K":
            if spec.cls == "U":
                return {"status": "blocked", "value": None,
                        "note": "outcome of the earlier attempt is unknown and cannot be verified; not repeated"}
            wait_until = u["t"] + self.tb.lag_s(tool) + (self.cfg.delta_assumed_s if self.wait_bound else 0.0)
            time.sleep(max(0.0, wait_until - time.time()))
            found = self.tb.by_content(tool, args, own_only=True) if self.own_only else self.tb.by_content(tool, args)
            done = all(found) if tool == "batch_insert" else bool(found)
            if done:
                self.uncertain.pop(step, None)
                return {"status": "ok", "value": value_from_found(tool, [] if tool == "batch_insert" else found, args),
                        "note": "verified: earlier attempt took effect"}
        t = time.time()
        res = self.send(step, tool, args, key=key)
        if res.kind == "ok" or exists_refusal(res):
            self.uncertain.pop(step, None)
            return obs_from(res, tool)
        self.uncertain[step] = {"tool": tool, "t": t}
        return obs_from(res, tool, note=f"{res.exc_type}: outcome unknown - the request may have taken effect")


class GuardWait(Guard):
    name = "Gw"
    wait_bound = True


class VerifyOwn(VerifyBeforeRetry):
    name = "C5a"
    own_only = True


class GuardWaitOwn(GuardWait):
    name = "Gwa"
    own_only = True


class Rollback(Layer):
    """Log-based compensation on failure, then restart (RAC / saga style)."""
    name = "C8"

    def __init__(self, *a):
        super().__init__(*a)
        self.done: list[tuple] = []
        self.restarted = False
        self.comp = 0

    def write(self, step, tool, args):
        res = self.send(step, tool, args)
        if res.kind == "ok" or exists_refusal(res):
            self.done.append((step, tool, args, effect_value(tool, res.value)))
            return obs_from(res, tool)
        if self.restarted:
            self.uncertain[step] = {"tool": tool}
            return obs_from(res, tool)
        for s, t, a, v in reversed(self.done):
            for eid in _effect_ids(t, v, a):
                self.disp.compensate(s, t, a, eid)
                self.comp += 1
        self.done.clear()
        self.restarted = True
        return {"status": "restart", "value": None, "note": "failure: compensated completed steps, restarting"}

    def finish(self):
        r = super().finish()
        r["compensations"] = self.comp
        return r


DELETE_TARGET = {   # agent deletion tool -> (contract tools it can undo, argument holding the effect id)
    "delete_issue": (("issue_create",), "issue_number"), "delete_comment": (("issue_comment",), "comment_id"),
    "delete_record": (("row_insert", "batch_insert"), "id"), "delete_event": (("event_append",), "entry_id"),
    "reverse_transfer": (("transfer_create",), "transfer_id"),
}


def _kept_id(tool, value: dict, found: list):
    """Keep the effect later steps already refer to (the recorded one); otherwise the earliest."""
    ids = _effect_ids(tool, value, {})
    for i in ids:
        if any(str(i) == str(f) for f in found):
            return i
    return sorted(found, key=_sort_key)[0]


def _effect_ids(tool, value, args) -> list:
    v = value or {}
    if tool == "issue_create":
        return [v.get("number")] if v.get("number") else []
    if tool == "issue_comment":
        return [v.get("comment_id")] if v.get("comment_id") else []
    if tool == "row_insert":
        return [v.get("id")] if v.get("id") else []
    if tool == "batch_insert":
        return list((v.get("ids") or {}).values())
    if tool == "event_append":
        return [v.get("entry_id")] if v.get("entry_id") else []
    if tool in ("stock_adjust", "counter_incr"):
        return ["unit"]
    if tool == "transfer_create":
        return [v.get("transfer_id")] if v.get("transfer_id") else []
    if tool == "object_put":
        return [args.get("name")]
    return []        # mail: irreversible


class Oracle(Layer):
    """Knows the hidden outcome of the focal write. Re-issues only if the original will never execute."""
    name = "O"

    def __init__(self, *a):
        super().__init__(*a)
        self.keys: dict[str, str] = {}

    def write(self, step, tool, args):
        cfg = self.cfg
        focal = step == cfg.focal_step
        if focal and cfg.mode == "crash_after" and (Path(cfg.state_dir) / "focal_fired").exists():
            return {"status": "ok", "value": {}, "note": "oracle: the write before the crash took effect"}
        key = self.keys.setdefault(step, str(uuid.uuid4())) if SPECS[tool].cls == "K" else None
        res = self.send(step, tool, args, key=key)
        if res.kind == "ok" or exists_refusal(res):
            return obs_from(res, tool)
        if focal:
            if cfg.mode == "partial":
                from .episode import BATCH_K
                n = len(args["items"])
                res = self.send(step, tool, args, batch_only=list(range(BATCH_K, n)))
                return obs_from(res, tool)
            if (INTENDED_EXEC.get(cfg.mode) or 0) >= 1:
                return {"status": "ok", "value": self._own_effect(tool, args),
                        "note": "oracle: the write took effect (or will)"}
        return obs_from(self.send(step, tool, args, key=key), tool)

    def _own_effect(self, tool, args) -> dict:
        """The oracle waits until its own effect is visible (it knows when an in-flight request lands) and
        returns its id, ignoring the background actor's look-alikes (it knows those too)."""
        if SPECS[tool].cls == "U":
            return {}
        import json as _json
        actor_path = Path(self.cfg.state_dir) / "actor.jsonl"
        deadline = time.time() + self.cfg.delay_s + 10
        while time.time() < deadline:
            actor = set()
            if actor_path.exists():
                for line in actor_path.read_text(encoding="utf-8").splitlines():
                    try:
                        actor.add(str(_json.loads(line).get("id")))
                    except Exception:
                        pass
            found = [f for f in self.tb.by_content(tool, args) if str(f) not in actor]
            if found:
                return value_from_found(tool, found, args)
            time.sleep(0.5)
        return {}


# ---------------------------------------------------------------------------------------------- Settle
class Settle(Layer):
    name = "S"

    def __init__(self, cfg, tb, disp, journal=True, tag=True, horizon=True, compensate=True, prevent=False,
                 designed=False, plan_free=False):
        super().__init__(cfg, tb, disp)
        self.use_tag, self.horizon, self.do_comp = tag, horizon, compensate
        self.prevent, self.designed, self.plan_free = prevent, designed, plan_free
        self.j = Journal(Path(cfg.state_dir) / "journal.jsonl" if journal else None)
        self.comp = 0
        self.dups: list[str] = []

    # ----- identity
    def intent(self, step, tool, args) -> str:
        if self.plan_free:   # S-np: identity from the call itself (tool + normalized intent fields), no plan
            from .driver_llm import KEYS, norm
            return tool + ":" + "|".join(norm(args.get(k)) for k in KEYS[tool])
        return step          # one intent per plan step (LLM driver: the harness's mapping of the call to a step)

    # ----- reads
    def _find(self, tool, args, rec, wait_lag=True):
        lag = self.tb.lag_s(tool)
        if wait_lag and lag:
            time.sleep(lag)
        if tool == "batch_insert":
            if self.use_tag:
                return self.tb.batch_by_tag(rec["op"], len(args["items"]))
            return {i: ids for i, ids in enumerate(self.tb.by_content(tool, args))}
        if self.use_tag and SPECS[tool].cls == "A":
            return self.tb.by_tag(tool, args, rec["op"])
        return self.tb.by_content(tool, args)            # content-only contract (C) or the S-t ablation

    def _executed_aggregate(self, tool, args, rec) -> int:
        return (self.tb.aggregate(tool, args) - int(rec["pre"])) // self.tb.unit(tool, args)

    # ----- write path
    def write(self, step, tool, args):
        it = self.intent(step, tool, args)
        rec = self.j.get(it)
        if rec:
            st = rec.get("state")
            if st in ("confirmed", "reconcile"):
                return {"status": "ok", "value": rec.get("value") or {}, "note": "already done (journal)"}
            if st == "unknown":
                return {"status": "unknown", "value": None, "note": "outcome unknown and not attributable; not repeated"}
            return self._resolve(it, step, tool, args)                # left in doubt (e.g. by a crash)
        spec = SPECS[tool]
        pre = self.tb.aggregate(tool, args) if spec.cls == "U" else None
        self.j.put(it, state="intent", op=str(uuid.uuid4()), step=step, tool=tool, args=args, pre=pre, attempts=[])
        return self._send(it, step, tool, args)

    def _send(self, it, step, tool, args, batch_only=None, reissue=False) -> dict:
        rec = self.j.get(it)
        spec = SPECS[tool]
        tag = rec["op"] if (spec.cls == "A" and self.use_tag) else None
        key = rec["op"] if spec.cls == "K" else None
        self.j.put(it, state="dispatching", attempts=rec["attempts"] + [time.time()])
        res = self.disp.dispatch(step, tool, args, tag=tag, key=key, batch_only=batch_only)
        if res.kind == "ok" or exists_refusal(res):
            val = effect_value(tool, res.value)
            if tool == "batch_insert" and rec.get("value"):
                merged = dict(rec["value"].get("ids", {}))
                merged.update(val.get("ids", {}))
                val = {"ids": merged}
            self.j.put(it, state="reconcile" if (reissue or len(rec["attempts"]) > 1) else "confirmed", value=val)
            return {"status": "ok", "value": val, "note": "done"}
        self.j.put(it, state="in_doubt", last=res.signature(), doubt=True)
        return self._resolve(it, step, tool, args)

    def _resolve(self, it, step, tool, args) -> dict:
        rec = self.j.get(it)
        spec = SPECS[tool]
        n_att = len(rec.get("attempts", []))
        if spec.cls == "K":                                     # same key: re-issue is always safe
            if n_att >= MAX_ATTEMPTS:
                return self._unknown(it)
            return self._send(it, step, tool, args, reissue=True)
        if spec.cls == "U":
            if self.tb.aggregate_shared(tool):
                return self._unknown(it, "shared aggregate: the outcome cannot be attributed")
            if self._executed_aggregate(tool, args, rec) >= 1:
                self.j.put(it, state="reconcile" if n_att > 1 else "confirmed", value={})
                return {"status": "ok", "value": {}, "note": "resolved: took effect"}
            if n_att >= MAX_ATTEMPTS:
                return self._unknown(it)
            if self.prevent:
                self._wait_bound(rec, tool)
                if self._executed_aggregate(tool, args, rec) >= 1:
                    self.j.put(it, state="confirmed", value={})
                    return {"status": "ok", "value": {}, "note": "resolved after the in-flight bound"}
            return self._send(it, step, tool, args, reissue=True)
        if tool == "batch_insert":
            found = self._find(tool, args, rec)
            missing = [i for i, ids in found.items() if not ids]
            if not missing:
                val = {"ids": {i: sorted(ids, key=_sort_key)[0] for i, ids in found.items()}}
                self.j.put(it, state="reconcile" if n_att > 1 else "confirmed", value=val)
                return {"status": "ok", "value": val, "note": "resolved: all rows present"}
            if n_att >= MAX_ATTEMPTS:
                return self._unknown(it)
            if self.prevent:
                self._wait_bound(rec, tool)
                found = self._find(tool, args, rec)
                missing = [i for i, ids in found.items() if not ids]
                if not missing:
                    val = {"ids": {i: sorted(ids, key=_sort_key)[0] for i, ids in found.items()}}
                    self.j.put(it, state="confirmed", value=val)
                    return {"status": "ok", "value": val, "note": "resolved after the in-flight bound"}
            return self._send(it, step, tool, args, batch_only=missing, reissue=True)
        found = self._find(tool, args, rec)
        if found and self.designed and spec.cls == "C":         # S2: a content match proves nothing (P4)
            return self._unknown(it, "content-only write: a matching record exists but cannot be attributed to "
                                     "this operation; not repeated, outcome unknown")
        if found:
            val = value_from_found(tool, found, args)
            self.j.put(it, state="reconcile" if n_att > 1 else "confirmed", value=val)
            return {"status": "ok", "value": val, "note": "resolved: the earlier attempt took effect"}
        if n_att >= MAX_ATTEMPTS:
            return self._unknown(it)
        if spec.reversible == "none" or spec.cls == "C" or self.prevent:
            if self.horizon or self.prevent:                    # irreversible (or Sp): wait out the in-flight bound
                last = max(rec["attempts"])
                time.sleep(max(0.0, last + self.cfg.delta_assumed_s + self.tb.lag_s(tool) - time.time()))
                found = self._find(tool, args, rec)
                if found and self.designed and spec.cls == "C":
                    return self._unknown(it, "content-only write: a matching record exists but cannot be "
                                             "attributed to this operation; not repeated, outcome unknown")
                if found:
                    val = value_from_found(tool, found, args)
                    self.j.put(it, state="confirmed", value=val)
                    return {"status": "ok", "value": val, "note": "resolved after the in-flight bound"}
        return self._send(it, step, tool, args, reissue=True)

    def _wait_bound(self, rec, tool) -> None:
        last = max(rec["attempts"])
        time.sleep(max(0.0, last + self.cfg.delta_assumed_s + self.tb.lag_s(tool) - time.time()))

    def _unknown(self, it, why="outcome unknown after repeated attempts") -> dict:
        self.j.put(it, state="unknown")
        return {"status": "unknown", "value": None, "note": why}

    # ----- reconciliation horizon
    def finish(self) -> dict:
        if self.prevent:                   # Sp: no reconciliation sweep (prevention only)
            uncertain = sorted(r["step"] for r in self.j.all() if r.get("state") in ("unknown", "in_doubt", "dispatching")
                               or (r.get("tool") and SPECS[r["tool"]].cls == "C" and len(r.get("attempts", [])) > 1))
            return {"uncertain": uncertain, "compensations": 0, "duplicates_unrecoverable": []}
        recs = [r for r in self.j.all() if r.get("state") in ("confirmed", "reconcile", "in_doubt", "dispatching")]
        ambiguous = [r for r in recs if r["state"] != "confirmed"]
        if self.designed and recs:         # S2: always wait out the bound, so a delayed redelivery is caught
            last = max(max(r.get("attempts") or [0]) for r in recs)
            lag = max(self.tb.lag_s(r["tool"]) for r in recs)
            time.sleep(max(0.0, last + self.cfg.delta_assumed_s + lag - time.time()))
        elif self.horizon and ambiguous:
            last = max(max(r.get("attempts") or [0]) for r in ambiguous)
            lag = max(self.tb.lag_s(r["tool"]) for r in ambiguous)
            time.sleep(max(0.0, last + self.cfg.delta_assumed_s + lag - time.time()))
        for r in recs:                     # the sweep also catches transport redelivery of 'confirmed' ops
            self._reconcile(r)
        uncertain = sorted(r["step"] for r in self.j.all() if r.get("state") == "unknown"
                           or (r.get("tool") and SPECS[r["tool"]].cls == "C" and len(r.get("attempts", [])) > 1)
                           or (self.designed and r.get("tool") and SPECS[r["tool"]].cls == "C" and r.get("doubt")))
        return {"uncertain": uncertain, "compensations": self.comp, "duplicates_unrecoverable": self.dups}

    def _reconcile(self, r) -> None:
        tool, args, step, it = r["tool"], r["args"], r["step"], r["intent"]
        spec = SPECS[tool]
        if spec.cls == "K":
            return
        if r["state"] in ("in_doubt", "dispatching"):
            self._resolve(it, step, tool, args)
            r = self.j.get(it)
        if spec.cls == "U":
            if self.tb.aggregate_shared(tool):
                return
            n = self._executed_aggregate(tool, args, r)
            if n == 0 and len(r.get("attempts", [])) < MAX_ATTEMPTS:
                self._send(it, step, tool, args, reissue=True)
            elif n > 1:
                self._surplus(step, tool, args, ["unit"] * (n - 1))
            self.j.put(it, state="confirmed")
            return
        found = self._find(tool, args, r, wait_lag=False)
        if tool == "batch_insert":
            extra = []
            for i, ids in found.items():
                extra += sorted(ids, key=_sort_key)[1:]
            if extra:
                self._surplus(step, tool, args, extra)
            self.j.put(it, state="confirmed")
            return
        if not found and len(r.get("attempts", [])) < MAX_ATTEMPTS:
            self._send(it, step, tool, args, reissue=True)
            found = self._find(tool, args, r)
        if len(found) > 1:
            keep = _kept_id(tool, (self.j.get(it) or {}).get("value") or {}, found)
            self._surplus(step, tool, args, [e for e in found if str(e) != str(keep)])
        self.j.put(it, state="confirmed")

    # ----- the agent's own deletions (LLM driver): attribution also decides what may be deleted
    def guard_delete(self, name: str, a: dict) -> str | None:
        """Allow an agent deletion only of a proven surplus effect of one of this task's operations (it carries
        the operation's tag and is not the effect the operation keeps). Refuse deleting an effect that carries
        none of this task's tags (another actor's look-alike: P4) or the only effect of a completed operation.
        Without tags (S-t) nothing can be attributed, so nothing is guarded."""
        if not self.use_tag or name not in DELETE_TARGET:
            return None
        tools, field = DELETE_TARGET[name]
        target = str(a.get(field))
        if "transfer_create" in tools:
            return ("refused: this task's transfers carry an idempotency key and are applied at most once, so none "
                    "of them is a duplicate; nothing was reversed")
        for r in self.j.all():
            if r.get("tool") not in tools or not r.get("op"):
                continue
            found = self._find(r["tool"], r["args"], r)
            if r["tool"] == "batch_insert":
                for i, ids in found.items():
                    if target in {str(x) for x in ids}:
                        keep = sorted(ids, key=_sort_key)[0]
                        if len(ids) > 1 and str(keep) != target:
                            return None
                        return (f"refused: record {target} is the only copy of row {i} of this task's operation "
                                f"'{r['step']}'; deleting it would undo a completed step")
                continue
            if target in {str(x) for x in found}:
                keep = _kept_id(r["tool"], r.get("value") or {}, found)
                if len(found) > 1 and str(keep) != target:
                    return None
                return (f"refused: {target} is the effect of this task's operation '{r['step']}'; deleting it would "
                        "undo a completed step")
        return (f"refused: {target} carries none of this task's operation ids, so it was not created by this task "
                "(it may belong to another user); it was left untouched")

    def _surplus(self, step, tool, args, effect_ids: list) -> None:
        spec = SPECS[tool]
        if spec.reversible == "none" or not self.do_comp:
            self.dups.append(step)
            return
        for e in effect_ids:                                    # evidence: these carry our tag (or our delta)
            self.disp.compensate(step, tool, args, e)
            self.comp += 1


LAYERS = {
    "C0": Layer, "C1": NeverRetry, "C7": Reflect, "C2": SdkRetry, "C3": Keys, "C5": VerifyBeforeRetry, "C6": Guard,
    "Gw": GuardWait, "C8": Rollback, "O": Oracle, "S": Settle, "C5a": VerifyOwn, "Gwa": GuardWaitOwn,
}


def make_layer(cfg: EpisodeConfig, tb: Toolbox, disp) -> Layer:
    c = cfg.condition
    if c.startswith("S"):
        flags = {"S": {}, "S-j": {"journal": False}, "S-t": {"tag": False}, "S-h": {"horizon": False},
                 "S-c": {"compensate": False}, "Sp": {"prevent": True, "compensate": False},
                 "S2": {"designed": True}, "S-np": {"plan_free": True}}[c]
        lay = Settle(cfg, tb, disp, **flags)
        lay.name = c
        return lay
    return LAYERS[c](cfg, tb, disp)
