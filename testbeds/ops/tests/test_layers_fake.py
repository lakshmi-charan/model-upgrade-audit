"""Offline logic test of the recovery layers and the scripted driver against an in-memory fake of the services.
No Docker, no network, no models. It checks the theory's predictions on the fake (not a result of the paper):
Settle converges on attributable reversible writes in every ambiguous world; naive retry duplicates when the
write committed; content verification is fooled by look-alikes. Run:  python -m pytest code/tests -q
"""
from __future__ import annotations

import itertools
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from actrec.contracts import SPECS  # noqa: E402
from actrec.driver_scripted import run_scripted  # noqa: E402
from actrec.episode import BATCH_K, EpisodeConfig  # noqa: E402
from actrec.services.base import CallResult  # noqa: E402
from actrec.tasks import make  # noqa: E402

DELAY = 0.4


class FakeTB:
    """Effects: dict tool -> list of {id, args, tag, alive, actor}."""

    def __init__(self, variant="exact"):
        self.variant = variant
        self.eff: dict[str, list[dict]] = {t: [] for t in SPECS}
        self.agg: dict[tuple, int] = {}
        self.ids = itertools.count(1)
        self.lock = threading.Lock()

    def lag_s(self, tool):
        return 0.0

    def aggregate_shared(self, tool):
        return self.variant == "agg_shared" and SPECS[tool].cls == "U"

    def addr(self, to):
        return to

    def _match(self, tool, e, args):
        f = SPECS[tool].intent
        return all(repr(e["args"].get(k)) == repr(args.get(k)) for k in f)

    def apply(self, tool, args, tag=None, key=None, batch_only=None, actor=False):
        with self.lock:
            if SPECS[tool].cls == "U":
                k = (tool, args.get("sku") or args.get("name"))
                self.agg[k] = self.agg.get(k, 0) + (int(args["delta"]) if tool == "stock_adjust" else 1)
                self.eff[tool].append({"id": next(self.ids), "args": args, "tag": None, "alive": True, "actor": actor})
                return {}
            if SPECS[tool].cls == "K":
                kk = key or args.get("name")
                if any(e["tag"] == kk and e["alive"] for e in self.eff[tool]):
                    return {"statuses": ["EXISTS"], "transfer_id": kk}
                self.eff[tool].append({"id": kk, "args": args, "tag": kk, "alive": True, "actor": actor})
                return {"statuses": ["CREATED"], "transfer_id": kk}
            if tool == "batch_insert":
                ids = {}
                for i in (batch_only if batch_only is not None else range(len(args["items"]))):
                    eid = next(self.ids)
                    self.eff[tool].append({"id": eid, "args": args["items"][i], "row": i,
                                           "tag": f"{tag}:{i}" if tag else None, "alive": True, "actor": actor})
                    ids[i] = eid
                return {"ids": ids}
            eid = next(self.ids)
            self.eff[tool].append({"id": eid, "args": args, "tag": tag, "alive": True, "actor": actor})
            return {"number": eid, "comment_id": eid, "id": eid, "entry_id": f"{eid}-0"}

    # ---- toolbox API used by layers
    def by_tag(self, tool, args, tag):
        return [e["id"] for e in self.eff[tool] if e["alive"] and e["tag"] == tag]

    def batch_by_tag(self, tag, n):
        return {i: [e["id"] for e in self.eff["batch_insert"] if e["alive"] and e["tag"] == f"{tag}:{i}"] for i in range(n)}

    def by_content(self, tool, args):
        if tool == "batch_insert":
            return [[e["id"] for e in self.eff[tool] if e["alive"] and e["args"] == it] for it in args["items"]]
        if SPECS[tool].cls == "K" and tool == "transfer_create":
            return [e["id"] for e in self.eff[tool] if e["alive"] and e["args"].get("memo") == args.get("memo")]
        return [e["id"] for e in self.eff[tool] if e["alive"] and self._match(tool, e, args)]

    def aggregate(self, tool, args):
        return self.agg.get((tool, args.get("sku") or args.get("name")), 0)

    def unit(self, tool, args):
        return int(args["delta"]) if tool == "stock_adjust" else 1

    def compensate(self, tool, args, eid):
        with self.lock:
            if SPECS[tool].cls == "U":
                k = (tool, args.get("sku") or args.get("name"))
                self.agg[k] -= self.unit(tool, args)
                return "ok"
            if SPECS[tool].reversible == "none":
                return "irreversible"
            for e in self.eff[tool]:
                if str(e["id"]) == str(eid) and e["alive"]:
                    e["alive"] = False
                    return "204"
            return "404"

    # ---- truth
    def standing(self, tool, args):
        if SPECS[tool].cls == "U":
            actor = sum((int(e["args"]["delta"]) if tool == "stock_adjust" else 1)
                        for e in self.eff[tool] if e["actor"] and self._match(tool, e, args))
            return (self.aggregate(tool, args) - actor) // self.unit(tool, args)
        if tool == "issue_comment":
            return len([e for e in self.eff[tool] if e["alive"] and not e["actor"]
                        and e["args"].get("body") == args.get("body")])
        return len([e for e in self.eff[tool] if e["alive"] and not e["actor"] and e["args"] == args])

    def collateral(self):
        return sum(1 for es in self.eff.values() for e in es if e["actor"] and not e["alive"])


class FakeDisp:
    def __init__(self, cfg, tb):
        self.cfg, self.tb, self.fired = cfg, tb, False
        self.threads = []

    def dispatch(self, step, tool, args, *, tag=None, key=None, batch_only=None, deadline_s=None):
        focal = step == self.cfg.focal_step and not self.fired
        mode = self.cfg.mode if focal else "none"
        if focal:
            self.fired = True
            if self.cfg.variant == "lookalike" and tool in ("issue_create", "row_insert", "event_append", "mail_send"):
                self.tb.apply(tool, args, actor=True)
            if self.cfg.variant == "agg_shared" and SPECS[tool].cls == "U":
                self.tb.apply(tool, args, actor=True)
        if mode == "none":
            return CallResult("ok", 0.01, value=self.tb.apply(tool, args, tag, key, batch_only))
        if mode == "lost_ack":
            self.tb.apply(tool, args, tag, key, batch_only)
            return CallResult("error", 0.01, exc_type="ToolTransportError", exc_msg="x")
        if mode == "not_exec_err":
            return CallResult("error", 0.01, exc_type="ToolTransportError", exc_msg="x")
        if mode == "late":
            th = threading.Timer(DELAY, lambda: self.tb.apply(tool, args, tag, key, batch_only))
            th.start()
            self.threads.append(th)
            return CallResult("timeout", 0.01, exc_type="ToolDeadline", exc_msg="x")
        if mode == "redelivery":
            self.tb.apply(tool, args, tag, key, batch_only)
            return CallResult("ok", 0.01, value=self.tb.apply(tool, args, tag, key, batch_only))
        if mode == "partial":
            n = len(args["items"])
            self.tb.apply(tool, args, tag, key, [i for i in range(BATCH_K) if batch_only is None or i in batch_only])
            return CallResult("error", 0.01, exc_type="ToolTransportError", exc_msg="x")
        raise ValueError(mode)

    def compensate(self, step, tool, args, eid):
        return self.tb.compensate(tool, args, eid)


def run(template, focal, mode, condition, variant="exact"):
    task = make(template, 0)
    with tempfile.TemporaryDirectory() as d:
        cfg = EpisodeConfig(episode_id="t", run="t", task=task.to_dict(), focal_step=focal, mode=mode,
                            variant=variant, condition=condition, delta_assumed_s=DELAY + 0.3, state_dir=d)
        tb = FakeTB(variant)
        disp = FakeDisp(cfg, tb)
        rep = run_scripted(cfg, tb, disp, False)
        for th in disp.threads:
            th.join()
        time.sleep(0.05)
        from actrec.tasks import resolve_args
        res = {}
        bad = []
        for st in task.steps:
            if st.tool == "batch_insert":
                rows = [len([e for e in tb.eff[st.tool] if e["alive"] and not e["actor"] and e["args"] == it])
                        for it in st.args["items"]]
                ok = all(r == 1 for r in rows)
            else:
                a = resolve_args(st.args, {k: {"number": 1} for k in ["open_ticket"]}) or st.args
                ok = tb.standing(st.tool, a if st.tool != "issue_comment" else st.args) == 1
            if not ok:
                bad.append(st.name)
        return {"bad": bad, "collateral": tb.collateral(), "claimed": rep["claimed"], "uncertain": rep["uncertain"]}


AMBIG = ["lost_ack", "not_exec_err", "late", "redelivery"]


@pytest.mark.parametrize("template,focal", [("T1", "open_issue"), ("T1", "emit_event"), ("T1", "log_row"),
                                            ("T7", "open_ticket"), ("T7", "triage_note"), ("T2", "pay_1"),
                                            ("T4", "stock_1"), ("T4", "count_restock")])
@pytest.mark.parametrize("mode", AMBIG)
def test_settle_converges_on_reversible(template, focal, mode):
    r = run(template, focal, mode, "S")
    assert r["bad"] == [] and r["collateral"] == 0, r


@pytest.mark.parametrize("mode", ["lost_ack", "not_exec_err", "late"])
def test_settle_mail_irreversible_with_known_bound(mode):
    r = run("T1", "page_oncall", mode, "S")
    assert r["bad"] == [], r


def test_settle_partial_batch():
    r = run("T5", "apply_batch", "partial", "S")
    assert r["bad"] == [], r


@pytest.mark.parametrize("mode", ["lost_ack", "not_exec_err", "late"])
def test_settle_lookalike(mode):
    r = run("T1", "open_issue", mode, "S", variant="lookalike")
    assert r["bad"] == [] and r["collateral"] == 0, r


def test_naive_duplicates_on_lost_ack():
    r = run("T1", "open_issue", "lost_ack", "C0")
    assert "open_issue" in r["bad"]


def test_vbr_fooled_by_lookalike_when_not_executed():
    r = run("T1", "open_issue", "not_exec_err", "C5", variant="lookalike")
    assert "open_issue" in r["bad"]          # the look-alike is taken as proof; ours is missing


def test_settle_without_tags_fooled_by_lookalike():
    r = run("T1", "open_issue", "not_exec_err", "S-t", variant="lookalike")
    assert "open_issue" in r["bad"] or r["collateral"] > 0


def test_settle_abstains_on_shared_aggregate():
    r = run("T4", "stock_1", "lost_ack", "S", variant="agg_shared")
    assert "stock_1" in r["uncertain"] and r["claimed"] != "completed"


def test_guard_blocks_unverifiable_aggregate():
    r = run("T4", "stock_1", "not_exec_err", "C6")
    assert "stock_1" in r["bad"]


def test_rollback_runs():
    r = run("T1", "log_row", "lost_ack", "C8")
    assert isinstance(r["bad"], list)
