"""Stage-3 environment validation (no model calls, no cost).

For every service x network fault mode, run N trials of one real write through the fault proxy and measure,
from ground truth read directly on the service:
  * how many times the write actually executed (vs. the intended outcome of the mode),
  * when it became visible (to confirm late commits land after the caller gave up),
  * what the caller observed (to test observation equivalence of paired modes),
  * whether the operation tag survives byte-exact (tag fidelity).
Writes runs/validation/<stamp>/trials.jsonl, summary.json and summary.md.
"""
from __future__ import annotations

import argparse
import collections
import statistics
import time
import uuid
from pathlib import Path

from .config import RUNS, append_jsonl, stack_secrets, write_json
from .faults.toxi import EQUIV_GROUPS, INTENDED_EXECUTIONS, NETWORK_MODES, NetFault, Toxi
from .services import base
from .services.gitea import Gitea
from .services.ledger_tb import Ledger
from .services.mail import Mail
from .services.objstore import Objects
from .services.pg import Pg
from .services.redisx import Streams

KEYED = {"tb_transfer", "s3_put"}
ALL_SERVICES = ["gitea_issue", "mail", "pg_row", "redis_stream", "redis_counter", "tb_transfer", "s3_put", "pg_batch"]
BATCH_N, BATCH_K = 6, 3


class Targets:
    def __init__(self, run: str):
        tok = stack_secrets()["gitea_token"]
        self.gitea = Gitea(tok, repo=f"val-{run}")
        self.gitea.ensure_repo()
        self.mail = Mail()
        self.pg = Pg(run)
        self.streams = Streams(run)
        self.objects = Objects("actrec")          # one bucket for the whole stack; runs use key prefixes
        self.objects.ensure_bucket()
        self.run = run
        self.ledger: Ledger | None = None

    def ledger_ready(self) -> Ledger:
        if self.ledger is None:
            self.ledger = Ledger()
            self.ledger.setup_accounts()
        return self.ledger


def do_write(t: Targets, service: str, tag: str, f: NetFault):
    """Perform the focal write. Returns (CallResult, truth_fn, fidelity_fn or None)."""
    route, hook, dl = f.route, f.hook, f.deadline_s
    rep = 2 if f.mode == "redelivery" else 1
    if service == "gitea_issue":
        res = t.gitea.issue_create(f"Validation {tag[:8]}", "Validation issue.", tag=tag, route=route, hook=hook,
                                   deadline_s=dl, repeat=rep)
        return res, lambda: t.gitea.truth_count(tag), lambda: bool(t.gitea.find_by_tag(tag))
    if service == "mail":
        res = t.mail.send("ops@actrec.invalid", f"Validation {tag[:8]}", "Validation mail.", tag=tag, route=route,
                          hook=hook, deadline_s=dl, repeat=rep)

        def fid():
            ids = t.mail.find_by_message_id(tag)
            return bool(ids) and t.mail.headers(ids[0]).get("X-Op-Id") == [tag]
        return res, lambda: t.mail.truth_count(tag), fid
    if service == "pg_row":
        res = t.pg.row_insert("validation", 1, tag=tag, route=route, hook=hook, deadline_s=dl, repeat=rep)
        return res, lambda: t.pg.truth_count(tag), lambda: bool(t.pg.rows_by_reference(tag))
    if service == "pg_batch":
        items = [(f"batch-{i}", i) for i in range(BATCH_N)]
        res = t.pg.batch_insert(items, tag=tag, route=route, hook=hook, hook_before=BATCH_K, deadline_s=dl + 2)
        return res, lambda: t.pg.truth_batch(tag), None
    if service == "redis_stream":
        res = t.streams.append("validation", tag[:8], tag=tag, route=route, hook=hook, deadline_s=dl, repeat=rep)
        return res, lambda: t.streams.truth_count(tag), lambda: bool(t.streams.entries_by_tag(tag))
    if service == "redis_counter":
        res = t.streams.counter_incr(tag, 1, route=route, hook=hook, deadline_s=dl, repeat=rep)
        return res, lambda: t.streams.truth_counter(tag), None
    if service == "tb_transfer":
        lg = t.ledger_ready()
        res = lg.transfer(1, tag, route=route, hook=hook, deadline_s=dl, repeat=rep)
        return res, lambda: lg.truth_count(tag), lambda: lg.truth_count(tag) == 1
    if service == "s3_put":
        key = f"{t.run}/ops/{tag}.json"
        res = t.objects.put(key, b'{"v":1}', tag=tag, conditional=True, route=route, hook=hook, deadline_s=dl,
                            repeat=rep)
        return res, lambda: t.objects.truth_count(key), lambda: t.objects.metadata(key).get("op-id") == tag
    raise ValueError(service)


def expected(service: str, mode: str) -> int:
    if service == "pg_batch":
        return {"none": BATCH_N, "partial": BATCH_K}[mode]
    e = INTENDED_EXECUTIONS[mode]
    return min(e, 1) if service in KEYED else e


def modes_for(service: str, modes: list[str]) -> list[str]:
    return [m for m in ("none", "partial") if m in modes or m == "partial"] if service == "pg_batch" else \
        [m for m in modes if m != "partial"]


def trial(t: Targets, service: str, mode: str, i: int, deadline_s: float, delay_s: float) -> dict:
    tag = str(uuid.uuid4())
    f = NetFault(service=_proxy_of(service), mode=mode, deadline_s=deadline_s, delay_s=delay_s)
    t0 = time.time()
    res, truth, fid = do_write(t, service, tag, f)
    t_ret = time.time()
    seen = []
    at_return = _safe(truth)
    first_change = 0.0 if at_return else None
    last = at_return
    while time.time() - t_ret < f.settle_s():
        time.sleep(0.5)
        v = _safe(truth)
        seen.append(v)
        if first_change is None and v:
            first_change = round(time.time() - t0, 2)
        last = v
    f.cleanup()
    time.sleep(0.3)
    final = _safe(truth)
    row = {
        "service": service, "mode": mode, "trial": i, "tag": tag,
        "call": res.to_dict(), "returned_after_s": round(t_ret - t0, 3),
        "truth_at_return": at_return, "truth_before_cleanup": last, "truth_final": final,
        "first_visible_after_start_s": first_change, "expected_executions": expected(service, mode),
    }
    row["outcome_as_intended"] = final == row["expected_executions"]
    if mode == "late":
        row["late_landed_after_caller_gave_up"] = (at_return == 0 and final >= 1)
    if mode == "none" and fid is not None:
        try:
            row["tag_fidelity"] = bool(fid())
        except Exception as e:  # noqa: BLE001
            row["tag_fidelity"] = f"error: {type(e).__name__}"
    return row


def _proxy_of(service: str) -> str:
    return {"gitea_issue": "gitea", "mail": "smtp", "pg_row": "pg", "pg_batch": "pg", "redis_stream": "redis",
            "redis_counter": "redis", "tb_transfer": "tb", "s3_put": "s3"}[service]


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {str(e)[:80]}"


def tvd(a: list[str], b: list[str]) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    keys = set(ca) | set(cb)
    return round(0.5 * sum(abs(ca[k] / max(len(a), 1) - cb[k] / max(len(b), 1)) for k in keys), 3)


def summarize(rows: list[dict]) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["service"], r["mode"])].append(r)
    cells = {}
    for (s, m), rs in sorted(by.items()):
        el = [r["call"]["elapsed_s"] for r in rs]
        cells[f"{s}|{m}"] = {
            "n": len(rs),
            "expected": rs[0]["expected_executions"],
            "realized": dict(collections.Counter(str(r["truth_final"]) for r in rs)),
            "as_intended_rate": round(sum(r["outcome_as_intended"] for r in rs) / len(rs), 3),
            "client_kinds": dict(collections.Counter(r["call"]["kind"] for r in rs)),
            "signatures": dict(collections.Counter(r["call"]["signature"] for r in rs)),
            "elapsed_median_s": round(statistics.median(el), 3),
            "late_landed_after_giveup_rate": (round(sum(bool(r.get("late_landed_after_caller_gave_up")) for r in rs)
                                                    / len(rs), 3) if m == "late" else None),
            "tag_fidelity_rate": (round(sum(r.get("tag_fidelity") is True for r in rs) / len(rs), 3)
                                  if m == "none" and "tag_fidelity" in rs[0] else None),
        }
    equiv = {}
    services = sorted({r["service"] for r in rows})
    for s in services:
        for g, ms in EQUIV_GROUPS.items():
            sig = {m: [r["call"]["kind"] + ":" + str(r["call"]["exc_type"]) for r in by.get((s, m), [])] for m in ms}
            raw = {m: [r["call"].get("raw_signature", "") for r in by.get((s, m), [])] for m in ms}
            present = [m for m in ms if sig[m]]
            if len(present) < 2:
                continue
            pairs = {f"{a} vs {b}": tvd(sig[a], sig[b]) for i, a in enumerate(present) for b in present[i + 1:]}
            raw_pairs = {f"{a} vs {b}": tvd(raw[a], raw[b]) for i, a in enumerate(present) for b in present[i + 1:]}
            med = {m: round(statistics.median([r["call"]["elapsed_s"] for r in by[(s, m)]]), 3) for m in present}
            equiv[f"{s}|{g}"] = {"tvd_kind_exctype": pairs, "tvd_raw_library_exception": raw_pairs,
                                 "elapsed_median_s": med, "equivalent": all(v == 0 for v in pairs.values())}
    return {"cells": cells, "observation_equivalence": equiv}


def to_md(summary: dict, meta: dict) -> str:
    L = ["# Stage-3 environment validation", "",
         f"Run: `{meta['stamp']}` · trials per cell: {meta['trials']} · tool deadline {meta['deadline_s']} s · "
         f"late delay {meta['delay_s']} s · Toxiproxy {meta['toxiproxy']}", "",
         "## Realized outcome per service and fault", "",
         "| service | mode | n | expected execs | realized execs (count) | as intended | caller saw | median elapsed s | late after give-up | tag fidelity |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in summary["cells"].items():
        s, m = k.split("|")
        L.append(f"| {s} | {m} | {c['n']} | {c['expected']} | {c['realized']} | {c['as_intended_rate']} | "
                 f"{c['client_kinds']} | {c['elapsed_median_s']} | {c['late_landed_after_giveup_rate'] or ''} | "
                 f"{c['tag_fidelity_rate'] if c['tag_fidelity_rate'] is not None else ''} |")
    L += ["", "## Observation equivalence of paired modes", "",
          "TVD = total variation distance between the (kind, exception type) distributions; 0 means the caller "
          "cannot tell the modes apart from the error alone.", "",
          "Tool-level = what the tool layer reports (transport failures normalized); raw = the client library's "
          "own exception type, for audit.", "",
          "| service | group | TVD tool-level | TVD raw library exception | median elapsed s | equivalent (tool level) |",
          "|---|---|---|---|---|---|"]
    for k, e in summary["observation_equivalence"].items():
        s, g = k.split("|")
        L.append(f"| {s} | {g} | {e['tvd_kind_exctype']} | {e['tvd_raw_library_exception']} | "
                 f"{e['elapsed_median_s']} | {e['equivalent']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--services", default=",".join(ALL_SERVICES))
    ap.add_argument("--modes", default=",".join(NETWORK_MODES))
    ap.add_argument("--deadline", type=float, default=4.0)
    ap.add_argument("--delay", type=float, default=12.0, help="in-flight delay for mode 'late'")
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = stamp.replace("-", "")
    out = RUNS / "validation" / stamp
    toxi = Toxi()
    toxi.ensure_proxies()
    meta = {"stamp": stamp, "trials": a.trials, "deadline_s": a.deadline, "delay_s": a.delay,
            "toxiproxy": toxi.version(), "services": a.services.split(","), "modes": a.modes.split(",")}
    write_json(out / "meta.json", meta)
    t = Targets(run)
    rows = []
    modes = a.modes.split(",")
    for s in a.services.split(","):
        for m in modes_for(s, modes):
            for i in range(a.trials):
                r = trial(t, s, m, i, a.deadline, a.delay)
                rows.append(r)
                append_jsonl(out / "trials.jsonl", r)
                print(f"{s:14s} {m:13s} #{i:02d} caller={r['call']['kind']:7s} execs={r['truth_final']} "
                      f"expected={r['expected_executions']} ok={r['outcome_as_intended']}", flush=True)
    left = base.wait_inflight(30)
    summary = summarize(rows)
    summary["inflight_threads_left_at_end"] = left
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(to_md(summary, meta), encoding="utf-8")
    print(f"\nwrote {out}")
    bad = {k: v["as_intended_rate"] for k, v in summary["cells"].items() if v["as_intended_rate"] < 1}
    print("cells not fully as intended:", bad or "none")


if __name__ == "__main__":
    main()
