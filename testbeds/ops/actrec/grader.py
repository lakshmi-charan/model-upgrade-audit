"""Ground-truth grading (parent process only; agents and layers never import this module).

For every plan step it counts, on the real services via the direct ports:
  executions  how many times the step's effect was ever created (including later compensated ones)
  standing    how many of the step's effects stand at quiescence (aggregates: net units)
Sources: service state; PostgreSQL audit triggers; Redis stream `entries-added` and the atomic counter audit
list; TigerBeetle's immutable transfers; the Mailpit capture store (mail cannot be deleted by any tool); the
dispatcher's compensation log (harness instrumentation) for deleted issues/comments/objects. The background
actor's own effects are identified from the actor log and excluded; any of them that disappeared is
collateral damage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import SPECS, Toolbox
from .durable import Log
from .services.ledger_tb import CODE, CODE_REVERSAL, memo_code
from .tasks import TaskInstance


def norm(x) -> str:
    """Case/whitespace/trailing-period-insensitive text comparison (agents may re-type given strings)."""
    return re.sub(r"[\s.]+$", "", re.sub(r"\s+", " ", str(x or "").strip().casefold()))


def _comment_text(body: str) -> str:
    return (body or "").split("\n\nop-id:")[0]


def _comp_ok(rec: dict) -> bool:
    r = str(rec.get("result"))
    return r in ("204", "1", "ok", "deleted", "CREATED") or r.startswith("ok")


def grade(state_dir: Path, task: TaskInstance, tb: Toolbox) -> dict:
    state_dir = Path(state_dir)
    actor = Log(state_dir / "actor.jsonl").records()
    dlog = Log(state_dir / "dispatch.jsonl").records()
    comps = [r for r in dlog if r.get("type") == "compensate" and _comp_ok(r)]
    actor_ids = {(a["tool"], str(a.get("id"))) for a in actor if a.get("id") is not None}
    actor_agg = {}
    for a in actor:
        if "delta" in a:
            actor_agg[a["tool"]] = actor_agg.get(a["tool"], 0) + int(a["delta"])
    steps, collateral = {}, 0
    audit = tb.pg.audit()
    issue_numbers: dict[str, list[int]] = {}
    comment_actor_checked: set[int] = set()        # count each actor comment at most once per episode

    for st in task.steps:
        tool, a = st.tool, st.args
        spec = SPECS[tool]
        g: dict = {"tool": tool, "cls": spec.cls, "reversible": spec.reversible}
        comp_here = [c for c in comps if (tool, str(c["effect_id"])) not in actor_ids and c["tool"] == tool and (
            c["step"] == st.name or (c["step"] == "agent" and _content_match(tool, c.get("args") or {}, a)))]
        if tool == "issue_create":
            all_t = [i["number"] for i in tb.gitea.list_issues(route="direct") if norm(i.get("title")) == norm(a["title"])]
            nums = [n for n in all_t if ("issue_create", str(n)) not in actor_ids]
            issue_numbers[st.name] = nums
            g["standing"] = len(nums)
            g["executions"] = len(nums) + len(comp_here)
            gone = [x for x in actor if x["tool"] == tool and x.get("id") and x["id"] not in all_t]
            collateral += len(gone)
        elif tool == "issue_comment":
            ref = a["issue"]["$ref"].split(".")[0] if isinstance(a.get("issue"), dict) else None
            nums = issue_numbers.get(ref, []) if ref else [int(a["issue"])]
            ids = []
            all_ids = []
            for n in nums:
                cs = [c["id"] for c in tb.gitea.list_comments(n, route="direct")
                      if norm(_comment_text(c.get("body"))) == norm(a["body"])]
                all_ids += cs
                ids += [c for c in cs if ("issue_comment", str(c)) not in actor_ids]
            g["standing"] = len(ids)
            g["executions"] = len(ids) + len(comp_here)
            # actor comments sit on the agent's issue; they count as collateral only if they are no longer
            # among ANY comment of that issue (not only among comments with this step's text)
            present = {int(c["id"]) for n in nums for c in tb.gitea.list_comments(n, route="direct")}
            if comment_actor_checked.isdisjoint(nums):
                gone = [x for x in actor if x["tool"] == tool and x.get("id") and int(x["id"]) not in present]
                collateral += len(gone)
                comment_actor_checked.update(nums)
        elif tool in ("mail_send", "mail_send_plain"):
            to = tb.addr(a["to"])
            msgs = [m for m in tb.mail.all_for(to) if norm(m.get("Subject")) == norm(a["subject"])
                    and (tool, m.get("MessageID")) not in actor_ids]
            g["standing"] = g["executions"] = len(msgs)
        elif tool == "row_insert":
            ins = [x for x in audit if x["tbl"] == "rows" and x["op"] == "I" and x["origin"] == "agent"
                   and x["item"] == a["item"] and x["amount"] == int(a["amount"])]
            dels = {x["row_id"] for x in audit if x["tbl"] == "rows" and x["op"] == "D"}
            g["executions"] = len(ins)
            g["standing"] = sum(1 for x in ins if x["row_id"] not in dels)
            collateral += sum(1 for x in audit if x["tbl"] == "rows" and x["op"] == "D" and x["origin"] == "actor"
                              and x["item"] == a["item"])
        elif tool == "batch_insert":
            per = []
            for it in a["items"]:
                ins = [x for x in audit if x["tbl"] == "rows" and x["op"] == "I" and x["origin"] == "agent"
                       and x["item"] == it["item"] and x["amount"] == int(it["amount"])]
                dels = {x["row_id"] for x in audit if x["tbl"] == "rows" and x["op"] == "D"}
                per.append((len(ins), sum(1 for x in ins if x["row_id"] not in dels)))
            g["rows"] = per
            g["executions"] = sum(e for e, _ in per)
            g["standing"] = sum(s for _, s in per)
            g["expected"] = len(per)
            g["rows_wrong"] = sum(1 for _, s in per if s != 1)
        elif tool == "event_append":
            entries = tb.streams.kind_entries(a["kind"], route="direct")
            mine = [e for e, f in entries if f.get("_origin") != "actor" and f.get("data") == a["data"]]
            actor_standing = {e for e, f in entries if f.get("_origin") == "actor"}
            actor_all = {str(x["id"]) for x in actor if x["tool"] == tool and x.get("id")}
            added = tb.streams.kind_entries_added(a["kind"])
            deleted = max(0, added - len(entries))
            actor_deleted = len(actor_all - actor_standing)
            g["standing"] = len(mine)
            g["executions"] = len(mine) + max(0, deleted - actor_deleted)
            collateral += actor_deleted
        elif tool == "stock_adjust":
            rows = [x for x in audit if x["tbl"] == "stock" and x["op"] == "I" and x["origin"] == "agent"
                    and x["sku"] == a["sku"]]
            d = int(a["delta"])
            g["executions"] = sum(1 for x in rows if x["delta"] == d)
            g["standing"] = sum(x["delta"] for x in rows) // d
        elif tool == "counter_incr":
            au = tb.streams.counter_audit(a["name"])
            g["executions"] = sum(1 for x in au if x > 0)
            g["standing"] = sum(au)
        elif tool == "transfer_create":
            ts = tb.ledger.all_transfers("direct")
            mine = [t for t in ts if t.code == CODE and t.user_data_64 == memo_code(a["memo"])]
            rev = {int(t.user_data_128) for t in ts if t.code == CODE_REVERSAL}
            g["executions"] = len(mine)
            g["standing"] = sum(1 for t in mine if int(t.id) not in rev)
        elif tool == "object_put":
            ex = tb.objects.truth_count(tb.obj_key(a["name"]))
            g["standing"] = ex
            g["executions"] = ex + len(comp_here)
        g.setdefault("expected", 1)
        g["ok"] = (g.get("rows_wrong", 0) == 0) if tool == "batch_insert" else (g["standing"] == g["expected"])
        g["missing"] = g["standing"] < g["expected"] if tool != "batch_insert" else any(s == 0 for _, s in g["rows"])
        g["surplus"] = g["standing"] > g["expected"] if tool != "batch_insert" else any(s > 1 for _, s in g["rows"])
        g["dup_execs"] = (sum(max(0, e - 1) for e, _ in g["rows"]) if tool == "batch_insert"
                          else max(0, g["executions"] - 1))
        steps[st.name] = g
    inconsistent = (not all(g["ok"] for g in steps.values())) or collateral > 0
    return {
        "steps": steps,
        "residual_inconsistent": inconsistent,
        "collateral": collateral,
        "dup_execs": sum(g["dup_execs"] for g in steps.values()),
        "irreversible_dup_execs": sum(g["dup_execs"] for g in steps.values() if g["reversible"] == "none"),
        "missing_steps": sum(1 for g in steps.values() if g["missing"]),
        "surplus_steps": sum(1 for g in steps.values() if g["surplus"]),
        "compensations_logged": len([r for r in dlog if r.get("type") == "compensate"]),
        "dispatches": len([r for r in dlog if r.get("type") == "dispatch"]),
        "actor_effects": len(actor),
    }


def _content_match(tool: str, logged: dict, plan: dict) -> bool:
    """Attribute an agent's own deletion (logged with the deleted effect's content) to a plan step."""
    if tool == "issue_create":
        return norm(logged.get("title")) == norm(plan.get("title"))
    if tool == "issue_comment":
        return norm(_comment_text(logged.get("body"))) == norm(plan.get("body"))
    return False


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
