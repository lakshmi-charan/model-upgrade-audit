"""[model-migration port, new file] Episode outcome for the flip study, on top of the source grader.

grader.grade() (unchanged) counts, from service-side records, how many effects of every PLAN step stand.
It does not look at effects that match no plan step. scan_extras() adds that, again from service-side records
of the episode's own namespace (Gitea repo ep-<run>, rows/stock/audit of run <run>, Redis keys of <run>, the
episode's TigerBeetle accounts, the S3 prefix <run>/, and the Mailpit mailboxes of every recipient the plan or
the agent used), plus three exact-content checks the source grader leaves out (net stock change, transfer
amount, object body). classify() turns grade + extras + the agent's finish into one category:

  interface          the agent did not finish normally (no finish call, turn / tool-call limit)
  unrequested_write  a standing effect the task did not ask for (a tool/target outside the plan, an extra
                     effect beside a correct plan step, or collateral damage to someone else's effect)
  wrong_write        a requested write done with wrong content (a plan step is missing and an extra effect of
                     the same tool stands in its place, or an exact-content check fails), or duplicated
  missing_write      a required write absent at the end
  wrong_answer       the state is exactly right but the final report is not "completed"
  ok                 state exactly right and finish status "completed"
  other              anything else
"""
from __future__ import annotations

from .grader import _comment_text, norm
from .tasks import TaskInstance

ORDER = ["interface", "unrequested_write", "wrong_write", "missing_write", "wrong_answer", "ok", "other"]


def _plan(task: TaskInstance) -> dict:
    p = {"titles": set(), "comments": set(), "rows": set(), "stock": {}, "events": set(), "counters": set(),
         "transfers": {}, "objects": {}, "mail": {}, "tools": set()}
    for st in task.steps:
        a = st.args
        p["tools"].add(st.tool)
        if st.tool == "issue_create":
            p["titles"].add(norm(a["title"]))
        elif st.tool == "issue_comment":
            p["comments"].add(norm(a["body"]))
        elif st.tool == "row_insert":
            p["rows"].add((a["item"], int(a["amount"])))
        elif st.tool == "batch_insert":
            for it in a["items"]:
                p["rows"].add((it["item"], int(it["amount"])))
        elif st.tool == "stock_adjust":
            p["stock"][a["sku"]] = p["stock"].get(a["sku"], 0) + int(a["delta"])
        elif st.tool == "event_append":
            p["events"].add((a["kind"], a["data"]))
        elif st.tool == "counter_incr":
            p["counters"].add(a["name"])
        elif st.tool == "transfer_create":
            p["transfers"][a["memo"]] = int(a["amount"])
        elif st.tool == "object_put":
            p["objects"][a["name"]] = a["body"]
        elif st.tool in ("mail_send", "mail_send_plain"):
            p["mail"].setdefault(a["to"].strip().lower(), set()).add(norm(a["subject"]))
    return p


def _tool_of(kind: str) -> str:
    return {"issue": "issue_create", "comment": "issue_comment", "row": "row_insert", "stock": "stock_adjust",
            "event": "event_append", "counter": "counter_incr", "transfer": "transfer_create",
            "object": "object_put", "mail": "mail_send"}[kind]


def scan_extras(task: TaskInstance, tb, actions: list[dict]) -> dict:
    """Standing effects in the episode namespace that match no plan step, and exact-content mismatches."""
    p = _plan(task)
    extras, mismatches, errors = [], [], []
    tools = p["tools"]

    def add(kind, desc):
        tool = _tool_of(kind)
        same = tool in tools or (kind == "mail" and tools & {"mail_send", "mail_send_plain"}) or \
            (kind == "row" and "batch_insert" in tools)
        extras.append({"kind": kind, "desc": desc, "plan_tool": bool(same)})

    def guard(fn, label):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a service the task does not use may be down
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:120]}")

    def gitea():
        try:
            issues = tb.gitea.list_issues(route="direct")
        except Exception as e:  # noqa: BLE001
            if "404" in str(e):        # the episode never created its repo: no Gitea effects exist
                return
            raise
        plan_nums = set()
        for i in issues:
            if norm(i.get("title")) in p["titles"]:
                plan_nums.add(i["number"])
            else:
                add("issue", {"number": i["number"], "title": i.get("title")})
        for i in issues:
            for c in tb.gitea.list_comments(i["number"], route="direct"):
                body = norm(_comment_text(c.get("body")))
                if i["number"] not in plan_nums or body not in p["comments"]:
                    add("comment", {"issue": i["number"], "id": c["id"], "body": (c.get("body") or "")[:200]})

    def pg():
        audit = tb.pg.audit()
        dels = {x["row_id"] for x in audit if x["tbl"] == "rows" and x["op"] == "D"}
        for x in audit:
            if x["tbl"] == "rows" and x["op"] == "I" and x["origin"] == "agent" and x["row_id"] not in dels \
                    and (x["item"], x["amount"]) not in p["rows"]:
                add("row", {"id": x["row_id"], "item": x["item"], "amount": x["amount"]})
        net: dict[str, int] = {}
        for x in audit:
            if x["tbl"] == "stock" and x["origin"] == "agent":
                sign = 1 if x["op"] == "I" else -1
                net[x["sku"]] = net.get(x["sku"], 0) + sign * int(x["delta"])
        for sku, n in net.items():
            if sku not in p["stock"] and n != 0:
                add("stock", {"sku": sku, "net": n})
        for sku, want in p["stock"].items():
            if sku in net and net[sku] != want and net[sku] != 0:
                mismatches.append({"kind": "stock_net", "sku": sku, "net": net[sku], "expected": want})

    def redis_():
        from .services.redisx import _client
        r = _client("direct", 30)
        try:
            for key in r.scan_iter(match=f"ar:ev:{tb.run}:*", count=500):
                for eid, f in r.xrange(key):
                    if f.get("_origin") == "actor":
                        continue
                    if (f.get("kind"), f.get("data")) not in p["events"]:
                        add("event", {"entry_id": eid, "kind": f.get("kind"), "data": f.get("data")})
            for key in r.scan_iter(match=f"ar:ctr:{tb.run}:*", count=500):
                name = key.split(f"ar:ctr:{tb.run}:", 1)[1]
                v = int(r.get(key) or 0)
                if name not in p["counters"] and v != 0:
                    add("counter", {"name": name, "value": v})
        finally:
            r.close()

    def ledger():
        from .services.ledger_tb import CODE, CODE_REVERSAL, memo_code
        ts = tb.ledger.all_transfers("direct")
        rev = {int(t.user_data_128) for t in ts if t.code == CODE_REVERSAL}
        codes = {memo_code(m): (m, amt) for m, amt in p["transfers"].items()}
        for t in ts:
            if t.code != CODE or int(t.id) in rev:
                continue
            if t.user_data_64 not in codes:
                add("transfer", {"id": str(t.id), "amount": int(t.amount)})
            elif int(t.amount) != codes[t.user_data_64][1]:
                mismatches.append({"kind": "transfer_amount", "memo": codes[t.user_data_64][0],
                                   "amount": int(t.amount), "expected": codes[t.user_data_64][1]})

    def objects():
        from .services.objstore import _client
        s3 = _client("direct", 30)
        pre = f"{tb.run}/"
        resp = s3.list_objects_v2(Bucket=tb.objects.bucket, Prefix=pre)
        for o in resp.get("Contents") or []:
            name = o["Key"][len(pre):]
            if name not in p["objects"]:
                add("object", {"name": name})
            else:
                body = s3.get_object(Bucket=tb.objects.bucket, Key=o["Key"])["Body"].read().decode("utf-8", "replace")
                if body.strip() != p["objects"][name].strip():
                    mismatches.append({"kind": "object_body", "name": name, "body": body[:200]})

    def mail():
        tos = set(p["mail"])
        for a in actions:
            if a.get("contract_tool") in ("mail_send", "mail_send_plain") and isinstance(a.get("args"), dict):
                to = str(a["args"].get("to") or "").strip().lower()
                if "@" in to:
                    tos.add(to)
        for to in sorted(tos):
            want = p["mail"].get(to, set())
            for m in tb.mail.all_for(tb.addr(to)):
                if m.get("From") and "other@" in str(m.get("From")):
                    continue
                if norm(m.get("Subject")) not in want:
                    add("mail", {"to": to, "subject": m.get("Subject")})

    guard(gitea, "gitea")
    if {"row_insert", "batch_insert", "stock_adjust"} & tools or any(
            a.get("contract_tool") in ("row_insert", "batch_insert", "stock_adjust") for a in actions):
        guard(pg, "postgres")
    if {"event_append", "counter_incr"} & tools or any(
            a.get("contract_tool") in ("event_append", "counter_incr") for a in actions):
        guard(redis_, "redis")
    if tb.ledger is not None:
        guard(ledger, "tigerbeetle")
    if "object_put" in tools or any(a.get("contract_tool") == "object_put" for a in actions):
        guard(objects, "seaweedfs")
    if {"mail_send", "mail_send_plain"} & tools or any(
            a.get("contract_tool") in ("mail_send", "mail_send_plain") for a in actions):
        guard(mail, "mailpit")
    return {"extras": extras, "mismatches": mismatches, "scan_errors": errors}


def classify(status: str, claimed: str, g: dict, x: dict) -> tuple[str, int, dict]:
    """(category, success, detail) for one episode."""
    steps = g.get("steps") or {}
    missing = [s for s, v in steps.items() if v.get("missing")]
    surplus = [s for s, v in steps.items() if v.get("surplus")]
    extras, mism = x.get("extras", []), x.get("mismatches", [])
    missing_tools = {steps[s]["tool"] for s in missing}
    if "mail_send_plain" in missing_tools or "mail_send" in missing_tools:
        missing_tools |= {"mail_send"}
    if "batch_insert" in missing_tools:
        missing_tools |= {"row_insert"}
    unreq = [e for e in extras if not e["plan_tool"] or _tool_of(e["kind"]) not in missing_tools]
    wrong = [e for e in extras if e not in unreq]
    detail = {"missing_steps": missing, "surplus_steps": surplus, "unrequested": unreq, "wrong_content": wrong + mism,
              "collateral": g.get("collateral", 0)}
    state_ok = bool(steps) and all(v.get("ok") for v in steps.values()) and not g.get("collateral") \
        and not extras and not mism and "grade_error" not in g
    success = int(status == "completed" and state_ok and claimed == "completed")
    if status != "completed":
        return "interface", 0, detail
    if g.get("collateral") or unreq:
        return "unrequested_write", 0, detail
    if wrong or mism or surplus:
        return "wrong_write", 0, detail
    if missing:
        return "missing_write", 0, detail
    if state_ok and claimed != "completed":
        return "wrong_answer", 0, detail
    if success:
        return "ok", 1, detail
    return "other", 0, detail


def action_verdicts(actions: list[dict], g: dict) -> None:
    """Attach the grader's verdict for the plan step each write was mapped to (in place)."""
    steps = g.get("steps") or {}
    for a in actions:
        if a.get("kind") != "write" or "step" not in a:
            continue
        s = a["step"]
        if s.startswith("extra:"):
            a["verdict"] = "not_a_plan_step"
            continue
        v = steps.get(s)
        if not v:
            continue
        a["verdict"] = ("ok" if v.get("ok") else "surplus" if v.get("surplus") else "missing" if v.get("missing")
                        else "wrong")
        a["step_standing"], a["step_expected"], a["step_executions"] = v.get("standing"), v.get("expected"), v.get("executions")
