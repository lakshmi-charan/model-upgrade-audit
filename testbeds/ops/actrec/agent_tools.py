"""The tools an LLM agent sees: writes (routed through the recovery layer), reads, its own compensation tools,
wait and finish. Descriptions state each service's documented behaviour (no hints about faults or methods).
"""
from __future__ import annotations

import time

from .contracts import Toolbox


def _fn(name, desc, props, req):
    """A function tool in the Responses API format (flat: name/description/parameters at the top level)."""
    return {"type": "function", "name": name, "description": desc, "strict": False,
            "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False}}


S, I = {"type": "string"}, {"type": "integer"}

WRITE_TOOLS = {   # agent tool name -> contract tool
    "create_issue": "issue_create", "add_comment": "issue_comment", "send_email": "mail_send",
    "send_bulk_email": "mail_send_plain", "insert_record": "row_insert", "insert_records": "batch_insert",
    "append_event": "event_append", "adjust_stock": "stock_adjust", "increment_counter": "counter_incr",
    "create_transfer": "transfer_create", "put_object": "object_put",
}
TOOL_FOR = {v: k for k, v in WRITE_TOOLS.items()}

SCHEMAS = {
    "create_issue": _fn("create_issue", "Create an issue in the team tracker. Returns the issue number.",
                        {"title": S, "body": S}, ["title", "body"]),
    "add_comment": _fn("add_comment", "Add a comment to an issue. Returns the comment id.",
                       {"issue_number": I, "body": S}, ["issue_number", "body"]),
    "send_email": _fn("send_email", "Send an e-mail. Sent mail cannot be recalled.",
                      {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    "send_bulk_email": _fn("send_bulk_email", "Send an e-mail through the bulk relay. Sent mail cannot be recalled.",
                           {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    "insert_record": _fn("insert_record", "Insert one record into the operations table. Returns the row id.",
                         {"item": S, "amount": I}, ["item", "amount"]),
    "insert_records": _fn("insert_records", "Insert several records one at a time. Not atomic: rows inserted "
                          "before a failure remain.",
                          {"items": {"type": "array", "items": {"type": "object", "properties": {"item": S, "amount": I},
                                                                "required": ["item", "amount"], "additionalProperties": False}}},
                          ["items"]),
    "append_event": _fn("append_event", "Append an event to the event stream. Returns the entry id.",
                        {"kind": S, "data": S}, ["kind", "data"]),
    "adjust_stock": _fn("adjust_stock", "Adjust the stock level of a SKU by delta (may be negative). Only the total "
                        "level can be read back; other processes may also adjust stock.", {"sku": S, "delta": I},
                        ["sku", "delta"]),
    "increment_counter": _fn("increment_counter", "Increment a named counter by one. Only the value can be read back.",
                             {"name": S}, ["name"]),
    "create_transfer": _fn("create_transfer", "Create a ledger transfer. transfer_id (optional, 32 hex characters) is "
                           "an idempotency key: a transfer id is applied at most once; reusing it returns 'exists'.",
                           {"amount": I, "memo": S, "transfer_id": S}, ["amount", "memo"]),
    "put_object": _fn("put_object", "Store an object under a name. Creating a name that already exists is refused.",
                      {"name": S, "body": S}, ["name", "body"]),
    # reads
    "list_issues": _fn("list_issues", "List all issues (number, title, state).", {}, []),
    "get_issue": _fn("get_issue", "Get an issue with its body and comments.", {"issue_number": I}, ["issue_number"]),
    "search_issues": _fn("search_issues", "Full-text search over issues. The search index is updated asynchronously.",
                         {"query": S}, ["query"]),
    "search_sent_email": _fn("search_sent_email", "List sent e-mails to a recipient (subject, date).", {"to": S}, ["to"]),
    "find_records": _fn("find_records", "Find records by item (id, item, amount).", {"item": S}, ["item"]),
    "list_events": _fn("list_events", "List events of a kind (entry id, data).", {"kind": S}, ["kind"]),
    "get_stock": _fn("get_stock", "Read the total stock level of a SKU.", {"sku": S}, ["sku"]),
    "get_counter": _fn("get_counter", "Read a counter's value.", {"name": S}, ["name"]),
    "find_transfers": _fn("find_transfers", "Find transfers by memo (ids).", {"memo": S}, ["memo"]),
    "object_exists": _fn("object_exists", "Check whether an object name exists.", {"name": S}, ["name"]),
    # agent's own corrections
    "delete_issue": _fn("delete_issue", "Delete an issue.", {"issue_number": I}, ["issue_number"]),
    "delete_comment": _fn("delete_comment", "Delete a comment.", {"comment_id": I}, ["comment_id"]),
    "delete_record": _fn("delete_record", "Delete a record by id.", {"id": I}, ["id"]),
    "delete_event": _fn("delete_event", "Delete an event entry.", {"kind": S, "entry_id": S}, ["kind", "entry_id"]),
    "reverse_transfer": _fn("reverse_transfer", "Reverse a transfer (creates an opposite transfer).",
                            {"transfer_id": S, "amount": I}, ["transfer_id", "amount"]),
    "wait": _fn("wait", "Wait for a number of seconds (max 60).", {"seconds": I}, ["seconds"]),
    "finish": _fn("finish", "Finish the task. status: completed | partial | failed. List operations whose outcome "
                  "you are not certain about.", {"status": {"type": "string", "enum": ["completed", "partial", "failed"]},
                                                 "summary": S, "uncertain_operations": {"type": "array", "items": S}},
                  ["status", "summary", "uncertain_operations"]),
}

READS_FOR = {
    "issue_create": ["list_issues", "get_issue", "search_issues", "delete_issue"],
    "issue_comment": ["get_issue", "delete_comment"],
    "mail_send": ["search_sent_email"], "mail_send_plain": ["search_sent_email"],
    "row_insert": ["find_records", "delete_record"], "batch_insert": ["find_records", "delete_record"],
    "event_append": ["list_events", "delete_event"], "stock_adjust": ["get_stock"],
    "counter_incr": ["get_counter"], "transfer_create": ["find_transfers", "reverse_transfer"],
    "object_put": ["object_exists"],
}


def tools_for(task) -> list[dict]:
    names = []
    for st in task.steps:
        for n in [TOOL_FOR[st.tool]] + READS_FOR[st.tool]:
            if n not in names:
                names.append(n)
    return [SCHEMAS[n] for n in names + ["wait", "finish"]]


def to_contract_args(name: str, a: dict) -> tuple[str, dict]:
    tool = WRITE_TOOLS[name]
    if tool == "issue_comment":
        return tool, {"issue": a.get("issue_number"), "body": a.get("body", "")}
    return tool, dict(a)


def run_read(tb: Toolbox, disp, name: str, a: dict):
    """Reads and the agent's own corrections. Corrections go through the dispatcher's compensation log."""
    if name == "list_issues":
        return [{"number": i["number"], "title": i["title"], "state": i["state"]} for i in tb.gitea.list_issues()]
    if name == "get_issue":
        n = int(a["issue_number"])
        with tb.gitea._client("clean", 30) as c:
            r = c.get(f"{tb.gitea._p}/issues/{n}")
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}"}
            i = r.json()
        return {"number": n, "title": i["title"], "body": i.get("body"), "state": i["state"],
                "comments": [{"id": c["id"], "body": c["body"]} for c in tb.gitea.list_comments(n)]}
    if name == "search_issues":
        with tb.gitea._client("clean", 30) as c:
            r = c.get(f"{tb.gitea._p}/issues", params={"q": a["query"], "state": "all", "type": "issues"})
            return [{"number": i["number"], "title": i["title"]} for i in r.json()] if r.status_code == 200 else []
    if name == "search_sent_email":
        return [{"subject": m.get("Subject"), "date": m.get("Created")} for m in tb.mail.all_for(tb.addr(a["to"]))]
    if name == "find_records":
        import psycopg

        from .config import pg_dsn
        with psycopg.connect(pg_dsn("clean"), autocommit=True) as c:
            return [{"id": r[0], "item": r[1], "amount": r[2]} for r in c.execute(
                "select id,item,amount from ar_rows where run=%s and item=%s order by id", (tb.run, a["item"])).fetchall()]
    if name == "list_events":
        return [{"entry_id": e, "data": f.get("data")} for e, f in tb.streams.kind_entries(a["kind"])]
    if name == "get_stock":
        return {"sku": a["sku"], "level": tb.pg.stock_level(a["sku"])}
    if name == "get_counter":
        return {"name": a["name"], "value": tb.streams.counter_value(a["name"])}
    if name == "find_transfers":
        return {"transfer_ids": [str(x) for x in tb.ledger.by_memo(a["memo"])]}
    if name == "object_exists":
        return {"exists": bool(tb.objects.exists(tb.obj_key(a["name"])))}
    if name == "delete_issue":
        n = int(a["issue_number"])
        with tb.gitea._client("clean", 30) as c:          # remember what is deleted, for the grader's ledger
            r = c.get(f"{tb.gitea._p}/issues/{n}")
            title = r.json().get("title") if r.status_code == 200 else None
        return {"result": disp.compensate("agent", "issue_create", {"title": title}, n)}
    if name == "delete_comment":
        cid = int(a["comment_id"])
        with tb.gitea._client("clean", 30) as c:
            r = c.get(f"{tb.gitea._p}/issues/comments/{cid}")
            body = r.json().get("body") if r.status_code == 200 else None
        return {"result": disp.compensate("agent", "issue_comment", {"body": body}, cid)}
    if name == "delete_record":
        return {"result": disp.compensate("agent", "row_insert", {}, int(a["id"]))}
    if name == "delete_event":
        return {"result": disp.compensate("agent", "event_append", {"kind": a["kind"]}, a["entry_id"])}
    if name == "reverse_transfer":
        return {"result": disp.compensate("agent", "transfer_create", {"amount": a["amount"]}, a["transfer_id"])}
    if name == "wait":
        s = max(0, min(60, int(a.get("seconds", 0))))
        time.sleep(s)
        return {"waited_s": s}
    return {"error": f"unknown tool {name}"}
