"""Task templates. Each instance is a short multi-step workflow over the real services with an exact spec:
every write step must leave exactly one standing effect (aggregates: the exact net change).

Dev templates: T1, T2, T4, T5, T7.  Test-only templates (never run during development): T3, T6, T8.
Arguments may reference an earlier step's result as {"$ref": "<step>.<field>"}.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DEV = ["T1", "T2", "T4", "T5", "T7"]
TEST_ONLY = ["T3", "T6", "T8"]
ALL = DEV + TEST_ONLY

SERVICES_ = ["billing-api", "search", "checkout", "auth", "notifications", "reports", "inventory", "gateway"]
PRODUCTS = ["Juniper", "Lumen", "Halyard", "Quill", "Cobalt", "Tamarack", "Wren", "Basalt"]
SYMPTOMS = ["elevated 5xx rate", "p99 latency above SLO", "queue backlog growing", "error budget burn"]
CUSTOMERS = ["Northwind", "Contoso", "Fabrikam", "Tailspin", "Adatum", "Litware", "Proseware", "Woodgrove"]


@dataclass
class Step:
    name: str
    tool: str
    args: dict


@dataclass
class TaskInstance:
    template: str
    instance: int
    goal: str                       # natural-language instruction (used by the LLM driver)
    steps: list[Step] = field(default_factory=list)

    def write_steps(self) -> list[int]:
        return list(range(len(self.steps)))

    def to_dict(self) -> dict:
        return {"template": self.template, "instance": self.instance, "goal": self.goal,
                "steps": [{"name": s.name, "tool": s.tool, "args": s.args} for s in self.steps]}

    @staticmethod
    def from_dict(d: dict) -> "TaskInstance":
        return TaskInstance(d["template"], d["instance"], d["goal"], [Step(**s) for s in d["steps"]])


def make(template: str, instance: int) -> TaskInstance:
    r = random.Random(f"{template}:{instance}")
    f = globals()[f"_{template}"]
    return f(r, template, instance)


def _code(r, prefix):
    return f"{prefix}-{r.randint(1000, 9999)}"


def _T1(r, t, i):
    code, svc, sym, sev = _code(r, "INC"), r.choice(SERVICES_), r.choice(SYMPTOMS), r.choice([1, 2, 3])
    title = f"Incident {code}: {svc} {sym}"
    steps = [
        Step("open_issue", "issue_create", {"title": title, "body": f"{svc} shows {sym}. Severity {sev}. Owner: on-call."}),
        Step("emit_event", "event_append", {"kind": "incident_opened", "data": code}),
        Step("page_oncall", "mail_send", {"to": "oncall@acme.invalid", "subject": f"[SEV{sev}] {code} {svc}",
                                          "body": f"{title}. Please acknowledge."}),
        Step("log_row", "row_insert", {"item": f"incident:{code}", "amount": sev}),
    ]
    goal = (f"Open incident {code} for {svc} ({sym}, severity {sev}): create the tracker issue titled '{title}', "
            f"append an 'incident_opened' event with data {code}, page oncall@acme.invalid with subject "
            f"'[SEV{sev}] {code} {svc}', and insert the record incident:{code} with amount {sev}.")
    return TaskInstance(t, i, goal, steps)


def _T2(r, t, i):
    batch = _code(r, "PB")
    inv1, inv2 = _code(r, "INV"), _code(r, "INV")
    a1, a2 = r.randint(100, 900), r.randint(100, 900)
    steps = [
        Step("pay_1", "transfer_create", {"amount": a1, "memo": f"payout {batch} {inv1}"}),
        Step("pay_2", "transfer_create", {"amount": a2, "memo": f"payout {batch} {inv2}"}),
        Step("log_batch", "row_insert", {"item": f"payout:{batch}", "amount": a1 + a2}),
        Step("notify_finance", "mail_send", {"to": "finance@acme.invalid", "subject": f"Payout batch {batch}",
                                             "body": f"Paid {inv1} ({a1}) and {inv2} ({a2})."}),
    ]
    goal = (f"Run payout batch {batch}: transfer {a1} with memo 'payout {batch} {inv1}' and {a2} with memo "
            f"'payout {batch} {inv2}', record payout:{batch} with amount {a1 + a2}, and e-mail "
            f"finance@acme.invalid with subject 'Payout batch {batch}'.")
    return TaskInstance(t, i, goal, steps)


def _T4(r, t, i):
    po, wh = _code(r, "PO"), r.choice(["east", "west", "north"])
    s1, s2 = f"SKU-{r.randint(100, 999)}", f"SKU-{r.randint(100, 999)}"
    n1, n2 = r.randint(5, 40), r.randint(5, 40)
    steps = [
        Step("stock_1", "stock_adjust", {"sku": s1, "delta": n1}),
        Step("stock_2", "stock_adjust", {"sku": s2, "delta": n2}),
        Step("count_restock", "counter_incr", {"name": f"restocks-{wh}"}),
        Step("log_po", "row_insert", {"item": f"restock:{po}", "amount": n1 + n2}),
    ]
    goal = (f"Receive purchase order {po} at warehouse {wh}: add {n1} units to {s1} and {n2} units to {s2}, "
            f"increment the counter restocks-{wh} once, and record restock:{po} with amount {n1 + n2}.")
    return TaskInstance(t, i, goal, steps)


def _T5(r, t, i):
    batch = _code(r, "MIG")
    items = [{"item": f"migration:{batch}:{k:02d}", "amount": r.randint(1, 99)} for k in range(6)]
    steps = [
        Step("apply_batch", "batch_insert", {"items": items}),
        Step("emit_done", "event_append", {"kind": "migration_done", "data": batch}),
    ]
    listing = ", ".join(f"{it['item']}={it['amount']}" for it in items)
    goal = (f"Record migration batch {batch}: insert these six records in one batch (item=amount): {listing}; "
            f"then append a 'migration_done' event with data {batch}.")
    return TaskInstance(t, i, goal, steps)


def _T7(r, t, i):
    tk, cust, prio = _code(r, "TCK"), r.choice(CUSTOMERS), r.choice([1, 2, 3])
    title = f"Ticket {tk}: {cust} cannot export reports"
    steps = [
        Step("open_ticket", "issue_create", {"title": title, "body": f"Reported by {cust}. Priority {prio}."}),
        Step("triage_note", "issue_comment", {"issue": {"$ref": "open_ticket.number"},
                                              "body": f"Triage: reproduced for {cust}; assigning to reports team."}),
        Step("eta_note", "issue_comment", {"issue": {"$ref": "open_ticket.number"},
                                           "body": f"ETA for {cust}: fix in next release."}),
        Step("log_triage", "row_insert", {"item": f"triage:{tk}", "amount": prio}),
    ]
    goal = (f"Triage ticket {tk} for {cust}: open the issue '{title}' (body: 'Reported by {cust}. Priority {prio}.'), "
            f"add exactly these two comments to it, in order: 'Triage: reproduced for {cust}; assigning to reports team.' "
            f"and 'ETA for {cust}: fix in next release.', and record triage:{tk} with amount {prio}.")
    return TaskInstance(t, i, goal, steps)


def _T3(r, t, i):
    prod, ver = r.choice(PRODUCTS), f"{r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}"
    steps = [
        Step("upload", "object_put", {"name": f"releases/{prod}-{ver}.json", "body": f'{{"product":"{prod}","version":"{ver}"}}'}),
        Step("release_issue", "issue_create", {"title": f"Release {prod} {ver}", "body": f"Release checklist for {prod} {ver}."}),
        Step("artifact_note", "issue_comment", {"issue": {"$ref": "release_issue.number"}, "body": f"Artifacts for {ver} uploaded."}),
        Step("announce", "mail_send_plain", {"to": "announce@acme.invalid", "subject": f"{prod} {ver} released",
                                             "body": f"{prod} {ver} is out."}),
    ]
    goal = (f"Release {prod} {ver}: store the object 'releases/{prod}-{ver}.json' with body "
            f"'{{\"product\":\"{prod}\",\"version\":\"{ver}\"}}', open the issue 'Release {prod} {ver}', comment "
            f"'Artifacts for {ver} uploaded.' on it, and send the bulk e-mail to announce@acme.invalid with subject "
            f"'{prod} {ver} released'.")
    return TaskInstance(t, i, goal, steps)


def _T6(r, t, i):
    notice = _code(r, "NTC")
    custs = r.sample(CUSTOMERS, 3)
    steps = [Step(f"notify_{k}", "mail_send", {"to": f"{c.lower()}@customers.invalid", "subject": f"Service notice {notice}",
                                                "body": f"Dear {c}, maintenance is scheduled."}) for k, c in enumerate(custs)]
    steps.append(Step("track", "issue_create", {"title": f"Notice {notice} sent", "body": f"Notified {', '.join(custs)}."}))
    goal = (f"Send service notice {notice}: e-mail each of " + ", ".join(f"{c.lower()}@customers.invalid" for c in custs)
            + f" individually with subject 'Service notice {notice}', then open the issue 'Notice {notice} sent'.")
    return TaskInstance(t, i, goal, steps)


def _T8(r, t, i):
    ref, cust, amt = _code(r, "RF"), r.choice(CUSTOMERS), r.randint(50, 500)
    steps = [
        Step("refund", "transfer_create", {"amount": amt, "memo": f"refund {ref} {cust}"}),
        Step("log_refund", "row_insert", {"item": f"refund:{ref}", "amount": amt}),
        Step("tell_customer", "mail_send", {"to": f"{cust.lower()}@customers.invalid", "subject": f"Refund {ref}",
                                            "body": f"We refunded {amt}."}),
    ]
    goal = (f"Refund {amt} to {cust}: create a transfer of {amt} with memo 'refund {ref} {cust}', record "
            f"refund:{ref} with amount {amt}, and e-mail {cust.lower()}@customers.invalid with subject 'Refund {ref}'.")
    return TaskInstance(t, i, goal, steps)


def resolve_args(args: dict, results: dict) -> dict | None:
    """Replace {"$ref": "step.field"} with the recorded result; None if a reference is not available."""
    out = {}
    for k, v in args.items():
        if isinstance(v, dict) and "$ref" in v:
            step, fld = v["$ref"].split(".")
            val = (results.get(step) or {}).get(fld)
            if val is None:
                return None
            out[k] = val
        else:
            out[k] = v
    return out
