"""Tool contracts and the per-episode toolbox.

A ToolSpec is the *contract* of a write as the theory defines it (`docs/05_theory.md`):
  cls         K = native key, A = key-less but taggable (attributable), U = aggregate only, C = content only
  reversible  targeted | fungible | none
  intent      the argument fields that identify one logical intent (used for content matching)
  lag_s       documented visibility lag of the agent's read path in this variant
The Toolbox binds the contracts to one episode's namespace on the real services and offers three families
of calls: execute (a write, via the dispatcher only), reads (by tag, by content, aggregate), compensate.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .config import GITEA_ADMIN
from .services.base import CallResult, nohook
from .services.gitea import Gitea
from .services.ledger_tb import Ledger
from .services.mail import Mail
from .services.objstore import Objects
from .services.pg import Pg
from .services.redisx import Streams


@dataclass(frozen=True)
class ToolSpec:
    name: str
    service: str            # proxy name used for faults
    cls: str                # K | A | U | C
    reversible: str         # targeted | fungible | none
    intent: tuple[str, ...]
    doc: str


SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("issue_create", "gitea", "A", "targeted", ("title",),
             "Create an issue in the team tracker. Returns the issue number."),
    ToolSpec("issue_comment", "gitea", "A", "targeted", ("issue", "body"),
             "Add a comment to an issue. Returns the comment id."),
    ToolSpec("mail_send", "smtp", "A", "none", ("to", "subject"),
             "Send an e-mail. Sent mail cannot be recalled."),
    ToolSpec("mail_send_plain", "smtp", "C", "none", ("to", "subject"),
             "Send an e-mail through the bulk relay (no custom headers). Sent mail cannot be recalled."),
    ToolSpec("row_insert", "pg", "A", "targeted", ("item", "amount"),
             "Insert one record into the operations table. Returns the row id."),
    ToolSpec("batch_insert", "pg", "A", "targeted", ("items",),
             "Insert several records, one at a time (not atomic: rows before a failure remain)."),
    ToolSpec("event_append", "redis", "A", "targeted", ("kind", "data"),
             "Append an event to the event stream. Returns the entry id."),
    ToolSpec("stock_adjust", "pg", "U", "fungible", ("sku", "delta"),
             "Adjust the stock level of a SKU by delta. Only the total level can be read."),
    ToolSpec("counter_incr", "redis", "U", "fungible", ("name",),
             "Increment a named counter by one. Only the counter value can be read."),
    ToolSpec("transfer_create", "tb", "K", "targeted", ("amount", "memo"),
             "Create a ledger transfer. The transfer id is an idempotency key: the same id is applied at most once."),
    ToolSpec("object_put", "s3", "K", "targeted", ("name",),
             "Store an object under a name; creating an existing name is refused (conditional create)."),
]}


def intent_of(tool: str, args: dict) -> tuple:
    return (tool,) + tuple(repr(args.get(f)) for f in SPECS[tool].intent)


class Toolbox:
    """All service access for one episode. Namespaced by `run` so episodes never see each other."""

    def __init__(self, run: str, gitea_token: str, variant: str = "exact", tb_accounts: tuple | None = None):
        self.run, self.variant = run, variant
        self.gitea = Gitea(gitea_token, repo=f"ep-{run}")
        self.mail = Mail()
        self.pg = Pg(run)
        self.streams = Streams(run)
        self.objects = Objects("actrec")
        self.ledger = Ledger(*(int(x) for x in tb_accounts), namespace=run) if tb_accounts else None

    # ------------------------------------------------------------------ contract queries
    def spec(self, tool: str) -> ToolSpec:
        return SPECS[tool]

    def lag_s(self, tool: str) -> float:
        return 3.0 if (self.variant == "lagged" and tool in ("issue_create",)) else 0.0

    def aggregate_shared(self, tool: str) -> bool:
        return self.variant == "agg_shared" and SPECS[tool].cls == "U"

    def obj_key(self, name: str) -> str:
        return f"{self.run}/{name}"

    def addr(self, to: str) -> str:
        """Plus-addressing keeps each episode's mail separate in the shared capture store."""
        local, dom = to.split("@", 1)
        return f"{local}+{self.run}@{dom}"

    # ------------------------------------------------------------------ writes (dispatcher only)
    def execute(self, tool: str, args: dict, *, tag: str | None = None, key: str | None = None, route: str = "clean",
                hook=nohook, deadline_s: float = 4.0, repeat: int = 1, batch_only: list[int] | None = None,
                batch_hook_before: int = 0) -> CallResult:
        g = dict(route=route, hook=hook, deadline_s=deadline_s)
        if tool == "issue_create":
            return self.gitea.issue_create(args["title"], args["body"], tag=tag, repeat=repeat, **g)
        if tool == "issue_comment":
            return self.gitea.comment_create(int(args["issue"]), args["body"], tag=tag, repeat=repeat, **g)
        if tool == "mail_send":
            return self.mail.send(self.addr(args["to"]), args["subject"], args["body"], tag=tag, repeat=repeat, **g)
        if tool == "mail_send_plain":
            return self.mail.send(self.addr(args["to"]), args["subject"], args["body"], tag=None, repeat=repeat, **g)
        if tool == "row_insert":
            return self.pg.row_insert(args["item"], int(args["amount"]), tag=tag, repeat=repeat, **g)
        if tool == "batch_insert":
            items = [(it["item"], int(it["amount"])) for it in args["items"]]
            idx = batch_only if batch_only is not None else list(range(len(items)))
            return self._batch(items, idx, tag, route, hook, batch_hook_before, deadline_s)
        if tool == "event_append":
            return self.streams.append(args["kind"], args["data"], tag=tag, repeat=repeat, per_kind=True, **g)
        if tool == "stock_adjust":
            return self.pg.stock_adjust(args["sku"], int(args["delta"]), repeat=repeat, **g)
        if tool == "counter_incr":
            return self.streams.counter_add(args["name"], 1, repeat=repeat, **g)
        if tool == "transfer_create":
            return self.ledger.transfer(int(args["amount"]), key or str(uuid.uuid4()), repeat=repeat,
                                        memo=args["memo"], **g)
        if tool == "object_put":
            return self.objects.put(self.obj_key(args["name"]), args["body"].encode(), tag=tag or key,
                                    conditional=True, repeat=repeat, **g)
        raise ValueError(tool)

    def _batch(self, items, idx, tag, route, hook, hook_before, deadline_s) -> CallResult:
        import psycopg

        from .config import pg_dsn
        from .services.base import call_with_deadline

        def fn():
            ids = {}
            with psycopg.connect(pg_dsn(route), autocommit=True) as c:
                c.execute("select 1").fetchone()
                for n, i in enumerate(idx):
                    if n == hook_before:
                        hook()
                    item, amount = items[i]
                    ref = f"{tag}:{i}" if tag else None
                    ids[i] = c.execute("insert into ar_rows(run, item, amount, reference) values (%s,%s,%s,%s) "
                                       "returning id", (self.run, item, amount, ref)).fetchone()[0]
            return {"ids": ids}
        return call_with_deadline(fn, deadline_s + 2)

    # ------------------------------------------------------------------ reads
    def by_tag(self, tool: str, args: dict, tag: str) -> list:
        """Standing effects carrying the tag (A) or the key (K). Empty for U/C tools."""
        if tool == "issue_create":
            return self.gitea.search_by_tag(tag) if self.variant == "lagged" else self.gitea.find_by_tag(tag)
        if tool == "issue_comment":
            return self.gitea.comments_by_tag(int(args["issue"]), tag)
        if tool == "mail_send":
            return self.mail.find_by_message_id(tag)
        if tool == "row_insert":
            return self.pg.rows_by_reference(tag)
        if tool == "event_append":
            return [e for e, f in self.streams.kind_entries(args["kind"]) if f.get("op_id") == tag]
        if tool == "transfer_create":
            return [tag] if self.ledger.exists(tag) else []
        if tool == "object_put":
            return [args["name"]] if self.objects.exists(self.obj_key(args["name"])) else []
        return []

    def batch_by_tag(self, tag: str, n: int) -> dict[int, list[int]]:
        return {i: self.pg.rows_by_reference(f"{tag}:{i}") for i in range(n)}

    def by_content(self, tool: str, args: dict, own_only: bool = False) -> list:
        """Standing effects whose visible content matches the intent fields (what content verification sees).
        own_only (C5a/Gwa): on services that record an author (Gitea), accept only the agent's own records."""
        own = GITEA_ADMIN if own_only else None
        if tool == "issue_create":
            t = args["title"]
            return (self.gitea.search_by_title(t, own=own) if self.variant == "lagged"
                    else self.gitea.find_by_title(t, own=own))
        if tool == "issue_comment":
            return self.gitea.comments_by_body(int(args["issue"]), args["body"], own=own)
        if tool in ("mail_send", "mail_send_plain"):
            return self.mail.find_by_content(args["subject"], self.addr(args["to"]))
        if tool == "row_insert":
            return self.pg.rows_by_content(args["item"], int(args["amount"]))
        if tool == "batch_insert":
            return [self.pg.rows_by_content(it["item"], int(it["amount"])) for it in args["items"]]
        if tool == "event_append":
            return [e for e, f in self.streams.kind_entries(args["kind"]) if f.get("data") == args["data"]]
        if tool == "transfer_create":
            return self.ledger.by_memo(args["memo"])
        if tool == "object_put":
            return [args["name"]] if self.objects.exists(self.obj_key(args["name"])) else []
        return []

    def aggregate(self, tool: str, args: dict) -> int:
        if tool == "stock_adjust":
            return self.pg.stock_level(args["sku"])
        if tool == "counter_incr":
            return self.streams.counter_value(args["name"])
        raise ValueError(tool)

    def unit(self, tool: str, args: dict) -> int:
        """Change in the aggregate caused by one execution."""
        return int(args["delta"]) if tool == "stock_adjust" else 1

    # ------------------------------------------------------------------ compensation (targeted / fungible)
    def compensate(self, tool: str, args: dict, effect_id) -> str:
        if tool == "issue_create":
            return str(self.gitea.issue_delete(int(effect_id)))
        if tool == "issue_comment":
            return str(self.gitea.comment_delete(int(effect_id)))
        if tool in ("row_insert", "batch_insert"):
            return str(self.pg.row_delete(int(effect_id)))
        if tool == "event_append":
            return str(self.streams.delete_kind_entry(args["kind"], effect_id))
        if tool == "stock_adjust":
            return self.pg.stock_adjust(args["sku"], -int(args["delta"])).kind
        if tool == "counter_incr":
            return self.streams.counter_add(args["name"], -1).kind
        if tool == "transfer_create":
            tid = int(effect_id) if str(effect_id).isdigit() else self.ledger.id_for(effect_id)
            return self.ledger.reverse(tid, int(args["amount"]))
        if tool == "object_put":
            from .services.objstore import _client
            _client("clean", 30).delete_object(Bucket=self.objects.bucket, Key=self.obj_key(args["name"]))
            return "deleted"
        return "irreversible"
