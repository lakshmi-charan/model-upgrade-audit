"""PostgreSQL rows (class A via a free-text `reference` column; reversible: delete by row id) and stock
adjustments (class U: only an aggregate SUM is readable by the agent; fungible compensation)."""
from __future__ import annotations

import psycopg

from ..config import pg_dsn
from .base import CallResult, call_with_deadline, nohook

SCHEMA = """
create table if not exists ar_rows (
  id bigserial primary key, run text not null, item text not null, amount integer not null,
  reference text, created_at timestamptz not null default clock_timestamp());
alter table ar_rows add column if not exists origin text not null default 'agent';
create index if not exists ar_rows_ref on ar_rows(reference);
create index if not exists ar_rows_run on ar_rows(run);
create table if not exists ar_stock (
  id bigserial primary key, run text not null, sku text not null, delta integer not null,
  created_at timestamptz not null default clock_timestamp());
alter table ar_stock add column if not exists origin text not null default 'agent';
create index if not exists ar_stock_sku on ar_stock(run, sku);
-- grader-only audit (no agent tool reads it): every insert and delete, written by the database itself
create table if not exists ar_audit (
  id bigserial primary key, op text not null, tbl text not null, row_id bigint, run text, item text,
  amount integer, reference text, sku text, delta integer, origin text,
  at timestamptz not null default clock_timestamp());
create index if not exists ar_audit_run on ar_audit(run);
create or replace function ar_audit_fn() returns trigger language plpgsql as $$
begin
  if tg_table_name = 'ar_rows' then
    if tg_op = 'INSERT' then
      insert into ar_audit(op,tbl,row_id,run,item,amount,reference,origin) values ('I','rows',new.id,new.run,new.item,new.amount,new.reference,new.origin);
    else
      insert into ar_audit(op,tbl,row_id,run,item,amount,reference,origin) values ('D','rows',old.id,old.run,old.item,old.amount,old.reference,old.origin);
    end if;
  else
    if tg_op = 'INSERT' then
      insert into ar_audit(op,tbl,row_id,run,sku,delta,origin) values ('I','stock',new.id,new.run,new.sku,new.delta,new.origin);
    else
      insert into ar_audit(op,tbl,row_id,run,sku,delta,origin) values ('D','stock',old.id,old.run,old.sku,old.delta,old.origin);
    end if;
  end if;
  return null;
end $$;
drop trigger if exists ar_rows_audit on ar_rows;
create trigger ar_rows_audit after insert or delete on ar_rows for each row execute function ar_audit_fn();
drop trigger if exists ar_stock_audit on ar_stock;
create trigger ar_stock_audit after insert or delete on ar_stock for each row execute function ar_audit_fn();
"""


def ensure_schema() -> None:
    with psycopg.connect(pg_dsn("direct"), autocommit=True) as c:
        c.execute(SCHEMA)


class Pg:
    def __init__(self, run: str):
        self.run = run

    def _conn(self, route: str):
        return psycopg.connect(pg_dsn(route), autocommit=True)

    # ---------------- writes ----------------
    def row_insert(self, item: str, amount: int, tag: str | None = None, route: str = "clean", hook=nohook,
                   deadline_s: float = 4.0, repeat: int = 1) -> CallResult:
        def fn():
            with self._conn(route) as c:
                c.execute("select 1").fetchone()
                hook()
                rid = None
                for _ in range(repeat):
                    rid = c.execute("insert into ar_rows(run, item, amount, reference) values (%s,%s,%s,%s) returning id",
                                    (self.run, item, amount, tag)).fetchone()[0]
                return {"id": rid}
        return call_with_deadline(fn, deadline_s)

    def batch_insert(self, items: list[tuple[str, int]], tag: str | None = None, route: str = "clean", hook=nohook,
                     hook_before: int = 0, deadline_s: float = 6.0) -> CallResult:
        """Non-atomic batch: one autocommit insert per row. The hook fires before row `hook_before`."""
        def fn():
            ids = []
            with self._conn(route) as c:
                c.execute("select 1").fetchone()
                for i, (item, amount) in enumerate(items):
                    if i == hook_before:
                        hook()
                    ref = f"{tag}:{i}" if tag else None
                    ids.append(c.execute("insert into ar_rows(run, item, amount, reference) values (%s,%s,%s,%s) returning id",
                                         (self.run, item, amount, ref)).fetchone()[0])
            return {"ids": ids}
        return call_with_deadline(fn, deadline_s)

    def stock_adjust(self, sku: str, delta: int, route: str = "clean", hook=nohook, deadline_s: float = 4.0,
                     repeat: int = 1) -> CallResult:
        def fn():
            with self._conn(route) as c:
                c.execute("select 1").fetchone()
                hook()
                for _ in range(repeat):
                    c.execute("insert into ar_stock(run, sku, delta) values (%s,%s,%s)", (self.run, sku, delta))
                return {"ok": True}
        return call_with_deadline(fn, deadline_s)

    def row_delete(self, row_id: int, route: str = "clean") -> int:
        with self._conn(route) as c:
            return c.execute("delete from ar_rows where id = %s", (row_id,)).rowcount

    # ---------------- agent-visible reads ----------------
    def rows_by_reference(self, tag: str, route: str = "clean") -> list[int]:
        with self._conn(route) as c:
            return [r[0] for r in c.execute("select id from ar_rows where run=%s and reference=%s order by id",
                                            (self.run, tag)).fetchall()]

    def actor_insert(self, item: str, amount: int) -> int:
        """Background actor (another process sharing the table): same content, no reference."""
        with self._conn("direct") as c:
            return c.execute("insert into ar_rows(run, item, amount, origin) values (%s,%s,%s,'actor') returning id",
                             (self.run, item, amount)).fetchone()[0]

    def actor_stock(self, sku: str, delta: int) -> None:
        with self._conn("direct") as c:
            c.execute("insert into ar_stock(run, sku, delta, origin) values (%s,%s,%s,'actor')", (self.run, sku, delta))

    def audit(self) -> list[dict]:
        with self._conn("direct") as c:
            cur = c.execute("select op,tbl,row_id,item,amount,reference,sku,delta,origin from ar_audit where run=%s "
                            "order by id", (self.run,))
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def rows_by_content(self, item: str, amount: int, route: str = "clean") -> list[int]:
        with self._conn(route) as c:
            return [r[0] for r in c.execute("select id from ar_rows where run=%s and item=%s and amount=%s order by id",
                                            (self.run, item, amount)).fetchall()]

    def stock_level(self, sku: str, route: str = "clean") -> int:
        with self._conn(route) as c:
            return int(c.execute("select coalesce(sum(delta),0) from ar_stock where run=%s and sku=%s",
                                 (self.run, sku)).fetchone()[0])

    # ---------------- grader side ----------------
    def truth_count(self, tag: str) -> int:
        return len(self.rows_by_reference(tag, route="direct"))

    def truth_batch(self, tag: str) -> int:
        with self._conn("direct") as c:
            return int(c.execute("select count(*) from ar_rows where run=%s and reference like %s",
                                 (self.run, f"{tag}:%")).fetchone()[0])

    def truth_stock_rows(self, sku: str) -> int:
        with self._conn("direct") as c:
            return int(c.execute("select count(*) from ar_stock where run=%s and sku=%s", (self.run, sku)).fetchone()[0])
