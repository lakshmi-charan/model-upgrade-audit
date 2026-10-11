"""TigerBeetle transfers (class K: the 128-bit transfer id is a native idempotency key).

Note: the TigerBeetle client retries internally and deduplicates by session, so a dropped reply is
re-requested by the client itself. That is part of the contract being measured, not something we add.
"""
from __future__ import annotations

import uuid
import zlib

import tigerbeetle as tb

from ..config import port
from .base import CallResult, call_with_deadline, nohook

LEDGER, CODE, CODE_REVERSAL = 1, 1, 2


def memo_code(memo: str) -> int:
    """The transfer's visible business reference (stored in user_data_64)."""
    return zlib.crc32(memo.encode()) or 1


def tag_to_id(tag: str) -> int:
    """128-bit transfer id from an op id / agent-supplied id (UUID or 32 hex); anything else is hashed."""
    try:
        v = uuid.UUID(str(tag)).int
    except ValueError:
        import hashlib
        v = int(hashlib.sha1(str(tag).encode()).hexdigest()[:32], 16)
    return v if 0 < v < (1 << 128) - 1 else 1


def _client(route: str) -> tb.ClientSync:
    return tb.ClientSync(cluster_id=0, replica_addresses=str(port("tb", route)))


class LedgerRefusal(RuntimeError):
    """The ledger answered, and refused the transfer (e.g. its id already exists with other accounts)."""


class Ledger:
    def __init__(self, debit: int | None = None, credit: int | None = None, namespace: str | None = None):
        self.debit, self.credit = debit, credit
        self.namespace = namespace

    def id_for(self, tag: str) -> int:
        """Transfer id for a key. Keys are scoped to the episode's run namespace (amendment 2): TigerBeetle ids
        are global, so without the scope an agent-chosen key (e.g. derived from an invoice number) collides
        with the same key chosen in another episode that replays the same task."""
        return tag_to_id(f"{self.namespace}:{tag}" if self.namespace else tag)

    def setup_accounts(self) -> None:
        c = _client("direct")
        try:
            self.debit, self.credit = tb.id(), tb.id()
            res = c.create_accounts([tb.Account(id=self.debit, ledger=LEDGER, code=CODE),
                                     tb.Account(id=self.credit, ledger=LEDGER, code=CODE)])
            bad = [r for r in res if r.status.name not in ("CREATED", "EXISTS")]
            if bad:
                raise RuntimeError(f"account setup failed: {[r.status.name for r in bad]}")
        finally:
            c.close()

    def transfer(self, amount: int, tag: str, route: str = "clean", hook=nohook, deadline_s: float = 4.0,
                 repeat: int = 1, memo: str = "") -> CallResult:
        tid = self.id_for(tag)
        tr = tb.Transfer(id=tid, debit_account_id=self.debit, credit_account_id=self.credit, amount=amount,
                         ledger=LEDGER, code=CODE, user_data_64=memo_code(memo) if memo else 0)

        def fn():
            c = _client(route)
            try:
                c.lookup_accounts([self.debit])     # handshake / session registration
                hook()
                statuses = []
                for _ in range(repeat):
                    statuses.append(c.create_transfers([tr])[0].status.name)
                if statuses[-1] not in ("CREATED", "EXISTS"):
                    # a definite answer from the ledger, surfaced to the caller as the service's own error
                    raise LedgerRefusal(f"transfer refused by the ledger: {statuses[-1]}")
                return {"statuses": statuses, "transfer_id": str(tid)}
            finally:
                c.close()
        return call_with_deadline(fn, deadline_s)

    def reverse(self, original_id: int, amount: int, route: str = "clean") -> str:
        """Compensation: a new transfer in the opposite direction that references the original."""
        c = _client(route)
        try:
            tr = tb.Transfer(id=tb.id(), debit_account_id=self.credit, credit_account_id=self.debit, amount=amount,
                             ledger=LEDGER, code=CODE_REVERSAL, user_data_128=int(original_id))
            return c.create_transfers([tr])[0].status.name
        finally:
            c.close()

    def all_transfers(self, route: str = "direct") -> list:
        c = _client(route)
        try:
            # tigerbeetle-python 0.17 requires every field; 0 means "no filter" for the unused ones
            f = tb.AccountFilter(account_id=self.debit, user_data_128=0, user_data_64=0, user_data_32=0, code=0,
                                 timestamp_min=0, timestamp_max=0, limit=8189,
                                 flags=tb.AccountFilterFlags.DEBITS | tb.AccountFilterFlags.CREDITS)
            return list(c.get_account_transfers(f))
        finally:
            c.close()

    def by_memo(self, memo: str, route: str = "clean") -> list[int]:
        code = memo_code(memo)
        rev = {int(t.user_data_128) for t in self.all_transfers(route) if t.code == CODE_REVERSAL}
        return [int(t.id) for t in self.all_transfers(route)
                if t.code == CODE and t.user_data_64 == code and int(t.id) not in rev]

    def exists(self, tag: str, route: str = "clean") -> int:
        c = _client(route)
        try:
            return len(c.lookup_transfers([self.id_for(tag)]))
        finally:
            c.close()

    # ---------------- grader side ----------------
    def truth_count(self, tag: str) -> int:
        c = _client("direct")
        try:
            return len(c.lookup_transfers([self.id_for(tag)]))
        finally:
            c.close()

    def truth_debits(self) -> int:
        c = _client("direct")
        try:
            return int(c.lookup_accounts([self.debit])[0].debits_posted)
        finally:
            c.close()
