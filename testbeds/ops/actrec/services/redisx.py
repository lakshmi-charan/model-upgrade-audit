"""Redis stream events (class A via an `op_id` field; reversible: XDEL) and counters (class U, fungible)."""
from __future__ import annotations

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from ..config import port
from .base import INFLIGHT_SOCKET_TIMEOUT_S, CallResult, call_with_deadline, nohook


def _client(route: str, timeout: float = INFLIGHT_SOCKET_TIMEOUT_S) -> redis.Redis:
    # Retry(NoBackoff(), 0): the library must not retry on its own (a hidden retry would duplicate XADD/INCR).
    return redis.Redis(host="localhost", port=port("redis", route), socket_timeout=timeout,
                       socket_connect_timeout=10, single_connection_client=True,
                       retry=Retry(NoBackoff(), 0), retry_on_timeout=False, decode_responses=True)


# INCRBY plus a grader-only audit list, executed atomically inside Redis (instrumentation, like a DB trigger).
_INCR_LUA = "redis.call('RPUSH', KEYS[2], ARGV[1]) return redis.call('INCRBY', KEYS[1], ARGV[1])"


class Streams:
    def __init__(self, run: str):
        self.run = run
        self.key = f"ar:events:{run}"          # validation stream (one key)

    def kkey(self, kind: str) -> str:
        return f"ar:ev:{self.run}:{kind}"

    def append(self, kind: str, data: str, tag: str | None = None, route: str = "clean", hook=nohook,
               deadline_s: float = 4.0, repeat: int = 1, per_kind: bool = False) -> CallResult:
        fields = {"kind": kind, "data": data}
        if tag:
            fields["op_id"] = tag
        key = self.kkey(kind) if per_kind else self.key

        def fn():
            r = _client(route)
            try:
                r.ping()
                hook()
                eid = None
                for _ in range(repeat):
                    eid = r.xadd(key, fields)
                return {"entry_id": eid}
            finally:
                r.close()
        return call_with_deadline(fn, deadline_s)

    def actor_append(self, kind: str, data: str) -> str:
        r = _client("direct", 30)
        try:
            return r.xadd(self.kkey(kind), {"kind": kind, "data": data, "_origin": "actor"})
        finally:
            r.close()

    def delete_kind_entry(self, kind: str, entry_id: str, route: str = "clean") -> int:
        r = _client(route, 30)
        try:
            return int(r.xdel(self.kkey(kind), entry_id))
        finally:
            r.close()

    def kind_entries(self, kind: str, route: str = "clean") -> list[tuple[str, dict]]:
        r = _client(route, 30)
        try:
            return r.xrange(self.kkey(kind))
        finally:
            r.close()

    def kind_entries_added(self, kind: str) -> int:
        r = _client("direct", 30)
        try:
            if not r.exists(self.kkey(kind)):
                return 0
            return int(r.xinfo_stream(self.kkey(kind))["entries-added"])
        finally:
            r.close()

    def counter_add(self, name: str, by: int, route: str = "clean", hook=nohook, deadline_s: float = 4.0,
                    repeat: int = 1) -> CallResult:
        key, audit = f"ar:ctr:{self.run}:{name}", f"ar:ctr-audit:{self.run}:{name}"

        def fn():
            r = _client(route)
            try:
                r.ping()
                hook()
                v = None
                for _ in range(repeat):
                    v = r.eval(_INCR_LUA, 2, key, audit, by)
                return {"value": v}
            finally:
                r.close()
        return call_with_deadline(fn, deadline_s)

    def actor_counter(self, name: str, by: int) -> None:
        r = _client("direct", 30)
        try:
            r.eval(_INCR_LUA, 2, f"ar:ctr:{self.run}:{name}", f"ar:ctr-audit:{self.run}:{name}:actor", by)
            r.incrby(f"ar:ctr:{self.run}:{name}", 0)
        finally:
            r.close()

    def counter_audit(self, name: str, actor: bool = False) -> list[int]:
        r = _client("direct", 30)
        try:
            k = f"ar:ctr-audit:{self.run}:{name}" + (":actor" if actor else "")
            return [int(x) for x in r.lrange(k, 0, -1)]
        finally:
            r.close()

    def counter_incr(self, name: str, by: int = 1, route: str = "clean", hook=nohook, deadline_s: float = 4.0,
                     repeat: int = 1) -> CallResult:
        key = f"ar:ctr:{self.run}:{name}"

        def fn():
            r = _client(route)
            try:
                r.ping()
                hook()
                v = None
                for _ in range(repeat):
                    v = r.incrby(key, by)
                return {"value": v}
            finally:
                r.close()
        return call_with_deadline(fn, deadline_s)

    def delete_entry(self, entry_id: str, route: str = "clean") -> int:
        r = _client(route, 30)
        try:
            return int(r.xdel(self.key, entry_id))
        finally:
            r.close()

    # ---------------- agent-visible reads ----------------
    def entries_by_tag(self, tag: str, route: str = "clean") -> list[str]:
        r = _client(route, 30)
        try:
            return [eid for eid, f in r.xrange(self.key) if f.get("op_id") == tag]
        finally:
            r.close()

    def entries_by_content(self, kind: str, data: str, route: str = "clean") -> list[str]:
        r = _client(route, 30)
        try:
            return [eid for eid, f in r.xrange(self.key) if f.get("kind") == kind and f.get("data") == data]
        finally:
            r.close()

    def counter_value(self, name: str, route: str = "clean") -> int:
        r = _client(route, 30)
        try:
            return int(r.get(f"ar:ctr:{self.run}:{name}") or 0)
        finally:
            r.close()

    # ---------------- grader side ----------------
    def truth_count(self, tag: str) -> int:
        return len(self.entries_by_tag(tag, route="direct"))

    def truth_counter(self, name: str) -> int:
        return self.counter_value(name, route="direct")
