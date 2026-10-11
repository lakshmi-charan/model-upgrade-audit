"""Common pieces of the service adapters.

A *tool call* has a deadline (what the agent waits for). The underlying request is run in a background thread
with a much longer socket timeout, so a request that is still in flight when the deadline passes keeps going
and may commit later. This mirrors a real agent tool call that gives up while the HTTP/SQL request is still
travelling, which is what produces genuine late commits.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

INFLIGHT_SOCKET_TIMEOUT_S = 900.0   # how long an abandoned request may stay in flight


@dataclass
class CallResult:
    kind: str                      # "ok" | "error" | "timeout"
    elapsed_s: float
    value: Any = None              # parsed response on success
    exc_type: str | None = None    # normalized, what the tool layer reports to the caller
    exc_msg: str | None = None
    background: dict = field(default_factory=dict)   # filled when an abandoned request finishes
    raw_exc_type: str | None = None  # library exception, kept for audit only
    raw_exc_msg: str | None = None

    def signature(self) -> str:
        """What the caller can observe, normalized: used to test observation equivalence of fault pairs."""
        if self.kind == "ok":
            return "ok"
        msg = re.sub(r"[0-9a-f]{8,}|\d+", "#", (self.exc_msg or ""))[:80]
        return f"{self.kind}:{self.exc_type}:{msg}"

    def raw_signature(self) -> str:
        if self.kind == "ok":
            return "ok"
        return f"{self.kind}:{self.raw_exc_type or self.exc_type}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signature"] = self.signature()
        d["raw_signature"] = self.raw_signature()
        return d


_INFLIGHT: list[threading.Thread] = []

# Transport failures that happen before a complete reply is read. The tool layer reports all of them the
# same way, as an HTTP/SQL/SMTP client library wrapper in a real agent framework would. Without this, an
# incidental detail (e.g. one leaked byte of an SMTP "250" reply being parsed as reply code 2) would let a
# policy distinguish worlds that are meant to be observation-equivalent. Raw exceptions are kept for audit.
_TRANSPORT_NAMES = ("ConnectionError", "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError",
                    "RemoteProtocolError", "ReadError", "WriteError", "ConnectError", "ConnectionClosedError",
                    "EndpointConnectionError", "SMTPServerDisconnected", "OperationalError", "IncompleteRead",
                    "ProtocolError", "TimeoutError", "ReadTimeout")
TRANSPORT_MSG = "connection closed before a complete reply was received"


def _normalize(exc_type: str | None, exc_msg: str | None) -> tuple[str | None, str | None]:
    name = (exc_type or "").rsplit(".", 1)[-1]
    if name in _TRANSPORT_NAMES:
        return "ToolTransportError", TRANSPORT_MSG
    # smtplib reports a reply that was cut after the first digit as SMTPDataError(<1-digit code>, b'')
    if name in ("SMTPDataError", "SMTPResponseException") and re.match(r"^\(\d, b?''\)$", (exc_msg or "").strip()):
        return "ToolTransportError", TRANSPORT_MSG
    return exc_type, exc_msg


def call_with_deadline(fn: Callable[[], Any], deadline_s: float) -> CallResult:
    """Run fn in a daemon thread; return at the deadline with kind='timeout' if it has not finished."""
    box: dict = {}
    t0 = time.perf_counter()

    def target():
        try:
            box["value"] = fn()
            box["kind"] = "ok"
        except BaseException as e:  # noqa: BLE001 - we record every failure verbatim
            box["kind"] = "error"
            box["exc_type"] = f"{type(e).__module__}.{type(e).__name__}"
            box["exc_msg"] = str(e)[:300]
        box["finished_after_s"] = round(time.perf_counter() - t0, 3)

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(deadline_s)
    el = round(time.perf_counter() - t0, 3)
    if th.is_alive():
        _INFLIGHT.append(th)
        res = CallResult(kind="timeout", elapsed_s=el, exc_type="ToolDeadline",
                         exc_msg=f"no response within {deadline_s:.0f}s")
        res.background = box  # the dict keeps filling in after we return
        return res
    if box.get("kind") == "ok":
        return CallResult(kind="ok", elapsed_s=el, value=box.get("value"))
    et, em = _normalize(box.get("exc_type"), box.get("exc_msg"))
    return CallResult(kind="error", elapsed_s=el, exc_type=et, exc_msg=em,
                      raw_exc_type=box.get("exc_type"), raw_exc_msg=box.get("exc_msg"))


def inflight_count() -> int:
    _INFLIGHT[:] = [t for t in _INFLIGHT if t.is_alive()]
    return len(_INFLIGHT)


def wait_inflight(max_s: float) -> int:
    end = time.time() + max_s
    for t in list(_INFLIGHT):
        t.join(max(0.0, end - time.time()))
    return inflight_count()


def nohook() -> None:
    return None


def tag_line(tag: str) -> str:
    """The persisted marker written into a free-text field."""
    return f"op-id: {tag}"
