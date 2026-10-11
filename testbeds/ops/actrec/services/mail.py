"""Mail over SMTP into Mailpit (class A via client Message-ID; irreversible: sent mail cannot be recalled).

The SMTP dialogue is driven step by step so the fault hook fires after DATA was accepted (354) and right
before the message body is sent. A downstream fault then loses only the final 250 reply (the message is
queued: lost acknowledgement); an upstream fault truncates the body (never queued: not executed).
"""
from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage
from email.policy import SMTP as SMTP_POLICY

import httpx

from ..config import MAILPIT_API, port
from .base import INFLIGHT_SOCKET_TIMEOUT_S, CallResult, call_with_deadline, nohook

DOMAIN = "actrec.invalid"


def message_id_for(tag: str) -> str:
    return f"{tag}@{DOMAIN}"


def _payload(m: EmailMessage) -> bytes:
    data = m.as_bytes(policy=SMTP_POLICY)
    data = re.sub(rb"(?m)^\.", b"..", data)       # dot-stuffing
    if not data.endswith(b"\r\n"):
        data += b"\r\n"
    return data + b".\r\n"


class Mail:
    sender = f"agent@{DOMAIN}"

    def send(self, to: str, subject: str, body: str, tag: str | None = None, route: str = "clean", hook=nohook,
             deadline_s: float = 4.0, repeat: int = 1, extra_headers: dict | None = None) -> CallResult:
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = self.sender, to, subject
        if tag:
            m["Message-ID"] = f"<{message_id_for(tag)}>"
            m["X-Op-Id"] = tag
        for k, v in (extra_headers or {}).items():
            m[k] = v
        m.set_content(body)
        payload = _payload(m)

        def fn():
            s = smtplib.SMTP("localhost", port("smtp", route), timeout=INFLIGHT_SOCKET_TIMEOUT_S)
            try:
                s.ehlo()
                for i in range(repeat):
                    code, msg = s.mail(self.sender)
                    if code != 250:
                        raise smtplib.SMTPResponseException(code, msg)
                    code, msg = s.rcpt(to)
                    if code not in (250, 251):
                        raise smtplib.SMTPRecipientsRefused({to: (code, msg)})
                    code, msg = s.docmd("DATA")
                    if code != 354:
                        raise smtplib.SMTPDataError(code, msg)
                    if i == 0:
                        hook()
                    s.send(payload)
                    code, msg = s.getreply()
                    if code != 250:
                        raise smtplib.SMTPDataError(code, msg)
                try:
                    s.quit()
                except Exception:
                    pass
                return {"queued": True}
            finally:
                try:
                    s.close()
                except Exception:
                    pass
        return call_with_deadline(fn, deadline_s)

    def actor_send(self, to: str, subject: str, body: str, message_id: str) -> None:
        """Background actor: same recipient/subject/body, its own Message-ID, sent on the direct port."""
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = f"other@{DOMAIN}", to, subject
        m["Message-ID"] = f"<{message_id}>"
        m["X-Origin"] = "actor"
        m.set_content(body)
        with smtplib.SMTP("localhost", port("smtp", "direct"), timeout=30) as s:
            s.send_message(m)

    def all_for(self, to: str) -> list[dict]:
        """Grader: every captured message to a recipient (summary objects)."""
        out, start = [], 0
        with self._api() as c:
            while True:
                r = c.get("/search", params={"query": f'to:"{to}"', "start": start, "limit": 200})
                r.raise_for_status()
                msgs = r.json().get("messages", [])
                out += [m for m in msgs if any((x.get("Address") or "").lower() == to.lower() for x in m.get("To") or [])]
                if len(msgs) < 200:
                    return out
                start += 200

    # ---------------- agent-visible reads (Mailpit HTTP API, not faulted) ----------------
    def _api(self) -> httpx.Client:
        return httpx.Client(base_url=MAILPIT_API, timeout=30)

    def find_by_message_id(self, tag: str) -> list[str]:
        mid = message_id_for(tag)
        with self._api() as c:
            r = c.get("/search", params={"query": f"message-id:{mid}", "limit": 200})
            r.raise_for_status()
            return [m["ID"] for m in r.json().get("messages", []) if m.get("MessageID") == mid]

    def find_by_content(self, subject: str, to: str) -> list[str]:
        """Content read path: same subject and recipient (what verify-before-retry can use).
        Search by recipient only and compare the subject exactly, so search tokenization cannot hide a match."""
        return [m["ID"] for m in self.all_for(to) if m.get("Subject") == subject]

    def headers(self, msg_id: str) -> dict:
        with self._api() as c:
            return c.get(f"/message/{msg_id}/headers").json()

    # ---------------- grader side ----------------
    def truth_count(self, tag: str) -> int:
        return len(self.find_by_message_id(tag))
