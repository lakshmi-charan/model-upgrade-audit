"""Gitea issues and comments (class A, reversible: targeted delete)."""
from __future__ import annotations

import httpx

from ..config import GITEA_ADMIN, port
from .base import INFLIGHT_SOCKET_TIMEOUT_S, CallResult, call_with_deadline, nohook, tag_line


class Gitea:
    def __init__(self, token: str, repo: str, owner: str = GITEA_ADMIN):
        self.token, self.owner, self.repo = token, owner, repo
        self.h = {"Authorization": f"token {token}"}

    def _client(self, route: str, timeout: float = INFLIGHT_SOCKET_TIMEOUT_S) -> httpx.Client:
        return httpx.Client(base_url=f"http://localhost:{port('gitea', route)}/api/v1", headers=self.h,
                            timeout=httpx.Timeout(timeout, connect=10.0))

    @property
    def _p(self) -> str:
        return f"/repos/{self.owner}/{self.repo}"

    def ensure_repo(self) -> int:
        with self._client("direct", 30) as c:
            r = c.post("/user/repos", json={"name": self.repo, "private": False, "auto_init": False})
            return r.status_code  # 201 created, 409 exists

    # ---------------- writes ----------------
    def issue_create(self, title: str, body: str, tag: str | None = None, route: str = "clean", hook=nohook,
                     deadline_s: float = 4.0, repeat: int = 1) -> CallResult:
        full = body + (f"\n\n{tag_line(tag)}\n" if tag else "")

        def fn():
            with self._client(route) as c:
                c.get("/version")          # handshake on the same keep-alive connection
                hook()
                out = None
                for _ in range(repeat):
                    r = c.post(f"{self._p}/issues", json={"title": title, "body": full})
                    r.raise_for_status()
                    out = {"number": r.json()["number"]}
                return out
        return call_with_deadline(fn, deadline_s)

    def comment_create(self, number: int, body: str, tag: str | None = None, route: str = "clean", hook=nohook,
                       deadline_s: float = 4.0, repeat: int = 1) -> CallResult:
        full = body + (f"\n\n{tag_line(tag)}\n" if tag else "")

        def fn():
            with self._client(route) as c:
                c.get("/version")
                hook()
                out = None
                for _ in range(repeat):
                    r = c.post(f"{self._p}/issues/{number}/comments", json={"body": full})
                    r.raise_for_status()
                    out = {"comment_id": r.json()["id"]}
                return out
        return call_with_deadline(fn, deadline_s)

    def comment_delete(self, comment_id: int, route: str = "clean") -> int:
        with self._client(route, 30) as c:
            return c.delete(f"{self._p}/issues/comments/{comment_id}").status_code

    def list_comments(self, number: int, route: str = "clean") -> list[dict]:
        with self._client(route, 30) as c:
            r = c.get(f"{self._p}/issues/{number}/comments")
            return r.json() if r.status_code == 200 else []

    def comments_by_tag(self, number: int, tag: str, route: str = "clean") -> list[int]:
        line = tag_line(tag)
        return [c["id"] for c in self.list_comments(number, route) if line in (c.get("body") or "").splitlines()]

    def comments_by_body(self, number: int, body: str, route: str = "clean", own: str | None = None) -> list[int]:
        """Content read path; with `own`, only comments written by that account (author filter)."""
        return [c["id"] for c in self.list_comments(number, route)
                if (c.get("body") or "").split("\n\n" + "op-id:")[0].strip() == body.strip()
                and (own is None or (c.get("user") or {}).get("login") == own)]

    def add_webhook(self, url: str) -> int:
        with self._client("direct", 30) as c:
            r = c.post(f"{self._p}/hooks", json={"type": "gitea", "active": True,
                                                  "events": ["issues", "issue_comment"],
                                                  "config": {"url": url, "content_type": "json"}})
            return r.status_code

    def issue_delete(self, number: int, route: str = "clean") -> int:
        """Targeted compensation. 204 = deleted, 404 = already gone (both mean neutralized)."""
        with self._client(route, 30) as c:
            return c.delete(f"{self._p}/issues/{number}").status_code

    # ---------------- agent-visible reads ----------------
    def list_issues(self, route: str = "clean") -> list[dict]:
        out, page = [], 1
        with self._client(route, 30) as c:
            while True:
                r = c.get(f"{self._p}/issues", params={"state": "all", "type": "issues", "limit": 50, "page": page})
                r.raise_for_status()
                batch = r.json()
                out += batch
                if len(batch) < 50:
                    return out
                page += 1

    def find_by_tag(self, tag: str, route: str = "clean") -> list[int]:
        """Exact read path: issues whose body contains the tag line exactly."""
        line = tag_line(tag)
        return [i["number"] for i in self.list_issues(route) if line in (i.get("body") or "").splitlines()]

    def find_by_title(self, title: str, route: str = "clean", own: str | None = None) -> list[int]:
        """Content read path (what verify-before-retry uses); with `own`, only issues opened by that account."""
        return [i["number"] for i in self.list_issues(route) if i.get("title") == title
                and (own is None or (i.get("user") or {}).get("login") == own)]

    def search_by_tag(self, tag: str, route: str = "clean") -> list[int]:
        """Lagged read path for tags: indexed search, then confirm the exact tag line in the body."""
        line = tag_line(tag)
        return [n for n in self.search(tag, route) if line in self.body(n, route).splitlines()]

    def search_by_title(self, title: str, route: str = "clean", own: str | None = None) -> list[int]:
        with self._client(route, 30) as c:
            r = c.get(f"{self._p}/issues", params={"q": title, "state": "all", "type": "issues", "limit": 50})
            r.raise_for_status()
            return [i["number"] for i in r.json() if i.get("title") == title
                    and (own is None or (i.get("user") or {}).get("login") == own)]

    def search(self, q: str, route: str = "clean") -> list[int]:
        """Indexed search (lagged, tokenized; partial matches possible)."""
        with self._client(route, 30) as c:
            r = c.get(f"{self._p}/issues", params={"q": q, "state": "all", "type": "issues", "limit": 50})
            r.raise_for_status()
            return [i["number"] for i in r.json()]

    def body(self, number: int, route: str = "clean") -> str:
        with self._client(route, 30) as c:
            return c.get(f"{self._p}/issues/{number}").json().get("body") or ""

    # ---------------- grader side ----------------
    def truth_count(self, tag: str) -> int:
        return len(self.find_by_tag(tag, route="direct"))
