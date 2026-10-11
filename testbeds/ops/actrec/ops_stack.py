"""[model-migration port, new file] Start, bootstrap and check only the services an ops subset needs.

Replaces stack.up() for this project: the compose project is `mmops` (docker/compose.yml), credentials come
from the process environment (code/run_ops.py generates them once into runs/.secrets/stack_env.json and never
prints them), and only the services required by the chosen task templates are started. Bootstrap steps are
the source's own (stack.gitea_bootstrap logic, pg.ensure_schema, Toxi.ensure_proxies, Objects.ensure_bucket).
"""
from __future__ import annotations

import os
import socket
import subprocess
import time

from .config import COMPOSE_FILE, GITEA_ADMIN, MAILPIT_API, PROJECT, register_secret, save_stack_secrets

TOOL_SERVICE = {   # contract tool -> compose service it writes to
    "issue_create": "gitea", "issue_comment": "gitea", "mail_send": "mailpit", "mail_send_plain": "mailpit",
    "row_insert": "postgres", "batch_insert": "postgres", "stock_adjust": "postgres", "event_append": "redis",
    "counter_incr": "redis", "transfer_create": "tigerbeetle", "object_put": "seaweedfs",
}


def services_for(tasks) -> list[str]:
    """Compose services needed by these task instances. toxiproxy is always needed (the agent's path to every
    service is the clean proxy); postgres also holds the grader's audit trigger tables only for pg tools."""
    need = {"toxiproxy"}
    for t in tasks:
        for st in t.steps:
            need.add(TOOL_SERVICE[st.tool])
    order = ["postgres", "gitea", "mailpit", "redis", "seaweedfs", "tigerbeetle", "toxiproxy"]
    return [s for s in order if s in need]


def compose_cmd(*args: str, services_tb: bool = True) -> list[str]:
    cmd = ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE)]
    if services_tb:
        cmd += ["--profile", "tb"]
    return cmd + list(args)


def compose(*args: str, timeout: float = 900) -> subprocess.CompletedProcess:
    return subprocess.run(compose_cmd(*args), capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace", env=dict(os.environ))


def _tcp(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def probe(service: str) -> bool:
    """One quick reachability probe (direct port, as the grader uses)."""
    try:
        if service == "postgres":
            import psycopg

            from .config import pg_dsn
            psycopg.connect(pg_dsn("direct")).close()
            return True
        import httpx
        if service == "gitea":
            return httpx.get("http://localhost:3000/api/healthz", timeout=2).status_code == 200
        if service == "mailpit":
            return httpx.get(MAILPIT_API + "/info", timeout=2).status_code == 200
        if service == "redis":
            return _tcp(6379)
        if service == "seaweedfs":
            return httpx.get("http://localhost:8333/", timeout=2).status_code < 500
        if service == "toxiproxy":
            return httpx.get("http://localhost:8474/version", timeout=2).status_code == 200
        if service == "tigerbeetle":
            return _tcp(3001)
    except Exception:
        return False
    return False


def wait_healthy(services: list[str], timeout: float = 300) -> dict:
    out = {}
    for s in services:
        t0 = time.time()
        while not probe(s):
            if time.time() - t0 > timeout:
                out[s] = None
                break
            time.sleep(1.0)
        else:
            out[s] = round(time.time() - t0, 1)
    return out


def gitea_token_ok(token: str) -> bool:
    import httpx
    try:
        r = httpx.get("http://localhost:3000/api/v1/user", headers={"Authorization": f"token {token}"}, timeout=5)
        return r.status_code == 200 and r.json().get("login") == GITEA_ADMIN
    except Exception:
        return False


def gitea_bootstrap() -> str:
    """Same steps as stack.gitea_bootstrap, through this project's compose command (no --env-file)."""
    pw = os.environ.get("GITEA_ADMIN_PASSWORD", "")
    if not pw:
        raise RuntimeError("GITEA_ADMIN_PASSWORD missing from the environment")
    register_secret(pw)
    base = ["exec", "-T", "-u", "git", "gitea", "gitea", "admin", "user"]
    r = compose(*base, "create", "--username", GITEA_ADMIN, "--password", pw, "--email",
                f"{GITEA_ADMIN}@actrec.invalid", "--admin", "--must-change-password=false", timeout=120)
    if r.returncode != 0 and "already exists" not in (r.stdout + r.stderr):
        raise RuntimeError("gitea admin create failed: " + (r.stderr or r.stdout)[:300].replace(pw, "<redacted>"))
    t = compose(*base, "generate-access-token", "--username", GITEA_ADMIN, "--token-name",
                f"mmops-{int(time.time())}", "--scopes", "all", "--raw", timeout=120)
    token = t.stdout.strip().splitlines()[-1] if t.returncode == 0 and t.stdout.strip() else ""
    if not token:
        raise RuntimeError("gitea token creation failed")
    register_secret(token)
    return token


def bootstrap(services: list[str]) -> dict:
    """Idempotent: proxies, schema, bucket, Gitea admin token (only for the services in use)."""
    from .config import SECRETS
    import json
    info = {}
    if "toxiproxy" in services:
        from .faults.toxi import Toxi
        info["proxies"] = len(Toxi().ensure_proxies())
    if "postgres" in services:
        from .services.pg import ensure_schema
        ensure_schema()
        info["pg_schema"] = "ok"
    if "seaweedfs" in services:
        from .services.objstore import Objects
        for _ in range(30):                     # seaweedfs' S3 gateway answers before the filer is ready
            try:
                Objects("actrec").ensure_bucket()
                info["bucket"] = "ok"
                break
            except Exception as e:  # noqa: BLE001
                info["bucket"] = f"{type(e).__name__}"
                time.sleep(2)
    if "gitea" in services:
        p = SECRETS / "stack.json"
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        tok = cur.get("gitea_token")
        if not tok or not gitea_token_ok(tok):
            cur["gitea_token"] = gitea_bootstrap()
            save_stack_secrets(cur)
            info["gitea_token"] = "new"
        else:
            register_secret(tok)
            info["gitea_token"] = "valid"
    return info
