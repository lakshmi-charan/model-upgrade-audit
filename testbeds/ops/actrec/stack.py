"""Start, bootstrap, check and stop the experiment stack (docker/compose.yml, project `actrec`)."""
from __future__ import annotations

import json
import socket
import subprocess
import time

import httpx
import psycopg

from .config import (COMPOSE_FILE, ENV, GITEA_ADMIN, MAILPIT_API, PROJECT, PROXIES, ROOT, pg_dsn,
                     register_secret, save_stack_secrets, write_json)
from .faults.toxi import Toxi

IMAGES = ["postgres:16", "gitea/gitea:1.24", "axllent/mailpit:latest", "redis:7", "chrislusf/seaweedfs:latest",
          "ghcr.io/tigerbeetle/tigerbeetle:latest", "ghcr.io/shopify/toxiproxy:latest", "temporalio/temporal:latest"]


def compose(*args: str, profiles: tuple[str, ...] = ("tb",), timeout: float = 600) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-p", PROJECT, "--env-file", str(ROOT / ".env"), "-f", str(COMPOSE_FILE)]
    for p in profiles:
        cmd += ["--profile", p]
    return subprocess.run(cmd + list(args), capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def _tcp(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _wait(name: str, probe, timeout: float = 180) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if probe():
                return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{name} not healthy after {timeout}s")


def health(include_tb: bool = True, include_engine: bool = False) -> dict:
    out = {}
    out["postgres"] = _wait("postgres", lambda: psycopg.connect(pg_dsn("direct")).close() is None)
    out["gitea"] = _wait("gitea", lambda: httpx.get("http://localhost:3000/api/healthz", timeout=2).status_code == 200)
    out["mailpit"] = _wait("mailpit", lambda: httpx.get(MAILPIT_API + "/info", timeout=2).status_code == 200)
    out["redis"] = _wait("redis", lambda: _tcp(6379))
    out["seaweedfs"] = _wait("seaweedfs", lambda: httpx.get("http://localhost:8333/", timeout=2).status_code < 500)
    out["toxiproxy"] = _wait("toxiproxy", lambda: httpx.get("http://localhost:8474/version", timeout=2).status_code == 200)
    if include_tb:
        out["tigerbeetle"] = _wait("tigerbeetle", lambda: _tcp(3001))
    if include_engine:
        out["temporal"] = _wait("temporal", lambda: _tcp(7233))
    return out


def gitea_bootstrap() -> str:
    pw = ENV.get("GITEA_ADMIN_PASSWORD", "")
    if not pw:
        raise RuntimeError("GITEA_ADMIN_PASSWORD is empty in .env")
    register_secret(pw)
    base = ["exec", "-T", "-u", "git", "gitea", "gitea", "admin", "user"]
    r = compose(*base, "create", "--username", GITEA_ADMIN, "--password", pw, "--email",
                f"{GITEA_ADMIN}@actrec.invalid", "--admin", "--must-change-password=false", timeout=120)
    if r.returncode != 0 and "already exists" not in (r.stdout + r.stderr):
        raise RuntimeError("gitea admin create failed: " + (r.stderr or r.stdout)[:300].replace(pw, "<redacted>"))
    t = compose(*base, "generate-access-token", "--username", GITEA_ADMIN, "--token-name",
                f"actrec-{int(time.time())}", "--scopes", "all", "--raw", timeout=120)
    token = t.stdout.strip().splitlines()[-1] if t.returncode == 0 and t.stdout.strip() else ""
    if not token:
        raise RuntimeError("gitea token creation failed")
    register_secret(token)
    return token


GITEA_OTHER = "actrec-other"     # a second, ordinary user: the follow-up study's look-alikes from another account


def gitea_other_bootstrap() -> str:
    """Create the second Gitea user (non-admin) with a random password and return an access token for it."""
    import secrets
    pw = secrets.token_urlsafe(24)
    register_secret(pw)
    base = ["exec", "-T", "-u", "git", "gitea", "gitea", "admin", "user"]
    r = compose(*base, "create", "--username", GITEA_OTHER, "--password", pw, "--email",
                f"{GITEA_OTHER}@actrec.invalid", "--must-change-password=false", timeout=120)
    if r.returncode != 0 and "already exists" not in (r.stdout + r.stderr):
        raise RuntimeError("gitea second user create failed: " + (r.stderr or r.stdout)[:300].replace(pw, "<redacted>"))
    t = compose(*base, "generate-access-token", "--username", GITEA_OTHER, "--token-name",
                f"actrec-other-{int(time.time())}", "--scopes", "all", "--raw", timeout=120)
    token = t.stdout.strip().splitlines()[-1] if t.returncode == 0 and t.stdout.strip() else ""
    if not token:
        raise RuntimeError("gitea second-user token creation failed")
    register_secret(token)
    return token


def image_lock() -> dict:
    lock = {}
    for img in IMAGES:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", img],
                           capture_output=True, text=True)
        lock[img] = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    write_json(ROOT / "docker" / "images.lock.json", lock)
    return lock


def up(engine: bool = False) -> dict:
    profiles = ("tb", "engine") if engine else ("tb",)
    r = compose("up", "-d", "--wait", "--wait-timeout", "240", profiles=profiles, timeout=900)
    if r.returncode != 0:
        # --wait fails for services without healthchecks on some compose versions; fall back to our probes
        r = compose("up", "-d", profiles=profiles, timeout=900)
        if r.returncode != 0:
            raise RuntimeError("compose up failed:\n" + r.stderr[-1500:])
    h = health(include_tb=True, include_engine=engine)
    proxies = Toxi().ensure_proxies()
    token = gitea_bootstrap()
    other = gitea_other_bootstrap()
    save_stack_secrets({"gitea_token": token, "gitea_other_token": other})
    from .services.pg import ensure_schema
    ensure_schema()
    lock = image_lock()
    info = {"health_s": h, "proxies": proxies, "images": lock}
    print(json.dumps({"health_s": h, "proxies": proxies}, indent=1))
    return info


def down() -> None:
    r = compose("down", "-v", "--remove-orphans", profiles=("tb", "engine"))
    print(r.stdout[-500:], r.stderr[-500:])


def status() -> None:
    r = compose("ps", "--format", "json", profiles=("tb", "engine"))
    print(r.stdout)
    print("proxies:", {k: (v[0], v[1]) for k, v in PROXIES.items()})
