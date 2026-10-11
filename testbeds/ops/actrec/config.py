"""Paths, ports and environment. Never prints secret values."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# [model-migration port] the testbed root is testbeds/ops (one level up from this package), overridable by
# ACTREC_ROOT; the compose project is "mmops" so the source study's volumes are never touched.
ROOT = Path(os.environ.get("ACTREC_ROOT") or Path(__file__).resolve().parents[1])
RUNS = ROOT / "runs"
SECRETS = RUNS / ".secrets"
COMPOSE_FILE = ROOT / "docker" / "compose.yml"
PROJECT = os.environ.get("ACTREC_PROJECT", "mmops")

TOXI_URL = "http://localhost:8474"

# name -> (clean proxy port, fault proxy base port, upstream inside the docker network, direct host port)
# Fault proxies exist once per worker slot w: port = base + 10*w, proxy name "<svc>_fault<w>". Parallel
# episodes therefore never share a faulted path.
PROXIES: dict[str, tuple[int, int, str, int]] = {
    "pg": (26000, 26100, "postgres:5432", 5432),
    "gitea": (26001, 26101, "gitea:3000", 3000),
    "smtp": (26002, 26102, "mailpit:1025", 1025),
    "redis": (26003, 26103, "redis:6379", 6379),
    "tb": (26004, 26104, "tigerbeetle:3001", 3001),
    "s3": (26005, 26105, "seaweedfs:8333", 8333),
}
MAX_WORKERS = 8
MAILPIT_API = "http://localhost:8025/api/v1"


def port(service: str, route: str) -> int:
    """route: 'clean' (agent path, never faulted), 'fault<w>' (focal call of worker w; 'fault' = worker 0),
    'direct' (grader and background actor only)."""
    clean, fault, _, direct = PROXIES[service]
    if route == "clean":
        return clean
    if route == "direct":
        return direct
    if route.startswith("fault"):
        w = int(route[5:] or 0)
        if not 0 <= w < MAX_WORKERS:
            raise ValueError(route)
        return fault + 10 * w
    raise ValueError(route)


_SECRET_VALUES: list[str] = []
_TOKEN_RES = [re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), re.compile(r"gsk_[A-Za-z0-9]{16,}")]


def register_secret(v: str | None) -> None:
    if v and len(v) >= 6 and v not in _SECRET_VALUES:
        _SECRET_VALUES.append(v)


def redact(x):
    if isinstance(x, dict):
        return {k: redact(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [redact(v) for v in x]
    if not isinstance(x, str):
        return x
    for s in _SECRET_VALUES:
        x = x.replace(s, "<redacted>")
    for p in _TOKEN_RES:
        x = p.sub("<redacted-token>", x)
    home = str(Path.home())
    for h in {home, home.replace("\\", "/")}:
        x = re.sub(re.escape(h), "~", x, flags=re.IGNORECASE)
    return x


STACK_ENV_KEYS = ("POSTGRES_PASSWORD", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "GITEA_ADMIN_PASSWORD")


def load_env() -> dict:
    # [model-migration port] stack credentials may come from the process environment (run_ops.py sets them
    # from testbeds/ops/runs/.secrets/stack_env.json); a testbed-local .env, if any, still wins.
    env: dict = {k: os.environ[k] for k in STACK_ENV_KEYS if os.environ.get(k)}
    for v in env.values():
        register_secret(v)
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            env[k.strip()] = v
            if any(t in k.upper() for t in ("KEY", "PASSWORD", "SECRET", "TOKEN")):
                register_secret(v)
    return env


ENV = load_env()
PG_USER, PG_DB = "actrec", "actrec"
PG_PASSWORD = ENV.get("POSTGRES_PASSWORD", "")
S3_KEY = ENV.get("MINIO_ROOT_USER", "actrec")
S3_SECRET = ENV.get("MINIO_ROOT_PASSWORD", "")
GITEA_ADMIN = "actrec-admin"


def pg_dsn(route: str) -> str:
    return (f"host=localhost port={port('pg', route)} dbname={PG_DB} user={PG_USER} "
            f"password={PG_PASSWORD} connect_timeout=10")


def stack_secrets() -> dict:
    p = SECRETS / "stack.json"
    if not p.exists():
        raise RuntimeError("stack not bootstrapped: run `python code/run.py stack up` first")
    d = json.loads(p.read_text(encoding="utf-8"))
    for v in d.values():
        if isinstance(v, str):
            register_secret(v)
    return d


def save_stack_secrets(d: dict) -> None:
    SECRETS.mkdir(parents=True, exist_ok=True)
    (SECRETS / "stack.json").write_text(json.dumps(d), encoding="utf-8")
    for v in d.values():
        if isinstance(v, str):
            register_secret(v)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(data), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(redact(row), ensure_ascii=False, default=str) + "\n")


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
