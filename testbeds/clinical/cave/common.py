"""Shared helpers: paths, config, .env loading, time parsing, JSON I/O."""
import json
import os
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
_CFG = None
_LOCK = threading.Lock()


def load_env(path=None):
    """Minimal .env loader (KEY=VALUE per line). Never prints values."""
    candidates = [Path(path)] if path else [ROOT / ".env", ROOT.parent / ".env", Path.cwd() / ".env"]
    for p in candidates:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
            return str(p)
    return None


def cfg():
    global _CFG
    if _CFG is None:
        path = os.environ.get("CAVE_CONFIG") or (ROOT / "config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def out_dir(*parts):
    p = ROOT / cfg().get("output_dir", "runs") / Path(*parts) if parts else ROOT / cfg().get("output_dir", "runs")
    p.mkdir(parents=True, exist_ok=True)
    return p


NOW_STR = "2023-11-13T10:15:00+00:00"


def now():
    return parse_dt(cfg().get("now", NOW_STR))


def parse_dt(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if len(s) == 10:  # date only
        s = s + "T00:00:00+00:00"
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        # handle fractional seconds of odd length etc.
        base = s.split(".")[0]
        tz = ""
        for sep in ["+", "-"]:
            idx = s.rfind(sep)
            if idx > 10:
                tz = s[idx:]
                break
        d = datetime.fromisoformat(base + tz)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def to_float(x):
    try:
        if isinstance(x, bool):
            return None
        return float(x)
    except (TypeError, ValueError):
        try:
            return float(str(x).strip().split()[0])
        except Exception:
            return None


__all__ = ["cfg", "load_env", "out_dir", "now", "parse_dt", "write_json", "read_json", "append_jsonl",
           "read_jsonl", "sha", "to_float", "timedelta", "timezone", "datetime", "ROOT", "PKG_DIR"]
