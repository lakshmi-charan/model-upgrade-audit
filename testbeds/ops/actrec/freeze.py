"""Pre-registration freeze: hash every file that can change what the test split measures or how it is analysed.

    python code/run.py freeze            write FREEZE.json (refuses if it already exists)
    python code/run.py freeze --check    compare the working tree with FREEZE.json

Test-split presets refuse to start unless the check passes, so no code, task, prompt, grader, analysis or
protocol file can change between the freeze and the end of the main run without it being detected. A deliberate
change after the freeze needs a new, dated amendment in docs/PROTOCOL.md and a re-freeze (FREEZE.json is then
renamed, never overwritten), so the history of freezes stays in the repository.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

from .config import ROOT

FROZEN_DIRS = ["code/actrec", "code/tools", "analysis", "docker"]
FROZEN_FILES = ["code/run.py", "docs/04_study_design.md", "docs/05_theory.md", "docs/06_analysis_plan.md",
                "docs/07_followup_plan.md", "docs/PROTOCOL.md", "requirements.txt"]
SKIP = ("__pycache__", ".pyc", ".tmp")
FREEZE = ROOT / "FREEZE.json"


def _files() -> list:
    out = []
    for d in FROZEN_DIRS:
        p = ROOT / d
        if p.exists():
            out += [f for f in p.rglob("*") if f.is_file() and not any(s in str(f) for s in SKIP)]
    out += [ROOT / f for f in FROZEN_FILES if (ROOT / f).exists()]
    return sorted(set(out))


def _sha(p) -> str:
    # normalise line endings so a checkout on another OS hashes identically
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest() -> dict:
    return {str(f.relative_to(ROOT)).replace("\\", "/"): _sha(f) for f in _files()}


def digest(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def write() -> dict:
    if FREEZE.exists():
        raise SystemExit("FREEZE.json exists; an amendment must rename it first (see docs/PROTOCOL.md)")
    missing = [f for f in ("docs/06_analysis_plan.md", "docs/PROTOCOL.md") if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"cannot freeze without {missing}")
    m = manifest()
    rec = {"frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": m, "digest": digest(m)}
    FREEZE.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def check() -> tuple[bool, list[str], str | None]:
    if not FREEZE.exists():
        return False, ["FREEZE.json missing"], None
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    now = manifest()
    diff = sorted({k for k in set(f["files"]) | set(now) if f["files"].get(k) != now.get(k)})
    return not diff, diff, f["digest"]


def main(argv: list[str]) -> None:
    if "--check" in argv:
        ok, diff, d = check()
        print(f"freeze {'OK' if ok else 'BROKEN'} (digest {d})")
        for x in diff:
            print("  changed:", x)
        sys.exit(0 if ok else 1)
    rec = write()
    print(f"frozen {len(rec['files'])} files at {rec['frozen_at_utc']}, digest {rec['digest']}")
