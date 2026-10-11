"""Freeze the confirmatory protocol: SHA-256 of code, analysis scripts, configs, subsets and the protocol.
Usage: python code/freeze.py write   -> FREEZE.json      |   python code/freeze.py check -> verify unchanged."""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["code/*.py", "code/mm/*.py", "analysis/*.py", "config/models.yaml", "docs/PROTOCOL.md",
            "testbeds/*/subsets/*.json", "testbeds/clinical/cave/*.py", "testbeds/clinical/config.yaml",
            "testbeds/clinical/data/refs_live.json", "testbeds/ops/actrec/*.py", "testbeds/mlops/*.py",
            "testbeds/mlops/data/cases.jsonl"]
def digest():
    out = {}
    for pat in PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file():
                out[str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
if __name__ == "__main__":
    f = ROOT / "FREEZE.json"
    if sys.argv[1:] == ["write"]:
        f.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "files": digest()}, indent=1), encoding="utf-8")
        print("frozen", len(digest()), "files")
    else:
        old = json.loads(f.read_text(encoding="utf-8"))["files"]; new = digest()
        changed = [k for k in old if new.get(k) != old[k]] + [k for k in new if k not in old]
        print("FREEZE OK" if not changed else "CHANGED: " + ", ".join(changed)); sys.exit(1 if changed else 0)
