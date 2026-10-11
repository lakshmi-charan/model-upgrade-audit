"""Audit: find graded episodes in which the environment (not the model) failed, e.g. the FHIR server or a
service refused connections. Such episodes violate the protocol's exclusion rule (episodes lost to
infrastructure errors must be re-run, not graded). Output: analysis/out/infra_scan.json.
Usage: python code/infra_scan.py [clinical=main,ops=main50,mlops=main40]"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PATTERNS = {
    "conn_refused": re.compile(r"WinError 10061|Connection refused|ConnectionRefused|actively refused", re.I),
    "new_conn": re.compile(r"Failed to establish a new connection|NewConnectionError", re.I),
    "max_retries": re.compile(r"Max retries exceeded", re.I),
    "reset": re.compile(r"ConnectionResetError|RemoteDisconnected|Connection aborted|WinError 10054", re.I),
    "timeout": re.compile(r"Read timed out|ConnectTimeout|ReadTimeout", re.I),
    "http5xx": re.compile(r"\b50[234] (Bad Gateway|Service Unavailable|Gateway Time-?out)", re.I),
    "send_err": re.compile(r"Error in sending the (GET|POST)", re.I),
}
smap = dict(x.split("=") for x in (sys.argv[1] if len(sys.argv) > 1 else "clinical=main,ops=main50,mlops=main40").split(","))
out = {"episodes": [], "summary": {}}
for tb, sub in smap.items():
    base = ROOT / "runs" / tb / sub
    if not base.exists():
        continue
    for f in base.rglob("*.json"):
        rel = f.relative_to(base).parts
        if len(rel) < 3 or not rel[-2].startswith("r"):
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = {k: len(p.findall(txt)) for k, p in PATTERNS.items()}
        hits = {k: v for k, v in hits.items() if v}
        label, rep, task = rel[0], rel[-2], f.stem
        key = f"{tb}|{label}|{rep}"
        s = out["summary"].setdefault(key, {"n": 0, "affected": 0})
        s["n"] += 1
        if hits:
            s["affected"] += 1
            try:
                succ = json.loads(txt).get("success")
            except Exception:
                succ = None
            out["episodes"].append({"testbed": tb, "label": label, "rep": rep, "task": task, "hits": hits,
                                    "success": succ, "mtime": f.stat().st_mtime})
p = ROOT / "analysis" / "out" / "infra_scan.json"
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in sorted(out["summary"].items()):
    if v["affected"]:
        print(f"{k:50s} affected {v['affected']}/{v['n']}")
print("total affected episodes:", len(out["episodes"]))
