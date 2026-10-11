"""Recompute every clinical task's reference (CAVE solve()) against the LIVE FHIR server and compare with the
frozen references. Writes testbeds/clinical/data/refs_live.json and a diff report. Free (no model calls)."""
import json, os, sys
from pathlib import Path
ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TB = ROOT / "testbeds" / "clinical"
os.environ["CAVE_CONFIG"] = str(TB / "config.yaml")
sys.path.insert(0, str(TB))
from cave import common, fhir
from cave.reference import solve
rt = TB / "runtime.json"
if rt.exists():
    common.cfg()["fhir_api_base"] = json.loads(rt.read_text())["fhir_api_base"]
print("base", fhir.base(), "ping", fhir.ping(), flush=True)
tasks = json.loads((TB / "data" / "base_tasks.json").read_text(encoding="utf-8")) + \
        json.loads((TB / "data" / "stress_tasks.json").read_text(encoding="utf-8"))
live, diffs = {}, []
for t in tasks:
    r = solve(t)
    live[t["id"]] = r
    a = json.dumps(r, sort_keys=True, default=str); b = json.dumps(t["ref"], sort_keys=True, default=str)
    if a != b:
        diffs.append({"id": t["id"], "frozen": t["ref"], "live": r})
(TB / "data" / "refs_live.json").write_text(json.dumps(live, indent=0, default=str), encoding="utf-8")
(ROOT / "runs" / "clinical_ref_diffs.json").write_text(json.dumps(diffs, indent=1, default=str), encoding="utf-8")
print(f"{len(tasks)} tasks, {len(diffs)} references differ from the frozen ones")
for d in diffs:
    print("==", d["id"]); print("  frozen:", json.dumps(d["frozen"], default=str)[:400]); print("  live:  ", json.dumps(d["live"], default=str)[:400])
