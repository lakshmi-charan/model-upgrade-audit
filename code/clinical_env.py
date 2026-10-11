"""Brings up and verifies the clinical testbed's FHIR server (MedAgentBench image) on the Windows host.

Steps: Docker engine (launch Docker Desktop if needed) -> reuse an existing medagentbench container (any port)
or create `mm-medagentbench` on host port 18080 -> wait for HAPI -> verify the stress-suite records; if they
are missing, rebuild them deterministically (CAVE build_stress, seed 7) and apply them, but only after checking
that the rebuilt tasks reproduce the frozen stress_tasks.json exactly. The resolved base URL is written to
testbeds/clinical/runtime.json, which run_clinical.py loads.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("MM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TB = ROOT / "testbeds" / "clinical"
RUNTIME = TB / "runtime.json"
IMAGE = "jyxsu6/medagentbench"


def sh(args, timeout=120):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return -1, "", f"{type(e).__name__}: {e}"


def port_open(port, host="127.0.0.1", timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def docker_engine(max_wait_s=900):
    rc, out, _ = sh(["docker", "info", "--format", "{{.ServerVersion}}"], 60)
    if rc == 0 and out:
        print("docker engine up:", out, flush=True)
        return True
    print("Docker engine not running; launching Docker Desktop", flush=True)
    for exe in (r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe")):
        if os.path.exists(exe):
            subprocess.Popen([exe], close_fds=True, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
            print("started", exe, flush=True)
            break
    else:
        print("Docker Desktop executable not found", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        time.sleep(15)
        rc, out, _ = sh(["docker", "info", "--format", "{{.ServerVersion}}"], 60)
        if rc == 0 and out:
            print(f"docker engine up after {time.time() - t0:.0f}s:", out, flush=True)
            return True
    print("Docker engine not available", flush=True)
    return False


def find_container():
    rc, out, err = sh(["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}\t{{.Ports}}"])
    print("containers:\n" + (out or err), flush=True)
    rows = [ln.split("\t") for ln in out.splitlines() if "medagentbench" in ln]
    # prefer the study's own container, then any other
    rows.sort(key=lambda r: 0 if r[3] == "mm-medagentbench" else 1)
    return rows[0] if rows else None


def host_port(cid):
    rc, out, _ = sh(["docker", "port", cid, "8080/tcp"])
    for ln in out.splitlines():
        if ":" in ln:
            return int(ln.rsplit(":", 1)[1])
    return None


def _fhir_up(base):
    import urllib.request
    try:
        req = urllib.request.Request(base + "metadata?_format=json")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server():
    if RUNTIME.exists():                       # server already up? then Docker's state does not matter
        base = json.loads(RUNTIME.read_text(encoding="utf-8")).get("fhir_api_base")
        if base and _fhir_up(base):
            print("FHIR server already answering at", base, flush=True)
            return base
    if not docker_engine():
        return None
    row = find_container()
    if row:
        cid = row[0]
        if not row[2].startswith("Up"):
            print("starting container", row[3], flush=True)
            sh(["docker", "start", cid], 180)
        port = host_port(cid)
    else:
        port = 18080
        if port_open(port):
            print(f"port {port} is busy; cannot create the container", flush=True)
            return None
        print(f"creating mm-medagentbench on host port {port} (first pull can take several minutes)", flush=True)
        rc, out, err = sh(["docker", "run", "-d", "--name", "mm-medagentbench", "-p", f"{port}:8080",
                           f"{IMAGE}:latest"], 3600)
        print("docker run:", rc, out[:80], err[-300:], flush=True)
        if rc != 0:
            return None
    if not port:
        print("could not resolve the container's host port", flush=True)
        return None
    base = f"http://localhost:{port}/fhir/"
    RUNTIME.write_text(json.dumps({"fhir_api_base": base}), encoding="utf-8")
    return base


def main() -> bool:
    base = ensure_server()
    if not base:
        return False
    os.environ["CAVE_CONFIG"] = str(TB / "config.yaml")
    import sys
    sys.path.insert(0, str(TB))
    from cave import common, fhir, tasks as ctasks
    common.cfg()["fhir_api_base"] = base
    common.cfg()["fhir_timeout"] = 8
    t0 = time.time()
    while not fhir.ping():
        if time.time() - t0 > 900:
            print("FHIR server did not answer within 15 min", flush=True)
            return False
        time.sleep(10)
    print(f"FHIR server answering at {base} after {time.time() - t0:.0f}s", flush=True)
    common.cfg()["fhir_timeout"] = 60
    frozen = json.loads((TB / "data" / "stress_tasks.json").read_text(encoding="utf-8"))
    baits = [t["bait_id"] for t in frozen if t.get("bait_id")]
    present = sum(fhir.read("Patient", b) is not None for b in baits)
    print(f"stress setup: {present}/{len(baits)} look-alike patients present", flush=True)
    if present == len(baits):
        return True
    if present:
        print("partial stress setup found; refusing to write more (manual check needed)", flush=True)
        return False
    print("rebuilding the stress setup deterministically", flush=True)
    rebuilt, setup = ctasks.build_stress()
    from cave.reference import solve
    keyf = lambda t: (t["id"], t["template"], t.get("mrn"), t.get("instruction"), t.get("hazard"), t.get("bait_id"))
    if sorted(map(keyf, rebuilt)) != sorted(map(keyf, frozen)):
        print("rebuilt stress tasks differ from the frozen ones; not applying", flush=True)
        return False
    ctasks.apply_setup(setup)
    bad = 0
    fz = {t["id"]: t for t in frozen}
    for t in rebuilt:
        if json.dumps(solve(t), sort_keys=True, default=str) != json.dumps(fz[t["id"]]["ref"], sort_keys=True, default=str):
            bad += 1
    present = sum(fhir.read("Patient", b) is not None for b in baits)
    print(f"applied {len(setup)} setup records; look-alikes now {present}/{len(baits)}; reference mismatches {bad}",
          flush=True)
    return present == len(baits) and bad == 0


if __name__ == "__main__":
    ok = main()
    print("ENVIRONMENT OK" if ok else "ENVIRONMENT NOT READY", flush=True)
    raise SystemExit(0 if ok else 2)
