"""Restart a hung Docker Desktop on Windows: stop it, shut down its WSL backend, relaunch, wait for the engine,
then start the study's FHIR container again. Touches nothing else."""
import subprocess, time, os
def sh(a, t=60):
    try:
        r = subprocess.run(a, capture_output=True, text=True, timeout=t); return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return -1, type(e).__name__
print("engine check:", sh(["docker", "info", "--format", "{{.ServerVersion}}"], 30), flush=True)
for im in ("Docker Desktop.exe", "com.docker.backend.exe", "com.docker.build.exe", "docker.exe"):
    print("taskkill", im, sh(["taskkill", "/F", "/T", "/IM", im], 60)[0], flush=True)
print("wsl --shutdown", sh(["wsl", "--shutdown"], 180), flush=True)
time.sleep(15)
exe = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
subprocess.Popen([exe], close_fds=True, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
print("launched Docker Desktop", flush=True)
t0 = time.time(); up = False
while time.time() - t0 < 900:
    time.sleep(15)
    rc, out = sh(["docker", "info", "--format", "{{.ServerVersion}}"], 30)
    if rc == 0 and out and "error" not in out.lower():
        up = True; print(f"engine up after {time.time()-t0:.0f}s: {out}", flush=True); break
if not up:
    print("engine still down after 15 min"); raise SystemExit(2)
print("start medagentbench:", sh(["docker", "start", "medagentbench"], 180), flush=True)
