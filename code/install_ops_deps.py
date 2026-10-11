"""Installs the ops testbed's extra packages (testbeds/ops/requirements.txt) into the project venv (PyPI only)."""
import subprocess
import sys
from pathlib import Path

req = Path(__file__).resolve().parent.parent / "testbeds" / "ops" / "requirements.txt"
r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(req)], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
import boto3, httpx, psycopg, redis, tigerbeetle  # noqa: E401,F401,E402
print("ops deps OK")
