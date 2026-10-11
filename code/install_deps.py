"""Installs the few Python packages the harness needs into the project venv (PyPI only)."""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pyyaml", "requests", "numpy", "pandas", "scipy",
                    "statsmodels", "matplotlib"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
import yaml, requests, numpy, scipy, pandas  # noqa
print("deps OK")
