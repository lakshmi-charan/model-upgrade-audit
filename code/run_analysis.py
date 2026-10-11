"""Runs the full analysis: confirmatory H1–H5, transitions + H6 erosion test,
FlipScout per testbed, and per-pair summaries. Outputs in analysis/out/; a text log on stdout."""
import subprocess, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
env = dict(os.environ, MM_PROJECT_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
def run(args):
    print("\n$", " ".join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(r.stdout[-20000:]); print(r.stderr[-3000:], flush=True)
run(["analysis/confirmatory.py", "--subset-map", "clinical=main,ops=main50,mlops=main40"])
run(["analysis/transitions.py", "clinical=main,ops=main50,mlops=main40"])
pairs = ("gpt-4.1-nano>gpt-5.6-luna,gpt-5-nano>gpt-5.6-luna,gpt-4.1-nano>gpt-5-nano,qwen3-30b-a3b-2507>qwen3.5-35b-a3b,"
         "mistral-small-3.2>mistral-small-2603,gemma-3-27b>gemma-4-31b,gemini-3.1-flash-lite>gemini-3.5-flash-lite")
for tb, sub in (("clinical", "main"), ("ops", "main50"), ("mlops", "main40")):
    run(["analysis/flipscout.py", "--testbed", tb, "--subset", sub, "--pairs", pairs, "--seeds", "200"])
    run(["analysis/flips.py", "--testbed", tb, "--subset", sub, "--pairs", pairs])
