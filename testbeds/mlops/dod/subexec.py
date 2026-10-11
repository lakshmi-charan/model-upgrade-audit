"""Run one component in the current interpreter's library environment (called by Executor._exec_in_env)."""
import json
import sys
import warnings

import pandas as pd

from dod.core import Version
from dod.domains import get_pipeline


def main(spec_path: str) -> None:
    warnings.simplefilter("ignore")
    spec = json.loads(open(spec_path, encoding="utf-8").read())
    pipe = get_pipeline(spec["domain"])
    v = Version.from_json(spec["version"])
    comp = pipe.components[spec["node"]]
    inputs = {k: pd.read_parquet(p) for k, p in spec["inputs"].items()}
    arts = {k: pipe.load_artifact(k, a) for k, a in v.artifacts}
    out = comp.variants[v.variant](inputs, v.p(), arts)
    out.to_parquet(spec["out"], index=False)


if __name__ == "__main__":
    main(sys.argv[1])
