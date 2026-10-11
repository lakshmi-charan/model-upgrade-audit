"""Core abstractions: versioned components, pipeline DAGs, and an executor that measures per-component cost.

A pipeline is a DAG of components. Each component has named code variants; a *version* fixes the variant,
its parameters, the reference artifacts it reads, and the library environment it runs in. A pipeline *state*
maps every component to a version. Raw inputs are "bronze" tables for one period.

The executor runs a state on a period, caching each component output on disk keyed by everything that
determines it (period, component, version, upstream keys). Every component execution is timed once, and the
measured seconds are stored with the cached output, so diagnosis cost is charged from real measurements even
when a later call is served from cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DOD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CACHE = Path(os.environ.get("DOD_CACHE", ROOT / "data" / "cache"))
ENVS = ROOT / "envs"
MAIN_ENV = "main"


def stable_hash(obj, n: int = 16) -> str:
    return hashlib.blake2b(json.dumps(obj, sort_keys=True, default=str).encode(), digest_size=n).hexdigest()


@dataclass(frozen=True)
class Version:
    variant: str
    params: tuple = ()          # sorted (key, value) pairs
    artifacts: tuple = ()       # sorted (artifact_name, artifact_version) pairs
    env: str = MAIN_ENV

    @staticmethod
    def make(variant: str, params: dict | None = None, artifacts: dict | None = None, env: str = MAIN_ENV):
        return Version(variant, tuple(sorted((params or {}).items())), tuple(sorted((artifacts or {}).items())), env)

    def p(self) -> dict:
        return dict(self.params)

    def a(self) -> dict:
        return dict(self.artifacts)

    def to_json(self) -> dict:
        return {"variant": self.variant, "params": self.p(), "artifacts": self.a(), "env": self.env}

    @staticmethod
    def from_json(d: dict) -> "Version":
        return Version.make(d["variant"], d.get("params"), d.get("artifacts"), d.get("env", MAIN_ENV))

    def key(self) -> str:
        return stable_hash(self.to_json())

    def short(self) -> str:
        bits = [self.variant] + [f"{k}={v}" for k, v in self.params] + [f"{k}@{v}" for k, v in self.artifacts]
        if self.env != MAIN_ENV:
            bits.append(f"env={self.env}")
        return ",".join(bits)


@dataclass
class Component:
    name: str
    inputs: list[str]                                   # upstream component names, or "raw:<table>"
    variants: dict[str, Callable] = field(default_factory=dict)
    doc: str = ""
    deterministic: bool = True

    def variant(self, name: str):
        def deco(fn):
            self.variants[name] = fn
            return fn
        return deco


@dataclass
class Pipeline:
    domain: str
    components: dict[str, Component]                    # in topological order
    output: str                                         # node consumed by the model
    raw_tables: list[str]
    load_raw: Callable[[str, str], pd.DataFrame]        # (period, table) -> bronze frame
    load_artifact: Callable[[str, str], object]         # (name, version) -> artifact

    def order(self) -> list[str]:
        return list(self.components)

    def upstream(self, node: str) -> list[str]:
        """All component ancestors of node (inclusive), in topological order."""
        need, stack = set(), [node]
        while stack:
            n = stack.pop()
            if n in need or n.startswith("raw:"):
                continue
            need.add(n)
            stack.extend(self.components[n].inputs)
        return [c for c in self.order() if c in need]

    def downstream(self, node: str) -> list[str]:
        out, frontier = [], {node}
        for c in self.order():
            if c != node and any(i in frontier for i in self.components[c].inputs):
                out.append(c)
                frontier.add(c)
        return out


State = dict  # component name -> Version


def state_to_json(state: State) -> dict:
    return {k: v.to_json() for k, v in state.items()}


def state_from_json(d: dict) -> State:
    return {k: Version.from_json(v) for k, v in d.items()}


class ExecError(RuntimeError):
    pass


def _replace(tmp: Path, dst: Path) -> None:
    """os.replace with retries (Windows: a reader or another worker may hold dst for a moment). If another worker
    already wrote the same content-addressed file, keep theirs."""
    for i in range(30):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    if dst.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
        return
    os.replace(tmp, dst)


class Executor:
    """Runs pipeline states on periods with content-addressed caching and per-component cost accounting."""

    def __init__(self, pipe: Pipeline, cache: Path = CACHE, use_cache: bool = True):
        self.pipe, self.use_cache = pipe, use_cache
        self.cache = cache / pipe.domain
        self.cache.mkdir(parents=True, exist_ok=True)
        self.charged_s = 0.0          # cost charged since the last reset (measured seconds of executed nodes)
        self.ledger: list[dict] = []
        self._code: dict = {}
        self._art: dict = {}
        from collections import OrderedDict
        self._mem: "OrderedDict[str, pd.DataFrame]" = OrderedDict()   # small LRU of recently read outputs
        self._mem_max = int(os.environ.get("DOD_MEM_FRAMES", "24"))
        self.free_keys: set[str] = set()   # outputs archived by production runs: reading them costs nothing
        self.charge_all = False            # charge every visited node, archived or not (full re-run baselines)

    def reset_charge(self) -> None:
        self.charged_s, self.ledger = 0.0, []

    # ------------------------------------------------------------------ keys
    def node_key(self, period: str, state: State, node: str, input_override: dict | None = None,
                 _memo: dict | None = None) -> str:
        memo = {} if _memo is None else _memo
        if node in memo:
            return memo[node]
        comp = self.pipe.components[node]
        ups = []
        for i in comp.inputs:
            if input_override and i in input_override:
                ups.append(("override", input_override[i]))
            elif i.startswith("raw:"):
                ups.append((i, period))
            else:
                ups.append((i, self.node_key(period, state, i, input_override, memo)))
        k = stable_hash({"node": node, "version": state[node].to_json(), "ups": ups,
                         "code": self.code_hash(node, state[node].variant),
                         "art": [self.artifact_hash(n, v) for n, v in state[node].artifacts]})
        memo[node] = k
        return k

    def code_hash(self, node: str, variant: str) -> str:
        """Fingerprint of a component variant's source code, so cached outputs never outlive a code change."""
        k = (node, variant)
        if k not in self._code:
            import inspect
            fn = self.pipe.components[node].variants.get(variant)
            try:
                src = inspect.getsource(fn) if fn else variant
            except (OSError, TypeError):
                src = variant
            self._code[k] = stable_hash(src, 8)
        return self._code[k]

    def artifact_hash(self, name: str, version: str) -> str:
        """Content fingerprint of a reference artifact, so cached outputs never outlive an artifact rebuild."""
        k = (name, version)
        if k not in self._art:
            try:
                self._art[k] = stable_hash(self.pipe.load_artifact(name, version), 8)
            except Exception:
                self._art[k] = "missing"
        return self._art[k]

    def _paths(self, node: str, key: str) -> tuple[Path, Path]:
        d = self.cache / node
        return d / f"{key}.parquet", d / f"{key}.json"

    # ------------------------------------------------------------------ execution
    def _exec_component(self, node: str, version: Version, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        comp = self.pipe.components[node]
        if version.variant not in comp.variants:
            raise ExecError(f"{node}: unknown variant {version.variant}")
        if version.env == MAIN_ENV:
            arts = {k: self.pipe.load_artifact(k, v) for k, v in version.artifacts}
            return comp.variants[version.variant](inputs, version.p(), arts)
        return self._exec_in_env(node, version, inputs)

    def _exec_in_env(self, node: str, version: Version, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run one component in another library environment (a separate venv) via a subprocess."""
        py = ENVS / version.env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.exists():
            raise ExecError(f"environment {version.env} not installed at {py}")
        tmp = self.cache / "_xfer" / stable_hash([node, version.to_json(), time.time_ns()])
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            names = {}
            for i, (k, df) in enumerate(inputs.items()):
                p = tmp / f"in{i}.parquet"
                df.to_parquet(p, index=False)
                names[k] = str(p)
            spec = {"domain": self.pipe.domain, "node": node, "version": version.to_json(), "inputs": names,
                    "out": str(tmp / "out.parquet")}
            (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), DOD_PROJECT_ROOT=str(ROOT))
            r = subprocess.run([str(py), "-m", "dod.subexec", str(tmp / "spec.json")], capture_output=True,
                               text=True, env=env, cwd=str(ROOT))
            if r.returncode != 0:
                raise ExecError(f"{node} in env {version.env} failed: {r.stderr[-1500:]}")
            return pd.read_parquet(tmp / "out.parquet")
        finally:
            for p in tmp.glob("*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                tmp.rmdir()
            except OSError:
                pass

    def node(self, period: str, state: State, node: str, input_override: dict | None = None,
             override_frames: dict | None = None, charge: bool = True, _memo: dict | None = None) -> pd.DataFrame:
        """Output of `node` under `state` on `period`. input_override maps an input name to an opaque id whose
        frame is given in override_frames (used by replay tests that feed archived inputs)."""
        memo = {} if _memo is None else _memo
        key = self.node_key(period, state, node, input_override)
        out_p, meta_p = self._paths(node, key)
        if self.use_cache and out_p.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            if meta.get("error"):
                if charge:
                    self._charge(node, meta["seconds"], cached=True, error=True, key=key)
                raise ExecError(meta["error"])
            if charge:
                self._charge(node, meta["seconds"], cached=True, key=key)
            return self._read(out_p)
        comp = self.pipe.components[node]
        inputs = {}
        for i in comp.inputs:
            if input_override and i in input_override:
                inputs[i] = override_frames[input_override[i]]
            elif i.startswith("raw:"):
                inputs[i] = self.pipe.load_raw(period, i[4:])
            else:
                inputs[i] = self.node(period, state, i, input_override, override_frames, charge, memo)
        t0 = time.perf_counter()
        try:
            out = self._exec_component(node, state[node], inputs)
            if not isinstance(out, pd.DataFrame):
                raise ExecError(f"{node} returned {type(out)}")
        except Exception as e:
            secs = time.perf_counter() - t0
            msg = f"{type(e).__name__}: {str(e)[:500]}"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            tmpm = meta_p.with_suffix(f".{os.getpid()}.tmp")
            tmpm.write_text(json.dumps({"seconds": secs, "error": msg}))
            _replace(tmpm, meta_p)
            if not out_p.exists():
                pd.DataFrame().to_parquet(out_p)
            if charge:
                self._charge(node, secs, cached=False, error=True, key=key)
            raise ExecError(msg) from e
        secs = time.perf_counter() - t0
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_p.with_suffix(f".{os.getpid()}.tmp")
        out.to_parquet(tmp, index=False)
        _replace(tmp, out_p)
        tmpm = meta_p.with_suffix(f".{os.getpid()}.tmp")
        tmpm.write_text(json.dumps({"seconds": secs, "rows": len(out), "version": state[node].to_json()}))
        _replace(tmpm, meta_p)
        if charge:
            self._charge(node, secs, cached=False, key=key)
        return out

    def _read(self, path: Path) -> pd.DataFrame:
        k = str(path)
        if k in self._mem:
            self._mem.move_to_end(k)
            return self._mem[k].copy(deep=True)
        df = pd.read_parquet(path)
        self._mem[k] = df
        while len(self._mem) > self._mem_max:
            self._mem.popitem(last=False)
        return df.copy(deep=True)

    def _charge(self, node: str, secs: float, cached: bool, error: bool = False, key: str | None = None) -> None:
        if key is not None and key in self.free_keys and not self.charge_all:
            return
        self.charged_s += secs
        self.ledger.append({"node": node, "seconds": round(secs, 4), "cached": cached, "error": error})

    def run(self, period: str, state: State, node: str | None = None) -> pd.DataFrame:
        return self.node(period, state, node or self.pipe.output)

    def all_keys(self, period: str, state: State) -> set[str]:
        memo: dict = {}
        for n in self.pipe.order():
            self.node_key(period, state, n, None, memo)
        return set(memo.values())

    def node_cost(self, period: str, state: State, node: str) -> float | None:
        """Measured seconds of one component execution, if it has been executed before."""
        _, meta_p = self._paths(node, self.node_key(period, state, node))
        if meta_p.exists():
            return json.loads(meta_p.read_text())["seconds"]
        return None


def _eq_cols(x: pd.Series, y: pd.Series, rtol: float = 1e-9) -> np.ndarray:
    """Element-wise equality of two aligned columns (NaN == NaN; floats to rtol; otherwise exact as strings)."""
    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not (
            pd.api.types.is_bool_dtype(x) ^ pd.api.types.is_bool_dtype(y)):
        xv = x.to_numpy(dtype="float64", na_value=np.nan)
        yv = y.to_numpy(dtype="float64", na_value=np.nan)
        return np.isclose(xv, yv, rtol=rtol, atol=0.0, equal_nan=True)
    xn, yn = pd.isna(x).to_numpy(), pd.isna(y).to_numpy()
    xs = x.to_numpy(dtype=object)
    ys = y.to_numpy(dtype=object)
    same = np.array([a == b if type(a) is type(b) else str(a) == str(b) for a, b in zip(xs, ys)], dtype=bool) \
        if (x.dtype != y.dtype) else (xs == ys)
    return (same & ~xn & ~yn) | (xn & yn)


def frames_equal(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-9) -> bool:
    """Exact comparison used by rollback and replay tests (column names, row count, values; floats to rtol)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return all(bool(_eq_cols(a[c], b[c], rtol).all()) for c in a.columns)


def diff_summary(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """How two frames differ: per-column fraction of differing values (aligned by position)."""
    out = {"rows": [len(a), len(b)], "cols_only_a": sorted(set(a.columns) - set(b.columns)),
           "cols_only_b": sorted(set(b.columns) - set(a.columns)), "col_diff_frac": {}}
    if len(a) != len(b):
        return out
    for c in a.columns:
        if c in b.columns and len(a):
            f = float(1.0 - _eq_cols(a[c], b[c]).mean())
            if f > 0:
                out["col_diff_frac"][c] = round(f, 6)
    return out
