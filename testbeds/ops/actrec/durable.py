"""Durable local state that survives a driver crash: an append-only JSONL log with fsync on every record.

* Checkpoint: step-level progress (which plan steps completed, with their results). Every condition gets it,
  like a framework checkpointer. It is written only AFTER a step's write returned, so a crash between the
  service committing and the checkpoint being written leaves the step 'in progress' on restart.
* Journal: the method's write-ahead intent log (op id, state machine, attempts). Only Settle uses it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # a torn last line after a crash is ignored
        return out


class Checkpoint:
    def __init__(self, path: Path):
        self.log = Log(path)
        self.done: dict[str, dict] = {}
        for r in self.log.records():
            if r.get("type") == "reset":
                self.done.clear()
            else:
                self.done[r["step"]] = r.get("value") or {}

    def add(self, step: str, value: dict | None) -> None:
        self.done[step] = value or {}
        self.log.append({"step": step, "value": value or {}})

    def reset(self) -> None:
        self.done.clear()
        self.log.append({"type": "reset"})


class Journal:
    """Latest state per intent; every transition is appended before it takes effect."""

    def __init__(self, path: Path | None):
        self.log = Log(path) if path else None
        self.ops: dict[str, dict] = {}
        if self.log:
            for r in self.log.records():
                self.ops.setdefault(r["intent"], {}).update(r)

    def get(self, intent: str) -> dict | None:
        return self.ops.get(intent)

    def put(self, intent: str, **fields) -> dict:
        rec = self.ops.setdefault(intent, {"intent": intent})
        rec.update(fields)
        if self.log:
            self.log.append({"intent": intent, **fields})
        return rec

    def all(self) -> list[dict]:
        return list(self.ops.values())
