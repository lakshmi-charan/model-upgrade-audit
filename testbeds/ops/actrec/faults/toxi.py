"""Toxiproxy control and the network fault modes.

Every service has two proxies: a *clean* one used for all ordinary calls and a *fault* one used only for the
single injected (focal) call. Toxics are attached to the fault proxy, so a fault never touches retries,
reads or compensations that go through the clean proxy. This is what lets an in-flight request commit late
while the agent's re-issue goes through normally.

Adapters call `hook()` after the connection is established and any protocol handshake is done, right before
the write's bytes are sent. The hook installs the toxic, so every mode acts on the same request bytes.

Mode         toxic on fault proxy (installed by hook)          intended hidden outcome
none         -                                                  executes, client sees success
lost_ack     limit_data downstream bytes=1                       executes, client sees connection error
not_exec_err limit_data upstream bytes=1                         never executes, client sees connection error
held         timeout downstream timeout=0 (hold)                 executes, client sees tool deadline
not_exec_to  timeout upstream timeout=T+2s (drop, then close)    never executes, client sees tool deadline
late         latency upstream latency=delta                      executes at ~t0+delta, client sees tool deadline
partial      (adapter-specific: hook before row k+1 = not_exec_err)
redelivery   - (the adapter sends the identical request twice)   executes twice
All outcomes are *intended*; stage-3 validation measures the realized outcome from ground truth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from ..config import MAX_WORKERS, PROXIES, TOXI_URL

NETWORK_MODES = ["none", "lost_ack", "not_exec_err", "held", "not_exec_to", "late", "redelivery"]
# observation-equivalent pairs/groups (same client-visible signature by design)
EQUIV_GROUPS = {"error": ["lost_ack", "not_exec_err"], "deadline": ["held", "not_exec_to", "late"]}
INTENDED_EXECUTIONS = {"none": 1, "lost_ack": 1, "not_exec_err": 0, "held": 1, "not_exec_to": 0, "late": 1,
                       "redelivery": 2}


class Toxi:
    def __init__(self, url: str = TOXI_URL):
        self.c = httpx.Client(base_url=url, timeout=10)

    def version(self) -> str:
        return self.c.get("/version").text.strip()

    def ensure_proxies(self, workers: int = MAX_WORKERS) -> dict:
        """(Re)create the clean proxy and one fault proxy per worker slot for every service. Idempotent."""
        out = {}
        existing = self.c.get("/proxies").json()
        for name, (clean, fault, upstream, _direct) in PROXIES.items():
            pairs = [(name, clean)] + [(f"{name}_fault{w}", fault + 10 * w) for w in range(workers)]
            for pname, p in pairs:
                spec = {"name": pname, "listen": f"0.0.0.0:{p}", "upstream": upstream, "enabled": True}
                if pname in existing:
                    cur = existing[pname]
                    if cur.get("listen", "").endswith(f":{p}") and cur.get("upstream") == upstream:
                        self.clear(pname)
                        self.c.post(f"/proxies/{pname}", json={"enabled": True})
                        out[pname] = "kept"
                        continue
                    self.c.delete(f"/proxies/{pname}")
                r = self.c.post("/proxies", json=spec)
                out[pname] = r.status_code
        return out

    def add(self, proxy: str, name: str, type_: str, stream: str, attributes: dict) -> int:
        r = self.c.post(f"/proxies/{proxy}/toxics",
                        json={"name": name, "type": type_, "stream": stream, "toxicity": 1.0, "attributes": attributes})
        return r.status_code

    def clear(self, proxy: str) -> None:
        r = self.c.get(f"/proxies/{proxy}/toxics")
        if r.status_code == 200:
            for t in r.json():
                self.c.delete(f"/proxies/{proxy}/toxics/{t['name']}")

    def bounce(self, proxy: str) -> None:
        """Disable and re-enable: drops every open connection through the proxy."""
        self.c.post(f"/proxies/{proxy}", json={"enabled": False})
        self.c.post(f"/proxies/{proxy}", json={"enabled": True})


@dataclass
class NetFault:
    """One network fault on one service's fault proxy. Use `route` for the focal call and `hook` inside it."""
    service: str
    mode: str
    deadline_s: float = 4.0
    delay_s: float = 12.0          # for modes 'late' / 'late_tail'
    worker: int = 0
    toxi: Toxi = field(default_factory=Toxi)
    installed_at: float | None = None

    def __post_init__(self):
        if self.mode == "late_tail":
            self.mode = "late"
        if self.mode not in NETWORK_MODES + ["partial"]:
            raise ValueError(self.mode)
        self.proxy = f"{self.service}_fault{self.worker}"

    @property
    def route(self) -> str:
        return "clean" if self.mode in ("none", "redelivery") else f"fault{self.worker}"

    def hook(self) -> None:
        m = self.mode
        if m in ("none", "redelivery"):
            return
        if m == "lost_ack":
            self.toxi.add(self.proxy, "f", "limit_data", "downstream", {"bytes": 1})
        elif m in ("not_exec_err", "partial"):
            self.toxi.add(self.proxy, "f", "limit_data", "upstream", {"bytes": 1})
        elif m == "held":
            self.toxi.add(self.proxy, "f", "timeout", "downstream", {"timeout": 0})
        elif m == "not_exec_to":
            self.toxi.add(self.proxy, "f", "timeout", "upstream", {"timeout": int((self.deadline_s + 2) * 1000)})
        elif m == "late":
            self.toxi.add(self.proxy, "f", "latency", "upstream", {"latency": int(self.delay_s * 1000), "jitter": 0})
        self.installed_at = time.time()

    def settle_s(self) -> float:
        """How long after the call returns the outcome can still change."""
        return {"late": self.delay_s + 5.0, "held": 2.0, "not_exec_to": 4.0}.get(self.mode, 2.0)

    def cleanup(self) -> None:
        self.toxi.clear(self.proxy)
        self.toxi.bounce(self.proxy)
