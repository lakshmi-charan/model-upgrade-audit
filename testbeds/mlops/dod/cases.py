"""Case generation. Each case starts from the correct (spec) pipeline of its period and applies a change set drawn
from the domain catalog: nothing, benign changes only, one defect, or two defects, mixed with benign decoys.
Defect types are split into dev and test so that the method is never tuned on test defect types."""
from __future__ import annotations

import random

from dod import bench
from dod.core import stable_hash
from dod.domains import get_domain

# Reference periods and the current periods paired with them.
PAIRS = {
    "acs": {"2014": [str(y) for y in range(2014, 2025)], "2016": [str(y) for y in range(2016, 2025)]},
    "tlc": {"2015-01": ["2015-01", "2015-07", "2016-07", "2017-07", "2018-07", "2019-07", "2020-07", "2021-07",
                        "2022-07", "2023-04", "2024-07", "2025-01", "2025-07", "2026-07"],
            "2022-01": ["2022-01", "2022-07", "2023-01", "2023-04", "2023-10", "2024-07", "2024-12", "2025-01",
                        "2025-04", "2025-10", "2026-04", "2026-07"]},
    "news": {"2012": [str(y) for y in range(2012, 2019)], "2013": [str(y) for y in range(2013, 2019)]},
    # v2 held-out domains (current periods: same season as the reference plus every dated source change)
    "divvy": {"2016-Q3": ["2016-Q3", "2016-Q4", "2017-Q1", "2017-Q3", "2017-Q4", "2018-Q1", "2018-Q2", "2018-Q3",
                          "2019-Q2", "2019-Q3", "2020-Q1", "2020-Q3", "2020-Q4", "2021-Q3", "2022-Q3", "2023-Q3",
                          "2024-Q2", "2024-Q3", "2025-Q3", "2026-Q3"],
              "2019-Q1": ["2019-Q1", "2019-Q2", "2019-Q3", "2019-Q4", "2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4",
                          "2021-Q1", "2022-Q1", "2023-Q1", "2024-Q1", "2024-Q2", "2025-Q1", "2026-Q1"]},
    "hmda": {"2012": [str(y) for y in range(2012, 2026)], "2015": [str(y) for y in range(2015, 2026)]},
    "bts": {"2015-01": ["2015-01", "2015-07", "2016-07", "2017-07", "2018-07", "2019-07", "2020-04", "2020-07",
                        "2021-07", "2022-07", "2023-07", "2024-07", "2025-07", "2026-07"],
            "2019-01": ["2019-01", "2019-04", "2019-07", "2020-01", "2020-04", "2020-07", "2021-01", "2021-07",
                        "2022-01", "2023-01", "2024-01", "2025-01", "2026-01"]},
}

# Development domain: every ACS case was available while the method was designed (its defect types were inspected
# during development), so ACS is reported as in-sample. Held-out test = TLC and news test defect types and periods.
DEV_DOMAINS = {"acs"}
# v2 held-out domains: no method was designed or tuned on them; every case is a test case.
HELDOUT_V2 = {"divvy", "hmda", "bts"}

DEV_DEFECTS = {
    "acs": {"acs.relp_minus20", "acs.key_digits", "acs.adjinc_ignored", "acs.clean_env_p30"},
    "tlc": {"tlc.ratecode_stale", "tlc.time_env_main", "tlc.tz_utc"},
    "news": {"news.no_lower", "news.mojibake"},
}


def _mk(domain, p0, p1, enc_id, items, tag, rng_seed) -> bench.Case:
    dom = get_domain(domain)
    spec = dom.spec_state(p1, enc_id)
    state = dict(spec)
    for it in items:
        state[it["component"]] = it["version"]
    tiers = sorted({it["tier"] for it in items if it["kind"] == "defect"})
    cid = f"{domain}-{p0}-{p1}-{tag}-" + stable_hash([it["id"] for it in items] + [rng_seed], 4)
    return bench.Case(case_id=cid, domain=domain, p0=p0, p1=p1, state=state, spec=spec,
                      injected=[{k: (v.to_json() if k == "version" else v) for k, v in it.items()} for it in items],
                      tier="+".join(tiers) if tiers else "none",
                      meta={"scenario": tag})


def generate(domain: str, seed: int = 7) -> list[tuple[str, bench.Case]]:
    """Returns (split, case) pairs. split is 'dev' or 'test'."""
    rng = random.Random(seed)
    out = []
    if domain in DEV_DOMAINS:
        return [("dev", c) for _, c in _generate(domain, seed)]
    if domain in HELDOUT_V2:
        return [("test", c) for _, c in _generate(domain, seed)]
    return _generate(domain, seed)


def _generate(domain: str, seed: int = 7) -> list[tuple[str, bench.Case]]:
    rng = random.Random(seed)
    out = []
    dev_def = DEV_DEFECTS.get(domain, set())
    for p0, p1s in PAIRS[domain].items():
        enc_id = f"p0_{p0}"
        dom = get_domain(domain)
        for p1 in p1s:
            cat = dom.catalog(p1, enc_id)
            defects = [c for c in cat if c["kind"] == "defect"]
            benign = [c for c in cat if c["kind"] != "defect"]

            def decoys(exclude_components, k):
                pool = [b for b in benign if b["component"] not in exclude_components]
                rng.shuffle(pool)
                chosen, comps = [], set(exclude_components)
                for b in pool:
                    if b["component"] not in comps and len(chosen) < k:
                        chosen.append(b)
                        comps.add(b["component"])
                return chosen
            # clean and benign-only cases (period split: alternate periods to dev/test)
            per_split = "dev" if (p1s.index(p1) % 3 == 0) else "test"
            out.append((per_split, _mk(domain, p0, p1, enc_id, [], "clean", seed)))
            out.append((per_split, _mk(domain, p0, p1, enc_id, decoys(set(), 2), "benign", seed)))
            # single defects with 0-2 decoys
            for d in defects:
                split = "dev" if d["id"] in dev_def else "test"
                items = [d] + decoys({d["component"]}, rng.randint(0, 2))
                out.append((split, _mk(domain, p0, p1, enc_id, items, "defect", seed)))
            # one two-defect case per pair (both defects from the same split)
            for split in ("dev", "test"):
                pool = [d for d in defects if (d["id"] in dev_def) == (split == "dev")]
                pairs = [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:] if a["component"] != b["component"]]
                if pairs:
                    a, b = rng.choice(pairs)
                    out.append((split, _mk(domain, p0, p1, enc_id, [a, b] + decoys({a["component"], b["component"]}, 1),
                                           "two", seed)))
    return out
