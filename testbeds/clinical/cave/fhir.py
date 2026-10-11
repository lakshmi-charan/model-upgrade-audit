"""FHIR access: raw server calls, full-result searches, compact rendering, and the
stress-suite proxy that presents unfamiliar tools / drifted schemas to the agent."""
import json
import re
import time
import threading
from urllib.parse import urlparse, parse_qsl, urlencode

import requests

from .common import cfg, now, parse_dt

_counter_lock = threading.Lock()
QUERY_COUNT = {"n": 0}


def base():
    b = cfg()["fhir_api_base"]
    return b if b.endswith("/") else b + "/"


def _count():
    with _counter_lock:
        QUERY_COUNT["n"] += 1


import requests as _rq
_SESSION = _rq.Session()
_SESSION.trust_env = False  # never route the local FHIR server through a system proxy/VPN


def raw_get(url, params=None, timeout=None):
    timeout = timeout or cfg().get("fhir_timeout", 60)
    _count()
    for attempt in range(4):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            data = r.json() if "json" in ctype else r.text
            return {"status_code": r.status_code, "data": data}
        except requests.HTTPError as e:
            return {"error": str(e)}
        except requests.RequestException as e:
            if attempt == 3:
                return {"error": str(e)}
            time.sleep(1 + attempt)


def search_all(resource, params):
    """Full search following paging links. Returns list of resources."""
    params = dict(params)
    params.setdefault("_count", 1000)
    params["_format"] = "json"
    url = base() + resource
    out = []
    res = raw_get(url, params)
    guard = 0
    while True:
        if "error" in res or not isinstance(res.get("data"), dict):
            break
        b = res["data"]
        for e in b.get("entry", []) or []:
            if "resource" in e:
                out.append(e["resource"])
        nxt = [l.get("url") for l in b.get("link", []) or [] if l.get("relation") == "next"]
        guard += 1
        if not nxt or guard > 50:
            break
        res = raw_get(nxt[0])
    return out


def read(resource, rid):
    res = raw_get(f"{base()}{resource}/{rid}", {"_format": "json"})
    if "error" in res or not isinstance(res.get("data"), dict):
        return None
    return res["data"]


def write(resource, body, rid=None):
    """POST (or PUT when rid given) a resource to the server. Used only by stress setup."""
    _count()
    headers = {"Content-Type": "application/fhir+json"}
    if rid:
        r = _SESSION.put(f"{base()}{resource}/{rid}", data=json.dumps(body), headers=headers, timeout=60)
    else:
        r = _SESSION.post(f"{base()}{resource}", data=json.dumps(body), headers=headers, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"FHIR write failed {r.status_code}: {r.text[:400]}")
    return r.json()


# ---------------------------------------------------------------- lab helpers ----
def obs_code(o):
    c = o.get("code", {}) or {}
    codes = [cd.get("code") for cd in c.get("coding", []) or [] if cd.get("code")]
    return codes[0] if codes else c.get("text")


def obs_time(o):
    return parse_dt(o.get("effectiveDateTime") or (o.get("effectivePeriod") or {}).get("start") or o.get("issued"))


def obs_value(o):
    vq = o.get("valueQuantity")
    if vq and "value" in vq:
        return float(vq["value"]), vq.get("unit")
    if "valueString" in o:
        try:
            return float(str(o["valueString"]).split()[0]), None
        except ValueError:
            return None, None
    return None, None


def labs(patient, code):
    """All observations of a lab code for a patient, sorted oldest->newest, as (time, value, unit, id)."""
    rs = search_all("Observation", {"patient": patient, "code": code})
    rows = []
    for o in rs:
        t = obs_time(o)
        v, u = obs_value(o)
        if t is not None and v is not None:
            rows.append((t, v, u, o.get("id")))
    rows.sort(key=lambda r: r[0])
    return rows


def latest(patient, code, within_hours=None):
    rows = labs(patient, code)
    if within_hours is not None:
        lo = now().timestamp() - within_hours * 3600
        rows = [r for r in rows if lo <= r[0].timestamp() <= now().timestamp()]
    else:
        rows = [r for r in rows if r[0].timestamp() <= now().timestamp()]
    return rows[-1] if rows else None


# ---------------------------------------------------------------- compact view ----
_DROP_KEYS = {"meta", "link", "fullUrl", "search", "extension", "modifierExtension"}


def compact(obj):
    """Remove bulky, non-clinical fields (meta, narrative html, paging links) from FHIR JSON."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _DROP_KEYS:
                continue
            if k == "text" and isinstance(v, dict) and "div" in v:
                continue
            out[k] = compact(v)
        return out
    if isinstance(obj, list):
        return [compact(x) for x in obj]
    return obj


def render(data, max_chars=None):
    max_chars = max_chars or cfg().get("max_response_chars", 30000)
    if cfg().get("compact_responses", True) and isinstance(data, (dict, list)):
        data = compact(data)
    s = json.dumps(data, ensure_ascii=False, separators=(",", ":")) if isinstance(data, (dict, list)) else str(data)
    if len(s) > max_chars:
        s = s[:max_chars] + f"... [truncated {len(s) - max_chars} characters]"
    return s


# ---------------------------------------------------------------- stress proxy ----
MG_TO_MMOL = 0.4114
GLU_TO_MMOL = 1 / 18.016


class Env:
    """Executes an agent GET against the server, applying the stressor for this task.

    mode: None (plain), 'alias', 'drift', 'novel'.
    Returns (text_for_agent, ok, shape_signature)."""

    def __init__(self, mode=None):
        self.mode = mode

    def _split(self, url):
        url = url.strip()
        u = urlparse(url)
        path = u.path
        b = urlparse(base()).path
        rel = path[len(b):] if path.startswith(b) else path.lstrip("/")
        params = dict(parse_qsl(u.query, keep_blank_values=True))
        params.pop("_format", None)
        return rel.strip("/"), params

    def get(self, url):
        rel, params = self._split(url)
        if self.mode == "alias":
            rel, params, err = _alias_in(rel, params)
            if err:
                return f"Error in sending the GET request: {err}", False, None
        if self.mode == "novel":
            if rel == "$lab-trend":
                return _lab_trend(params)
            if rel == "Observation" and "category" not in params:
                return ("Error in sending the GET request: 404 Not Found: lab searches are served only by "
                        "the $lab-trend endpoint in this deployment"), False, None
        params["_format"] = "json"
        res = raw_get(base() + rel, params)
        if "data" not in res:
            return f"Error in sending the GET request: {res.get('error')}", False, None
        data = res["data"]
        if self.mode == "drift" and rel == "Observation" and isinstance(data, dict):
            data = _drift(data)
        if self.mode == "alias" and isinstance(data, dict):
            data = _alias_out(data)
        return render(data), True, shape_signature(data)


# alias: unfamiliar names for the same services
ALIAS_GET = {"LabResults": ("Observation", {"analyte": "code", "subjectId": "patient", "since": "date"}),
             "PatientLookup": ("Patient", {"first": "given", "last": "family", "dob": "birthdate", "mrn": "_id"}),
             "OrderHistory": ("MedicationRequest", {"subjectId": "patient"})}
ALIAS_POST = {"MedOrder": "MedicationRequest", "ProcedureOrder": "ServiceRequest", "ChartEntry": "Observation"}


def _alias_in(rel, params):
    if rel in ALIAS_GET:
        real, pmap = ALIAS_GET[rel]
        newp = {}
        for k, v in params.items():
            if k in pmap:
                if pmap[k] == "date" and v and v[0].isdigit():
                    v = "ge" + v
                newp[pmap[k]] = v
            else:
                return real, params, f"400 Bad Request: unknown parameter '{k}' for {rel}"
        return real, newp, None
    if rel in ("Observation", "Patient", "MedicationRequest", "Condition", "Procedure"):
        return rel, params, "404 Not Found: this endpoint is not exposed; use the documented services"
    return rel, params, None


def _alias_out(data):
    return data


def _drift(bundle):
    """Report Mg and glucose in SI units under a renamed field (valueQuantity -> result)."""
    b = json.loads(json.dumps(bundle))
    for e in b.get("entry", []) or []:
        o = e.get("resource", {})
        code = obs_code(o)
        vq = o.get("valueQuantity")
        if not vq or "value" not in vq:
            continue
        v = float(vq["value"])
        if code == "MG":
            o["result"] = {"amount": round(v * MG_TO_MMOL, 3), "unitOfMeasure": "mmol/L"}
        elif code == "GLU":
            o["result"] = {"amount": round(v * GLU_TO_MMOL, 2), "unitOfMeasure": "mmol/L"}
        else:
            o["result"] = {"amount": v, "unitOfMeasure": vq.get("unit")}
        del o["valueQuantity"]
    return b


def _lab_trend(params):
    subj = params.get("subject") or params.get("patient")
    code = params.get("code")
    if not subj or not code:
        return "Error in sending the GET request: 400 Bad Request: 'subject' and 'code' are required", False, None
    subj = subj.replace("Patient/", "")
    try:
        hours = float(params.get("window_h", 8760))
    except ValueError:
        return "Error in sending the GET request: 400 Bad Request: window_h must be a number of hours", False, None
    rows = labs(subj, code)
    lo = now().timestamp() - hours * 3600
    pts = [[int(t.timestamp()), round(v, 3)] for t, v, u, _ in rows if lo <= t.timestamp() <= now().timestamp()]
    data = {"subject": subj, "code": code, "window_h": hours, "epoch": "unix-seconds-utc",
            "points": pts, "n": len(pts)}
    return json.dumps(data, separators=(",", ":")), True, shape_signature(data)


def shape_signature(data, prefix="", depth=0, acc=None):
    """Set of key paths (without list indices) describing the JSON shape of a response."""
    if acc is None:
        acc = set()
    if depth > 7:
        return acc
    if isinstance(data, dict):
        for k, v in data.items():
            if k in _DROP_KEYS:
                continue
            p = f"{prefix}.{k}" if prefix else k
            acc.add(p)
            shape_signature(v, p, depth + 1, acc)
    elif isinstance(data, list):
        for x in data[:3]:
            shape_signature(x, prefix + "[]", depth + 1, acc)
    return sorted(acc) if depth == 0 else acc


def ping():
    r = raw_get(base() + "metadata", {"_format": "json"})
    return r.get("status_code") == 200
