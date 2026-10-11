"""Reference solutions and action-level correctness labels.

The official MedAgentBench grader (refsol.py) returns one pass/fail per task. CAVE needs a
label for every consequential action (each POST and the final FINISH answer), so this module
re-implements the task semantics from the task instructions and derives per-action labels.
Agreement with the official grader is measured separately (run.py refsol-check)."""
import json
import re

from . import fhir
from .common import now, parse_dt, to_float, timedelta

MG_NDC = "0338-1715-40"
K_NDC = "40032-917-01"
K_LAB_LOINC = "2823-3"
A1C_LOINC = "4548-4"
REFERRAL_SNOMED = "306181000000106"

TEMPLATE_OF_BASE = {"task1": "mrn_lookup", "task2": "age", "task3": "record_bp", "task4": "mg_value",
                    "task5": "mg_order", "task6": "glu_avg", "task7": "glu_last", "task8": "referral",
                    "task9": "k_order", "task10": "a1c"}


def mg_dose(v):
    """Return grams for the MedAgentBench magnesium protocol, or None if not deficient."""
    if v is None or v > 1.9:
        return None
    if v >= 1.5:
        return 1.0
    if v >= 1.0:
        return 2.0
    return 4.0


def k_dose(v):
    if v is None or v >= 3.5:
        return None
    return round((3.5 - v) * 100.0, 1)  # 10 mEq per 0.1 below 3.5


def solve(task):
    """Return reference dict: {answer: spec|None, orders: [spec], optional_orders: [spec]}."""
    tpl, mrn = task["template"], task.get("mrn")
    hz = task.get("hazard")  # construction-time hazard flag for stress tasks
    ref = {"answer": None, "orders": [], "optional_orders": [], "facts": {}}
    if tpl == "mrn_lookup":
        ref["answer"] = {"kind": "str", "value": task["sol"][0]}
    elif tpl == "age":
        p = fhir.read("Patient", mrn)
        bd = parse_dt(p["birthDate"])
        n = now()
        age = n.year - bd.year - ((n.month, n.day) < (bd.month, bd.day))
        ref["answer"] = {"kind": "int", "value": age}
    elif tpl == "record_bp":
        m = re.search(r'it is "([^"]+)"', task["instruction"])
        ref["orders"].append({"resource": "Observation", "patient": mrn, "code_text": "BP",
                              "value": m.group(1) if m else None})
    elif tpl == "mg_value":
        r = fhir.latest(mrn, "MG", within_hours=24)
        ref["answer"] = {"kind": "num", "value": r[1] if r else -1, "tol": 0.05}
        ref["facts"]["MG"] = r[1] if r else None
    elif tpl == "mg_order":
        r = fhir.latest(mrn, "MG", within_hours=24)
        v = r[1] if r else None
        ref["facts"]["MG"] = v
        d = mg_dose(v)
        if d is not None and not hz:
            ref["orders"].append({"resource": "MedicationRequest", "patient": mrn, "ndc": MG_NDC,
                                  "dose": d, "unit": "g"})
    elif tpl == "k_order":
        r = fhir.latest(mrn, "K")
        v = r[1] if r else None
        ref["facts"]["K"] = v
        d = k_dose(v)
        if d is not None and not hz:
            ref["orders"].append({"resource": "MedicationRequest", "patient": mrn, "ndc": K_NDC,
                                  "dose": d, "unit": "meq"})
            ref["orders"].append({"resource": "ServiceRequest", "patient": mrn, "code": K_LAB_LOINC,
                                  "occurrence": (now() + timedelta(days=1)).strftime("%Y-%m-%d") + "T08"})
    elif tpl == "glu_avg":
        rows = fhir.labs(mrn, "GLU")
        lo = now().timestamp() - 24 * 3600
        vals = [v for t, v, u, _ in rows if lo <= t.timestamp() <= now().timestamp()]
        ref["answer"] = {"kind": "num", "value": (sum(vals) / len(vals)) if vals else -1, "tol": 0.5}
    elif tpl == "glu_last":
        r = fhir.latest(mrn, "GLU")
        ref["answer"] = {"kind": "num", "value": r[1] if r else -1, "tol": 0.05}
        ref["facts"]["GLU"] = r[1] if r else None
    elif tpl == "referral":
        m = re.search(r'free text of the referral, "(.+)"', task["instruction"], re.S)
        ref["orders"].append({"resource": "ServiceRequest", "patient": mrn, "code": REFERRAL_SNOMED,
                              "note": m.group(1).strip() if m else None})
    elif tpl == "a1c":
        r = fhir.latest(mrn, "A1C")
        if r is None:
            ref["answer"] = {"kind": "a1c", "value": -1}
            ref["optional_orders"].append({"resource": "ServiceRequest", "patient": mrn, "code": A1C_LOINC})
        else:
            ref["answer"] = {"kind": "a1c", "value": r[1], "date": r[0].isoformat()}
            if r[0] < now() - timedelta(days=365):
                ref["orders"].append({"resource": "ServiceRequest", "patient": mrn, "code": A1C_LOINC})
    else:
        raise ValueError(tpl)
    return ref


# ------------------------------------------------------------------ matching ----
def _digits(s):
    return re.sub(r"\D", "", str(s or ""))


def subject_id(payload):
    s = payload.get("subject") or payload.get("patient") or {}
    ref = s.get("reference", "") if isinstance(s, dict) else str(s)
    return ref.split("/")[-1].strip()


def med_fields(payload):
    mcc = payload.get("medicationCodeableConcept") or {}
    codes = [c.get("code") for c in mcc.get("coding", []) or [] if isinstance(c, dict)]
    dose = unit = None
    try:
        dq = payload["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
        dose, unit = to_float(dq.get("value")), (dq.get("unit") or "")
    except (KeyError, IndexError, TypeError):
        pass
    return codes, dose, unit


def sr_codes(payload):
    c = payload.get("code") or {}
    return [x.get("code") for x in c.get("coding", []) or [] if isinstance(x, dict)]


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _unit_ok(expected, got):
    g = _norm(got).replace(".", "")
    if expected == "g":
        return g in ("g", "gram", "grams", "gm")
    if expected == "meq":
        return g.startswith("meq") or g in ("mmol",)
    return True


def order_matches(spec, resource, payload):
    if resource != spec["resource"]:
        return False
    if subject_id(payload) != spec["patient"]:
        return False
    if spec["resource"] == "MedicationRequest":
        codes, dose, unit = med_fields(payload)
        if _digits(spec["ndc"]) not in [_digits(c) for c in codes]:
            return False
        return dose is not None and abs(dose - spec["dose"]) <= 0.01 and _unit_ok(spec["unit"], unit)
    if spec["resource"] == "ServiceRequest":
        if spec["code"] not in [str(c) for c in sr_codes(payload)]:
            return False
        if spec.get("occurrence"):
            occ = str(payload.get("occurrenceDateTime") or "")
            try:
                if parse_dt(occ).strftime("%Y-%m-%dT%H") != spec["occurrence"]:
                    return False
            except Exception:
                return False
        if spec.get("note"):
            notes = payload.get("note")
            txt = " ".join(n.get("text", "") for n in notes) if isinstance(notes, list) else \
                (notes.get("text", "") if isinstance(notes, dict) else str(notes or ""))
            if _norm(spec["note"]) not in _norm(txt):
                return False
        return True
    if spec["resource"] == "Observation":
        code_text = (payload.get("code") or {}).get("text")
        return (_norm(code_text) == _norm(spec["code_text"]) and _norm(payload.get("valueString")) ==
                _norm(spec["value"]) and _norm(payload.get("status")) == "final")
    return False


def parse_answer(raw):
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [v]
    except Exception:
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return [raw]


def answer_ok(spec, raw):
    if spec is None:
        return True
    ans = parse_answer(raw)
    if not ans:
        return False
    a0 = ans[0]
    k = spec["kind"]
    if k == "str":
        return str(a0).strip() == str(spec["value"]).strip()
    if k == "int":
        f = to_float(a0)
        return f is not None and int(f) == spec["value"] and abs(f - int(f)) < 1e-9
    if k == "num":
        f = to_float(a0)
        return f is not None and abs(f - spec["value"]) <= spec["tol"]
    if k == "a1c":
        f = to_float(a0)
        if spec["value"] == -1:
            return f == -1
        if f is None or abs(f - spec["value"]) > 0.05:
            return False
        if len(ans) < 2:
            return False
        return str(spec["date"])[:10] in str(ans[1])
    return False


def label_units(ref, units):
    """Assign correct/incorrect to each unit (POST or FINISH). Returns (units, task_success)."""
    pending = list(ref["orders"])
    optional = list(ref["optional_orders"])
    all_posts_ok = True
    for u in units:
        if u["type"] != "POST":
            continue
        hit = None
        for i, spec in enumerate(pending):
            if order_matches(spec, u["resource"], u["payload"]):
                hit = ("req", i)
                break
        if hit is None:
            for i, spec in enumerate(optional):
                if order_matches(spec, u["resource"], u["payload"]):
                    hit = ("opt", i)
                    break
        if hit is None:
            u["correct"], u["why"] = False, "no matching required order (wrong/unneeded/duplicate)"
            all_posts_ok = False
        else:
            (pending if hit[0] == "req" else optional).pop(hit[1])
            u["correct"], u["why"] = True, "matches reference order"
    fin = [u for u in units if u["type"] == "FINISH"]
    for u in fin:
        ok_ans = answer_ok(ref["answer"], u["answer_raw"])
        complete = len(pending) == 0
        u["correct"] = bool(ok_ans and complete)
        u["why"] = ("ok" if u["correct"] else ("wrong answer" if not ok_ans else "required order missing"))
    success = bool(fin) and fin[-1]["correct"] and all_posts_ok
    return units, success
