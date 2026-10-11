"""[model-migration port] Offline tests of the ops testbed: subsets, the chat driver, the per-episode record
written by code/run_ops.py, and the outcome categories. No Docker, no network, no model: the model is a
scripted stub of mm.llm.chat and the services are the in-memory fakes of test_layers_fake.py.
Run from the project root:  python -m pytest -q -p no:cacheprovider testbeds/ops/tests
(test_llm_driver_fake.py::test_behaviour_coding_of_a_lost_ack_episode needs the source study's analysis
folder and fails here by design; everything else should pass.)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "code"))

import pytest  # noqa: E402

import run_ops  # noqa: E402  (sets ACTREC_ROOT, puts testbeds/ops on sys.path; writes nothing on import)
import actrec.config as C  # noqa: E402
import actrec.contracts as CT  # noqa: E402
import actrec.episode as EP  # noqa: E402
import actrec.grader as GR  # noqa: E402
import actrec.ops_grade as OG  # noqa: E402
import actrec.services.ledger_tb as LT  # noqa: E402
from actrec.agent_tools import TOOL_FOR  # noqa: E402
from actrec.driver_chat import chat_tools  # noqa: E402
from actrec.tasks import make  # noqa: E402
from mm import llm as mmllm  # noqa: E402
from test_layers_fake import FakeDisp, FakeTB  # noqa: E402

REQUIRED = ["task_id", "label", "model", "provider", "temperature", "repeat", "success", "status", "error",
            "category", "actions", "step_map", "grade", "agent_cost", "n_turns", "n_tool_calls", "template", "world",
            "condition", "claimed", "messages", "extras"]
SPEC = {"label": "stub-model", "provider": "openrouter", "model": "stub/model"}


# ------------------------------------------------------------------ subsets
def test_subsets_disjoint_and_spread():
    d = run_ops._draw()
    ids = d["ids"]
    assert len(ids["smoke"]) == 4 and len(ids["pilot"]) == 40 and len(ids["main"]) == 100
    assert len({i.split("_")[0] for i in ids["smoke"]}) == 4
    assert not (set(ids["pilot"]) & set(ids["main"])) and not (set(ids["smoke"]) & (set(ids["pilot"]) | set(ids["main"])))
    for name in ("pilot", "main"):
        per = {t: sum(1 for i in ids[name] if i.startswith(t + "_")) for t in run_ops.TEMPLATES}
        assert min(per.values()) >= (5 if name == "pilot" else 12)
    assert all(200 <= int(i.split("_")[1]) < 400 for v in ids.values() for i in v)
    assert run_ops._draw() == d                       # deterministic


# ------------------------------------------------------------------ fakes for the episode path
class FakeToolbox:
    shared: dict = {}

    def __new__(cls, run, token, variant="exact", tb_accounts=None):
        tb = cls.shared.setdefault(run, FakeTB(variant))
        tb.run = run
        tb.ledger = None

        class _G:
            @staticmethod
            def ensure_repo():
                return 201
        tb.gitea = _G()
        return tb


class FakeDispatcher(FakeDisp):
    def drain(self, max_s=0):
        for th in self.threads:
            th.join()


class FakeLedger:
    def __init__(self, *a, **k):
        self.debit, self.credit = 1, 2

    def setup_accounts(self):
        pass


def _plan_args(st):
    return st.args


def fake_grade(state_dir, task, tb):
    steps = {}
    for st in task.steps:
        if st.tool == "batch_insert":
            rows = [len([e for e in tb.eff[st.tool] if e["alive"] and not e["actor"] and e["args"] == it])
                    for it in st.args["items"]]
            g = {"tool": st.tool, "rows": [(r, r) for r in rows], "standing": sum(rows), "expected": len(rows),
                 "rows_wrong": sum(1 for r in rows if r != 1)}
            g.update(ok=g["rows_wrong"] == 0, missing=any(r == 0 for r in rows), surplus=any(r > 1 for r in rows))
        else:
            n = tb.standing(st.tool, st.args)
            g = {"tool": st.tool, "standing": n, "expected": 1, "ok": n == 1, "missing": n < 1, "surplus": n > 1}
        steps[st.name] = g
    return {"steps": steps, "collateral": tb.collateral(),
            "residual_inconsistent": not all(v["ok"] for v in steps.values())}


def fake_scan(task, tb, actions):
    extras = []
    plan = {}
    for st in task.steps:
        plan.setdefault(st.tool, []).append(st.args)
    for tool, es in tb.eff.items():
        for e in es:
            if not e["alive"] or e["actor"] or tool == "batch_insert":
                continue
            ps = plan.get(tool, [])
            if tool == "issue_comment":
                hit = any(e["args"].get("body") == p.get("body") for p in ps)
            elif CT.SPECS[tool].cls == "U":
                hit = any(e["args"] == p for p in ps)
            else:
                hit = any(e["args"] == p for p in ps)
            if not hit:
                kind = {"issue_create": "issue", "issue_comment": "comment", "row_insert": "row",
                        "stock_adjust": "stock", "event_append": "event", "counter_incr": "counter",
                        "transfer_create": "transfer", "object_put": "object"}.get(tool, "mail")
                extras.append({"kind": kind, "desc": e["args"], "plan_tool": tool in plan})
    return {"extras": extras, "mismatches": [], "scan_errors": []}


@pytest.fixture
def fakes(monkeypatch):
    FakeToolbox.shared.clear()
    monkeypatch.setattr(CT, "Toolbox", FakeToolbox)
    monkeypatch.setattr(EP, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(GR, "grade", fake_grade)
    monkeypatch.setattr(OG, "scan_extras", fake_scan)
    monkeypatch.setattr(LT, "Ledger", FakeLedger)
    monkeypatch.setattr(C, "stack_secrets", lambda: {"gitea_token": "fake-token-123456"})
    monkeypatch.setattr(run_ops, "QUIESCE_S", 0.0)


def stub_chat_for(task, *, retry_once=True, extra_write=False, wrong_amount=False, malformed_first=False,
                  never_finish=False, claim="completed", calls_log=None):
    """A scripted model: calls the plan's writes in order with exact arguments, re-issues a failed write once
    (a naive agent), then finishes. Returns mm.llm.chat-shaped dicts."""
    st = {"k": 0, "n": 0, "issue": None, "last": None, "retried": False, "bad": malformed_first, "extra": extra_write}

    def tc(name, args):
        st["n"] += 1
        cid = f"call_{st['n']}"
        argstr = args if isinstance(args, str) else json.dumps(args)
        msg = {"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": argstr}}]}
        return {"text": "", "tool_calls": [{"id": cid, "name": name, "arguments": argstr}], "message": msg,
                "usage": {"prompt_tokens": 900, "completion_tokens": 50}, "cost": 0.0001, "cached": False,
                "finish_reason": "tool_calls", "latency_s": 0.01}

    def chat(spec, messages, temperature=0.0, max_tokens=0, salt="", stage="", tools=None, tool_choice=None):
        if calls_log is not None:
            calls_log.append({"salt": salt, "n_messages": len(messages), "tools": [t["function"]["name"] for t in tools]})
        # every assistant tool call must be answered before the next call
        open_ids = set()
        for m in messages:
            for c in m.get("tool_calls") or []:
                open_ids.add(c["id"])
            if m.get("role") == "tool":
                open_ids.discard(m["tool_call_id"])
        assert not open_ids, open_ids
        last = messages[-1]
        if last.get("role") == "tool":
            out = json.loads(last["content"])
            if isinstance(out.get("result"), dict) and "number" in out["result"]:
                st["issue"] = out["result"]["number"]
            if out.get("status") in ("error", "timeout") and retry_once and not st["retried"] and st["last"]:
                st["retried"] = True
                return tc(*st["last"])
        if st["bad"]:
            st["bad"] = False
            return tc("create_issue", "{not json")
        if never_finish:
            return tc("wait", {"seconds": 0})
        steps = task.steps
        if st["k"] < len(steps):
            s = steps[st["k"]]
            st["k"] += 1
            a = dict(s.args)
            name = TOOL_FOR[s.tool]
            if s.tool == "issue_comment":
                a = {"issue_number": st["issue"], "body": a["body"]}
            if wrong_amount and s.tool == "row_insert":
                a["amount"] = int(a["amount"]) + 1
            st["last"] = (name, a)
            st["retried"] = False
            return tc(name, a)
        if st["extra"]:
            st["extra"] = False
            return tc("insert_record", {"item": "unrequested:1", "amount": 1})
        return tc("finish", {"status": claim, "summary": "done", "uncertain_operations": []})
    return chat


def _run(task_id, world="none", **kw):
    task = make(*[(p if i == 0 else int(p)) for i, p in enumerate(task_id.split("_"))])
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x" / f"{task_id}.json"
        chat = stub_chat_for(task, **kw)
        row = run_ops.run_episode(SPEC, task_id, 0.0, 1, world, "test", out, [0], chat=chat)
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return row, rec


@pytest.mark.parametrize("tid", ["T1_201", "T2_202", "T3_203", "T4_204", "T5_205", "T6_206", "T7_207", "T8_208"])
def test_fault_free_episode_record(fakes, tid):
    row, rec = _run(tid)
    for k in REQUIRED:
        assert k in rec, k
    assert rec["status"] == "completed" and rec["success"] == 1 and rec["category"] == "ok", rec["category_detail"]
    assert rec["condition"] == "C0" and rec["world"] == "none"
    writes = [a for a in rec["actions"] if a["kind"] == "write"]
    assert writes and all(a["verdict"] == "ok" and not a["step"].startswith("extra:") for a in writes)
    assert rec["n_tool_calls"] == len(rec["actions"]) and rec["agent_cost"] > 0
    assert row["category"] == "ok" and row["r"] == 1


def test_lost_ack_naive_retry_duplicates(fakes):
    row, rec = _run("T1_201", world="lost_ack")
    assert rec["focal_step"] == "open_issue"
    assert rec["category"] == "wrong_write" and rec["success"] == 0
    assert rec["grade"]["steps"]["open_issue"]["standing"] == 2


def test_lost_ack_focal_skips_batch(fakes):
    assert run_ops.focal_step(make("T5", 205)) == "emit_done"


def test_unrequested_and_wrong_write(fakes):
    _, rec = _run("T1_201", extra_write=True)
    assert rec["category"] == "unrequested_write"
    _, rec = _run("T1_201", wrong_amount=True)
    assert rec["category"] == "wrong_write"


def test_wrong_answer_and_interface(fakes):
    _, rec = _run("T4_204", claim="partial")
    assert rec["category"] == "wrong_answer" and rec["success"] == 0
    _, rec = _run("T4_204", never_finish=True)
    assert rec["category"] == "interface" and rec["status"] == "turn_limit" and rec["n_turns"] == 30


def test_malformed_call_is_recorded(fakes):
    _, rec = _run("T7_207", malformed_first=True)
    assert rec["n_malformed"] == 1 and rec["actions"][0].get("malformed")
    assert rec["category"] == "ok"


def test_budget_stop_propagates_and_writes_nothing(fakes):
    def chat(*a, **k):
        raise mmllm.BudgetExceeded("cap")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "T1_201.json"
        with pytest.raises(mmllm.BudgetExceeded):
            run_ops.run_episode(SPEC, "T1_201", 0.0, 1, "none", "test", out, [0], chat=chat)
        assert not out.exists()


def test_model_api_error_is_not_written(fakes):
    def chat(*a, **k):
        raise RuntimeError("HTTP 400: bad tools")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "T1_201.json"
        with pytest.raises(run_ops.HarnessError):
            run_ops.run_episode(SPEC, "T1_201", 0.0, 1, "none", "test", out, [0], chat=chat)
        assert not out.exists()


def test_salt_has_temperature_repeat_and_task(fakes):
    log = []
    task = make("T8", 208)
    with tempfile.TemporaryDirectory() as d:
        run_ops.run_episode(SPEC, "T8_208", 0.7, 3, "none", "test", Path(d) / "a.json", [0],
                            chat=stub_chat_for(task, calls_log=log))
    assert log and all(c["salt"] == "T0.7|r3|T8_208|none" for c in log)
    assert "finish" in log[0]["tools"] and "create_transfer" in log[0]["tools"]


def test_chat_tools_format():
    ts = chat_tools(make("T7", 207))
    assert all(t["type"] == "function" and set(t["function"]) == {"name", "description", "parameters"} for t in ts)
    assert "additionalProperties" in json.dumps(ts)
    assert "additionalProperties" not in json.dumps(chat_tools(make("T7", 207), "gemini"))


# ------------------------------------------------------------------ the real namespace scan, on fake clients
class _FakeGitea:
    def __init__(self, issues, comments):
        self.issues, self.comments = issues, comments

    def list_issues(self, route="clean"):
        return self.issues

    def list_comments(self, n, route="clean"):
        return self.comments.get(n, [])


class _FakePg:
    def __init__(self, audit):
        self._a = audit

    def audit(self):
        return self._a


class _FakeMail:
    def __init__(self, box):
        self.box = box

    def all_for(self, to):
        return self.box.get(to, [])


class _FakeRedis:
    def __init__(self, streams, ctrs):
        self.streams, self.ctrs = streams, ctrs

    def scan_iter(self, match, count=0):
        pre = match[:-1]
        return [k for k in list(self.streams) + list(self.ctrs) if k.startswith(pre)]

    def xrange(self, k):
        return self.streams[k]

    def get(self, k):
        return self.ctrs.get(k)

    def close(self):
        pass


def test_scan_extras_on_fake_clients(monkeypatch):
    task = make("T1", 201)
    a = {s.name: s.args for s in task.steps}

    class TB:
        run = "r1"
        ledger = None
        gitea = _FakeGitea([{"number": 1, "title": a["open_issue"]["title"]}, {"number": 2, "title": "stray"}],
                           {1: [{"id": 9, "body": "a note\n\nop-id: x"}]})
        pg = _FakePg([{"tbl": "rows", "op": "I", "origin": "agent", "row_id": 5, "item": a["log_row"]["item"],
                       "amount": a["log_row"]["amount"]},
                      {"tbl": "rows", "op": "I", "origin": "agent", "row_id": 6, "item": "x", "amount": 1}])
        mail = _FakeMail({"oncall+r1@acme.invalid": [{"Subject": a["page_oncall"]["subject"]},
                                                     {"Subject": "again"}]})

        @staticmethod
        def addr(to):
            local, dom = to.split("@")
            return f"{local}+r1@{dom}"
    import actrec.services.redisx as RX
    fake_r = _FakeRedis({"ar:ev:r1:incident_opened": [("1-0", {"kind": "incident_opened",
                                                                "data": a["emit_event"]["data"]})]},
                        {"ar:ctr:r1:stray": "2"})
    monkeypatch.setattr(RX, "_client", lambda *x, **k: fake_r)
    x = OG.scan_extras(task, TB(), [{"contract_tool": "mail_send", "args": {"to": "oncall@acme.invalid"}}])
    kinds = sorted(e["kind"] for e in x["extras"])
    assert kinds == ["comment", "counter", "issue", "mail", "row"], x
    assert x["scan_errors"] == []
