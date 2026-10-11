"""Offline tests of the LLM driver loop, step mapping, crash-resume replay, cost cap and cache.
No network: the model is a scripted fake, services are the in-memory fakes from test_layers_fake."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

import actrec.driver_llm as D  # noqa: E402
import actrec.llm_client as L  # noqa: E402
from actrec.agent_tools import TOOL_FOR  # noqa: E402
from actrec.episode import EpisodeConfig  # noqa: E402
from actrec.tasks import make  # noqa: E402
from test_layers_fake import FakeDisp, FakeTB  # noqa: E402


class FakeLLM:
    """Calls the plan's writes in order with the exact arguments, retries a failed call once, then finishes.
    Returns Responses-API output items: an encrypted reasoning item followed by one function call."""

    def __init__(self, model, grid, reasoning_effort=None):
        self.n, self.k, self.just_retried, self.last = 0, 0, False, None
        self.seen_inputs = []

    def respond(self, instructions, items, tools, sample=0, episode=""):
        self.seen_inputs.append(list(items))
        writes = make("T1", 0).steps
        last_out = next((i for i in reversed(items) if i.get("type") == "function_call_output"), None)
        failed = last_out is not None and ('"status": "error"' in last_out["output"]
                                           or '"status": "timeout"' in last_out["output"])
        if failed and not self.just_retried and self.last:
            self.just_retried = True
            return self._tc(*self.last), {"usage": {}}
        self.just_retried = False
        if self.k >= len(writes):
            return self._tc("finish", {"status": "completed", "summary": "done", "uncertain_operations": []}), {"usage": {}}
        st = writes[self.k]
        self.k += 1
        self.last = (TOOL_FOR[st.tool], dict(st.args))
        return self._tc(*self.last), {"usage": {}}

    def _tc(self, name, args):
        self.n += 1
        return [{"type": "reasoning", "id": f"rs_{self.n}", "summary": [], "encrypted_content": "gAAAA-fake"},
                {"type": "function_call", "id": f"fc_{self.n}", "call_id": f"call_{self.n}", "name": name,
                 "arguments": json.dumps(args), "status": "completed"}]


def _run(condition, mode, focal="open_issue"):
    task = make("T1", 0)
    with tempfile.TemporaryDirectory() as d:
        cfg = EpisodeConfig(episode_id="t", run="t", task=task.to_dict(), focal_step=focal, mode=mode,
                            condition=condition, driver="llm:fake", delta_assumed_s=0.7, state_dir=d)
        tb = FakeTB()
        disp = FakeDisp(cfg, tb)
        made = []

        class Rec(FakeLLM):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                made.append(self)
        orig = D.LLM
        D.LLM = Rec
        try:
            rep = D.run_llm(cfg, tb, disp, False)
        finally:
            D.LLM = orig
        rep["_inputs"] = made[0].seen_inputs
        for th in disp.threads:
            th.join()
        return rep, tb, task


def test_naive_llm_duplicates_on_lost_ack():
    rep, tb, task = _run("C0", "lost_ack")
    assert tb.standing("issue_create", task.steps[0].args) == 2
    assert rep["claimed"] == "completed"


def test_settle_under_llm_converges():
    rep, tb, task = _run("S", "lost_ack")
    assert tb.standing("issue_create", task.steps[0].args) == 1
    assert rep["step_map"]


def test_step_mapper_normalizes_and_remembers():
    m = D.StepMapper(make("T7", 0))
    a = make("T7", 0).steps[0].args
    s1 = m.step_for("issue_create", {"title": a["title"].upper() + ".", "body": "x"}, {})
    s2 = m.step_for("issue_create", {"title": a["title"], "body": "y"}, {})
    assert s1 == s2 == "open_ticket"
    c1 = m.step_for("issue_comment", {"issue": 7, "body": "something paraphrased"}, {})
    c2 = m.step_for("issue_comment", {"issue": 7, "body": "something paraphrased"}, {})
    c3 = m.step_for("issue_comment", {"issue": 7, "body": "another"}, {})
    assert c1 == c2 == "triage_note" and c3 == "eta_note"


def test_cost_cap_refuses_before_spending(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(L, "CACHE", Path(d) / "cache")
        monkeypatch.setattr(L, "LEDGER", Path(d) / "ledger.jsonl")
        monkeypatch.setattr(L, "LOCK", Path(d) / ".lock")
        monkeypatch.setitem(L.ENV, "OPENAI_API_KEY", "sk-test-not-a-real-key-000000")
        monkeypatch.setitem(L.ENV, "MAX_SPEND_USD", "0")
        monkeypatch.setitem(L.ENV, "MAX_SPEND_PER_RUN_USD", "0")
        llm = L.LLM("gpt-5.6-luna", grid="g")
        posted = []
        monkeypatch.setattr(llm, "_post", lambda body: posted.append(body))
        with pytest.raises(L.BudgetExceeded):
            llm.respond("sys", [{"role": "user", "content": "hi"}], [])
        assert posted == []


def _iso(monkeypatch, d, cap="1"):
    monkeypatch.setattr(L, "CACHE", Path(d) / "cache")
    monkeypatch.setattr(L, "LEDGER", Path(d) / "ledger.jsonl")
    monkeypatch.setattr(L, "LOCK", Path(d) / ".lock")
    monkeypatch.setitem(L.ENV, "OPENAI_API_KEY", "sk-test-not-a-real-key-000000")
    monkeypatch.setitem(L.ENV, "MAX_SPEND_USD", cap)
    monkeypatch.setitem(L.ENV, "MAX_SPEND_PER_RUN_USD", cap)


FAKE_RESP = {"model": "gpt-5.6-luna-x", "status": "completed",
             "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400}, "output_tokens": 100,
                       "output_tokens_details": {"reasoning_tokens": 60}},
             "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                         "content": [{"type": "output_text", "text": "ok"}]}]}


def test_cache_hit_is_free_and_cost_uses_cached_rate(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _iso(monkeypatch, d)
        llm = L.LLM("gpt-5.6-luna", grid="g")
        monkeypatch.setattr(llm, "_post", lambda body: FAKE_RESP)
        o1, meta1 = llm.respond("sys", [{"role": "user", "content": "hi"}], [])
        o2, meta2 = llm.respond("sys", [{"role": "user", "content": "hi"}], [])
        assert not meta1["cached"] and meta2["cached"] and meta2["usd"] == 0 and o1 == o2
        assert abs(L.spent() - (600 * 0.20 + 400 * 0.02 + 100 * 1.20) / 1e6) < 1e-12
        assert L.tokens(FAKE_RESP["usage"]) == (1000, 400, 100, 60)


def test_request_body_is_responses_format(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _iso(monkeypatch, d)
        bodies = []
        for model, reasoning in (("gpt-5.6-luna", True), ("gpt-4.1-mini", False)):
            llm = L.LLM(model, grid="g")
            monkeypatch.setattr(llm, "_post", lambda body: bodies.append(body) or FAKE_RESP)
            llm.respond("sys", [{"role": "user", "content": model}], D.tools_for(make("T1", 0)))
            b = bodies[-1]
            assert b["instructions"] == "sys" and b["store"] is False and b["parallel_tool_calls"] is False
            assert "messages" not in b and "reasoning" not in b          # default reasoning: nothing sent
            assert ("include" in b) == reasoning
            assert all(t["type"] == "function" and "name" in t and "function" not in t for t in b["tools"])


def test_refused_optional_field_is_dropped_and_replay_still_hits(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        _iso(monkeypatch, d)

        class R:
            def __init__(self, code, text, js=None):
                self.status_code, self.text, self._js = code, text, js

            def json(self):
                return self._js
        calls = []

        def post(url, headers, json, timeout):  # noqa: A002
            calls.append(json)
            if "parallel_tool_calls" in json:
                return R(400, "Unsupported parameter: 'parallel_tool_calls'")
            return R(200, "", FAKE_RESP)
        monkeypatch.setattr(L.httpx, "post", post)
        llm = L.LLM("gpt-5.6-luna", grid="g")
        llm.respond("sys", [{"role": "user", "content": "a"}], [])
        assert len(calls) == 2 and "parallel_tool_calls" in llm.dropped
        llm.respond("sys", [{"role": "user", "content": "b"}], [])
        assert len(calls) == 3                                    # remembered: no second refusal
        fresh = L.LLM("gpt-5.6-luna", grid="g")                   # replay without the refusal: same cache entry
        _, meta = fresh.respond("sys", [{"role": "user", "content": "a"}], [])
        assert meta["cached"]


def test_reasoning_items_passed_back_and_unencrypted_dropped():
    assert D._keep({"type": "reasoning", "id": "rs_1", "summary": []}) is None
    k = D._keep({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x", "status": None})
    assert k == {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x"}
    rep, tb, task = _run("C0", "none")
    # every turn after the first carried the previous reasoning item and the function call output back
    fake_inputs = rep["_inputs"]
    second = fake_inputs[1]
    assert [i.get("type") for i in second[1:]] == ["reasoning", "function_call", "function_call_output"]


def test_resume_reexecutes_unanswered_call(monkeypatch):
    task = make("T1", 0)
    with tempfile.TemporaryDirectory() as d:
        cfg = EpisodeConfig(episode_id="t", run="t", task=task.to_dict(), focal_step="open_issue", mode="none",
                            condition="C0", driver="llm:fake", delta_assumed_s=0.7, state_dir=d)
        st = task.steps[0]
        conv = {"instructions": D.SYSTEM, "input": [
            {"role": "user", "content": task.goal},
            {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": TOOL_FOR[st.tool],
             "arguments": json.dumps(st.args)}],
            "turns": 1, "tool_calls": 0, "usd": 0.0, "cached_calls": 0, "paid_calls": 1, "tokens_in": 0,
            "tokens_out": 0, "finish": None, "nudges": 0, "mapper": {}, "stop": None}
        (Path(d) / "conversation.json").write_text(json.dumps(conv))
        tb = FakeTB()
        disp = FakeDisp(cfg, tb)

        class Done(FakeLLM):
            def respond(self, instructions, items, tools, sample=0, episode=""):
                self.seen_inputs.append(list(items))
                return self._tc("finish", {"status": "completed", "summary": "", "uncertain_operations": []}), {}
        monkeypatch.setattr(D, "LLM", Done)
        rep = D.run_llm(cfg, tb, disp, True)
        for th in disp.threads:
            th.join()
        saved = json.loads((Path(d) / "conversation.json").read_text())
        outs = [i for i in saved["input"] if i.get("type") == "function_call_output"]
        assert outs[0]["call_id"] == "call_0" and rep["claimed"] == "completed"
        assert tb.standing("issue_create", st.args) == 1


def test_model_error_record_is_unfinished():
    from actrec.runner import unfinished
    assert unfinished({"stop": "model error: RuntimeError: OpenAI HTTP 400"})
    assert unfinished({"stop": "budget: cap reached"})
    assert unfinished({"grade": {"grade_error": "x"}})
    assert not unfinished({"stop": None, "grade": {"residual_inconsistent": False}})


def test_behaviour_coding_of_a_lost_ack_episode():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
    import llm_behaviour as LB
    task = make("T1", 0)
    with tempfile.TemporaryDirectory() as d:
        cfg = EpisodeConfig(episode_id="t", run="t", task=task.to_dict(), focal_step="open_issue", mode="lost_ack",
                            condition="C0", driver="llm:fake", delta_assumed_s=0.7, state_dir=d)
        tb = FakeTB()
        disp = FakeDisp(cfg, tb)
        orig = D.LLM
        D.LLM = FakeLLM
        try:
            D.run_llm(cfg, tb, disp, False)
        finally:
            D.LLM = orig
        for th in disp.threads:
            th.join()
        c = LB.code_episode(json.loads((Path(d) / "conversation.json").read_text()))
    assert c["first_action"] == "reissue" and c["checked_before_reissue"] is False and c["reissues"] == 1
    assert c["claimed"] == "completed" and c["fault_tool"] == "create_issue"


def test_settle_guards_agent_deletions_by_attribution():
    """Look-alike world: after Settle confirmed the issue by its tag, deleting the other user's identical issue or
    the task's only issue is refused; deleting a proven surplus copy is allowed."""
    from actrec.layers import make_layer
    task = make("T1", 0)
    st = task.steps[0]
    with tempfile.TemporaryDirectory() as d:
        cfg = EpisodeConfig(episode_id="t", run="t", task=task.to_dict(), focal_step=st.name, mode="none",
                            condition="S", driver="llm:fake", delta_assumed_s=0.7, state_dir=d)
        tb = FakeTB()
        disp = FakeDisp(cfg, tb)
        lay = make_layer(cfg, tb, disp)
        other = tb.apply(st.tool, dict(st.args), actor=True)["number"]          # another user's look-alike
        mine = lay.write(st.name, st.tool, dict(st.args))["value"]["number"]
        assert "not created by this task" in lay.guard_delete("delete_issue", {"issue_number": other})
        assert "undo a completed step" in lay.guard_delete("delete_issue", {"issue_number": mine})
        rec = lay.j.get(st.name)
        extra = tb.apply(st.tool, dict(st.args), tag=rec["op"])["number"]      # a redelivered duplicate of ours
        assert lay.guard_delete("delete_issue", {"issue_number": extra}) is None
        assert "undo a completed step" in lay.guard_delete("delete_issue", {"issue_number": mine})
        assert make_layer(EpisodeConfig(**{**cfg.__dict__, "condition": "C0"}), tb, disp).guard_delete(
            "delete_issue", {"issue_number": other}) is None


def test_ledger_lock_retries_on_windows_permission_error(monkeypatch):
    """Amendment 1: on Windows, opening the lock file while another process deletes it raises PermissionError."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(L, "LOCK", Path(d) / ".lock")
        real, calls = L.os.open, []

        def flaky(path, flags, *a, **k):
            if str(path) == str(L.LOCK):
                calls.append(path)
                if len(calls) == 1:
                    raise PermissionError(13, "Permission denied")
            return real(path, flags, *a, **k)
        monkeypatch.setattr(L.os, "open", flaky)
        with L._Lock():
            pass
        monkeypatch.setattr(L.os, "open", real)
        assert len(calls) == 2


def test_ledger_ids_are_scoped_to_the_run():
    """Amendment 2: an agent-chosen transfer key must not collide with the same key in another episode."""
    from actrec.services.ledger_tb import Ledger
    a, b = Ledger(1, 2, namespace="run-a"), Ledger(1, 2, namespace="run-b")
    assert a.id_for("13050000000000000000000000000000") == a.id_for("13050000000000000000000000000000")
    assert a.id_for("13050000000000000000000000000000") != b.id_for("13050000000000000000000000000000")
    assert 0 < a.id_for("x") < (1 << 128) - 1
