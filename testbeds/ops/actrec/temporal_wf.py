"""C4: an ordinary workflow engine with retries (Temporal). The task plan is a workflow; every write is an
activity with a retry policy (3 attempts, 1 s initial backoff, 10 s start-to-close). Native-key tools get the
standard durable key derived from workflow id + activity id, as Temporal's documentation recommends; key-less
tools are simply retried (at-least-once). Durable progress lives in Temporal's event history, so after a
driver crash the workflow resumes on a new worker and the interrupted activity is retried.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import uuid
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, WorkflowAlreadyStartedError
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from .contracts import SPECS
from .layers import effect_value, exists_refusal, value_from_found

_CTX: dict = {}


@activity.defn(name="write_step")
def write_step(step: str, tool: str, args: dict) -> dict:
    info = activity.info()
    key = None
    if SPECS[tool].cls == "K":
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{info.workflow_id}/{info.activity_id}"))
    res = _CTX["disp"].dispatch(step, tool, args, key=key)
    if res.kind == "ok" or exists_refusal(res):
        return effect_value(tool, res.value)
    raise ApplicationError(res.signature())


@activity.defn(name="lookup_ref")
def lookup_ref(tool: str, args: dict) -> dict:
    found = _CTX["tb"].by_content(tool, args)
    return value_from_found(tool, found, args)


@workflow.defn(name="TaskWorkflow")
class TaskWorkflow:
    @workflow.run
    async def run(self, steps: list) -> dict:
        results: dict = {}
        status: dict = {}
        policy = RetryPolicy(initial_interval=timedelta(seconds=1), backoff_coefficient=2.0, maximum_attempts=3)
        for st in steps:
            args = {}
            missing = False
            for k, v in st["args"].items():
                if isinstance(v, dict) and "$ref" in v:
                    ref_step, fld = v["$ref"].split(".")
                    val = (results.get(ref_step) or {}).get(fld)
                    if val is None:
                        src = next(s for s in steps if s["name"] == ref_step)
                        got = await workflow.execute_activity("lookup_ref", args=[src["tool"], src["args"]],
                                                              start_to_close_timeout=timedelta(seconds=30))
                        val = (got or {}).get(fld)
                    if val is None:
                        missing = True
                    args[k] = val
                else:
                    args[k] = v
            if missing:
                status[st["name"]] = "missing_ref"
                continue
            try:
                results[st["name"]] = await workflow.execute_activity(
                    "write_step", args=[st["name"], st["tool"], args],
                    start_to_close_timeout=timedelta(seconds=10), retry_policy=policy)
                status[st["name"]] = "ok"
            except ActivityError:
                status[st["name"]] = "error"
        return {"status": status, "results": results}


def run_temporal(cfg, tb, disp, resumed: bool) -> dict:
    _CTX.update(tb=tb, disp=disp)

    async def main():
        client = await Client.connect("localhost:7233")
        tq = f"tq-{cfg.run}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            async with Worker(client, task_queue=tq, workflows=[TaskWorkflow], activities=[write_step, lookup_ref],
                              activity_executor=ex, workflow_runner=UnsandboxedWorkflowRunner()):
                wid = f"wf-{cfg.run}"
                try:
                    h = await client.start_workflow(TaskWorkflow.run, cfg.task["steps"], id=wid, task_queue=tq)
                except WorkflowAlreadyStartedError:
                    h = client.get_workflow_handle(wid)
                return await h.result()

    out = asyncio.run(main())
    status = out["status"]
    all_ok = all(v == "ok" for v in status.values()) and len(status) == len(cfg.task["steps"])
    return {"driver": "scripted", "condition": "C4", "engine": "temporal", "step_status": status,
            "claimed": "completed" if all_ok else "partial", "uncertain": [], "compensations": 0,
            "duplicates_reported": [], "restarts": 0}
