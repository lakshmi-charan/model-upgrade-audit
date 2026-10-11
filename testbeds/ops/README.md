# Testbed `ops`: operations agent on real services

Ported from the action-recovery project (`code/actrec/`, `docker/compose.yml`). An LLM agent completes short
multi-step operations tasks by calling tools that write to real services running in Docker. Each write step
must leave **exactly one** effect. Grading reads only service-side records.

## Services (docker/compose.yml, compose project `mmops`)
PostgreSQL (records, stock, plus an audit trigger table the grader reads), Gitea (issues, comments), Mailpit
(SMTP capture), Redis (event streams, counters), SeaweedFS (S3 objects), TigerBeetle (ledger transfers),
Toxiproxy (the agent's path to every service, and the fault proxies). Temporal is not used.
`--check-env` starts only the services the subset's templates need, plus Toxiproxy. smoke needs everything
except SeaweedFS; pilot and main need all seven.

## Task templates (actrec/tasks.py, unchanged)
| | task | writes |
|---|---|---|
| T1 | open an incident | issue, event, e-mail, record |
| T2 | payout batch | 2 transfers, record, e-mail |
| T3 | release | S3 object, issue, comment, bulk e-mail |
| T4 | receive purchase order | 2 stock adjustments, counter +1, record |
| T5 | migration batch | 6-row batch insert, event |
| T6 | service notice | 3 e-mails, issue |
| T7 | ticket triage | issue, 2 comments in order, record |
| T8 | refund | transfer, record, e-mail |

An instance is `make(template, n)`, seeded by `"<template>:<n>"`. Task ids are `T<k>_<n>`.

## Subsets (subsets/*.json, frozen, seed 20261006, instances 200-399)
The source used instance 0 (dev), 100-102 (test) and 900 (harness check), so this project uses only new
instances. `smoke`: 4 tasks from 4 templates (T1, T2, T6, T7). `pilot`: 40 tasks (5 per template). `main`:
100 tasks (12 or 13 per template). The three subsets do not overlap. They are drawn together in
`run_ops._draw()`.

## Agent (actrec/driver_chat.py)
This is `driver_llm.run_llm` on chat completions through `mm.llm.chat`. It keeps the source's system prompt,
the task goal plus the cue "Each action must take effect exactly once.", and the tools from
`agent_tools.tools_for(task)` converted to chat function tools: the writes, the reads and the agent's own
delete/reverse tools for the task's services, plus `wait` and `finish(status, summary, uncertain_operations)`.
The limits are also unchanged: up to 3 nudges, MAX_TURNS 30 and MAX_TOOL_CALLS 40. `driver_llm.StepMapper`
maps each write to a plan step, and every write goes through the recovery layer.
- **Condition C0, the plain agent** (`layers.Layer`). A write goes once to the service. The model sees the
  status and the result or error, and it alone decides whether to check, retry or report. No recovery layer
  is used: not Settle, keys, guard, verify-before-retry or reflection.
- **World `none`** (main setting): no faults.
- **World `lost_ack`** (secondary setting, `--world lost_ack`): the first non-batch write commits, but its
  reply is cut by Toxiproxy. The agent sees `ToolTransportError` for a write that did take effect, so a blind
  retry duplicates it.
- The source's crash/resume and Temporal paths are not used.

Differences from the source:
- There are no Responses-API reasoning items.
- A write with unusable arguments returns an error to the model. In the source it killed the child process.
- Episodes run in threads of one process, so the job's spend cap covers all workers.

## State isolation (as actrec.runner.run_episode)
Every attempt gets a fresh namespace `run = <episode-hash8>-<random6>`:
- its own Gitea repo `ep-<run>`
- plus-addressed mail `local+<run>@domain`
- rows and stock tagged with `run`
- Redis keys `ar:*:<run>:*`
- the S3 prefix `<run>/`
- two new TigerBeetle accounts

The fault proxy is per worker slot (`<svc>_fault<w>`, w < 8). Parallel workers are therefore safe, up to 8
(enforced). The source itself ran 3. Nothing is reset between episodes because nothing is shared.

## Grading
`grader.grade()` is unchanged. For every plan step it counts the standing effects and the executions from
service records: Gitea lists, the PostgreSQL audit trigger, Redis `entries-added` and the counter audit,
TigerBeetle's immutable transfers, the Mailpit capture store and the compensation log.
`ops_grade.scan_extras()` adds two things, also from service records of the episode's own namespace:
- effects that match no plan step;
- exact-content checks of net stock change, transfer amount and object body.

`success = 1` when all of these hold:
- the agent called finish;
- every plan step has exactly the expected effect;
- there are no extra effects, mismatches or collateral;
- the finish status is `completed`.

`category` is one of:
- `ok`
- `interface`: no finish, nudges used up, or the turn or tool-call limit was reached
- `unrequested_write`
- `wrong_write`: wrong content, or a duplicate
- `missing_write`
- `wrong_answer`: the state is right but the status is partial or failed
- `other`

The rules are in `ops_grade.classify`. Each action record stores its tool, arguments, mapped step, observed
status, tool output and the grader verdict for its step.

## Commands (project root, project venv)
```
python code/install_ops_deps.py                       # once: httpx, psycopg, redis, boto3, tigerbeetle
python code/run_ops.py --check-env --subset main      # start/verify Docker services, bootstrap (free)
python code/run_ops.py --models gpt-4.1-nano,gpt-5.6-luna --subset smoke --repeats 1 --temperature 0 --workers 4
python code/run_ops.py --models gpt-4.1-nano,gpt-5.6-luna --subset pilot --repeats 3 --temperature 0 --workers 4
python code/run_ops.py --models gpt-4.1-nano --subset pilot --repeats 3 --world lost_ack
python -m pytest -q -p no:cacheprovider testbeds/ops/tests           # offline (no Docker, no network)
```
Output goes to `runs/ops/<subset>[-lost_ack]/<label>/T<T>/r<k>/<task_id>.json` and
`runs/ops/<subset>[-lost_ack]/episodes.jsonl`. Episodes that cannot be measured are not written. These are
service, grading or model-API errors. They are logged to `errors.jsonl` and re-run on the next call.

Exit codes: 0 done, 2 environment not ready, 3 budget or quota stop, 4 some episodes errored.

`--check-env` does the following:
- It generates random stack credentials once, in `runs/.secrets/stack_env.json` under this folder (never
  printed).
- If the Docker engine is down, it launches Docker Desktop and waits up to 15 minutes.
- It stops a running action-recovery stack (`docker stop`; that stack uses the same host ports), then runs
  `docker compose -p mmops -f testbeds/ops/docker/compose.yml --profile tb up -d <services>`.
- It bootstraps the stack: Toxiproxy proxies, the PostgreSQL schema, the S3 bucket and a Gitea admin token.

## Expected tokens
The source's cost ledger (`runs/cost_ledger.jsonl`, test grids) shows about 800-1,250 input tokens per call
and 40-65 output tokens per call, with 3.6-6 calls per episode. That is about 3-7.5k input and 0.2-0.4k
output tokens per episode, plus reasoning tokens for reasoning models. Wall time is mostly model latency, plus
3 s of quiescence before grading.

`tests/test_llm_driver_fake.py::test_behaviour_coding_of_a_lost_ack_episode` needs the source's `analysis/`
folder and fails here by design.
