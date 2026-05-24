# v3 — Tool supervisor (dynamic routing + guardrails)

## What this version solves vs v2

**v2** fixed *state shape* (typed summaries) but still ran a **fixed pipeline**:
`documents → role_requirements → compliance` on every query.

**v3** adds a **supervisor** that chooses the next specialist via **forced tool calls**
(`route_to_agent`, `signal_complete`, `request_clarification`) using Anthropic's
`tool_use` API directly (Haiku for routing, Sonnet for specialist summaries).

Specialists still write **structured summaries** into typed state—not raw blobs.

## Architecture

- **Supervisor** (`claude-haiku-4-5-20251001`): routing only, via tools.
- **Specialists** (`claude-sonnet-4-20250514`): `documents_agent`, `role_requirements_agent`, `compliance_agent`.
- **LangGraph** `GraphState`: `agent_status`, `routing_history`, `call_counts`, `gate_blocks`, summaries, etc.

Demo question (same as v1/v2):

> Is this new hire fully compliant and ready to start given their documents and role requirements?

## Five failure modes and fixes

| # | Failure | Fix in code |
|---|---------|-------------|
| **1** | Runaway retries / cost blow-up | **Circuit breaker**: `MAX_CALLS_PER_AGENT = 3`, `MAX_TURNS_PER_QUERY = 20` → force partial `finalize` |
| **2** | Re-running a finished specialist | **Routing gate**: block `route_to_agent` when `agent_status == "success"`; log to `routing_history` |
| **3** | Supervisor stuck re-targeting a gated agent | **Gate escalation**: after `GATE_ESCALATION_THRESHOLD` blocks for an agent, bypass to next unvisited agent in sequence |
| **4** | Premature `signal_complete` | **Enforcement**: all required agents must be `success`; order `documents → role_requirements → compliance`; reject and reinject correction |
| **5** | Compound queries ("documents AND compliance") | **Intent detection**: `detect_compound_intents()` requires every detected domain to complete before `signal_complete` |

## How to run

```bash
# from repo root, with .env containing ANTHROPIC_API_KEY
python v3_tool_supervisor/agent.py
```

Requires: `anthropic`, `langgraph`, `langchain-anthropic`, `python-dotenv` (see root `requirements.txt`).

## What to look for in the output

`main()` runs four terminal demos and prints **`routing_history`**, **`call_counts`**, **`agent_status`**, and **`gate_blocks`** after each:

1. **Happy path** — supervisor routes three specialists in order, then `signal_complete:accepted`.
2. **Failure 1 (circuit breaker)** — `documents_agent` fails twice (simulated); on the next turn you should see `circuit_breaker:tripped` and partial finalize.
3. **Failure 2 (routing gate)** — `documents_agent` pre-marked `success`; look for `routing_gate:blocked documents_agent` (and eventually `gate_escalation:force_route` if the supervisor retries).
4. **Failure 5 (compound query)** — query with "documents AND compliance"; all required intents must reach `success` before completion.

## What the next version fixes

**v4** (planned) adds a **plan-first** supervisor: explicit decomposition and reconciliation before/alongside tool routing—useful when the task needs multi-step hypotheses, not just "which specialist next."
