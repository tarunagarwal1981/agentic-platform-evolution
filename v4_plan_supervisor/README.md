# v4 — Plan-based supervisor with intent contract

## What this version solves vs v3

**v3** routes dynamically via forced tool calls but the supervisor has no memory of *why* it's routing — it infers intent turn-by-turn from raw agent status. This creates three recurring problems:

1. **Routing gate bypass** — the gate blocks re-routing to a finished agent but cannot suggest where to go next (it has no declared plan).
2. **Specialists read raw query** — each agent re-parses the user's free-form text independently, leading to drift when entity extraction is ambiguous.
3. **Finalize fires before required fields exist** — nothing enforced that `compliance_clearance`, `missing_items`, etc. are populated before the answer is assembled.

**v4** fixes all three by generating an **intent contract** upfront (before any specialist fires) and wiring it into every node.

## Architecture

```
START → planner → supervisor ⇆ documents_agent
                             ⇆ role_requirements_agent
                             ⇆ compliance_agent
                  supervisor → display_contract_check → finalize → END
                                      ↓ (if blocked)
                               supervisor (correction injected)
```

- **Planner** (`claude-sonnet-4-20250514`): generates a typed intent contract once, before any specialist runs.
- **Supervisor** (`claude-haiku-4-5-20251001`): routes via forced tool calls, now with the declared `agent_sequence` in context.
- **Specialists** (`claude-sonnet-4-20250514`): read `extracted_params` from the contract, not the raw query.
- **Display contract check**: enforces required output fields before `finalize` runs.

## Intent contract schema

```json
{
  "intent": "hr_onboarding_compliance_check",
  "confidence": 0.95,
  "required_entities": ["candidate_id", "role_id", "office_location"],
  "agent_sequence": ["documents_agent", "role_requirements_agent", "compliance_agent"],
  "extracted_params": {
    "candidate_id": "CAND-10492",
    "role_id": "ROLE-ENG-2",
    "office_location": "NYC"
  },
  "display_contract": {
    "output_type": "compliance_report",
    "required_fields": ["ready_to_start", "missing_items", "risk_flags", "compliance_clearance"]
  }
}
```

## Three failure modes covered

| # | Failure | Fix |
|---|---------|-----|
| **1** | Routing gate bypass — gate blocks but has nowhere to redirect | **Plan-aware gate**: when blocking, reads `agent_sequence` from contract and returns `plan_suggestion` (next correct agent) in the tool result |
| **2** | Specialists drift on entity extraction | **Plan Honor Contract**: each specialist node reads `intent_contract["extracted_params"]` instead of parsing the raw query — see `# Reading from plan params, not raw query` comment in each node |
| **3** | Finalize fires before required output fields are populated | **Display contract enforcement**: `node_display_contract_check` runs before `finalize`, checks all `required_fields`, blocks and reinjects a correction message if any are missing |

## Two practical wins

**HITL resume optimization** — when a human-in-the-loop pause happens mid-workflow, the intent contract is already serialized in state. On resume, the workflow reuses it verbatim and skips the planner entirely. `routing_history` records `"hitl_resume: reusing prior plan"` to make this auditable.

**Capability manifest as source of truth** — `agent_sequence` in the contract is the single authoritative list of what must run and in what order. The routing gate, signal_complete enforcement, and display contract check all read from this one field rather than hardcoding sequences in three places.

## How to run

```bash
# from repo root, with .env containing ANTHROPIC_API_KEY
python v4_plan_supervisor/agent.py
```

Requires: `anthropic`, `langgraph`, `langchain-anthropic`, `python-dotenv` (see root `requirements.txt`).

## What to look for in the output

`main()` runs four demos:

1. **Demo 1 — Happy path**: prints the full `intent_contract` so you can see what the planner extracted. Watch `routing_history` show `planner:generated` before any specialist entry.

2. **Demo 2 — Routing gate bypass**: after all agents succeed, an attempted re-route to `role_requirements_agent` fires the gate. Look for `routing_gate:blocked role_requirements_agent` and `plan_suggestion: all agents complete — use signal_complete`.

3. **Demo 3 — Display contract enforcement**: starts with a partial state (documents + role done, compliance skipped). Watch `display_contract:blocked missing=['compliance_clearance', ...]`, then the supervisor routes `compliance_agent`, then `display_contract:passed` and finalize proceeds.

4. **Demo 4 — HITL resume**: planner runs once, `documents_agent` runs, then a simulated HITL pause. On resume, the prior contract is reused verbatim. Look for `hitl_resume: reusing prior plan` in `routing_history` and the `[hitl] CONFIRMED` message at the end.
