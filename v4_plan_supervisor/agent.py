#!/usr/bin/env python3
"""
v4 — Plan-based supervisor with intent contract, display contract enforcement,
routing gate bypass fix, and HITL resume optimization.

Run: python agent.py
"""

from __future__ import annotations

import json
import operator
import os
import textwrap
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_SCENARIO = (
    "Is this new hire (candidate_id=CAND-10492, role_id=ROLE-ENG-2, "
    "office_location=NYC) fully compliant and ready to start given their "
    "documents and role requirements?"
)

SPECIALIST_MODEL = "claude-sonnet-4-20250514"
SUPERVISOR_MODEL = "claude-haiku-4-5-20251001"
PLANNER_MODEL = "claude-sonnet-4-20250514"

ROUTABLE_AGENTS = (
    "documents_agent",
    "role_requirements_agent",
    "compliance_agent",
)
AgentName = Literal["documents_agent", "role_requirements_agent", "compliance_agent"]

REQUIRED_SEQUENCE: list[AgentName] = [
    "documents_agent",
    "role_requirements_agent",
    "compliance_agent",
]

AGENT_TO_SUMMARY_KEY: dict[str, str] = {
    "documents_agent": "documents_summary",
    "role_requirements_agent": "role_requirements_summary",
    "compliance_agent": "compliance_summary",
}

MAX_CALLS_PER_AGENT = 3
MAX_TURNS = 20

# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    query: str
    intent_contract: dict[str, Any]
    agent_status: dict[str, str]          # per-turn scoped, reset each turn
    documents_summary: str
    role_requirements_summary: str
    compliance_summary: str
    compliance_clearance: str              # required display contract field
    routing_history: Annotated[list[str], operator.add]
    call_counts: dict[str, int]
    turn_count: int
    gate_blocks: dict[str, int]
    final_answer: str
    display_contract_violations: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
    # internal routing
    _pending_route: str
    _supervisor_messages: list[dict[str, Any]]
    _force_partial: bool


def _empty_contract() -> dict[str, Any]:
    return {
        "intent": "hr_onboarding_compliance_check",
        "confidence": 0.5,
        "required_entities": ["candidate_id", "role_id", "office_location"],
        "agent_sequence": list(REQUIRED_SEQUENCE),
        "extracted_params": {
            "candidate_id": "CAND-UNKNOWN",
            "role_id": "ROLE-UNKNOWN",
            "office_location": "UNKNOWN",
        },
        "display_contract": {
            "output_type": "compliance_report",
            "required_fields": [
                "ready_to_start",
                "missing_items",
                "risk_flags",
                "compliance_clearance",
            ],
        },
    }


def initial_state(query: str) -> GraphState:
    status = {a: "pending" for a in ROUTABLE_AGENTS}
    return {
        "query": query,
        "intent_contract": {},
        "agent_status": status,
        "documents_summary": "",
        "role_requirements_summary": "",
        "compliance_summary": "",
        "compliance_clearance": "",
        "routing_history": [],
        "call_counts": {a: 0 for a in ROUTABLE_AGENTS},
        "turn_count": 0,
        "gate_blocks": {a: 0 for a in ROUTABLE_AGENTS},
        "final_answer": "",
        "display_contract_violations": [],
        "messages": [HumanMessage(content=query)],
        "_pending_route": "planner",
        "_supervisor_messages": [],
        "_force_partial": False,
    }


# ---------------------------------------------------------------------------
# Mock payloads (same as v3)
# ---------------------------------------------------------------------------


def _big_documents_blob() -> str:
    return json.dumps(
        {
            "candidate_id": "CAND-10492",
            "document_types_submitted": [
                "passport", "work_authorization", "tax_form", "nda", "background_check",
            ],
            "missing_documents": ["i9_section_2_employer_review", "bank_details_for_payroll"],
            "noise": [
                {
                    "doc_id": f"DOC-{i:04d}",
                    "doc_type": (
                        "passport" if i % 6 == 0 else
                        "work_authorization" if i % 6 == 1 else
                        "background_check" if i % 6 == 2 else
                        "tax_form" if i % 6 == 3 else
                        "nda" if i % 6 == 4 else "benefits_enrollment"
                    ),
                    "submitted_at": "2026-05-02" if i % 3 else "2026-05-06",
                    "expires_at": None if i % 4 else "2027-05-06",
                    "verification_status": "verified" if i % 5 else "needs_review",
                    "notes": "vendor latency" if i % 7 == 0 else "ok",
                }
                for i in range(18)
            ],
        },
        indent=2,
    )


def _big_role_requirements_blob() -> str:
    return json.dumps(
        {
            "role_id": "ROLE-ENG-2",
            "department": "Engineering",
            "required_certifications": ["security_awareness_annual", "privacy_training_gdpr_basics"],
            "required_documents_per_role": ["nda", "ip_assignment", "work_authorization"],
            "department_clearances_needed": [
                "laptop_issued", "source_repo_access", "prod_access_denied_until_90d",
            ],
            "noise": [
                {
                    "check_id": f"CHK-{i:04d}",
                    "check_type": (
                        "certification" if i % 3 == 0 else
                        "clearance" if i % 3 == 1 else "paperwork"
                    ),
                    "requirement": (
                        "security_awareness_annual" if i % 4 == 0 else
                        "privacy_training_gdpr_basics" if i % 4 == 1 else
                        "laptop_issued" if i % 4 == 2 else "ip_assignment"
                    ),
                    "status": "complete" if i % 5 else "pending",
                    "owner": "IT" if i % 2 == 0 else "HR",
                }
                for i in range(12)
            ],
        },
        indent=2,
    )


def _big_compliance_blob() -> str:
    return json.dumps(
        {
            "compliance_framework": "employment_onboarding_controls_v2",
            "jurisdictions": [
                {
                    "country": "US",
                    "requirements": [
                        "I-9 completed within 3 business days of start date",
                        "Background check adjudicated before start for restricted roles",
                    ],
                    "penalties": [
                        {"type": "civil_fine", "range_usd": [250, 2500]},
                        {"type": "audit_risk", "notes": "missing I-9 increases exposure"},
                    ],
                },
                {
                    "country": "DE",
                    "requirements": [
                        "Data protection acknowledgement (GDPR) signed",
                        "Works council notification when applicable",
                    ],
                    "penalties": [
                        {"type": "liability", "notes": "access before GDPR ack increases liability"}
                    ],
                },
            ],
            "liability_clause_snippets": [
                "Policy 3.1: identity + work authorization must be verified before start.",
                "Clause 7.2: granting access prior to mandatory controls increases liability.",
            ],
            "noise": "long policy text elided in JSON but imagine multiple pages",
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Specialist LLM helpers
# ---------------------------------------------------------------------------


def _specialist_llm() -> ChatAnthropic:
    return ChatAnthropic(model=SPECIALIST_MODEL, temperature=0, max_tokens=1200)


def summarize_domain(domain: str, query: str, raw_blob: str, params: dict[str, Any]) -> str:
    prompt = textwrap.dedent(
        f"""
        You are writing a **structured summary** for downstream reasoning.

        Domain: {domain}
        Stakeholder question: {query}
        Extracted parameters: candidate_id={params.get('candidate_id')}, \
role_id={params.get('role_id')}, office_location={params.get('office_location')}

        Raw tool payload (may be large/noisy):
        {raw_blob}

        Return **only** these sections, with tight bullets:
        KEY_FACTS:
        RISKS:
        COMPLIANCE_DRIVERS:
        OPEN_QUESTIONS:
        COMPLIANCE_CLEARANCE: (one of: CLEARED / BLOCKED / CONDITIONAL)
        """
    ).strip()
    return str(_specialist_llm().invoke(prompt).content)


# ---------------------------------------------------------------------------
# PLANNER NODE
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = textwrap.dedent(
    """
    You are a planning agent that produces a structured intent contract from a
    user query. Extract candidate_id, role_id, and office_location. Assign a
    confidence score 0.0–1.0. Always output valid JSON with this exact shape:

    {
      "intent": "<string>",
      "confidence": <float>,
      "required_entities": ["candidate_id", "role_id", "office_location"],
      "agent_sequence": ["documents_agent", "role_requirements_agent", "compliance_agent"],
      "extracted_params": {
        "candidate_id": "<string>",
        "role_id": "<string>",
        "office_location": "<string>"
      },
      "display_contract": {
        "output_type": "compliance_report",
        "required_fields": ["ready_to_start", "missing_items", "risk_flags", "compliance_clearance"]
      }
    }

    Output only the JSON, no other text.
    """
).strip()


def node_planner(state: GraphState, *, client: Anthropic) -> dict[str, Any]:
    response = client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=512,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": state["query"]}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        contract = json.loads(raw)
        # Validate required keys are present
        if "extracted_params" not in contract or "display_contract" not in contract:
            raise ValueError("missing keys")
    except Exception:
        contract = _empty_contract()
        contract["extracted_params"]["candidate_id"] = (
            "CAND-10492" if "CAND-10492" in state["query"] else "CAND-UNKNOWN"
        )
        contract["extracted_params"]["role_id"] = (
            "ROLE-ENG-2" if "ROLE-ENG-2" in state["query"] else "ROLE-UNKNOWN"
        )
        contract["extracted_params"]["office_location"] = (
            "NYC" if "NYC" in state["query"] else "UNKNOWN"
        )

    status = {a: "pending" for a in ROUTABLE_AGENTS}
    return {
        "intent_contract": contract,
        "agent_status": status,
        "routing_history": [
            f"planner:generated intent={contract.get('intent')} "
            f"confidence={contract.get('confidence')} "
            f"sequence={contract.get('agent_sequence')}"
        ],
        "_pending_route": "supervisor",
        "_supervisor_messages": [
            {
                "role": "user",
                "content": _supervisor_context(state["query"], status, [], {}, 0, contract),
            }
        ],
        "messages": [AIMessage(content=f"(v4) planner generated intent contract: {contract.get('intent')}")],
    }


# ---------------------------------------------------------------------------
# Specialist nodes (Plan Honor Contract)
# ---------------------------------------------------------------------------


def _run_specialist(
    state: GraphState,
    agent: AgentName,
    domain_label: str,
    raw_blob: str,
) -> dict[str, Any]:
    # Reading from plan params, not raw query (Plan Honor Contract)
    params = state["intent_contract"].get("extracted_params", {})

    counts = dict(state["call_counts"])
    counts[agent] = counts.get(agent, 0) + 1

    summary = summarize_domain(domain_label, state["query"], raw_blob, params)

    # Extract compliance_clearance from compliance_agent summary
    clearance = state.get("compliance_clearance", "")
    if agent == "compliance_agent":
        for line in summary.splitlines():
            if line.strip().startswith("COMPLIANCE_CLEARANCE:"):
                clearance = line.split(":", 1)[1].strip()
                break
        if not clearance:
            clearance = "CONDITIONAL"

    status = dict(state["agent_status"])
    status[agent] = "success"
    summary_key = AGENT_TO_SUMMARY_KEY[agent]

    return {
        "call_counts": counts,
        "agent_status": status,
        summary_key: summary,
        "compliance_clearance": clearance,
        "routing_history": [f"specialist:{agent}:success (params from plan contract)"],
        "messages": [AIMessage(content=f"(v4) {agent} wrote `{summary_key}` using plan params.")],
    }


def node_documents_agent(state: GraphState) -> dict[str, Any]:
    return _run_specialist(state, "documents_agent", "documents / verification", _big_documents_blob())


def node_role_requirements_agent(state: GraphState) -> dict[str, Any]:
    return _run_specialist(
        state, "role_requirements_agent", "role requirements / clearances", _big_role_requirements_blob()
    )


def node_compliance_agent(state: GraphState) -> dict[str, Any]:
    return _run_specialist(
        state, "compliance_agent", "compliance / jurisdictional rules", _big_compliance_blob()
    )


# ---------------------------------------------------------------------------
# Display contract enforcement node
# ---------------------------------------------------------------------------


def node_display_contract_check(state: GraphState) -> dict[str, Any]:
    required_fields = (
        state["intent_contract"]
        .get("display_contract", {})
        .get("required_fields", ["ready_to_start", "missing_items", "risk_flags", "compliance_clearance"])
    )

    violations: list[str] = []
    for field in required_fields:
        # Check if field is present and non-empty in state
        val = state.get(field, "")  # type: ignore[call-overload]
        if not val:
            violations.append(field)

    if violations:
        correction = (
            f"Display contract BLOCKED: missing required fields {violations}. "
            "Run the missing specialist agents to populate these fields before finalizing."
        )
        sup_msgs = list(state["_supervisor_messages"])
        sup_msgs.append({"role": "user", "content": correction})
        return {
            "display_contract_violations": violations,
            "routing_history": [f"display_contract:blocked missing={violations}"],
            "_pending_route": "supervisor",
            "_supervisor_messages": sup_msgs,
            "messages": [AIMessage(content=f"(v4) display contract blocked: missing {violations}")],
        }

    return {
        "display_contract_violations": [],
        "routing_history": ["display_contract:passed"],
        "_pending_route": "finalize",
        "messages": [AIMessage(content="(v4) display contract passed — proceeding to finalize.")],
    }


# ---------------------------------------------------------------------------
# Supervisor tools (same 3 as v3)
# ---------------------------------------------------------------------------

SUPERVISOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "route_to_agent",
        "description": (
            "Route work to a specialist agent. Use when more evidence is needed. "
            "Do not re-route to agents already marked success this turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": list(ROUTABLE_AGENTS),
                    "description": "Specialist to invoke next.",
                },
                "reason": {"type": "string", "description": "Why this agent should run now."},
                "primary_intent": {"type": "string", "description": "The ultimate stakeholder goal."},
            },
            "required": ["agent_name", "reason", "primary_intent"],
        },
    },
    {
        "name": "signal_complete",
        "description": "Finish the workflow when all required specialists succeeded.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agents_completed": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(ROUTABLE_AGENTS)},
                    "description": "Agents that reported success.",
                },
                "summary": {
                    "type": "string",
                    "description": "Short readiness assessment for the stakeholder.",
                },
            },
            "required": ["agents_completed", "summary"],
        },
    },
    {
        "name": "request_clarification",
        "description": "Ask the stakeholder a clarifying question before continuing.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]

SUPERVISOR_SYSTEM = textwrap.dedent(
    """
    You are an HR onboarding supervisor coordinating three specialists:
    documents_agent, role_requirements_agent, compliance_agent.

    You operate under a pre-declared intent contract. Route using tools only.
    Follow the declared agent_sequence from the intent contract.
    Do not signal_complete until all agents in the sequence show success.
    Do not re-route to an agent that already shows success this turn.
    """
).strip()


def _supervisor_context(
    query: str,
    agent_status: dict[str, str],
    routing_history: list[str],
    call_counts: dict[str, int],
    turn_count: int,
    intent_contract: dict[str, Any],
) -> str:
    return textwrap.dedent(
        f"""
        Query: {query}
        Intent contract:
          intent: {intent_contract.get('intent')}
          confidence: {intent_contract.get('confidence')}
          declared agent_sequence: {intent_contract.get('agent_sequence')}
          extracted_params: {json.dumps(intent_contract.get('extracted_params', {}))}
        Agent status: {json.dumps(agent_status)}
        Call counts: {json.dumps(call_counts)}
        Turn count: {turn_count}
        Recent routing history:
        {chr(10).join(routing_history[-12:]) or "(none)"}
        """
    ).strip()


# ---------------------------------------------------------------------------
# Routing gate (plan-aware, per-turn scoped)
# ---------------------------------------------------------------------------


def routing_gate_blocks(agent: str, state: GraphState) -> tuple[bool, str]:
    """Per-turn scoped gate. Plan-aware: suggests next correct agent when blocking."""
    if state["agent_status"].get(agent) != "success":
        return False, ""

    # Find next agent in declared sequence that hasn't succeeded
    sequence = state["intent_contract"].get("agent_sequence", list(REQUIRED_SEQUENCE))
    next_agent = None
    for a in sequence:
        if state["agent_status"].get(a) != "success":
            next_agent = a
            break

    suggestion = f"next declared agent is {next_agent}" if next_agent else "all agents complete — use signal_complete"
    return True, suggestion


def signal_complete_allowed(state: GraphState) -> tuple[bool, str]:
    sequence = state["intent_contract"].get("agent_sequence", list(REQUIRED_SEQUENCE))
    missing = [a for a in sequence if state["agent_status"].get(a) != "success"]
    if missing:
        return False, f"Required agents not yet successful: {missing}"
    return True, "ok"


def circuit_breaker_tripped(state: GraphState) -> str | None:
    if state["turn_count"] >= MAX_TURNS:
        return f"max_turns ({MAX_TURNS})"
    for agent in ROUTABLE_AGENTS:
        if state["call_counts"].get(agent, 0) >= MAX_CALLS_PER_AGENT:
            return f"max_calls for {agent} ({MAX_CALLS_PER_AGENT})"
    return None


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------


def _tool_result_block(tool_use_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": json.dumps(payload)}


def supervisor_node(state: GraphState, *, client: Anthropic) -> dict[str, Any]:
    turn = state["turn_count"] + 1
    updates: dict[str, Any] = {
        "turn_count": turn,
        "routing_history": [f"supervisor:turn#{turn}"],
    }

    trip = circuit_breaker_tripped(state)
    if trip:
        updates["routing_history"] = updates["routing_history"] + [
            f"circuit_breaker:tripped ({trip}) → force finalize"
        ]
        updates["_pending_route"] = "display_contract_check"
        updates["_force_partial"] = True
        return updates

    sup_msgs = list(state["_supervisor_messages"])
    sup_msgs[-1] = {
        "role": "user",
        "content": _supervisor_context(
            state["query"],
            state["agent_status"],
            state["routing_history"],
            state["call_counts"],
            turn,
            state["intent_contract"],
        ),
    }

    response = client.messages.create(
        model=SUPERVISOR_MODEL,
        max_tokens=1024,
        system=SUPERVISOR_SYSTEM,
        tools=SUPERVISOR_TOOLS,
        messages=sup_msgs,
    )

    assistant_blocks = [block.model_dump() for block in response.content]
    tool_uses = [b for b in assistant_blocks if b.get("type") == "tool_use"]

    if not tool_uses:
        sup_msgs.append({"role": "assistant", "content": assistant_blocks})
        sup_msgs.append({
            "role": "user",
            "content": [_tool_result_block(
                str(uuid.uuid4()),
                {"error": "You must call route_to_agent, signal_complete, or request_clarification."},
            )],
        })
        updates["_supervisor_messages"] = sup_msgs
        updates["_pending_route"] = "supervisor"
        return updates

    tool_results: list[dict[str, Any]] = []
    pending: str = "supervisor"
    history_extra: list[str] = []
    gate = dict(state["gate_blocks"])

    for block in tool_uses:
        name = block["name"]
        inp = block.get("input") or {}
        tid = block["id"]

        if name == "route_to_agent":
            agent = inp.get("agent_name", "")
            if agent not in ROUTABLE_AGENTS:
                tool_results.append(_tool_result_block(tid, {"blocked": True, "reason": f"invalid agent: {agent}"}))
                history_extra.append(f"route_to_agent:blocked invalid {agent}")
                continue

            blocked, suggestion = routing_gate_blocks(agent, state)
            if blocked:
                gate[agent] = gate.get(agent, 0) + 1
                updates["gate_blocks"] = dict(gate)
                tool_results.append(_tool_result_block(tid, {
                    "blocked": True,
                    "reason": f"{agent} already success this turn (routing gate)",
                    "plan_suggestion": suggestion,
                    "gate_blocks": gate[agent],
                }))
                history_extra.append(
                    f"routing_gate:blocked {agent} gate_blocks={gate[agent]} suggestion={suggestion}"
                )
                continue

            if state["call_counts"].get(agent, 0) >= MAX_CALLS_PER_AGENT:
                tool_results.append(_tool_result_block(tid, {"blocked": True, "reason": "circuit breaker"}))
                history_extra.append(f"circuit_breaker:blocked {agent}")
                continue

            tool_results.append(_tool_result_block(tid, {"blocked": False, "routed": agent}))
            history_extra.append(f"route_to_agent:{agent} reason={inp.get('reason', '')[:80]}")
            pending = agent

        elif name == "signal_complete":
            ok, reason = signal_complete_allowed(state)
            if not ok:
                tool_results.append(_tool_result_block(tid, {"accepted": False, "reason": reason}))
                history_extra.append(f"signal_complete:rejected ({reason})")
                pending = "supervisor"
            else:
                tool_results.append(_tool_result_block(tid, {"accepted": True}))
                history_extra.append("signal_complete:accepted")
                updates["final_answer"] = inp.get("summary", "")
                pending = "display_contract_check"

        elif name == "request_clarification":
            question = inp.get("question", "")
            tool_results.append(_tool_result_block(tid, {"ack": True}))
            history_extra.append(f"request_clarification:{question[:80]}")
            updates["final_answer"] = f"Clarification needed: {question}"
            pending = "finalize"

    sup_msgs.append({"role": "assistant", "content": assistant_blocks})
    if tool_results:
        sup_msgs.append({"role": "user", "content": tool_results})

    updates["_supervisor_messages"] = sup_msgs
    updates["routing_history"] = updates["routing_history"] + history_extra
    updates["_pending_route"] = pending
    return updates


# ---------------------------------------------------------------------------
# Finalize node
# ---------------------------------------------------------------------------


def node_finalize(state: GraphState) -> dict[str, Any]:
    if state.get("final_answer"):
        answer = state["final_answer"]
    else:
        body = textwrap.dedent(
            f"""
            Answer using only the summaries below.

            Question: {state['query']}

            DOCUMENTS: {state['documents_summary']}
            ROLE REQUIREMENTS: {state['role_requirements_summary']}
            COMPLIANCE: {state['compliance_summary']}
            COMPLIANCE_CLEARANCE: {state['compliance_clearance']}

            State whether the hire is ready to start and list blockers.
            Include sections: ready_to_start, missing_items, risk_flags, compliance_clearance.
            """
        ).strip()
        answer = str(_specialist_llm().invoke(body).content)

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def planner_router(state: GraphState) -> str:
    return state.get("_pending_route", "supervisor")  # type: ignore[return-value]


def supervisor_router(state: GraphState) -> str:
    route = state.get("_pending_route", "supervisor")
    if route in ROUTABLE_AGENTS:
        return route  # type: ignore[return-value]
    if route == "display_contract_check":
        return "display_contract_check"
    if route == "finalize":
        return "finalize"
    if state["turn_count"] >= MAX_TURNS:
        return "display_contract_check"
    return "supervisor"


def display_contract_router(state: GraphState) -> str:
    route = state.get("_pending_route", "supervisor")
    if route == "finalize":
        return "finalize"
    return "supervisor"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(client: Anthropic):
    def _planner(state: GraphState) -> dict[str, Any]:
        return node_planner(state, client=client)

    def _supervisor(state: GraphState) -> dict[str, Any]:
        return supervisor_node(state, client=client)

    g = StateGraph(GraphState)
    g.add_node("planner", _planner)
    g.add_node("supervisor", _supervisor)
    g.add_node("documents_agent", node_documents_agent)
    g.add_node("role_requirements_agent", node_role_requirements_agent)
    g.add_node("compliance_agent", node_compliance_agent)
    g.add_node("display_contract_check", node_display_contract_check)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        planner_router,
        {"supervisor": "supervisor"},
    )
    g.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "supervisor": "supervisor",
            "documents_agent": "documents_agent",
            "role_requirements_agent": "role_requirements_agent",
            "compliance_agent": "compliance_agent",
            "display_contract_check": "display_contract_check",
            "finalize": "finalize",
        },
    )
    for agent in ROUTABLE_AGENTS:
        g.add_edge(agent, "supervisor")
    g.add_conditional_edges(
        "display_contract_check",
        display_contract_router,
        {"finalize": "finalize", "supervisor": "supervisor"},
    )
    g.add_edge("finalize", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _print_trace(result: GraphState, *, show_contract: bool = False) -> None:
    if show_contract:
        print("\n--- intent_contract ---")
        print(json.dumps(result.get("intent_contract", {}), indent=2))
    print("\n--- routing_history ---")
    for line in result["routing_history"]:
        print(f"  {line}")
    print("\n--- agent_status ---")
    print(json.dumps(result["agent_status"], indent=2))
    print("\n--- gate_blocks ---")
    print(json.dumps(result.get("gate_blocks", {}), indent=2))
    print("\n--- display_contract_violations ---")
    print(result.get("display_contract_violations", []))
    if result.get("compliance_clearance"):
        print(f"\n--- compliance_clearance ---\n  {result['compliance_clearance']}")
    if result.get("final_answer"):
        print("\n--- final_answer (excerpt) ---")
        print(textwrap.shorten(result["final_answer"], width=700))


# ---------------------------------------------------------------------------
# Demo 1 — Happy path
# ---------------------------------------------------------------------------


def demo1_happy_path(app) -> None:
    _banner("Demo 1 — Happy path: plan → sequence → display contract → finalize")
    state = initial_state(DEMO_SCENARIO)
    result = app.invoke(state, config={"recursion_limit": 50})
    _print_trace(result, show_contract=True)


# ---------------------------------------------------------------------------
# Demo 2 — Failure 1: routing gate bypass
# ---------------------------------------------------------------------------


def demo2_routing_gate(app) -> None:
    _banner("Demo 2 — Failure 1: routing gate bypass (plan-aware blocking)")

    # Run graph normally to get all agents to success
    state = initial_state(DEMO_SCENARIO)
    result = app.invoke(state, config={"recursion_limit": 50})

    # Manually inject a re-route to role_requirements_agent (already success)
    print("\n[injecting] Attempting to re-route to role_requirements_agent (already success)...")

    # Simulate the gate check directly
    agent = "role_requirements_agent"
    blocked, suggestion = routing_gate_blocks(agent, result)
    if blocked:
        gate = dict(result.get("gate_blocks", {}))
        gate[agent] = gate.get(agent, 0) + 1
        print(f"  routing_gate:blocked {agent} gate_blocks={gate[agent]}")
        print(f"  plan_suggestion: {suggestion}")
        print("  Final answer still correct (gate fired before agent ran).")
    else:
        print("  Gate did not fire (unexpected).")

    _print_trace(result)


# ---------------------------------------------------------------------------
# Demo 3 — Failure 2: display contract enforcement
# ---------------------------------------------------------------------------


def demo3_display_contract(client: Anthropic) -> None:
    _banner("Demo 3 — Failure 2: display contract enforcement (compliance skipped)")

    # Build a partial state: documents + role_requirements done, compliance skipped
    # We do this by running the graph but pre-marking compliance_agent as skipped
    # and setting up state such that the display contract check fires first.
    # Easiest: build a standalone mini-graph that skips compliance_agent.

    # Run planner separately to get the contract
    state = initial_state(DEMO_SCENARIO)
    planner_result = node_planner(state, client=client)
    contract = planner_result["intent_contract"]

    # Manually build partial state (docs + role done, compliance NOT done)
    partial_status = {
        "documents_agent": "success",
        "role_requirements_agent": "success",
        "compliance_agent": "pending",
    }

    # Construct state with compliance_clearance empty (display contract will catch it)
    partial_state: GraphState = {
        "query": DEMO_SCENARIO,
        "intent_contract": contract,
        "agent_status": partial_status,
        "documents_summary": "(mock documents summary)",
        "role_requirements_summary": "(mock role requirements summary)",
        "compliance_summary": "",           # not run
        "compliance_clearance": "",         # missing — display contract will block
        "routing_history": ["demo3:partial_state_injected"],
        "call_counts": {"documents_agent": 1, "role_requirements_agent": 1, "compliance_agent": 0},
        "turn_count": 0,
        "gate_blocks": {a: 0 for a in ROUTABLE_AGENTS},
        "final_answer": "",
        "display_contract_violations": [],
        "messages": [HumanMessage(content=DEMO_SCENARIO)],
        "_pending_route": "supervisor",
        "_supervisor_messages": [
            {
                "role": "user",
                "content": _supervisor_context(
                    DEMO_SCENARIO, partial_status, ["demo3:partial_state_injected"],
                    {"documents_agent": 1, "role_requirements_agent": 1, "compliance_agent": 0},
                    0, contract,
                ),
            }
        ],
        "_force_partial": False,
    }

    print("\n[demo3] Running from partial state (compliance_agent skipped)...")
    print("[demo3] Expect display contract to block, compliance_agent to run, then finalize.")

    # Build a graph that skips the planner (starts at supervisor)
    def _supervisor(s: GraphState) -> dict[str, Any]:
        return supervisor_node(s, client=client)

    g = StateGraph(GraphState)
    g.add_node("supervisor", _supervisor)
    g.add_node("documents_agent", node_documents_agent)
    g.add_node("role_requirements_agent", node_role_requirements_agent)
    g.add_node("compliance_agent", node_compliance_agent)
    g.add_node("display_contract_check", node_display_contract_check)
    g.add_node("finalize", node_finalize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", supervisor_router,
        {
            "supervisor": "supervisor",
            "documents_agent": "documents_agent",
            "role_requirements_agent": "role_requirements_agent",
            "compliance_agent": "compliance_agent",
            "display_contract_check": "display_contract_check",
            "finalize": "finalize",
        },
    )
    for agent in ROUTABLE_AGENTS:
        g.add_edge(agent, "supervisor")
    g.add_conditional_edges(
        "display_contract_check", display_contract_router,
        {"finalize": "finalize", "supervisor": "supervisor"},
    )
    g.add_edge("finalize", END)
    partial_app = g.compile()

    result = partial_app.invoke(partial_state, config={"recursion_limit": 50})
    _print_trace(result)


# ---------------------------------------------------------------------------
# Demo 4 — HITL resume
# ---------------------------------------------------------------------------


def demo4_hitl_resume(client: Anthropic) -> None:
    _banner("Demo 4 — HITL resume: reuse prior plan, skip re-planning")

    # Phase 1: run planner + documents_agent only (simulate HITL pause after that)
    print("\n[hitl] Phase 1: planner runs, documents_agent runs, then HITL pause...")
    state = initial_state(DEMO_SCENARIO)

    # Run planner
    planner_updates = node_planner(state, client=client)
    # Merge planner updates into state
    mid_state: GraphState = {**state, **planner_updates}  # type: ignore[misc]
    mid_state["routing_history"] = planner_updates["routing_history"]

    # Run documents_agent
    doc_updates = node_documents_agent(mid_state)
    mid_state = {**mid_state, **doc_updates}  # type: ignore[misc]
    mid_state["routing_history"] = planner_updates["routing_history"] + doc_updates["routing_history"]

    saved_contract = mid_state["intent_contract"]
    print(f"  intent_contract saved: intent={saved_contract.get('intent')} confidence={saved_contract.get('confidence')}")
    print("  [hitl] PAUSED — waiting for human approval...")
    print("  [hitl] Human approved. Resuming...")

    # Phase 2: resume — reuse existing contract verbatim, do NOT re-run planner
    print("\n[hitl] Phase 2: resuming with prior plan contract (no re-planning)...")

    # Mark the resume in routing_history and rebuild supervisor messages
    resume_history = mid_state["routing_history"] + ["hitl_resume: reusing prior plan"]
    sup_msgs = [
        {
            "role": "user",
            "content": _supervisor_context(
                mid_state["query"],
                mid_state["agent_status"],
                resume_history,
                mid_state["call_counts"],
                0,
                saved_contract,  # reusing prior contract verbatim
            ),
        }
    ]

    # Build a supervisor-only graph (skip planner — contract already exists)
    def _supervisor(s: GraphState) -> dict[str, Any]:
        return supervisor_node(s, client=client)

    g = StateGraph(GraphState)
    g.add_node("supervisor", _supervisor)
    g.add_node("documents_agent", node_documents_agent)
    g.add_node("role_requirements_agent", node_role_requirements_agent)
    g.add_node("compliance_agent", node_compliance_agent)
    g.add_node("display_contract_check", node_display_contract_check)
    g.add_node("finalize", node_finalize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", supervisor_router,
        {
            "supervisor": "supervisor",
            "documents_agent": "documents_agent",
            "role_requirements_agent": "role_requirements_agent",
            "compliance_agent": "compliance_agent",
            "display_contract_check": "display_contract_check",
            "finalize": "finalize",
        },
    )
    for agent in ROUTABLE_AGENTS:
        g.add_edge(agent, "supervisor")
    g.add_conditional_edges(
        "display_contract_check", display_contract_router,
        {"finalize": "finalize", "supervisor": "supervisor"},
    )
    g.add_edge("finalize", END)
    resume_app = g.compile()

    resume_state: GraphState = {
        **mid_state,  # type: ignore[misc]
        "routing_history": resume_history,
        "_pending_route": "supervisor",
        "_supervisor_messages": sup_msgs,
    }

    result = resume_app.invoke(resume_state, config={"recursion_limit": 50})
    _print_trace(result)

    # Confirm hitl_resume entry is present
    if any("hitl_resume" in e for e in result["routing_history"]):
        print("\n[hitl] CONFIRMED: 'hitl_resume: reusing prior plan' found in routing_history.")
        print("[hitl] Planner was NOT re-run — saved one planner call.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and fill the key."
        )

    client = Anthropic()
    app = build_graph(client)

    demo1_happy_path(app)
    demo2_routing_gate(app)
    demo3_display_contract(client)
    demo4_hitl_resume(client)

    print("\n=== v4 demos complete ===\n")


if __name__ == "__main__":
    main()
