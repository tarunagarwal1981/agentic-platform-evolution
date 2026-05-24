#!/usr/bin/env python3
"""
v3 — Tool supervisor with forced tool-call routing and guardrails.

Teaching goal
-------------
Show a **supervisor** that routes via Anthropic tool_use (not text parsing),
three specialist agents that write **structured summaries** into typed state,
and five failure-mode fixes: circuit breaker, routing gate, gate escalation,
signal_complete enforcement, and compound-query support.

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
from typing_extensions import NotRequired

# ---------------------------------------------------------------------------
# Demo scenario (same as v1 / v2)
# ---------------------------------------------------------------------------

DEMO_SCENARIO = (
    "Is this new hire fully compliant and ready to start given their "
    "documents and role requirements?"
)

COMPOUND_DEMO_QUERY = (
    "Check documents AND run compliance for this new hire — are they ready "
    "to start given role requirements?"
)

SPECIALIST_MODEL = "claude-sonnet-4-20250514"
SUPERVISOR_MODEL = "claude-haiku-4-5-20251001"

ROUTABLE_AGENTS = (
    "documents_agent",
    "role_requirements_agent",
    "compliance_agent",
)
AgentName = Literal[
    "documents_agent",
    "role_requirements_agent",
    "compliance_agent",
]

REQUIRED_SEQUENCE: list[AgentName] = [
    "documents_agent",
    "role_requirements_agent",
    "compliance_agent",
]

AGENT_TO_SUMMARY_KEY: dict[AgentName, str] = {
    "documents_agent": "documents_summary",
    "role_requirements_agent": "role_requirements_summary",
    "compliance_agent": "compliance_summary",
}

MAX_CALLS_PER_AGENT = 3
MAX_TURNS_PER_QUERY = 20
GATE_ESCALATION_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    agent_status: dict[str, str]
    documents_summary: str
    role_requirements_summary: str
    compliance_summary: str
    routing_history: Annotated[list[str], operator.add]
    call_counts: dict[str, int]
    turn_count: int
    final_answer: str
    gate_blocks: dict[str, int]
    supervisor_messages: list[dict[str, Any]]
    pending_route: NotRequired[str | None]
    demo_mode: NotRequired[str | None]
    compound_intents: NotRequired[list[str]]
    force_partial_complete: NotRequired[bool]
    supervisor_correction_pending: NotRequired[bool]


def initial_state(
    query: str,
    *,
    demo_mode: str | None = None,
    agent_status_override: dict[str, str] | None = None,
    supervisor_extra: str | None = None,
) -> GraphState:
    status = {agent: "pending" for agent in ROUTABLE_AGENTS}
    if agent_status_override:
        status.update(agent_status_override)
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "agent_status": status,
        "documents_summary": "",
        "role_requirements_summary": "",
        "compliance_summary": "",
        "routing_history": [],
        "call_counts": {agent: 0 for agent in ROUTABLE_AGENTS},
        "turn_count": 0,
        "final_answer": "",
        "gate_blocks": {agent: 0 for agent in ROUTABLE_AGENTS},
        "supervisor_messages": [
            {
                "role": "user",
                "content": _supervisor_context(query, status, [], {}, 0)
                + (f"\n\n{supervisor_extra}" if supervisor_extra else ""),
            }
        ],
        "pending_route": None,
        "demo_mode": demo_mode,
        "compound_intents": detect_compound_intents(query),
        "force_partial_complete": False,
        "supervisor_correction_pending": False,
    }


# ---------------------------------------------------------------------------
# Failure 5 — compound query: detect multiple intents
# ---------------------------------------------------------------------------


def detect_compound_intents(query: str) -> list[str]:
    """Map query text to specialist domains that must all complete."""
    q = query.lower()
    intents: list[str] = []
    if any(k in q for k in ("document", "paperwork", "verification", "i-9", "i9")):
        intents.append("documents")
    if any(
        k in q
        for k in ("role", "certification", "clearance", "requirement", "onboarding")
    ):
        intents.append("role_requirements")
    if any(k in q for k in ("compliance", "jurisdiction", "liability", "legal")):
        intents.append("compliance")
    if " and " in q and len(intents) < 2:
        # Broaden for explicit compound phrasing ("documents AND compliance")
        if "document" in q and "documents" not in intents:
            intents.append("documents")
        if "compliance" in q and "compliance" not in intents:
            intents.append("compliance")
        if "role" in q and "role_requirements" not in intents:
            intents.append("role_requirements")
    if not intents:
        intents = ["documents", "role_requirements", "compliance"]
    return intents


def agents_required_for_intents(intents: list[str]) -> list[AgentName]:
    mapping = {
        "documents": "documents_agent",
        "role_requirements": "role_requirements_agent",
        "compliance": "compliance_agent",
    }
    ordered: list[AgentName] = []
    for key in ("documents", "role_requirements", "compliance"):
        if key in intents:
            agent = mapping[key]
            if agent not in ordered:
                ordered.append(agent)
    return ordered or list(REQUIRED_SEQUENCE)


# ---------------------------------------------------------------------------
# Mock payloads (same patterns as v2)
# ---------------------------------------------------------------------------


def _big_documents_blob() -> str:
    return json.dumps(
        {
            "candidate_id": "CAND-10492",
            "document_types_submitted": [
                "passport",
                "work_authorization",
                "tax_form",
                "nda",
                "background_check",
            ],
            "missing_documents": [
                "i9_section_2_employer_review",
                "bank_details_for_payroll",
            ],
            "noise": [
                {
                    "doc_id": f"DOC-{i:04d}",
                    "doc_type": (
                        "passport"
                        if i % 6 == 0
                        else "work_authorization"
                        if i % 6 == 1
                        else "background_check"
                        if i % 6 == 2
                        else "tax_form"
                        if i % 6 == 3
                        else "nda"
                        if i % 6 == 4
                        else "benefits_enrollment"
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
            "required_certifications": [
                "security_awareness_annual",
                "privacy_training_gdpr_basics",
            ],
            "required_documents_per_role": [
                "nda",
                "ip_assignment",
                "work_authorization",
            ],
            "department_clearances_needed": [
                "laptop_issued",
                "source_repo_access",
                "prod_access_denied_until_90d",
            ],
            "noise": [
                {
                    "check_id": f"CHK-{i:04d}",
                    "check_type": (
                        "certification"
                        if i % 3 == 0
                        else "clearance"
                        if i % 3 == 1
                        else "paperwork"
                    ),
                    "requirement": (
                        "security_awareness_annual"
                        if i % 4 == 0
                        else "privacy_training_gdpr_basics"
                        if i % 4 == 1
                        else "laptop_issued"
                        if i % 4 == 2
                        else "ip_assignment"
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
                        {
                            "type": "audit_risk",
                            "notes": "missing I-9 increases exposure",
                        },
                    ],
                },
                {
                    "country": "DE",
                    "requirements": [
                        "Data protection acknowledgement (GDPR) signed",
                        "Works council notification when applicable",
                    ],
                    "penalties": [
                        {
                            "type": "liability",
                            "notes": "access before GDPR ack increases liability",
                        }
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


def _specialist_llm() -> ChatAnthropic:
    return ChatAnthropic(model=SPECIALIST_MODEL, temperature=0, max_tokens=1200)


def summarize_domain(domain: str, query: str, raw_blob: str) -> str:
    prompt = textwrap.dedent(
        f"""
        You are writing a **structured summary** for downstream reasoning.

        Domain: {domain}
        Stakeholder question: {query}

        Raw tool payload (may be large/noisy):
        {raw_blob}

        Return **only** these sections, with tight bullets:
        KEY_FACTS:
        RISKS:
        COMPLIANCE_DRIVERS:
        OPEN_QUESTIONS:
        """
    ).strip()
    return str(_specialist_llm().invoke(prompt).content)


# ---------------------------------------------------------------------------
# Supervisor tools (Anthropic tool_use — not LangChain tools)
# ---------------------------------------------------------------------------

SUPERVISOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "route_to_agent",
        "description": (
            "Route work to a specialist agent. Use when more evidence is needed. "
            "Do not re-route to agents already marked success."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": list(ROUTABLE_AGENTS),
                    "description": "Specialist to invoke next.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this agent should run now.",
                },
                "primary_intent": {
                    "type": "string",
                    "description": (
                        "The ultimate stakeholder goal, not a prerequisite step."
                    ),
                },
            },
            "required": ["agent_name", "reason", "primary_intent"],
        },
    },
    {
        "name": "signal_complete",
        "description": (
            "Finish the workflow when all required specialists succeeded in order."
        ),
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
            "properties": {
                "question": {"type": "string"},
            },
            "required": ["question"],
        },
    },
]

SUPERVISOR_SYSTEM = textwrap.dedent(
    """
    You are an HR onboarding supervisor coordinating three specialists:
    documents_agent, role_requirements_agent, compliance_agent.

    Route using tools only — never output routing decisions as plain text.
    Required order for this scenario: documents_agent, then role_requirements_agent,
    then compliance_agent. Do not signal_complete until all three show success
    in agent_status.

    Prefer route_to_agent for the next pending specialist. Use signal_complete
    only when every required agent has succeeded. Use request_clarification only
    if the query is ambiguous.
    """
).strip()


def _supervisor_context(
    query: str,
    agent_status: dict[str, str],
    routing_history: list[str],
    call_counts: dict[str, int],
    turn_count: int,
    *,
    compound_intents: list[str] | None = None,
) -> str:
    return textwrap.dedent(
        f"""
        Query: {query}
        Compound intents detected: {compound_intents or detect_compound_intents(query)}
        Agent status: {json.dumps(agent_status)}
        Call counts: {json.dumps(call_counts)}
        Turn count: {turn_count}
        Recent routing history:
        {chr(10).join(routing_history[-12:]) or "(none)"}
        """
    ).strip()


# ---------------------------------------------------------------------------
# Failure 1 — circuit breaker
# ---------------------------------------------------------------------------


def circuit_breaker_tripped(state: GraphState) -> str | None:
    if state["turn_count"] >= MAX_TURNS_PER_QUERY:
        return f"max_turns ({MAX_TURNS_PER_QUERY})"
    for agent in ROUTABLE_AGENTS:
        if state["call_counts"].get(agent, 0) >= MAX_CALLS_PER_AGENT:
            return f"max_calls for {agent} ({MAX_CALLS_PER_AGENT})"
    return None


# ---------------------------------------------------------------------------
# Failure 2 — routing gate
# ---------------------------------------------------------------------------


def routing_gate_blocks(agent: AgentName, state: GraphState) -> bool:
    return state["agent_status"].get(agent) == "success"


# ---------------------------------------------------------------------------
# Failure 3 — gate escalation
# ---------------------------------------------------------------------------


def next_unvisited_agent(state: GraphState) -> AgentName | None:
    for agent in REQUIRED_SEQUENCE:
        if state["agent_status"].get(agent) != "success":
            return agent
    return None


def gate_escalation_target(state: GraphState) -> AgentName | None:
    for agent in ROUTABLE_AGENTS:
        if state["gate_blocks"].get(agent, 0) >= GATE_ESCALATION_THRESHOLD:
            nxt = next_unvisited_agent(state)
            if nxt and nxt != agent:
                return nxt
    return None


# ---------------------------------------------------------------------------
# Failure 4 — signal_complete enforcement
# ---------------------------------------------------------------------------


def sequence_violation(state: GraphState) -> str | None:
    """documents before role_requirements before compliance."""
    idx = {agent: state["agent_status"].get(agent) for agent in REQUIRED_SEQUENCE}
    if idx["role_requirements_agent"] == "success" and idx["documents_agent"] != "success":
        return "role_requirements_agent succeeded before documents_agent"
    if idx["compliance_agent"] == "success" and idx["role_requirements_agent"] != "success":
        return "compliance_agent succeeded before role_requirements_agent"
    return None


def signal_complete_allowed(state: GraphState) -> tuple[bool, str]:
    required = agents_required_for_intents(
        state.get("compound_intents") or detect_compound_intents(state["query"])
    )
    missing = [
        a for a in required if state["agent_status"].get(a) != "success"
    ]
    if missing:
        return (
            False,
            f"Required agents not successful: {missing}. Complete them in order: "
            f"{REQUIRED_SEQUENCE}.",
        )
    seq_err = sequence_violation(state)
    if seq_err:
        return False, seq_err
    return True, "ok"


# ---------------------------------------------------------------------------
# Specialist nodes
# ---------------------------------------------------------------------------


def _run_specialist(
    state: GraphState,
    agent: AgentName,
    domain_label: str,
    raw_blob: str,
) -> dict[str, Any]:
    counts = dict(state["call_counts"])
    counts[agent] = counts.get(agent, 0) + 1
    history = [
        f"specialist:{agent}:call#{counts[agent]}",
    ]

    # Failure 1 demo: first two documents_agent calls fail
    if (
        state.get("demo_mode") == "circuit_breaker"
        and agent == "documents_agent"
        and counts[agent] <= 2
    ):
        status = dict(state["agent_status"])
        status[agent] = "failed"
        return {
            "call_counts": counts,
            "agent_status": status,
            "routing_history": history
            + [f"{agent}:simulated_failure (demo Failure 1)"],
            "messages": [
                AIMessage(content=f"(v3) {agent} failed (simulated for circuit-breaker demo).")
            ],
        }

    summary = summarize_domain(domain_label, state["query"], raw_blob)
    status = dict(state["agent_status"])
    status[agent] = "success"
    summary_key = AGENT_TO_SUMMARY_KEY[agent]
    return {
        "call_counts": counts,
        "agent_status": status,
        summary_key: summary,
        "routing_history": history + [f"{agent}:success"],
        "messages": [
            AIMessage(content=f"(v3) {agent} wrote `{summary_key}`.")
        ],
    }


def node_documents_agent(state: GraphState) -> dict[str, Any]:
    return _run_specialist(
        state, "documents_agent", "documents / verification", _big_documents_blob()
    )


def node_role_requirements_agent(state: GraphState) -> dict[str, Any]:
    return _run_specialist(
        state,
        "role_requirements_agent",
        "role requirements / clearances",
        _big_role_requirements_blob(),
    )


def node_compliance_agent(state: GraphState) -> dict[str, Any]:
    return _run_specialist(
        state,
        "compliance_agent",
        "compliance / jurisdictional rules",
        _big_compliance_blob(),
    )


# ---------------------------------------------------------------------------
# Supervisor node — forced tool calls
# ---------------------------------------------------------------------------


def _append_supervisor_exchange(
    messages: list[dict[str, Any]],
    assistant_content: list[dict[str, Any]],
    tool_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out = list(messages)
    out.append({"role": "assistant", "content": assistant_content})
    if tool_results:
        out.append({"role": "user", "content": tool_results})
    return out


def _tool_result_block(tool_use_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(payload),
    }


def supervisor_node(state: GraphState, *, client: Anthropic) -> dict[str, Any]:
    turn = state["turn_count"] + 1
    updates: dict[str, Any] = {
        "turn_count": turn,
        "routing_history": [f"supervisor:turn#{turn}"],
        "pending_route": None,
    }

    # --- Failure 1: circuit breaker (hard stop) ---
    trip = circuit_breaker_tripped(state)
    if trip:
        updates["routing_history"] = updates["routing_history"] + [
            f"circuit_breaker:tripped ({trip}) → force signal_complete (partial)"
        ]
        updates["force_partial_complete"] = True
        updates["pending_route"] = "finalize"
        return updates

    # --- Failure 3: gate escalation bypass ---
    escalated = gate_escalation_target(state)
    if escalated:
        updates["routing_history"] = updates["routing_history"] + [
            f"gate_escalation:force_route → {escalated}",
        ]
        updates["pending_route"] = escalated
        return updates

    if turn >= MAX_TURNS_PER_QUERY:
        updates["routing_history"] = updates["routing_history"] + [
            f"circuit_breaker:tripped (max_turns {MAX_TURNS_PER_QUERY}) → finalize"
        ]
        updates["force_partial_complete"] = True
        updates["pending_route"] = "finalize"
        return updates

    # Refresh supervisor context
    sup_msgs = list(state["supervisor_messages"])
    if state.get("supervisor_correction_pending"):
        sup_msgs.append(
            {
                "role": "user",
                "content": (
                    "CORRECTION: signal_complete was rejected. Route remaining "
                    "specialists in order until all required agents show success."
                ),
            }
        )
        updates["supervisor_correction_pending"] = False
    else:
        sup_msgs[-1] = {
            "role": "user",
            "content": _supervisor_context(
                state["query"],
                state["agent_status"],
                state["routing_history"],
                state["call_counts"],
                turn,
                compound_intents=state.get("compound_intents"),
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
        # Nudge: supervisor must use a tool
        updates["supervisor_messages"] = _append_supervisor_exchange(
            sup_msgs,
            assistant_blocks,
            [
                _tool_result_block(
                    str(uuid.uuid4()),
                    {"error": "You must call route_to_agent, signal_complete, or request_clarification."},
                )
            ],
        )
        updates["pending_route"] = "supervisor"
        return updates

    tool_results: list[dict[str, Any]] = []
    pending: str | None = None
    history_extra: list[str] = []
    gate = dict(state["gate_blocks"])

    for block in tool_uses:
        name = block["name"]
        inp = block.get("input") or {}
        tid = block["id"]

        if name == "route_to_agent":
            agent = inp.get("agent_name")
            if agent not in ROUTABLE_AGENTS:
                tool_results.append(
                    _tool_result_block(
                        tid,
                        {"blocked": True, "reason": f"invalid agent_name: {agent}"},
                    )
                )
                history_extra.append(f"route_to_agent:blocked invalid {agent}")
                continue

            agent = agent  # type: ignore[assignment]

            # Failure 2: routing gate
            if routing_gate_blocks(agent, state):  # type: ignore[arg-type]
                gate[agent] = gate.get(agent, 0) + 1
                updates["gate_blocks"] = dict(gate)
                tool_results.append(
                    _tool_result_block(
                        tid,
                        {
                            "blocked": True,
                            "reason": f"{agent} already success (routing gate)",
                            "gate_blocks": gate[agent],
                        },
                    )
                )
                history_extra.append(
                    f"routing_gate:blocked {agent} (gate_blocks={gate[agent]})"
                )
                continue

            if state["call_counts"].get(agent, 0) >= MAX_CALLS_PER_AGENT:
                tool_results.append(
                    _tool_result_block(
                        tid,
                        {"blocked": True, "reason": "circuit breaker (per-agent max)"},
                    )
                )
                history_extra.append(f"circuit_breaker:blocked {agent}")
                continue

            tool_results.append(_tool_result_block(tid, {"blocked": False, "routed": agent}))
            history_extra.append(
                f"route_to_agent:{agent} reason={inp.get('reason', '')[:80]}"
            )
            pending = agent

        elif name == "signal_complete":
            ok, reason = signal_complete_allowed(state)
            if not ok:
                tool_results.append(
                    _tool_result_block(tid, {"accepted": False, "reason": reason})
                )
                history_extra.append(f"signal_complete:rejected ({reason})")
                updates["supervisor_correction_pending"] = True
                pending = "supervisor"
            else:
                tool_results.append(_tool_result_block(tid, {"accepted": True}))
                history_extra.append("signal_complete:accepted")
                updates["final_answer"] = inp.get("summary", "")
                pending = "finalize"

        elif name == "request_clarification":
            question = inp.get("question", "")
            tool_results.append(_tool_result_block(tid, {"ack": True}))
            history_extra.append(f"request_clarification:{question[:80]}")
            updates["final_answer"] = f"Clarification needed: {question}"
            pending = "finalize"

    updates["supervisor_messages"] = _append_supervisor_exchange(
        sup_msgs, assistant_blocks, tool_results
    )
    updates["routing_history"] = updates["routing_history"] + history_extra
    updates["pending_route"] = pending or "supervisor"
    return updates


def node_finalize(state: GraphState) -> dict[str, Any]:
    if state.get("final_answer"):
        answer = state["final_answer"]
    elif state.get("force_partial_complete"):
        answer = textwrap.dedent(
            f"""
            Partial answer (circuit breaker or forced completion):

            Documents: {textwrap.shorten(state.get('documents_summary') or '(none)', 400)}
            Role requirements: {textwrap.shorten(state.get('role_requirements_summary') or '(none)', 400)}
            Compliance: {textwrap.shorten(state.get('compliance_summary') or '(none)', 400)}

            Agent status: {json.dumps(state['agent_status'])}
            """
        ).strip()
    else:
        body = textwrap.dedent(
            f"""
            Answer using only the summaries below.

            Question: {state['query']}

            DOCUMENTS: {state['documents_summary']}
            ROLE REQUIREMENTS: {state['role_requirements_summary']}
            COMPLIANCE: {state['compliance_summary']}

            State whether the hire is ready to start and list blockers.
            """
        ).strip()
        answer = str(_specialist_llm().invoke(body).content)

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def supervisor_router(state: GraphState) -> str:
    route = state.get("pending_route") or "supervisor"
    if route == "supervisor":
        if state["turn_count"] >= MAX_TURNS_PER_QUERY:
            return "finalize"
        return "supervisor"
    if route in ROUTABLE_AGENTS:
        return route
    if route == "finalize":
        return "finalize"
    return "supervisor"


def build_graph(client: Anthropic):
    def _supervisor(state: GraphState) -> dict[str, Any]:
        return supervisor_node(state, client=client)

    g = StateGraph(GraphState)
    g.add_node("supervisor", _supervisor)
    g.add_node("documents_agent", node_documents_agent)
    g.add_node("role_requirements_agent", node_role_requirements_agent)
    g.add_node("compliance_agent", node_compliance_agent)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "supervisor": "supervisor",
            "documents_agent": "documents_agent",
            "role_requirements_agent": "role_requirements_agent",
            "compliance_agent": "compliance_agent",
            "finalize": "finalize",
        },
    )
    for agent in ROUTABLE_AGENTS:
        g.add_edge(agent, "supervisor")
    g.add_edge("finalize", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def _print_banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def _print_trace(result: GraphState) -> None:
    print("\n--- routing_history ---")
    for line in result["routing_history"]:
        print(f"  {line}")
    print("\n--- call_counts ---")
    print(json.dumps(result["call_counts"], indent=2))
    print("\n--- agent_status ---")
    print(json.dumps(result["agent_status"], indent=2))
    print("\n--- gate_blocks ---")
    print(json.dumps(result.get("gate_blocks", {}), indent=2))
    if result.get("final_answer"):
        print("\n--- final_answer (excerpt) ---")
        print(textwrap.shorten(result["final_answer"], width=700))


def run_demo(
    app,
    title: str,
    query: str,
    *,
    demo_mode: str | None = None,
    agent_status_override: dict[str, str] | None = None,
    supervisor_extra: str | None = None,
    recursion_limit: int = 40,
) -> GraphState:
    _print_banner(title)
    state = initial_state(
        query,
        demo_mode=demo_mode,
        agent_status_override=agent_status_override,
        supervisor_extra=supervisor_extra,
    )
    result = app.invoke(state, config={"recursion_limit": recursion_limit})
    _print_trace(result)
    return result


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and fill the key."
        )

    client = Anthropic()
    app = build_graph(client)

    # 1) Happy path
    run_demo(app, "Demo 1 — Happy path (supervised sequence + signal_complete)", DEMO_SCENARIO)

    # 2) Failure 1 — circuit breaker after repeated documents_agent failures
    run_demo(
        app,
        "Demo 2 — Failure 1: circuit breaker (documents_agent fails twice, then trips)",
        DEMO_SCENARIO,
        demo_mode="circuit_breaker",
        recursion_limit=50,
    )

    # 3) Failure 2 — routing gate blocks re-route to documents_agent already success
    run_demo(
        app,
        "Demo 3 — Failure 2: routing gate (documents_agent pre-marked success)",
        DEMO_SCENARIO,
        demo_mode="routing_gate",
        agent_status_override={"documents_agent": "success"},
        supervisor_extra=(
            "Routing gate drill: call route_to_agent(documents_agent) at least once "
            "even though documents_agent is already success, so the gate can block it. "
            "After repeated blocks, proceed to remaining agents."
        ),
        recursion_limit=50,
    )

    # 4) Compound query (Failure 5)
    run_demo(
        app,
        "Demo 4 — Failure 5: compound query (documents AND compliance)",
        COMPOUND_DEMO_QUERY,
        recursion_limit=50,
    )

    print("\n=== v3 demos complete ===\n")


if __name__ == "__main__":
    main()
