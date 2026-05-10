#!/usr/bin/env python3
"""
v2 — LangGraph with typed state + structured summaries.

Teaching goal
-------------
Show a real upgrade: instead of stuffing raw JSON into one growing chat, each
domain gets a **short, labeled summary** saved into **typed state fields**.

But also show the next failure mode: **static edges**. This graph *always*
executes documents → role_requirements → compliance → final answer—even when a different
route would be more appropriate (we carry a hypothetical flag that nothing
respects).

Run: python agent.py
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired


# Demo scenario mirrors v1 exactly.
DEMO_SCENARIO = (
    "Is this new hire fully compliant and ready to start given their documents and role requirements?"
)

MODEL = "claude-sonnet-4-20250514"


class GraphState(TypedDict):
    """
    Typed LangGraph state (the v2 upgrade).

    The `routing_hint_*` keys are **never read by the graph** — they exist to
    make the routing gap obvious in traces and debugger output.

    `routing_hint_prefer_checks` is optional: omit it when calling ``invoke``
    if you only need ``messages``, ``query``, and the summary fields (nothing
    in this graph reads it, so absent vs present behaves the same at runtime).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str

    # What we wish we could exploit for conditional routing (but don't):
    routing_hint_prefer_checks: NotRequired[
        Literal["all", "compliance_only", "documents_only"]
    ]

    documents_summary: str
    role_requirements_summary: str
    compliance_summary: str
    final_answer: str


def _llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=MODEL,
        temperature=0,
        max_tokens=1200,
    )


# ---------------------------------------------------------------------------
# Mock tools (same universe as v1, but we only show summaries to later nodes)
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
                "Policy 3.1: identity + work authorization must be verified before start; exceptions require HR+Legal approval.",
                "Clause 7.2: granting access prior to mandatory controls increases liability for data incidents.",
            ],
            "noise": "long policy text elided in JSON but imagine multiple pages",
        },
        indent=2,
    )


def summarize_domain(
    domain: str,
    query: str,
    raw_blob: str,
) -> str:
    """Force the model to compress tool output into a stable, human-legible shape."""
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
        DOLLAR_DRIVERS:
        OPEN_QUESTIONS:
        """
    ).strip()
    return str(_llm().invoke(prompt).content)


def node_documents(state: GraphState) -> dict:
    raw = _big_documents_blob()
    summary = summarize_domain("documents / verification", state["query"], raw)
    return {
        "documents_summary": summary,
        "messages": [
            AIMessage(
                content="(v2) documents node wrote `documents_summary` "
                "(raw blob not stored in state)."
            )
        ],
    }


def node_role_requirements(state: GraphState) -> dict:
    raw = _big_role_requirements_blob()
    summary = summarize_domain("role requirements / clearances", state["query"], raw)
    return {
        "role_requirements_summary": summary,
        "messages": [
            AIMessage(
                content="(v2) role requirements node wrote `role_requirements_summary`."
            )
        ],
    }


def node_compliance(state: GraphState) -> dict:
    raw = _big_compliance_blob()
    summary = summarize_domain("compliance / jurisdictional rules", state["query"], raw)
    return {
        "compliance_summary": summary,
        "messages": [
            AIMessage(content="(v2) compliance node wrote `compliance_summary`.")
        ],
    }


def node_final_answer(state: GraphState) -> dict:
    """Only sees summaries — the v2 win."""
    body = textwrap.dedent(
        f"""
        Answer the stakeholder using **only** the summaries below.

        Question: {state["query"]}

        DOCUMENTS SUMMARY:
        {state["documents_summary"]}

        ROLE REQUIREMENTS SUMMARY:
        {state["role_requirements_summary"]}

        COMPLIANCE SUMMARY:
        {state["compliance_summary"]}

        Output:
        1) Whether the new hire is ready to start (yes/no/conditional) with blockers
        2) What would change the decision and what to verify next
        """
    ).strip()
    answer = str(_llm().invoke(body).content)
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def build_graph():
    """
    **Static graph** on purpose — v2's pedagogical scar.

    There is zero `add_conditional_edges`: every execution path is identical.
    """
    g = StateGraph(GraphState)
    g.add_node("documents", node_documents)
    g.add_node("role_requirements", node_role_requirements)
    g.add_node("compliance", node_compliance)
    g.add_node("answer", node_final_answer)

    g.add_edge(START, "documents")
    g.add_edge("documents", "role_requirements")
    g.add_edge("role_requirements", "compliance")
    g.add_edge("compliance", "answer")
    g.add_edge("answer", END)
    return g.compile()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and fill the key."
        )

    app = build_graph()

    result = app.invoke(
        {
            "messages": [],
            "query": DEMO_SCENARIO,
            # This flag is intentionally ignored by routing — try switching it in
            # your head; the graph will not care. That's the v2 lesson.
            "routing_hint_prefer_checks": "compliance_only",
            "documents_summary": "",
            "role_requirements_summary": "",
            "compliance_summary": "",
            "final_answer": "",
        }
    )

    print("=== v2 LangGraph run complete ===\n")
    print(
        "Notice: `routing_hint_prefer_checks=compliance_only` yet we still ran "
        "documents → role_requirements → compliance because edges are hardcoded.\n"
    )
    print("--- documents_summary (excerpt) ---")
    print(textwrap.shorten(result["documents_summary"], width=900))
    print("\n--- role_requirements_summary (excerpt) ---")
    print(textwrap.shorten(result["role_requirements_summary"], width=900))
    print("\n--- compliance_summary (excerpt) ---")
    print(textwrap.shorten(result["compliance_summary"], width=900))
    print("\n=== final_answer ===\n")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
