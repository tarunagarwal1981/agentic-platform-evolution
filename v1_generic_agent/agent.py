#!/usr/bin/env python3
"""
v1 — Generic Anthropic tool-calling loop (NO LangGraph).

Teaching goal
-------------
Demonstrate **context collapse** when **raw tool payloads** are pasted straight
into the conversational transcript. The model still "works," but the thread
state balloons and you often see **repeated / low-value tool calls** because
reasoning over a giant, undifferentiated blob is hard.

Run:  python agent.py   (from this directory or with PYTHONPATH set)
"""

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Demo scenario (fixed across all versions)
# ---------------------------------------------------------------------------

DEMO_SCENARIO = (
    "Is this new hire fully compliant and ready to start given their documents and role requirements?"
)

MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Mock enterprise tools — intentionally verbose on purpose
# ---------------------------------------------------------------------------

def _chunky_records(prefix: str, rows: int) -> list[dict[str, Any]]:
    """Manufacture plausible-looking noise so token pressure is obvious."""
    return [
        {
            "row_id": f"{prefix}-{i:04d}",
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
            "verification_status": (
                "verified" if i % 5 else "needs_review"
            ),
            "verification_source": (
                "vendor_api" if i % 2 == 0 else "manual_review"
            ),
            "notes": (
                "Image quality borderline; reviewer requested re-upload"
                if i % 5 == 0
                else "Standard onboarding packet item"
            ),
        }
        for i in range(rows)
    ]


def documents_tool(_: dict[str, Any]) -> dict[str, Any]:
    """Document intake + verification facts—far more detail than the final answer needs."""
    return {
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
        "documents": _chunky_records("doc", 18),
        "risk_flags": [
            "work_authorization_expires_within_12_months",
            "background_check_vendor_delay",
        ],
        "analyst_commentary": textwrap.dedent(
            """
            Document completeness looks close on paper, but the raw verification
            stream is noisy. Several items are in "needs_review" due to image
            quality or vendor latency—watch row-level notes and expiry dates.
            """
        ).strip(),
    }


def role_requirements_tool(_: dict[str, Any]) -> dict[str, Any]:
    """Role + department requirements—noisy checklist facts."""
    return {
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
        "onboarding_checklist_items": [
            {
                "item": f"checklist_item_{i}",
                "status": "complete" if i % 4 else "pending",
                "owner": "IT" if i % 3 == 0 else "HR",
                "notes": "Waiting on manager approval" if i % 5 == 0 else "",
            }
            for i in range(12)
        ],
        "department_messages": [
            "Role requires NDA + IP assignment before repository access is granted.",
            "Security training must be completed by end of first week.",
        ],
    }


def compliance_tool(_: dict[str, Any]) -> dict[str, Any]:
    """Compliance frameworks + liability clauses—dense text-ish JSON."""
    return {
        "compliance_framework": "employment_onboarding_controls_v2",
        "jurisdictions": [
            {
                "country": "US",
                "requirements": [
                    "I-9 completed within 3 business days of start date",
                    "Background check must be adjudicated before start for restricted roles",
                ],
                "penalties": [
                    {"type": "civil_fine", "range_usd": [250, 2500]},
                    {"type": "audit_risk", "notes": "missing I-9 increases audit exposure"},
                ],
            },
            {
                "country": "DE",
                "requirements": [
                    "Data protection acknowledgement (GDPR) signed",
                    "Works council notification for certain departments (if applicable)",
                ],
                "penalties": [
                    {
                        "type": "liability",
                        "notes": "access before GDPR acknowledgement increases liability",
                    }
                ],
            },
        ],
        "penalty_for_non_compliant_hire": {
            "summary": "Fines + legal exposure + policy breach escalation",
            "internal_consequence": "security exception ticket + VP approval required",
        },
        "clause_snippets": [
            textwrap.dedent(
                """
                Policy 3.1: A new hire may not start work until identity and work
                authorization are verified per jurisdiction. Any exception must be
                documented with compensating controls and approved by HR + Legal.
                """
            ).strip(),
            textwrap.dedent(
                """
                Clause 7.2 (Liability): If access is granted prior to completion of
                mandatory onboarding controls, the company assumes heightened liability
                for data handling incidents. Penalties may include termination for cause
                of responsible approvers.
                """
            ).strip(),
        ],
    }


TOOL_REGISTRY = {
    "documents_tool": documents_tool,
    "role_requirements_tool": role_requirements_tool,
    "compliance_tool": compliance_tool,
}

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "documents_tool",
        "description": "Fetch noisy new-hire document intake and verification details.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "role_requirements_tool",
        "description": "Fetch noisy role requirements, certifications, and clearance checklist.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compliance_tool",
        "description": "Fetch onboarding compliance rules, jurisdictional requirements, and liability clauses.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# Agent state (the thing we are critiquing)
# ---------------------------------------------------------------------------


@dataclass
class ThreadState:
    """Everything is 'just messages' — classic v1."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_roundtrips: int = 0

    def transcript_char_count(self) -> int:
        return len(json.dumps(self.messages))


def handle_tool_calls(
    assistant_msg: dict[str, Any],
    state: ThreadState,
) -> None:
    """Execute tools and append **raw JSON** results into the thread."""
    for block in assistant_msg.get("content", []):
        if block.get("type") != "tool_use":
            continue
        name = block["name"]
        tool_input = block.get("input", {}) or {}
        tool_fn = TOOL_REGISTRY[name]
        payload = tool_fn(tool_input)
        state.tool_roundtrips += 1
        # v1 smell: paste the entire blob back; no summaries, no typed state slots.
        state.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(payload, indent=2),
                    }
                ],
            }
        )


def run_turn(
    client: Anthropic,
    state: ThreadState,
    extra_user_nudge: str | None = None,
) -> dict[str, Any]:
    """One model call + optional tool execution loop (single 'turn' bundle)."""
    if extra_user_nudge:
        state.messages.append({"role": "user", "content": extra_user_nudge})

    assistant = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=TOOLS_SPEC,
        messages=_only_user_assistant_blocks(state.messages),
    )

    # Anthropic returns a list of blocks; map to plain dict for re-submission.
    assistant_msg = {
        "role": "assistant",
        "content": [block.model_dump() for block in assistant.content],
    }
    state.messages.append(assistant_msg)

    stop_reason = assistant.stop_reason
    while stop_reason == "tool_use":
        handle_tool_calls(assistant_msg, state)
        assistant = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SPEC,
            messages=_only_user_assistant_blocks(state.messages),
        )
        assistant_msg = {
            "role": "assistant",
            "content": [block.model_dump() for block in assistant.content],
        }
        state.messages.append(assistant_msg)
        stop_reason = assistant.stop_reason

    return assistant_msg


def _only_user_assistant_blocks(
    messages: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Anthropic Messages API wants user/assistant turns.
    tool_result blocks ride inside 'user' messages — keep as-is.
    """
    return list(messages)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an HR onboarding analyst helping determine if a new hire is fully compliant and ready to start.

    You have three tools. Use them when you need facts. Be concise in your
    final natural-language answer, but do not omit material compliance risks or readiness blockers.

    Important nuance (for the demo): after you have enough information, STOP
    calling tools. If you are uncertain, prefer *re-reading prior tool output*
    over calling the same tool again with the same intent.
    """
).strip()


def _print_banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY. Copy .env.example to .env and fill the key."
        )

    client = Anthropic(api_key=api_key)
    state = ThreadState()
    state.messages.append({"role": "user", "content": DEMO_SCENARIO})

    _print_banner("v1 — First wave (expect three tool calls)")
    first_assistant = run_turn(client, state)
    _log_state("After first wave", state, first_assistant)

    # Deliberate second user nudge: mimics a 'supervisor' ping in chat products.
    # With a bloated transcript, models often re-call tools instead of citing prior JSON.
    _print_banner(
        "v1 — Supervisor ping: 'Please double-check onboarding compliance documents'"
    )
    second_assistant = run_turn(
        client,
        state,
        extra_user_nudge=(
            "Supervisor request: double-check whether all required compliance documents are actually "
            "present and verified. If needed, call tools again."
        ),
    )
    _log_state("After supervisor ping", state, second_assistant)

    print("\n--- Takeaway ---")
    print(
        "If you see **extra tool calls** above, that's the story: huge raw blobs in "
        "the thread make stable recall hard, so the stack 're-grounds' via tools."
    )


def _log_state(label: str, state: ThreadState, assistant_msg: dict[str, Any]) -> None:
    text = _assistant_text(assistant_msg)
    print(f"\n[{label}] transcript chars ≈ {state.transcript_char_count():,}")
    print(f"[{label}] tool round-trips so far: {state.tool_roundtrips}")
    if text:
        print(f"\n[{label}] assistant text preview:\n{textwrap.shorten(text, width=700)}")


def _assistant_text(assistant_msg: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in assistant_msg.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


if __name__ == "__main__":
    main()
