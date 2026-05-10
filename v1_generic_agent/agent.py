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
    "What is the total cost impact if we delay this shipment by three days?"
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
            "sku_family": "WIDGET-ALPHA" if i % 2 == 0 else "WIDGET-BETA",
            "on_hand": 120 + (i % 37),
            "safety_stock": 40 + (i % 11),
            "atp_date": "2025-06-02" if i % 3 else "2025-06-05",
            "notes": "Promotional buffer reserved for Q3 campaign"
            if i % 5 == 0
            else "Standard rolling forecast",
        }
        for i in range(rows)
    ]


def inventory_tool(_: dict[str, Any]) -> dict[str, Any]:
    """Warehouse / ATP style facts—far more detail than the final answer needs."""
    return {
        "shipment_id": "SHP-99821",
        "promise_date_baseline": "2025-06-04",
        "promise_date_plus_3d": "2025-06-07",
        "skus": _chunky_records("inv", 18),
        "risk_flags": [
            "lane_constrained_drayage",
            "promo_allocation_overlap",
        ],
        "analyst_commentary": textwrap.dedent(
            """
            Inventory is *technically* sufficient for a 3-day slip on paper, but
            several SKUs share promotional allocation pools. The raw ATP table
            often disagrees with marketing holds—watch row-level notes.
            """
        ).strip(),
    }


def logistics_tool(_: dict[str, Any]) -> dict[str, Any]:
    """Carrier + detention style facts."""
    return {
        "shipment_id": "SHP-99821",
        "mode": "ocean_fcl",
        "baseline_demurrage_curve": [
            {"day": 0, "est_usd": 0},
            {"day": 1, "est_usd": 850},
            {"day": 2, "est_usd": 1700},
            {"day": 3, "est_usd": 2650},
        ],
        "per_diem_reefer": 175,
        "appointments": _chunky_records("appt", 12),
        "carrier_messages": [
            "If we slide 3d, we likely miss the Friday gate; weekend storage applies.",
            "Alternate routing exists but needs manual approval (not modeled here).",
        ],
    }


def contract_tool(_: dict[str, Any]) -> dict[str, Any]:
    """SLA / chargeback clauses—dense text-ish JSON."""
    return {
        "shipment_id": "SHP-99821",
        "customer_id": "CUST-7712",
        "sla": {
            "on_time_delivery": "must_arrive_by_2025_06_05_local",
            "penalty_formula": "2.5% of line value per business day late, capped at 15%",
            "exceptions": ["force_majeure", "customer_caused_delay"],
        },
        "clause_snippets": [
            textwrap.dedent(
                """
                Section 9.4: Carrier demurrage, detention, and per-diem passes through
                to Customer unless Delay is attributable to Supplier manufacturing.
                Allocation disputes between programs are borne by Supplier.
                """
            ).strip(),
            textwrap.dedent(
                """
                Exhibit C: Expedite is pre-authorized up to $4,500 if required to
                recover OTD; amounts above require VP approval within 4 business hours.
                """
            ).strip(),
        ],
        "line_value_usd": 185_000,
    }


TOOL_REGISTRY = {
    "inventory_tool": inventory_tool,
    "logistics_tool": logistics_tool,
    "contract_tool": contract_tool,
}

TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "inventory_tool",
        "description": "Fetch noisy inventory/ATP details for a shipment delay analysis.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "logistics_tool",
        "description": "Fetch carrier, demurrage, and appointment noise for delay analysis.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "contract_tool",
        "description": "Fetch contractual penalties and pass-through clauses.",
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
    You are an operations analyst helping answer delay-cost questions.

    You have three tools. Use them when you need facts. Be concise in your
    final natural-language answer, but do not omit material $ impact drivers.

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
    _print_banner("v1 — Supervisor ping: 'Please double-check inventory exposure'")
    second_assistant = run_turn(
        client,
        state,
        extra_user_nudge=(
            "Supervisor request: double-check whether inventory exposure actually "
            "supports a 3-day slip without an allocation conflict. "
            "If needed, call tools again."
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
