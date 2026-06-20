#!/usr/bin/env python3
"""
v5 — Knowledge graph layer (STUB)

What this pattern solves
------------------------
The plan-based supervisor (v4) works well when facts are independent.
It breaks when the answer depends on **relationships between facts** —
not just what each agent found, but how those findings relate to each
other.

Flat state has no provenance. When finalize synthesizes an answer from
three natural language summaries, there is no way to tell whether a
number came from a verified agent output or was invented by the
synthesis step. Date arithmetic, threshold checks, and gap calculations
done in natural language are unreliable.

The knowledge graph layer fixes this by replacing flat summaries with
typed facts. Every fact has:
- A declared producer agent
- A freshness window
- A confidence score
- A set of aliases for deterministic lookup
- A derivation chain (what it was derived from)

Synthesis reads facts by ID and applies deterministic rules.
It does not reconstruct relationships from prose.

What is coming in a future revision of this repo
--------------------------------------------------
A runnable implementation that demonstrates:

- Typed fact writes from specialist agents into a fact store
  (KG_FACT_WRITES_ENABLED)
- Planner reads fact freshness before deciding whether to re-run
  an agent (KG_PLANNER_ENABLED)
- Deterministic comparison rules for date arithmetic, threshold
  checks, and gap calculations — no LLM in the loop
  (KG_SYNTHESIS_RULES_ENABLED)
- Confidence evaluator before finalize runs, checking every required
  fact meets threshold for the query's risk level
  (KG_CONFIDENCE_EVALUATOR_ENABLED)
- Deterministic alias matcher for fact lookup — word-boundary regex
  against fact aliases, ~1ms, no LLM involvement

The failure mode this version is built around: the dual-task problem.
Asking one model call to simultaneously classify a workflow AND extract
relevant fact IDs across two domain ontologies produces fact leakage on
compound queries. The fix is separation: LLM classifies the workflow,
deterministic alias matcher finds the facts. Two concerns, two
mechanisms.

Same demo scenario throughout: onboarding compliance check.
'Is this new hire fully compliant and ready to start given their
documents and role requirements?'
"""
from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
