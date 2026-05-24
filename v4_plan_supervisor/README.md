# v4 — Plan supervisor *(coming soon in code)*

## What this version will do (conceptually)

Add a **planning** layer ahead of (or intertwined with) tool calls:

1. Build a lightweight plan from the stakeholder question (assumptions, unknowns, needed evidence).
2. Execute plan steps via tools/subgraphs.
3. Reconcile: update the plan when evidence contradicts assumptions; surface confidence and remaining risks.

This is the natural next step after v3’s “which tool next?” supervisor.

## What still breaks in the real world (honest limits)

Even plan supervision needs:

- evals and golden tasks,
- policy constraints (PII, spend limits),
- human-in-the-loop checkpoints for irreversible actions.

This repo keeps those out of scope on purpose.

## End of the series (for this repo)

If you have followed v1 → v2 → v3 → v4 in prose and code, you have a mental model for **state hygiene**, **graph structure**, **tool supervision**, and **planning**—the four pressure points that show up in almost every serious agent build.
