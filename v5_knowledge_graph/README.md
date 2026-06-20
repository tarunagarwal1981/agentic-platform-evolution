# v5 — Knowledge Graph Layer

## What this version solves

The plan-based supervisor (v4) exposes a ceiling when facts have
relationships that flat state can't represent. Finalize synthesizing
answers from natural language summaries has no provenance — you can't
tell whether a number was produced by an agent or invented in synthesis.

This version introduces typed facts with declared producers, freshness
windows, confidence scores, and deterministic comparison rules.

## The core shift

Before: agents write natural language summaries to state.
After: agents write typed facts to a fact store. Synthesis reads
facts by ID and applies deterministic rules. No reconstruction
from prose.

## The failure that stayed behind a flag

KG_LLM_FACT_EXTRACTION is permanently OFF. Asking one model call
to classify a workflow AND extract fact IDs across two domain
ontologies simultaneously produces fact leakage on compound queries.
The deterministic alias matcher handles fact lookup instead.

## Five capability flags (all live in production)

- KG_FACT_WRITES_ENABLED — agents write typed facts as they run
- KG_PLANNER_ENABLED — planner checks fact freshness before re-running agents
- KG_RENDER_BLOCKS_ENABLED — display layer renders facts with provenance
- KG_SYNTHESIS_RULES_ENABLED — deterministic rules for date arithmetic
  and threshold checks
- KG_CONFIDENCE_EVALUATOR_ENABLED — confidence pass before finalize

## Status

Stub. Runnable implementation coming with the next series post.

## How to follow along

Read the companion Medium post:
[When a Sequence Is Not Enough](#) ← update with live URL after publish

The series starts at v1:
[I Built the Same AI Agent Four Ways](https://tarunaga.medium.com/i-built-the-same-ai-agent-four-ways-heres-what-each-version-couldn-t-do-cf90c6b19d22)
