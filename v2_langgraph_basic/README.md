# v2 — LangGraph + typed state + structured summaries

## What this version does

`agent.py` builds a **LangGraph** whose `TypedDict` state carries **`documents_summary`**, **`role_requirements_summary`**, and **`compliance_summary`**—each populated by an LLM call that reads a (still large) mocked tool payload and writes a compact, schema-ish string.

**Invoking the graph**: `routing_hint_prefer_checks` is `NotRequired` in `GraphState`—you can omit it in `app.invoke({...})` and avoid a misleading `KeyError` on first run. No node reads that field anyway; `main()` only sets it so traces show an ignored “prefer compliance only” hint.

That is real progress: summaries are **first-class citizens** of state, not just more chat lines.

## What breaks (the lesson)

- **Hardcoded routing**: edges are wired as `documents → role_requirements → compliance → answer` for **every** query. Nothing branches on intent, ambiguity, missing data, or “we already know enough.”
- **Conditional workflows don’t exist here**: imagine a stakeholder question that should **only** hit compliance—we still churn through documents and role requirements. That wastes tokens, latency, and attention.
- (Updated domain) In this demo, you might want to run only compliance checks—yet we still churn through documents and role requirements because edges are static.
- Operational reality: conditional routing eventually needs guards, thresholds, retries, optional subgraphs—we are not modeling that yet.

## What the next version fixes

**v3** (planned) sketches a **tool supervisor**: a routing layer whose job is to decide *whether* each tool fires, not just to always march down a fixed conveyor belt.
