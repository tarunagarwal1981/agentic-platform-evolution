# Agentic Platform Evolution

A small Python tour of how **agent architecture** evolves when you move from “one loop with tools” toward **graphs, supervision, and planning**—without pretending any of these snippets are production systems.

Every version answers the **same operational question** so you can compare behavior and failure modes apples-to-apples:

> “Is this new hire fully compliant and ready to start given their documents and role requirements?”

## The four versions

| Version | Folder | Idea | What breaks (on purpose) |
|--------:|--------|------|--------------------------|
| **v1** | `v1_generic_agent/` | Generic tool-calling agent; raw tool blobs land directly in conversational state. | **Context collapse**: state balloons, reasoning degrades, the “supervisor” re-invokes tools. |
| **v2** | `v2_langgraph_basic/` | **LangGraph** with a **typed state** and **structured summaries** per domain. | **Static edges**: every run follows the same node path—no conditional routing when the query only needs part of the world. |
| **v3** | `v3_tool_supervisor/` | *(Stub)* Tool-calling supervisor that decides *which tools* run and when. | Coming soon in code; see folder README for the narrative. |
| **v4** | `v4_plan_supervisor/` | *(Stub)* Plan-first supervisor: decompose, execute, reconcile. | Coming soon in code; see folder README for the narrative. |

## Model

Examples use **Anthropic Claude** `claude-sonnet-4-20250514`. Set `ANTHROPIC_API_KEY` (see `.env.example`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your real key
python v1_generic_agent/agent.py
python v2_langgraph_basic/agent.py
```

Read each version’s `README.md` for what to watch for in the logs.

## Further reading

- **Medium article (placeholder):** [Medium Post Link](https://medium.com/) — swap in the real URL when the post is live.

This repo is **illustrative**: verbose comments, exaggerated payloads, and taught failure modes are there to support a article walkthrough—not to ship to prod.
