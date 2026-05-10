# v1 — Generic agent (raw tool output in state)

## What this version does

`agent.py` runs a **single conversational loop** around the Anthropic Messages API. When the model chooses tools, we execute `inventory_tool`, `logistics_tool`, and `contract_tool`, then drop the **entire raw JSON** back into the thread as “tool results.”

There is **no LangGraph**: just messages + tool I/O. That is the point—most teams start here.

## What breaks (the lesson)

- **State balloons**: every tool returns a intentionally chatty payload. Nothing compacts it for the next turn.
- **Context collapse**: as the thread grows, the model’s effective “working memory” for details erodes; you see **noisy or repeated tool calls** (we nudge the demo so a second wave of tool calls is likely).
- **No supervision layer**: there is no graph—only reactive tool use—so behavior is brittle when tools disagree or when you need phased reasoning.

## What the next version fixes

**v2** introduces **LangGraph** plus **typed state** and **short, structured summaries** written to explicit fields instead of dumping raw payloads into one undifferentiated transcript.
