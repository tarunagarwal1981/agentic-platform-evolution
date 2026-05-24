#!/usr/bin/env python3
"""
v4 — Plan supervisor pattern (STUB)

What this pattern solves
------------------------
**Planning before (or interleaved with) tool use** helps when tasks require
hypothesis → evidence → revise cycles. A plan supervisor owns a lightweight
working plan: assumptions, unknowns, next evidence requests, and termination
conditions. Tool routing becomes a consequence of the plan, not the only control
structure.

What is coming in a future revision of this repo
--------------------------------------------------
A LangGraph sketch that:
- emits a short structured plan from the stakeholder question,
- executes plan steps through tool-equipped subgraphs,
- reconciles plan vs evidence (including explicit "we were wrong" updates).

v3 focuses on *which tool*; v4 focuses on *why we are calling anything at all*.
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
