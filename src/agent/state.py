"""Shared graph state — the 'notebook' passed between nodes."""

import operator
from typing import Annotated, TypedDict

from agent.schemas import Report


class RetrievedDoc(TypedDict):
    text: str
    source: str
    score: float


class AgentState(TypedDict, total=False):
    question: str
    retrieved_docs: list[RetrievedDoc]
    sufficient: bool
    grade_reason: str
    next_tool: str
    tool_input: str
    tool_calls_made: Annotated[list[str], operator.add]
    tool_results: Annotated[list[str], operator.add]
    loop_count: int
    report: Report
