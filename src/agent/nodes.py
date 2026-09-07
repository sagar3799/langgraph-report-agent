"""Real node implementations: Qdrant retrieval, Gemini grading/report-writing, tool calls."""

import logging

from agent.llm import StructuredOutputError, generate_structured, get_chat_model
from agent.retrieval.qdrant_store import search
from agent.schemas import GradeResult, Report, ReportSection
from agent.state import AgentState
from agent.tools.calculator import CalculatorError, calculate
from agent.tools.web_search import WebSearchError, web_search

logger = logging.getLogger(__name__)

MAX_LOOPS = 3
TOP_K = 4


def retrieve_node(state: AgentState) -> dict:
    docs = search(state["question"], top_k=TOP_K)
    return {"retrieved_docs": docs}


def _format_context(state: AgentState) -> str:
    doc_lines = [f"[{d['source']}] {d['text']}" for d in state.get("retrieved_docs", [])]
    tool_lines = state.get("tool_results", [])
    parts = []
    if doc_lines:
        parts.append("Retrieved documents:\n" + "\n\n".join(doc_lines))
    if tool_lines:
        parts.append("Prior tool results:\n" + "\n\n".join(tool_lines))
    return "\n\n".join(parts) if parts else "(no context retrieved yet)"


def grade_node(state: AgentState) -> dict:
    loop_count = state.get("loop_count", 0)
    llm = get_chat_model()

    try:
        grade = generate_structured(
            llm,
            system_prompt=(
                "You grade whether the given context is sufficient to write a confident, accurate "
                "answer to the user's question. Be strict: if the context doesn't directly address "
                "the question, mark it insufficient. Prefer 'web_search' for missing facts/current "
                "info, 'calculator' only when the question needs arithmetic on numbers already "
                "present in the context."
            ),
            user_prompt=f"Question: {state['question']}\n\nContext:\n{_format_context(state)}",
            schema=GradeResult,
        )
    except StructuredOutputError as exc:
        logger.warning("Grading failed, defaulting to insufficient: %s", exc)
        return {
            "sufficient": False,
            "grade_reason": "grading call failed; defaulting to insufficient",
            "next_tool": "none",
            "tool_input": "",
            "loop_count": loop_count + 1,
        }

    next_tool = grade.next_tool
    tool_input = grade.tool_input
    if not grade.sufficient and next_tool == "none":
        # Contradictory grade (insufficient but no tool picked) — don't waste a loop on a no-op.
        next_tool = "web_search"
        tool_input = state["question"]

    return {
        "sufficient": grade.sufficient,
        "grade_reason": grade.reason,
        "next_tool": next_tool,
        "tool_input": tool_input,
        "loop_count": loop_count + 1,
    }


def call_tool_node(state: AgentState) -> dict:
    tool_name = state.get("next_tool", "none")
    tool_input = state.get("tool_input", "")

    if tool_name == "web_search":
        try:
            results = web_search(tool_input)
            result_text = "\n".join(f"{r['title']}: {r['snippet']} ({r['url']})" for r in results)
        except WebSearchError as exc:
            result_text = f"web_search error: {exc}"
    elif tool_name == "calculator":
        try:
            result_text = f"{tool_input} = {calculate(tool_input)}"
        except CalculatorError as exc:
            result_text = f"calculator error: {exc}"
    else:
        result_text = "no tool call was made"

    return {
        "tool_calls_made": [tool_name],
        "tool_results": [result_text],
    }


def route_after_grade(state: AgentState) -> str:
    if state.get("sufficient"):
        return "write_report"
    if state.get("loop_count", 0) >= MAX_LOOPS:
        return "write_report"
    return "call_tool"


def write_report_node(state: AgentState) -> dict:
    llm = get_chat_model()
    sources = sorted({d["source"] for d in state.get("retrieved_docs", [])})

    try:
        report = generate_structured(
            llm,
            system_prompt=(
                "You write a structured report answering the user's question using only the given "
                "context. If the context doesn't fully answer it, say so explicitly in the summary "
                "and set confidence to 'low'. Never invent facts not present in the context."
            ),
            user_prompt=f"Question: {state['question']}\n\nContext:\n{_format_context(state)}",
            schema=Report,
        )
        return {"report": report}
    except StructuredOutputError as exc:
        logger.warning("Report generation failed after retry, returning degraded report: %s", exc)
        fallback = Report(
            title="Report generation failed",
            summary=(
                "The model could not produce a schema-valid report after one retry. "
                "See sections for the raw context that was available."
            ),
            sections=[ReportSection(heading="Available context", content=_format_context(state))],
            sources=sources,
            confidence="low",
        )
        return {"report": fallback}
