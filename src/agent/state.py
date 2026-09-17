"""Shared graph state — the 'notebook' passed between nodes."""

import operator
from typing import Annotated, TypedDict

from agent.schemas import Report, SectionPlan


class RetrievedDoc(TypedDict):
    id: str
    text: str
    source: str
    score: float
    # Sigmoid-normalized `score` -- a [0,1] heuristic confidence signal, NOT a calibrated
    # probability (raw cross-encoder scores run roughly -10 to +7 in testing, unbounded).
    normalized_confidence: float


class AgentState(TypedDict, total=False):
    question: str
    # Cross-section evidence pool from a research run (see ResearchState.evidence_pool below),
    # keyed by chunk id. Merged into this section's rerank candidates by retrieve_node so two
    # sections don't independently cite the same chunk as if it were two different sources.
    # Empty/absent for a standalone single-question run -- this field is optional on purpose.
    memory_pool: dict[str, RetrievedDoc]
    retrieved_docs: list[RetrievedDoc]
    sufficient: bool
    grade_reason: str
    next_tool: str
    tool_input: str
    tool_calls_made: Annotated[list[str], operator.add]
    tool_results: Annotated[list[str], operator.add]
    loop_count: int
    # Set only when run_sections() is regenerating a section that failed evidence verification --
    # fed into grading/report-writing context so the retry doesn't repeat the same unsupported
    # claims. Empty/absent otherwise.
    verifier_feedback: str
    report: Report
    trace: Annotated[list[str], operator.add]


class SectionVerification(TypedDict):
    total_claims: int
    unsupported_claims: list[str]
    coverage: float  # (total_claims - len(unsupported_claims)) / total_claims, clamped [0,1]
    retries_used: int


class SectionResult(TypedDict):
    title: str
    sub_question: str
    report: Report
    retrieved_docs: list[RetrievedDoc]
    tool_results: list[str]
    verification: SectionVerification


class ResearchState(TypedDict, total=False):
    """State for the topic-level research graph: planner -> per-section subgraph -> assemble."""

    topic: str
    sections: list[SectionPlan]
    # Shared research memory: every chunk retrieved by any section so far, keyed by chunk id.
    evidence_pool: dict[str, RetrievedDoc]
    section_results: Annotated[list[SectionResult], operator.add]
    # Sections run sequentially by default (max_workers=1), matching the project's rate-limit
    # budgeting. >1 fans sections out across threads -- architected, not load-tested; see
    # run_sections() in nodes.py.
    max_workers: int
    report: Report
    # Self-critic loop, mirroring the grade/call_tool loop in AgentState above but operating on
    # the whole assembled report instead of a single question.
    critic_accepted: bool
    critic_issues: list[str]
    critic_loop_count: int
    trace: Annotated[list[str], operator.add]
