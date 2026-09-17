"""Structured (Pydantic) contracts for LLM outputs and tool results."""

from typing import Literal

from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    """Output of the 'is this enough info?' grading step."""

    sufficient: bool = Field(description="True if the retrieved context can answer the question")
    reason: str = Field(description="One-sentence justification for the sufficiency verdict")
    next_tool: Literal["web_search", "calculator", "none"] = Field(
        default="none",
        description="Which tool to call next if sufficient is False. 'none' if sufficient is True.",
    )
    tool_input: str = Field(
        default="",
        description=(
            "Input to pass to next_tool: a search query string for web_search, or an arithmetic "
            "expression for calculator. Empty if next_tool is 'none'."
        ),
    )


class ReportSection(BaseModel):
    heading: str
    content: str


class Report(BaseModel):
    """Final structured output of the agent."""

    title: str
    summary: str = Field(description="2-3 sentence executive summary.")
    sections: list[ReportSection]
    sources: list[str] = Field(default_factory=list, description="Doc IDs or URLs used as evidence")
    confidence: Literal["low", "medium", "high"]
    # Populated only for an assembled multi-section research report (see assemble_report_node);
    # None/empty for a plain single-question Report, which isn't run through the Verifier.
    evidence_coverage: float | None = Field(
        default=None,
        description="Fraction of factual claims across the report traced to cited evidence "
        "(0-1). None if the report wasn't run through evidence verification.",
    )
    source_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-source confidence in [0,1] -- a sigmoid-normalized reranker score for "
        "knowledge-base sources, or a fixed baseline for web sources lacking a per-item "
        "relevance score. A heuristic signal, not a calibrated probability.",
    )


class SectionPlan(BaseModel):
    """One planned section of a multi-section research report."""

    title: str = Field(description="Short section heading")
    sub_question: str = Field(
        description="A specific, self-contained question this section should answer"
    )


class Plan(BaseModel):
    """Output of the planner: an ordered list of sections for a research topic."""

    sections: list[SectionPlan] = Field(
        description="3-5 sections that together cover the topic without major overlap"
    )


class VerificationResult(BaseModel):
    """Output of the evidence verifier: does every claim in a section trace to its evidence?"""

    total_claims: int = Field(
        description="Total number of distinct factual claims made in the section (don't count "
        "generic framing sentences, only concrete facts/figures/assertions)"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims from the section NOT directly backed by the given evidence. "
        "Empty if every claim is supported.",
    )


class CriticResult(BaseModel):
    """Output of the self-critic: a holistic pass over the ASSEMBLED report that per-section
    verification structurally can't do -- repetition across sections, thin/missing coverage of
    a planned section, contradictions between sections."""

    accepted: bool = Field(
        description="True if the report is complete, non-repetitive, and internally consistent"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific whole-report problems found (e.g. 'Sections X and Y both cover "
        "the same point about Z'). Empty if accepted.",
    )
