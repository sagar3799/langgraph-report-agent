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
