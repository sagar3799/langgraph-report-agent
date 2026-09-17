"""Graph wiring:
- `build_graph()`: retrieve -> grade -> (loop: call_tool -> grade) -> write_report. The
  original single-question graph; also reused, unchanged, as the per-section subgraph inside
  the research graph below (see run_sections() in nodes.py).
- `build_research_graph()`: planner -> [human_review, if enable_hitl] -> run_sections (fans out
  to build_graph() once per planned section, verifying each section's evidence support as it
  goes) -> assemble_report -> critic (loop: revise_report -> critic, bounded), a holistic pass
  over the whole assembled report that per-section verification structurally can't do
  (repetition across sections, thin coverage of a planned section, contradictions). Turns a
  topic into a self-reviewed multi-section report.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from agent.nodes import (
    assemble_report_node,
    call_tool_node,
    critic_node,
    grade_node,
    human_review_node,
    planner_node,
    retrieve_node,
    revise_report_node,
    route_after_critic,
    route_after_grade,
    run_sections_node,
    write_report_node,
)
from agent.state import AgentState, ResearchState

# The default checkpoint serializer warns (soon: blocks) on deserializing custom types it
# doesn't recognize -- our Pydantic schemas stored directly in ResearchState (SectionPlan,
# edited by a human during the HITL pause; Report, read back if a run is ever resumed after
# report-writing) trip that check. Allowlisting them explicitly here is the documented fix,
# found by actually running scripts/research_hitl_demo.py and reading the warning it printed.
_CHECKPOINT_ALLOWED_MODULES = [
    ("agent.schemas", "SectionPlan"),
    ("agent.schemas", "Report"),
    ("agent.schemas", "ReportSection"),
]


def _default_checkpointer() -> InMemorySaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MODULES)
    return InMemorySaver(serde=serde)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_node("write_report", write_report_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"call_tool": "call_tool", "write_report": "write_report"},
    )
    graph.add_edge("call_tool", "grade")
    graph.add_edge("write_report", END)

    return graph.compile()


def build_research_graph(
    *, enable_hitl: bool = False, checkpointer: BaseCheckpointSaver | None = None
):
    """Build the research graph.

    enable_hitl=True inserts a human-in-the-loop pause after planning, before any section
    research spends an LLM call, so a human can edit the proposed outline (see
    human_review_node). This requires a checkpointer -- one (InMemorySaver) is created
    automatically if `checkpointer` isn't supplied -- and every invoke/resume on a graph built
    this way MUST pass a config with a stable "thread_id" (see scripts/research_hitl_demo.py).

    Off by default: plain programmatic use (scripts, the eval harness in Phase 5) has no human
    available to satisfy an approval step, and forcing one there would just hang.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("run_sections", run_sections_node)
    graph.add_node("assemble_report", assemble_report_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revise_report", revise_report_node)

    graph.set_entry_point("planner")
    if enable_hitl:
        graph.add_node("human_review", human_review_node)
        graph.add_edge("planner", "human_review")
        graph.add_edge("human_review", "run_sections")
    else:
        graph.add_edge("planner", "run_sections")
    graph.add_edge("run_sections", "assemble_report")
    graph.add_edge("assemble_report", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise_report": "revise_report", "end": END},
    )
    graph.add_edge("revise_report", "critic")

    if enable_hitl:
        return graph.compile(checkpointer=checkpointer or _default_checkpointer())
    return graph.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv

    from agent.logging_config import configure_logging

    load_dotenv()
    configure_logging()

    app = build_graph()
    result = app.invoke({"question": "What is LangGraph?"})
    print("sufficient:", result["sufficient"])
    print("loop_count:", result["loop_count"])
    print("tool_calls_made:", result["tool_calls_made"])
    print("trace:")
    for line in result.get("trace", []):
        print(" ", line)
    print("report:", result["report"].model_dump_json(indent=2))
