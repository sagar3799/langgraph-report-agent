"""Graph wiring: retrieve -> grade -> (loop: call_tool -> grade) -> write_report."""

from langgraph.graph import END, StateGraph

from agent.nodes import (
    call_tool_node,
    grade_node,
    retrieve_node,
    route_after_grade,
    write_report_node,
)
from agent.state import AgentState


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


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"question": "What is LangGraph?"})
    print("sufficient:", result["sufficient"])
    print("loop_count:", result["loop_count"])
    print("tool_calls_made:", result["tool_calls_made"])
    print("report:", result["report"].model_dump_json(indent=2))
