"""End-to-end graph wiring test with faked nodes — no real LLM/Qdrant calls.

This mirrors the project's own build methodology: verify the graph's edges and
loop-back logic in isolation from whether the AI logic is any good.
"""

from agent import graph as graph_module


def _fake_retrieve(_state):
    return {"retrieved_docs": [{"text": "doc", "source": "s", "score": 1.0}]}


def _make_fake_grade(sufficient_after_loop: int):
    def _fake_grade(state):
        loop_count = state.get("loop_count", 0)
        return {
            "sufficient": loop_count >= sufficient_after_loop,
            "grade_reason": "fake",
            "next_tool": "calculator",
            "tool_input": "1+1",
            "loop_count": loop_count + 1,
        }

    return _fake_grade


def _fake_call_tool(_state):
    return {"tool_calls_made": ["calculator"], "tool_results": ["1+1 = 2"]}


def _fake_write_report(state):
    from agent.schemas import Report, ReportSection

    return {
        "report": Report(
            title="t",
            summary="s",
            sections=[ReportSection(heading="h", content="c")],
            sources=[d["source"] for d in state.get("retrieved_docs", [])],
            confidence="high",
        )
    }


def test_graph_loops_until_sufficient_then_writes_report(monkeypatch):
    monkeypatch.setattr(graph_module, "retrieve_node", _fake_retrieve)
    monkeypatch.setattr(graph_module, "grade_node", _make_fake_grade(sufficient_after_loop=2))
    monkeypatch.setattr(graph_module, "call_tool_node", _fake_call_tool)
    monkeypatch.setattr(graph_module, "write_report_node", _fake_write_report)

    app = graph_module.build_graph()
    result = app.invoke({"question": "2+2?"})

    assert result["sufficient"] is True
    assert result["loop_count"] == 3
    assert result["tool_calls_made"] == ["calculator", "calculator"]
    assert result["report"].confidence == "high"


def test_graph_skips_tool_loop_when_immediately_sufficient(monkeypatch):
    monkeypatch.setattr(graph_module, "retrieve_node", _fake_retrieve)
    monkeypatch.setattr(graph_module, "grade_node", _make_fake_grade(sufficient_after_loop=0))
    monkeypatch.setattr(graph_module, "call_tool_node", _fake_call_tool)
    monkeypatch.setattr(graph_module, "write_report_node", _fake_write_report)

    app = graph_module.build_graph()
    result = app.invoke({"question": "2+2?"})

    assert result.get("tool_calls_made", []) == []
    assert result["loop_count"] == 1
