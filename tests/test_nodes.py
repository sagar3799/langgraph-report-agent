from agent import nodes
from agent.schemas import GradeResult


def test_grade_node_defaults_to_web_search_on_contradictory_grade(monkeypatch):
    """If the model says insufficient but picks no tool, don't waste a loop on a no-op."""
    monkeypatch.setattr(nodes, "get_chat_model", lambda: object())
    monkeypatch.setattr(
        nodes,
        "generate_structured",
        lambda *a, **k: GradeResult(sufficient=False, reason="r", next_tool="none", tool_input=""),
    )

    result = nodes.grade_node({"question": "what year was X released?", "loop_count": 0})

    assert result["sufficient"] is False
    assert result["next_tool"] == "web_search"
    assert result["tool_input"] == "what year was X released?"


def test_grade_node_respects_explicit_tool_choice(monkeypatch):
    monkeypatch.setattr(nodes, "get_chat_model", lambda: object())
    monkeypatch.setattr(
        nodes,
        "generate_structured",
        lambda *a, **k: GradeResult(
            sufficient=False, reason="r", next_tool="calculator", tool_input="2+2"
        ),
    )

    result = nodes.grade_node({"question": "q", "loop_count": 0})

    assert result["next_tool"] == "calculator"
    assert result["tool_input"] == "2+2"
