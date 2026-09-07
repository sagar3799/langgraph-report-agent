from agent.nodes import MAX_LOOPS, route_after_grade


def test_routes_to_write_report_when_sufficient():
    assert route_after_grade({"sufficient": True, "loop_count": 1}) == "write_report"


def test_routes_to_call_tool_when_insufficient_and_under_limit():
    assert route_after_grade({"sufficient": False, "loop_count": 1}) == "call_tool"


def test_forces_write_report_at_loop_limit_to_avoid_infinite_loop():
    assert route_after_grade({"sufficient": False, "loop_count": MAX_LOOPS}) == "write_report"


def test_missing_keys_default_to_insufficient_and_no_loops():
    assert route_after_grade({}) == "call_tool"
