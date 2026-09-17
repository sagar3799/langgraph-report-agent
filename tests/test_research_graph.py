"""Tests for the topic-level research graph: planner -> per-section fan-out -> assemble.

Same methodology as test_graph.py -- fake nodes/subgraph, verify wiring and data-shuffling
logic in isolation from whether the AI is any good.
"""

from langgraph.types import Command

from agent import graph as graph_module
from agent import nodes as nodes_module
from agent.schemas import Report, ReportSection, SectionPlan, VerificationResult


def _fake_critic_accepts(_state):
    return {"critic_accepted": True, "critic_issues": [], "critic_loop_count": 1}


def _fake_planner(_state):
    return {
        "sections": [
            SectionPlan(title="A", sub_question="what is a?"),
            SectionPlan(title="B", sub_question="what is b?"),
        ],
        "trace": ["🗂️ Planned 2 section(s): A, B"],
    }


def _fake_assemble(state):
    section_results = state.get("section_results", [])
    sections = [
        ReportSection(heading=sr["title"], content=sr["report"].summary) for sr in section_results
    ]
    return {
        "report": Report(
            title=state["topic"],
            summary="assembled",
            sections=sections,
            sources=sorted({s for sr in section_results for s in sr["report"].sources}),
            confidence="high",
        )
    }


def test_research_graph_wires_planner_through_to_assemble(monkeypatch):
    monkeypatch.setattr(graph_module, "planner_node", _fake_planner)
    monkeypatch.setattr(graph_module, "assemble_report_node", _fake_assemble)
    monkeypatch.setattr(graph_module, "critic_node", _fake_critic_accepts)

    def _fake_run_sections_node(state):
        results = [
            {
                "title": section.title,
                "sub_question": section.sub_question,
                "report": Report(
                    title=section.title,
                    summary=f"summary for {section.title}",
                    sections=[],
                    sources=[f"{section.title}.txt"],
                    confidence="high",
                ),
            }
            for section in state["sections"]
        ]
        return {"section_results": results, "evidence_pool": {}}

    monkeypatch.setattr(graph_module, "run_sections_node", _fake_run_sections_node)

    app = graph_module.build_research_graph()
    result = app.invoke({"topic": "Electric vehicles"})

    assert [s.heading for s in result["report"].sections] == ["A", "B"]
    assert result["report"].sources == ["A.txt", "B.txt"]
    assert result["report"].confidence == "high"


def test_run_sections_merges_retrieved_docs_into_shared_evidence_pool(monkeypatch):
    """Sequential path: a doc retrieved for section 1 should be visible to section 2's memory
    pool, and the evidence pool returned should contain docs from every section."""

    seen_pools = []

    def _fake_subgraph_invoke(payload):
        seen_pools.append(dict(payload.get("memory_pool", {})))
        question = payload["question"]
        doc_id = f"doc-for-{question}"
        return {
            "report": Report(
                title=question, summary="s", sections=[], sources=["src"], confidence="high"
            ),
            "retrieved_docs": [{"id": doc_id, "text": "t", "source": "src", "score": 1.0}],
        }

    class _FakeSubgraph:
        def invoke(self, payload):
            return _fake_subgraph_invoke(payload)

    # run_sections() does `from agent.graph import build_graph` locally at call time, so
    # patching the attribute on the agent.graph module (graph_module is that same module) works.
    monkeypatch.setattr(graph_module, "build_graph", lambda: _FakeSubgraph())
    monkeypatch.setattr(
        nodes_module,
        "verify_section",
        lambda report, docs, tools: VerificationResult(total_claims=2, unsupported_claims=[]),
    )

    sections = [
        SectionPlan(title="First", sub_question="q1"),
        SectionPlan(title="Second", sub_question="q2"),
    ]
    results, pool, trace = nodes_module.run_sections(sections, evidence_pool={}, max_workers=1)

    assert len(results) == 2
    assert set(pool.keys()) == {"doc-for-q1", "doc-for-q2"}
    # Section 2 should have seen section 1's retrieved doc in its memory pool.
    assert seen_pools[0] == {}
    assert "doc-for-q1" in seen_pools[1]
    assert len(trace) == 2
    assert results[0]["verification"]["coverage"] == 1.0


def test_run_sections_regenerates_a_section_with_unsupported_claims_once(monkeypatch):
    """A section that fails verification should be regenerated exactly once (MAX_VERIFY_RETRIES
    == 1), with the unsupported claims fed back in, and the retry's result used regardless of
    whether it still has unsupported claims -- bounded, not looped forever."""

    invocations = []

    def _fake_subgraph_invoke(payload):
        invocations.append(payload)
        attempt = len(invocations)
        return {
            "report": Report(
                title="t", summary=f"attempt {attempt}", sections=[], sources=["src"],
                confidence="medium",
            ),
            "retrieved_docs": [],
            "tool_results": [],
        }

    class _FakeSubgraph:
        def invoke(self, payload):
            return _fake_subgraph_invoke(payload)

    monkeypatch.setattr(graph_module, "build_graph", lambda: _FakeSubgraph())

    verify_calls = []

    def _fake_verify(report, docs, tools):
        verify_calls.append(report.summary)
        # Always fails, to prove the retry is bounded rather than infinite.
        return VerificationResult(total_claims=2, unsupported_claims=["a bad claim"])

    monkeypatch.setattr(nodes_module, "verify_section", _fake_verify)

    sections = [SectionPlan(title="Only", sub_question="q1")]
    results, _pool, trace = nodes_module.run_sections(sections, evidence_pool={}, max_workers=1)

    assert len(invocations) == 2  # original attempt + exactly one bounded retry
    assert "verifier_feedback" not in invocations[0]
    assert "a bad claim" in invocations[1]["verifier_feedback"]
    assert len(verify_calls) == 2
    assert results[0]["verification"]["retries_used"] == 1
    assert results[0]["verification"]["unsupported_claims"] == ["a bad claim"]
    assert "1 unsupported claim(s) after 1 retry/retries" in trace[0]


def test_compute_source_confidence_uses_reranker_score_for_kb_and_baseline_for_web():
    retrieved_docs = [
        {"id": "1", "text": "t", "source": "kb.pdf", "score": 5.0, "normalized_confidence": 0.9},
        {"id": "2", "text": "t", "source": "kb.pdf", "score": 1.0, "normalized_confidence": 0.7},
    ]
    confidence = nodes_module.compute_source_confidence(
        retrieved_docs, report_sources=["kb.pdf", "https://example.com/article"]
    )

    assert confidence["kb.pdf"] == 0.8  # mean of 0.9 and 0.7
    assert confidence["https://example.com/article"] == nodes_module.WEB_SOURCE_BASELINE_CONFIDENCE


def test_compute_source_confidence_ignores_retrieved_but_uncited_sources():
    """A research run's shared evidence pool accumulates every chunk retrieved along the way,
    including irrelevant ones a section's grader looked at and discarded. Those shouldn't get a
    confidence score if they never made it into the report's actual sources -- caught via a real
    end-to-end run where an unrelated indexed PDF showed up with a misleading 0.0 confidence."""
    retrieved_docs = [
        {"id": "1", "text": "t", "source": "cited.pdf", "normalized_confidence": 0.9},
        {"id": "2", "text": "t", "source": "noise.pdf", "normalized_confidence": 0.0},
    ]
    confidence = nodes_module.compute_source_confidence(
        retrieved_docs, report_sources=["cited.pdf"]
    )

    assert confidence == {"cited.pdf": 0.9}


def test_assemble_report_computes_claim_weighted_evidence_coverage():
    state = {
        "topic": "Topic",
        "evidence_pool": {},
        "section_results": [
            {
                "title": "A",
                "report": Report(
                    title="A", summary="s", sections=[], sources=[], confidence="high"
                ),
                "verification": {
                    "total_claims": 4, "unsupported_claims": [], "coverage": 1.0, "retries_used": 0
                },
            },
            {
                "title": "B",
                "report": Report(
                    title="B", summary="s", sections=[], sources=[], confidence="high"
                ),
                "verification": {
                    "total_claims": 6, "unsupported_claims": ["x", "y"], "coverage": 4 / 6,
                    "retries_used": 1,
                },
            },
        ],
    }

    result = nodes_module.assemble_report_node(state)

    # 8 supported out of 10 total claims across both sections, not a plain average of 1.0/0.667.
    assert result["report"].evidence_coverage == 0.8
    assert "unsupported claim(s): x; y" in result["report"].sections[1].content


def test_route_after_critic_accepted_ends_immediately():
    state = {"critic_accepted": True, "critic_loop_count": 1}
    assert nodes_module.route_after_critic(state) == "end"


def test_route_after_critic_rejected_under_budget_revises():
    state = {"critic_accepted": False, "critic_loop_count": 1}
    assert nodes_module.route_after_critic(state) == "revise_report"


def test_route_after_critic_rejected_at_budget_stops_anyway():
    """MAX_CRITIC_LOOPS bounds the loop -- a report that's still rejected after using its one
    revision attempt gets returned as-is rather than looping forever."""
    state = {"critic_accepted": False, "critic_loop_count": nodes_module.MAX_CRITIC_LOOPS}
    assert nodes_module.route_after_critic(state) == "end"


def test_research_graph_revises_once_when_critic_rejects_then_accepts(monkeypatch):
    """Full graph run: critic rejects the first assembled report, revise_report runs once, the
    critic re-checks and accepts -- proving the loop-back edge (assemble -> critic ->
    revise_report -> critic -> end) actually wires up, not just the routing function alone."""
    monkeypatch.setattr(graph_module, "planner_node", _fake_planner)
    monkeypatch.setattr(graph_module, "assemble_report_node", _fake_assemble)

    def _fake_run_sections_node(state):
        return {"section_results": [], "evidence_pool": {}}

    monkeypatch.setattr(graph_module, "run_sections_node", _fake_run_sections_node)

    critic_calls = []

    def _fake_critic(state):
        critic_calls.append(state["report"].summary)
        accepted = state["report"].summary == "revised"
        return {
            "critic_accepted": accepted,
            "critic_issues": [] if accepted else ["repeats itself"],
            "critic_loop_count": state.get("critic_loop_count", 0) + 1,
        }

    def _fake_revise(state):
        return {"report": state["report"].model_copy(update={"summary": "revised"})}

    monkeypatch.setattr(graph_module, "critic_node", _fake_critic)
    monkeypatch.setattr(graph_module, "revise_report_node", _fake_revise)

    app = graph_module.build_research_graph()
    result = app.invoke({"topic": "Electric vehicles"})

    assert critic_calls == ["assembled", "revised"]  # critiqued once, revised, critiqued again
    assert result["report"].summary == "revised"


def test_research_graph_critic_loop_is_bounded_not_infinite(monkeypatch):
    """If the critic never accepts, the graph must still terminate (MAX_CRITIC_LOOPS), not loop
    forever -- same discipline as MAX_LOOPS on the grade/call_tool loop elsewhere."""
    monkeypatch.setattr(graph_module, "planner_node", _fake_planner)
    monkeypatch.setattr(graph_module, "assemble_report_node", _fake_assemble)
    monkeypatch.setattr(graph_module, "run_sections_node", lambda state: {"section_results": []})

    critic_call_count = 0

    def _always_reject_critic(state):
        nonlocal critic_call_count
        critic_call_count += 1
        return {
            "critic_accepted": False,
            "critic_issues": ["still bad"],
            "critic_loop_count": state.get("critic_loop_count", 0) + 1,
        }

    monkeypatch.setattr(graph_module, "critic_node", _always_reject_critic)
    monkeypatch.setattr(graph_module, "revise_report_node", lambda state: {})

    app = graph_module.build_research_graph()
    result = app.invoke({"topic": "Electric vehicles"})

    assert critic_call_count == nodes_module.MAX_CRITIC_LOOPS
    assert result["critic_accepted"] is False  # honestly reports it never got accepted


def test_build_research_graph_default_has_no_hitl_pause(monkeypatch):
    """The default graph (enable_hitl=False) must NOT require a thread_id or human approval --
    programmatic callers (scripts, the eval harness) have no human to satisfy one."""
    monkeypatch.setattr(graph_module, "planner_node", _fake_planner)
    monkeypatch.setattr(graph_module, "run_sections_node", lambda state: {"section_results": []})
    monkeypatch.setattr(graph_module, "assemble_report_node", _fake_assemble)
    monkeypatch.setattr(graph_module, "critic_node", _fake_critic_accepts)

    app = graph_module.build_research_graph()
    result = app.invoke({"topic": "Electric vehicles"})  # no config/thread_id at all

    assert "__interrupt__" not in result
    assert result["report"].title == "Electric vehicles"


def test_human_review_pauses_with_proposed_outline_and_resumes_with_edit(monkeypatch):
    """Real end-to-end interrupt/resume round trip through build_research_graph(enable_hitl=
    True) -- only planner/run_sections/assemble/critic are faked (to avoid real Gemini calls);
    the interrupt() mechanism, human_review_node, and the checkpointer are all real, unmocked."""
    monkeypatch.setattr(graph_module, "planner_node", _fake_planner)

    def _fake_run_sections_node(state):
        results = [
            {
                "title": s.title,
                "sub_question": s.sub_question,
                "report": Report(
                    title=s.title, summary="s", sections=[], sources=[], confidence="high"
                ),
            }
            for s in state["sections"]
        ]
        return {"section_results": results, "evidence_pool": {}}

    monkeypatch.setattr(graph_module, "run_sections_node", _fake_run_sections_node)
    monkeypatch.setattr(graph_module, "assemble_report_node", _fake_assemble)
    monkeypatch.setattr(graph_module, "critic_node", _fake_critic_accepts)

    app = graph_module.build_research_graph(enable_hitl=True)
    config = {"configurable": {"thread_id": "test-thread"}}

    paused = app.invoke({"topic": "Electric vehicles"}, config)

    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["topic"] == "Electric vehicles"
    assert [s["title"] for s in payload["proposed_sections"]] == ["A", "B"]

    # Human removes section B before continuing.
    edited = [payload["proposed_sections"][0]]
    result = app.invoke(Command(resume=edited), config)

    assert "__interrupt__" not in result
    assert [s.heading for s in result["report"].sections] == ["A"]
