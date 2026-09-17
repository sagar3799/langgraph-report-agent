"""Real node implementations: Qdrant retrieval, Gemini grading/report-writing, tool calls.

Each node logs what it did (visible in the terminal running streamlit/uvicorn) and appends
a short human-readable line to state["trace"], which the UI renders as a step-by-step view
of what the agent actually did — retrieval isn't a black box.
"""

import logging

from langgraph.types import interrupt

from agent.llm import StructuredOutputError, generate_structured, get_chat_model
from agent.retrieval.qdrant_store import search
from agent.schemas import (
    CriticResult,
    GradeResult,
    Plan,
    Report,
    ReportSection,
    SectionPlan,
    VerificationResult,
)
from agent.state import AgentState, ResearchState, RetrievedDoc, SectionResult
from agent.tools.calculator import CalculatorError, calculate
from agent.tools.web_search import WebSearchError, web_search

logger = logging.getLogger(__name__)

MAX_LOOPS = 3
TOP_K = 4
MAX_SECTIONS = 5
MAX_VERIFY_RETRIES = 1
# Counts total critic EVALUATIONS, same convention as MAX_LOOPS counting grade calls above (not
# revise calls) -- so 2 means "the initial critique, plus one recheck after one revision
# attempt," i.e. exactly one bounded revision, not two. Setting this to 1 would reject-and-stop
# without ever revising anything, since the very first critique already hits the cap.
MAX_CRITIC_LOOPS = 2
# Web search results have no per-item reranker score the way KB chunks do (see search() in
# qdrant_store.py) -- this is a documented, fixed stand-in, not a measured confidence.
WEB_SOURCE_BASELINE_CONFIDENCE = 0.6


def retrieve_node(state: AgentState) -> dict:
    logger.info("[retrieve] searching knowledge base for: %r", state["question"])
    memory_pool = state.get("memory_pool") or {}
    docs = search(state["question"], top_k=TOP_K, extra_candidates=list(memory_pool.values()))
    logger.info("[retrieve] got %d chunk(s): %s", len(docs), [d["source"] for d in docs])

    if docs:
        from_memory = sum(1 for d in docs if d["id"] in memory_pool)
        memory_note = f", {from_memory} from shared research memory" if from_memory else ""
        sources = ", ".join(f"{d['source']} ({d['score']:.2f})" for d in docs)
        trace_line = f"🔎 Retrieved {len(docs)} chunk(s){memory_note}: {sources}"
    else:
        trace_line = "🔎 Retrieved 0 chunks (knowledge base empty or nothing matched)"

    return {"retrieved_docs": docs, "trace": [trace_line]}


def _format_context(state: AgentState) -> str:
    doc_lines = [f"[{d['source']}] {d['text']}" for d in state.get("retrieved_docs", [])]
    tool_lines = state.get("tool_results", [])
    parts = []
    if doc_lines:
        parts.append("Retrieved documents:\n" + "\n\n".join(doc_lines))
    if tool_lines:
        parts.append("Prior tool results:\n" + "\n\n".join(tool_lines))
    # The grader sometimes works out the actual answer in its own reasoning (e.g. applying a
    # known formula) while judging sufficiency. Without this, write_report only sees the raw
    # docs/tool output and can end up contradicting a grade that already solved it.
    if state.get("sufficient") and state.get("grade_reason"):
        parts.append(f"Grading assessment (already judged sufficient): {state['grade_reason']}")
    if state.get("verifier_feedback"):
        feedback = state["verifier_feedback"]
        parts.append(f"Evidence verifier feedback from a previous attempt: {feedback}")
    return "\n\n".join(parts) if parts else "(no context retrieved yet)"


def grade_node(state: AgentState) -> dict:
    loop_count = state.get("loop_count", 0)
    logger.info("[grade] loop %d: asking Gemini whether context is sufficient...", loop_count + 1)
    llm = get_chat_model()

    try:
        grade = generate_structured(
            llm,
            system_prompt=(
                "You grade whether the given context is sufficient to write a confident, accurate "
                "answer to the user's question. Be strict about missing FACTS: if the question "
                "needs a specific real-world fact, name, date, or figure that the context doesn't "
                "contain, mark it insufficient and use 'web_search'.\n\n"
                "If the question is PURE ARITHMETIC on numbers already given in the question or "
                "context (addition, percentages, multiplication, totals) — mark it insufficient "
                "and use 'calculator' with the exact expression to evaluate, even though you could "
                "probably compute it yourself. Don't skip the tool and hope the number gets "
                "computed correctly later — a wrong or missing number is a real failure a human "
                "would notice immediately.\n\n"
                "If it instead needs a named formula or principle our calculator can't evaluate "
                "directly (square roots, geometric relationships, non-arithmetic reasoning) — mark "
                "it sufficient; the model can apply the formula itself in the report, and web "
                "search for a generic formula rarely returns a directly quotable answer anyway.\n\n"
                "Watch for a specific trap: a web search can surface a real page about a "
                "different, unrelated thing that merely shares a name or keyword with what the "
                "question is actually about (e.g. a search for 'Aurora's SLA' can return real "
                "docs about the unrelated Apache Aurora software project, which happens to use "
                "the word 'meaning' as jargon). A keyword match is NOT sufficiency — if the "
                "retrieved content isn't actually about the specific subject the question means, "
                "mark it insufficient rather than treating the coincidence as an answer."
            ),
            user_prompt=f"Question: {state['question']}\n\nContext:\n{_format_context(state)}",
            schema=GradeResult,
        )
    except StructuredOutputError as exc:
        logger.warning("[grade] grading call failed, defaulting to insufficient: %s", exc)
        return {
            "sufficient": False,
            "grade_reason": "grading call failed; defaulting to insufficient",
            "next_tool": "none",
            "tool_input": "",
            "loop_count": loop_count + 1,
            "trace": ["🧠 Grading failed (API error) — defaulting to insufficient"],
        }

    next_tool = grade.next_tool
    tool_input = grade.tool_input
    if not grade.sufficient and next_tool == "none":
        # Contradictory grade (insufficient but no tool picked) — don't waste a loop on a no-op.
        next_tool = "web_search"
        tool_input = state["question"]

    logger.info(
        "[grade] sufficient=%s reason=%r next_tool=%s", grade.sufficient, grade.reason, next_tool
    )
    verdict = "sufficient" if grade.sufficient else f"insufficient -> will call {next_tool}"
    trace_line = f"🧠 Graded as {verdict} ({grade.reason})"

    return {
        "sufficient": grade.sufficient,
        "grade_reason": grade.reason,
        "next_tool": next_tool,
        "tool_input": tool_input,
        "loop_count": loop_count + 1,
        "trace": [trace_line],
    }


def call_tool_node(state: AgentState) -> dict:
    tool_name = state.get("next_tool", "none")
    tool_input = state.get("tool_input", "")
    logger.info("[call_tool] invoking %s(%r)", tool_name, tool_input)

    if tool_name == "web_search":
        try:
            results = web_search(tool_input)
            fetched = sum(1 for r in results if r.get("content"))
            result_text = "\n\n".join(
                f"{r['title']} ({r['url']}):\n{r['content'] or r['snippet']}" for r in results
            )
            trace_line = (
                f"🌐 web_search({tool_input!r}) -> {len(results)} result(s), "
                f"{fetched} full page(s) fetched"
            )
        except WebSearchError as exc:
            result_text = f"web_search error: {exc}"
            trace_line = f"🌐 web_search({tool_input!r}) failed: {exc}"
    elif tool_name == "calculator":
        try:
            value = calculate(tool_input)
            result_text = f"{tool_input} = {value}"
            trace_line = f"🧮 calculator({tool_input!r}) = {value}"
        except CalculatorError as exc:
            result_text = f"calculator error: {exc}"
            trace_line = f"🧮 calculator({tool_input!r}) failed: {exc}"
    else:
        result_text = "no tool call was made"
        trace_line = "⚠️ call_tool reached with no tool selected (should not normally happen)"

    logger.info("[call_tool] result: %s", result_text[:200])

    return {
        "tool_calls_made": [tool_name],
        "tool_results": [result_text],
        "trace": [trace_line],
    }


def route_after_grade(state: AgentState) -> str:
    if state.get("sufficient"):
        return "write_report"
    if state.get("loop_count", 0) >= MAX_LOOPS:
        return "write_report"
    return "call_tool"


def write_report_node(state: AgentState) -> dict:
    logger.info("[write_report] asking Gemini to write the final structured report...")
    llm = get_chat_model()
    sources = sorted({d["source"] for d in state.get("retrieved_docs", [])})

    try:
        report = generate_structured(
            llm,
            system_prompt=(
                "You write a structured report answering the user's question using only the given "
                "context. If the context doesn't fully answer it — including when a search found "
                "nothing relevant, or only found a similarly-named entity that isn't confirmed to "
                "be the one asked about — say so explicitly in the summary and set confidence to "
                "'low'. Never invent facts, or a relationship between two entities (e.g. never "
                "call one company the 'parent' of another), not present in the context.\n\n"
                "The ONE exception: if the context includes a 'Grading assessment' line that "
                "applies a well-known MATHEMATICAL OR LOGICAL formula to numbers already given in "
                "the question, that is a certain derivation, not a guess — present it directly "
                "and set confidence to 'high' rather than hedging just because it wasn't copied "
                "from a document."
            ),
            user_prompt=f"Question: {state['question']}\n\nContext:\n{_format_context(state)}",
            schema=Report,
        )
        logger.info("[write_report] done, confidence=%s", report.confidence)
        return {"report": report, "trace": [f"✅ Report written (confidence: {report.confidence})"]}
    except StructuredOutputError as exc:
        logger.warning("[write_report] failed after retry, returning degraded report: %s", exc)
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
        return {
            "report": fallback,
            "trace": [f"❌ Report generation failed after retry: {exc}"],
        }


def planner_node(state: ResearchState) -> dict:
    logger.info("[planner] breaking topic into sections: %r", state["topic"])
    llm = get_chat_model()

    try:
        plan = generate_structured(
            llm,
            system_prompt=(
                "You break a research topic into 3-5 sections that together cover it without "
                "major overlap. Each section needs a short heading and a specific, "
                "self-contained sub-question that could be researched independently of the "
                "other sections."
            ),
            user_prompt=f"Topic: {state['topic']}",
            schema=Plan,
        )
        sections = plan.sections[:MAX_SECTIONS]
    except StructuredOutputError as exc:
        logger.warning("[planner] planning call failed, falling back to one section: %s", exc)
        sections = [SectionPlan(title=state["topic"], sub_question=state["topic"])]

    trace_line = f"🗂️ Planned {len(sections)} section(s): " + ", ".join(s.title for s in sections)
    logger.info("[planner] %s", trace_line)
    return {"sections": sections, "trace": [trace_line]}


def human_review_node(state: ResearchState) -> dict:
    """Pause after planning, before any section spends an LLM call, so a human can edit the
    proposed outline (remove/add sections) before continuing. Uses LangGraph's `interrupt()`
    primitive -- only reachable when build_research_graph(enable_hitl=True) wired this node in
    with a checkpointer attached; resuming requires invoking with the same config (same
    thread_id) and a Command(resume=<edited sections>).

    Per LangGraph's own documented behavior, the graph resumes execution from the START of this
    node and re-runs it -- harmless here since nothing above the interrupt() call does real work
    (no LLM/network calls happen in this node), so re-running it on resume costs nothing.
    """
    proposed = state.get("sections", [])
    payload = {
        "topic": state.get("topic"),
        "proposed_sections": [
            {"title": s.title, "sub_question": s.sub_question} for s in proposed
        ],
    }
    edited = interrupt(payload)
    sections = [
        section if isinstance(section, SectionPlan) else SectionPlan(**section)
        for section in edited
    ]
    logger.info("[human_review] resumed with %d section(s) after human edit", len(sections))
    return {
        "sections": sections,
        "trace": [f"🧑‍💻 Human review: proceeding with {len(sections)} section(s)"],
    }


def verify_section(
    report: Report, retrieved_docs: list[RetrievedDoc], tool_results: list[str]
) -> VerificationResult:
    """One Gemini call: does every factual claim in this section trace back to its evidence?

    This is the Evidence Verifier from Phase 2 of the build plan. It's deliberately scoped to
    ONE section at a time (not the whole assembled report -- that's the Self-Critic's job,
    still to come) so a thin-evidence section can be caught and flagged/regenerated without
    treating the rest of a perfectly good report as suspect.
    """
    logger.info("[verify] checking evidence support for section %r", report.title)
    llm = get_chat_model()

    evidence_lines = [f"[{d['source']}] {d['text']}" for d in retrieved_docs] + list(tool_results)
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "(no evidence retrieved)"
    section_text = report.summary + "\n\n" + "\n\n".join(
        f"{s.heading}: {s.content}" for s in report.sections
    )

    try:
        return generate_structured(
            llm,
            system_prompt=(
                "You check whether a written section is fully backed by the evidence it was "
                "supposed to be based on. Count the distinct factual claims in the section "
                "(don't count generic framing sentences, only concrete facts, figures, or "
                "assertions), and list any that are NOT directly supported by the evidence."
            ),
            user_prompt=f"Section content:\n{section_text}\n\nEvidence:\n{evidence_text}",
            schema=VerificationResult,
        )
    except StructuredOutputError as exc:
        logger.warning("[verify] verification call failed for %r: %s", report.title, exc)
        # Fail closed: an unverifiable section is reported as unsupported rather than silently
        # assumed fine -- consistent with grade_node's "default to insufficient" behavior above.
        return VerificationResult(total_claims=1, unsupported_claims=["(verification call failed)"])


def _section_coverage(verification: VerificationResult) -> float:
    if verification.total_claims <= 0:
        return 1.0
    supported = verification.total_claims - len(verification.unsupported_claims)
    return max(0.0, min(1.0, supported / verification.total_claims))


def compute_source_confidence(
    retrieved_docs: list[RetrievedDoc], report_sources: list[str]
) -> dict[str, float]:
    """Per-source confidence: mean normalized reranker score for KB sources (real signal, see
    RetrievedDoc.normalized_confidence), or a fixed documented baseline for anything cited that
    didn't come through the reranker (i.e. a web search citation).

    Only scores sources that actually made it into `report_sources` -- a research run's shared
    evidence pool accumulates every chunk retrieved along the way, including ones a section's
    grader considered and discarded as irrelevant noise; those shouldn't get a confidence score
    attached to them since they were never actually used as evidence for anything.
    """
    cited = set(report_sources)
    scores_by_source: dict[str, list[float]] = {}
    for doc in retrieved_docs:
        if doc["source"] in cited:
            scores_by_source.setdefault(doc["source"], []).append(doc["normalized_confidence"])

    confidence = {
        source: round(sum(scores) / len(scores), 3) for source, scores in scores_by_source.items()
    }
    for source in report_sources:
        confidence.setdefault(source, WEB_SOURCE_BASELINE_CONFIDENCE)
    return confidence


def run_sections(
    sections: list[SectionPlan],
    evidence_pool: dict[str, RetrievedDoc],
    max_workers: int = 1,
) -> tuple[list[SectionResult], dict[str, RetrievedDoc], list[str]]:
    """Run the existing per-question subgraph once per planned section, then verify each
    section's evidence support and regenerate (bounded) if it fails.

    Sequential by default (max_workers=1), matching this project's rate-limit budgeting —
    free-tier Gemini doesn't survive many concurrent calls. max_workers > 1 fans sections out
    across threads instead: that path is real and functional but has NOT been load-tested
    against the free tier honestly enough to call it proven, and it trades away one thing the
    sequential path gets for free — threads dispatched together snapshot the evidence pool at
    dispatch time, so sections running in the same batch don't see each other's retrieved
    evidence the way strictly sequential sections do. Written as a plain function, not baked
    into the graph's edges, so swapping to LangGraph's `Send`-based fan-out later is a refactor
    of this one function rather than a rewrite of the graph.

    Verification is embedded here, per section, rather than as a separate whole-batch graph
    node — deliberately, so that only sections that actually fail verification cost extra
    Gemini calls (regenerating just the one section, not re-verifying sections that already
    passed), which matters given the same rate-limit budgeting.
    """
    from agent.graph import build_graph  # local import: avoids a nodes<->graph import cycle

    subgraph = build_graph()
    pool = dict(evidence_pool)
    trace: list[str] = []
    results: list[SectionResult] = []

    def _run_one(section: SectionPlan) -> tuple[SectionResult, list[RetrievedDoc], str]:
        logger.info("[run_sections] researching section %r", section.title)
        result = subgraph.invoke({"question": section.sub_question, "memory_pool": dict(pool)})
        report = result["report"]
        retrieved_docs = result.get("retrieved_docs", [])
        tool_results = result.get("tool_results", [])

        verification = verify_section(report, retrieved_docs, tool_results)
        retries_used = 0
        while verification.unsupported_claims and retries_used < MAX_VERIFY_RETRIES:
            retries_used += 1
            logger.info(
                "[verify] section %r has %d unsupported claim(s), regenerating (attempt %d/%d)",
                section.title, len(verification.unsupported_claims),
                retries_used, MAX_VERIFY_RETRIES,
            )
            feedback = (
                "A previous draft of this section made claims not supported by the evidence -- "
                "do not repeat them, and state only what the evidence actually supports: "
                + "; ".join(verification.unsupported_claims)
            )
            result = subgraph.invoke({
                "question": section.sub_question,
                "memory_pool": dict(pool),
                "verifier_feedback": feedback,
            })
            report = result["report"]
            retrieved_docs = result.get("retrieved_docs", [])
            tool_results = result.get("tool_results", [])
            verification = verify_section(report, retrieved_docs, tool_results)

        coverage = _section_coverage(verification)
        section_result: SectionResult = {
            "title": section.title,
            "sub_question": section.sub_question,
            "report": report,
            "retrieved_docs": retrieved_docs,
            "tool_results": tool_results,
            "verification": {
                "total_claims": verification.total_claims,
                "unsupported_claims": verification.unsupported_claims,
                "coverage": coverage,
                "retries_used": retries_used,
            },
        }
        verdict = (
            "fully supported"
            if not verification.unsupported_claims
            else f"{len(verification.unsupported_claims)} unsupported claim(s) "
            f"after {retries_used} retry/retries"
        )
        trace_line = (
            f"📄 Section {section.title!r} done (confidence: {report.confidence}, "
            f"evidence coverage: {coverage:.0%}, {verdict})"
        )
        return section_result, retrieved_docs, trace_line

    if max_workers <= 1:
        for section in sections:
            section_result, docs, trace_line = _run_one(section)
            results.append(section_result)
            trace.append(trace_line)
            for doc in docs:
                pool[doc["id"]] = doc
    else:
        from concurrent.futures import ThreadPoolExecutor

        trace.append(
            f"⚡ Running {len(sections)} section(s) in parallel (max_workers={max_workers}) — "
            "architected, not load-tested against free-tier rate limits"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for section_result, docs, trace_line in executor.map(_run_one, sections):
                results.append(section_result)
                trace.append(trace_line)
                for doc in docs:
                    pool[doc["id"]] = doc

    return results, pool, trace


def run_sections_node(state: ResearchState) -> dict:
    max_workers = state.get("max_workers", 1)
    results, pool, trace = run_sections(
        state["sections"], state.get("evidence_pool", {}), max_workers=max_workers
    )
    return {"section_results": results, "evidence_pool": pool, "trace": trace}


def _combine_confidence(confidences: list[str]) -> str:
    if not confidences:
        return "low"
    if any(c == "low" for c in confidences):
        return "low"
    if any(c == "medium" for c in confidences):
        return "medium"
    return "high"


def assemble_report_node(state: ResearchState) -> dict:
    section_results = state.get("section_results", [])
    logger.info("[assemble] combining %d section report(s)", len(section_results))

    sections = []
    all_sources: set[str] = set()
    confidences = []
    total_claims = 0
    supported_claims = 0
    for sr in section_results:
        sub_report = sr["report"]
        body_parts = [sub_report.summary] + [
            f"**{s.heading}**\n{s.content}" for s in sub_report.sections
        ]
        verification = sr.get("verification")
        if verification and verification["unsupported_claims"]:
            flagged = "; ".join(verification["unsupported_claims"])
            body_parts.append(f"_Evidence verifier flagged unsupported claim(s): {flagged}_")
        sections.append(ReportSection(heading=sr["title"], content="\n\n".join(body_parts)))
        all_sources.update(sub_report.sources)
        confidences.append(sub_report.confidence)
        if verification:
            total_claims += verification["total_claims"]
            unsupported = len(verification["unsupported_claims"])
            supported_claims += verification["total_claims"] - unsupported

    evidence_coverage = supported_claims / total_claims if total_claims > 0 else None
    evidence_pool = state.get("evidence_pool", {})
    source_confidence = compute_source_confidence(list(evidence_pool.values()), sorted(all_sources))

    report = Report(
        title=state["topic"],
        summary=f"A {len(sections)}-section research report on {state['topic']}.",
        sections=sections,
        sources=sorted(all_sources),
        confidence=_combine_confidence(confidences),
        evidence_coverage=evidence_coverage,
        source_confidence=source_confidence,
    )
    coverage_note = f"{evidence_coverage:.0%}" if evidence_coverage is not None else "n/a"
    trace_line = f"✅ Assembled {len(sections)}-section report (evidence coverage: {coverage_note})"
    return {"report": report, "trace": [trace_line]}


def critic_node(state: ResearchState) -> dict:
    """One call on the ASSEMBLED report: catches what per-section verification structurally
    can't -- repetition across sections, a planned section that's missing or too thin to
    actually address its topic, contradictions between sections."""
    report = state["report"]
    logger.info("[critic] reviewing assembled report %r", report.title)
    llm = get_chat_model()

    planned_titles = [s.title for s in state.get("sections", [])]
    report_text = report.summary + "\n\n" + "\n\n".join(
        f"## {s.heading}\n{s.content}" for s in report.sections
    )
    if report.evidence_coverage is not None:
        coverage_note = f"{report.evidence_coverage:.0%}"
    else:
        coverage_note = "not computed"

    try:
        critic = generate_structured(
            llm,
            system_prompt=(
                "You review a finished multi-section research report holistically. Look for: "
                "sections that repeat the same point instead of covering distinct ground, "
                "planned sections that are missing or so thin they don't actually address their "
                "topic, and contradictions between sections. Per-claim citation support has "
                "already been checked section-by-section -- don't re-relitigate individual "
                "sentences, only flag genuine whole-report-level problems. Accept the report if "
                "none exist; don't invent issues to seem thorough."
            ),
            user_prompt=(
                f"Planned sections: {', '.join(planned_titles)}\n"
                f"Evidence-verifier coverage (fraction of claims traced to evidence): "
                f"{coverage_note}\n\nAssembled report:\n{report_text}"
            ),
            schema=CriticResult,
        )
    except StructuredOutputError as exc:
        logger.warning("[critic] critique call failed, accepting report as-is: %s", exc)
        return {
            "critic_accepted": True,
            "critic_loop_count": state.get("critic_loop_count", 0) + 1,
            "trace": [f"⚠️ Self-critic call failed ({exc}), accepting report as-is"],
        }

    if critic.accepted:
        trace_line = "🧐 Self-critic accepted the report"
    else:
        issues = "; ".join(critic.issues)
        trace_line = f"🧐 Self-critic flagged {len(critic.issues)} issue(s): {issues}"
    logger.info("[critic] accepted=%s issues=%s", critic.accepted, critic.issues)

    return {
        "critic_accepted": critic.accepted,
        "critic_issues": critic.issues,
        "critic_loop_count": state.get("critic_loop_count", 0) + 1,
        "trace": [trace_line],
    }


def route_after_critic(state: ResearchState) -> str:
    if state.get("critic_accepted", True):
        return "end"
    if state.get("critic_loop_count", 0) >= MAX_CRITIC_LOOPS:
        return "end"
    return "revise_report"


def revise_report_node(state: ResearchState) -> dict:
    """Rewrite the assembled report to address the critic's issues, constrained to evidence
    already collected across every section -- the revision may reorganize/tighten/merge
    sections, but must not introduce a fact that isn't in that evidence."""
    report = state["report"]
    issues = state.get("critic_issues", [])
    logger.info("[revise_report] revising %r for %d issue(s)", report.title, len(issues))
    llm = get_chat_model()

    evidence_lines: list[str] = []
    for sr in state.get("section_results", []):
        evidence_lines += [f"[{d['source']}] {d['text']}" for d in sr.get("retrieved_docs", [])]
        evidence_lines += sr.get("tool_results", [])
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "(no evidence collected)"

    try:
        revised = generate_structured(
            llm,
            system_prompt=(
                "You are given a finished multi-section report and specific issues a reviewer "
                "found with it. Rewrite the report to address every issue -- you may reorganize, "
                "tighten, merge, or split sections -- but do NOT introduce any fact that isn't in "
                "the evidence below. If the evidence genuinely doesn't support fixing an issue "
                "(e.g. filling in a missing section), say so plainly in that section rather than "
                "inventing something to look complete."
            ),
            user_prompt=(
                "Issues to fix:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
                f"Current report:\n{report.model_dump_json(indent=2)}\n\n"
                f"Evidence:\n{evidence_text}"
            ),
            schema=Report,
        )
    except StructuredOutputError as exc:
        logger.warning("[revise_report] revision call failed, keeping prior report: %s", exc)
        return {"trace": [f"⚠️ Report revision failed ({exc}), keeping prior version"]}

    # The revision call rewrites content/structure only -- preserve the coverage/confidence
    # metadata assemble_report_node derived directly from evidence, rather than let the model
    # (which doesn't recompute claim-level verification) guess new values for them.
    revised.evidence_coverage = report.evidence_coverage
    revised.source_confidence = report.source_confidence

    return {"report": revised, "trace": ["✏️ Report revised based on self-critic feedback"]}
