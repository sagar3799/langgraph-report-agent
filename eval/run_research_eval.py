"""Eval harness for the topic-level research graph: run eval/research_topics.json through it
and score retrieval precision, citation coverage, section completeness, and latency.

This is a separate harness from eval/run_eval.py (the original single-question one) rather than
a replacement -- a multi-section research report needs different dimensions than "did it call
the right tool," and the single-question graph is still real, tested, and reused as-is inside
this one as the per-section subgraph.

Kept deliberately small (2 topics): a full research run costs 15-30+ Gemini calls (planner +
per-section retrieve/grade/tool loops + verify + critic), against a free tier already flagged
as the binding constraint on this whole project -- a larger topic set risks not finishing
before hitting a rate limit, which would produce a worse eval report than a small one that
actually completes.

Usage:
    python eval/run_research_eval.py
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.graph import build_research_graph  # noqa: E402
from agent.logging_config import configure_logging  # noqa: E402
from agent.observability import get_invoke_config  # noqa: E402

TOPICS_PATH = Path(__file__).resolve().parent / "research_topics.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# A report that's mostly unsupported claims or drops half the planned sections shouldn't "pass"
# just because it happens to mention the right words -- these thresholds require all three
# dimensions to hold, not just one.
KEYWORD_COVERAGE_THRESHOLD = 0.7
SECTION_COMPLETENESS_THRESHOLD = 0.75
CITATION_COVERAGE_THRESHOLD = 0.5
# A section's retrieved chunk counts as a "precise" hit if the reranker's normalized confidence
# clears this bar. A heuristic proxy for retrieval precision, not a hand-labeled relevance
# judgment -- there's no gold-labeled relevant-doc set to compare against, so this measures
# "how often did retrieval surface something the reranker itself considered a decent match"
# rather than true precision against ground truth. Documented as such in the results, not
# presented as more rigorous than it is.
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.5


def report_text(report) -> str:
    return " ".join([report.title, report.summary] + [s.content for s in report.sections]).lower()


def retrieval_precision(result: dict) -> float | None:
    """Fraction of all chunks retrieved across every section whose reranker-normalized
    confidence cleared RETRIEVAL_CONFIDENCE_THRESHOLD. None if nothing was ever retrieved
    (can't compute precision over an empty set)."""
    all_docs = [
        doc
        for sr in result.get("section_results", [])
        for doc in sr.get("retrieved_docs", [])
    ]
    if not all_docs:
        return None
    hits = sum(1 for d in all_docs if d["normalized_confidence"] >= RETRIEVAL_CONFIDENCE_THRESHOLD)
    return hits / len(all_docs)


def score_case(case: dict, result: dict, latency_seconds: float) -> dict:
    report = result["report"]
    text = report_text(report)

    keywords = case.get("expected_keywords", [])
    matched = [k for k in keywords if k.lower() in text]
    keyword_coverage = len(matched) / len(keywords) if keywords else 1.0

    planned = len(result.get("sections", []))
    written = len(report.sections)
    section_completeness = written / planned if planned > 0 else 0.0

    precision = retrieval_precision(result)
    citation_coverage = report.evidence_coverage

    passed = (
        keyword_coverage >= KEYWORD_COVERAGE_THRESHOLD
        and section_completeness >= SECTION_COMPLETENESS_THRESHOLD
        and citation_coverage is not None
        and citation_coverage >= CITATION_COVERAGE_THRESHOLD
    )

    return {
        "id": case["id"],
        "topic": case["topic"],
        "passed": passed,
        "confidence": report.confidence,
        "critic_accepted": result.get("critic_accepted"),
        "keyword_coverage": round(keyword_coverage, 2),
        "keywords_matched": ",".join(matched) if matched else "none",
        "keywords_missed": ",".join(k for k in keywords if k not in matched) or "none",
        "section_completeness": round(section_completeness, 2),
        "sections_planned": planned,
        "sections_written": written,
        "citation_coverage": round(citation_coverage, 2) if citation_coverage is not None else None,
        "retrieval_precision": round(precision, 2) if precision is not None else None,
        "latency_seconds": round(latency_seconds, 1),
    }


def main() -> None:
    load_dotenv()
    configure_logging()
    RESULTS_DIR.mkdir(exist_ok=True)

    cases = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    # enable_hitl=False (the default): an eval run has no human available to approve an outline,
    # and pausing forever would just be a hang, not a result -- see build_research_graph()'s
    # own docstring for this reasoning.
    graph = build_research_graph()

    rows = []
    for case in cases:
        print(f"Running {case['id']}: {case['topic']}")
        t0 = time.monotonic()
        try:
            trace_config = get_invoke_config(source="research_eval", topic_id=case["id"])
            result = graph.invoke({"topic": case["topic"]}, config=trace_config)
            latency = time.monotonic() - t0
            row = score_case(case, result, latency)
        except Exception as exc:  # a topic failing outright is still a result worth recording
            latency = time.monotonic() - t0
            row = {
                "id": case["id"],
                "topic": case["topic"],
                "passed": False,
                "confidence": "error",
                "critic_accepted": None,
                "keyword_coverage": 0.0,
                "keywords_matched": "error",
                "keywords_missed": "error",
                "section_completeness": 0.0,
                "sections_planned": 0,
                "sections_written": 0,
                "citation_coverage": None,
                "retrieval_precision": None,
                "latency_seconds": round(latency, 1),
                "detail": f"exception: {exc}",
            }
        rows.append(row)
        time.sleep(5)  # stay well under the free-tier rate limit between topics

    write_outputs(rows)


def write_outputs(rows: list[dict]) -> None:
    results_path = RESULTS_DIR / "research_results.json"
    results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md_path = RESULTS_DIR / "research_results.md"
    passed_count = sum(r["passed"] for r in rows)
    lines = [
        f"# Research Eval Results: {passed_count}/{len(rows)} passed\n",
        f"Thresholds: keyword coverage >= {KEYWORD_COVERAGE_THRESHOLD:.0%}, section "
        f"completeness >= {SECTION_COMPLETENESS_THRESHOLD:.0%}, citation coverage >= "
        f"{CITATION_COVERAGE_THRESHOLD:.0%}. Retrieval precision is reported but not gated on "
        "(heuristic proxy, not a hand-labeled ground truth -- see script docstring).\n",
        "| ID | Passed | Confidence | Critic | Keyword Cov. | Section Cov. | Citation Cov. | "
        "Retrieval Prec. | Latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        citation = r["citation_coverage"]
        citation_cell = f"{citation:.0%}" if citation is not None else "n/a"
        precision = r["retrieval_precision"]
        precision_cell = f"{precision:.0%}" if precision is not None else "n/a"
        lines.append(
            f"| {r['id']} | {mark} | {r['confidence']} | {r['critic_accepted']} | "
            f"{r['keyword_coverage']:.0%} | {r['section_completeness']:.0%} | {citation_cell} | "
            f"{precision_cell} | {r['latency_seconds']}s |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{passed_count}/{len(rows)} passed. Results written to {results_path} and {md_path}")


if __name__ == "__main__":
    main()
