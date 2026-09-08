"""Eval harness: run eval/questions.json through the agent and score the results.

Usage:
    python eval/run_eval.py
"""

import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.graph import build_graph  # noqa: E402
from agent.logging_config import configure_logging  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def report_text(report) -> str:
    return " ".join([report.title, report.summary] + [s.content for s in report.sections]).lower()


def score_case(case: dict, result: dict) -> dict:
    report = result["report"]
    tool_calls = result.get("tool_calls_made", [])
    text = report_text(report)

    if case["category"] == "edge":
        passed = report.confidence == "low"
        detail = f"confidence={report.confidence} (edge cases must be low-confidence)"
    else:
        expected_tool = case.get("expected_tool")
        tool_ok = expected_tool is None or expected_tool in tool_calls
        keywords = case.get("expected_keywords", [])
        keyword_ok = not keywords or any(k.lower() in text for k in keywords)
        passed = tool_ok and keyword_ok
        detail = f"tool_ok={tool_ok} (want={expected_tool}, got={tool_calls}), kw_ok={keyword_ok}"

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": passed,
        "confidence": report.confidence,
        "tool_calls": ",".join(tool_calls) if tool_calls else "none",
        "loop_count": result.get("loop_count", 0),
        "detail": detail,
    }


def main() -> None:
    load_dotenv()
    configure_logging()
    RESULTS_DIR.mkdir(exist_ok=True)

    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    graph = build_graph()

    rows = []
    for case in cases:
        print(f"Running {case['id']}: {case['question']}")
        try:
            result = graph.invoke({"question": case["question"]})
            row = score_case(case, result)
        except Exception as exc:  # a case failing outright is still a result worth recording
            row = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "passed": False,
                "confidence": "error",
                "tool_calls": "error",
                "loop_count": -1,
                "detail": f"exception: {exc}",
            }
        rows.append(row)
        time.sleep(4)  # stay well under the free-tier rate limit between cases

    write_outputs(rows)


def write_outputs(rows: list[dict]) -> None:
    csv_path = RESULTS_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = RESULTS_DIR / "results.md"
    passed_count = sum(r["passed"] for r in rows)
    lines = [
        f"# Eval Results: {passed_count}/{len(rows)} passed\n",
        "| ID | Category | Passed | Confidence | Tools Called | Loops | Detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| {r['id']} | {r['category']} | {mark} | {r['confidence']} | "
            f"{r['tool_calls']} | {r['loop_count']} | {r['detail']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{passed_count}/{len(rows)} passed. Results written to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
