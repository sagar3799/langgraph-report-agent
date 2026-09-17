"""One-off script: run the topic-level research graph (planner -> per-section subgraph fan-out
-> assemble) end to end, and print the resulting multi-section report.

This is Phase 1's demoable checkpoint: give it a topic, get back a structured multi-section
report, each section independently retrieved/graded/written against the knowledge base indexed
via scripts/ingest_docs.py.

Usage:
    python scripts/research_demo.py "Your topic here"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.graph import build_research_graph  # noqa: E402
from agent.logging_config import configure_logging  # noqa: E402


def main() -> None:
    load_dotenv()
    configure_logging()
    # Windows' default console codepage (cp1252) can't print the emoji used in trace lines.
    sys.stdout.reconfigure(encoding="utf-8")

    default_topic = "What is LangGraph and how does it compare to plain function calling?"
    topic = " ".join(sys.argv[1:]).strip() or default_topic

    graph = build_research_graph()
    result = graph.invoke({"topic": topic})

    print("topic:", topic)
    print(f"planned {len(result.get('sections', []))} section(s):")
    for section in result.get("sections", []):
        print(f"  - {section.title}: {section.sub_question}")
    print()
    print("trace:")
    for line in result.get("trace", []):
        print(" ", line)
    print()
    print("report:", result["report"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
