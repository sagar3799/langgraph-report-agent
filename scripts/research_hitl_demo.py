"""One-off script: run the topic-level research graph WITH the human-in-the-loop checkpoint
enabled -- start a run, watch it pause after planning with the proposed section outline, edit
it, resume, and get back a report that reflects the edit.

This is Phase 4's demoable checkpoint: the single most "production" thing in the whole build --
a real pause/persist/resume using LangGraph's interrupt() + Command primitives, not a fake
sleep-and-poll loop.

Usage:
    python scripts/research_hitl_demo.py "Your topic here"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.types import Command  # noqa: E402

from agent.graph import build_research_graph  # noqa: E402
from agent.logging_config import configure_logging  # noqa: E402


def main() -> None:
    load_dotenv()
    configure_logging()
    # Windows' default console codepage (cp1252) can't print the emoji used in trace lines.
    sys.stdout.reconfigure(encoding="utf-8")

    default_topic = "What is LangGraph and how does it compare to plain function calling?"
    topic = " ".join(sys.argv[1:]).strip() or default_topic

    graph = build_research_graph(enable_hitl=True)
    config = {"configurable": {"thread_id": f"hitl-demo-{abs(hash(topic))}"}}

    print(f"Starting research on: {topic!r}\n")
    paused = graph.invoke({"topic": topic}, config)

    interrupts = paused.get("__interrupt__")
    if not interrupts:
        print("No interrupt was raised -- HITL wiring isn't behaving as expected.")
        print(paused)
        return

    payload = interrupts[0].value
    proposed = payload["proposed_sections"]
    print(f"[PAUSED] Planned {len(proposed)} section(s) for topic {payload['topic']!r}:")
    for i, section in enumerate(proposed, start=1):
        print(f"  {i}. {section['title']}")
        print(f"     {section['sub_question']}")

    print()
    choice = input(
        "Press Enter to accept as-is, or type a comma-separated list of section numbers "
        "to KEEP (e.g. '1,3'): "
    ).strip()

    if choice:
        keep_indices = {int(n) - 1 for n in choice.split(",") if n.strip().isdigit()}
        edited = [s for i, s in enumerate(proposed) if i in keep_indices]
    else:
        edited = proposed

    print(f"\n[RESUMING] with {len(edited)} section(s): {[s['title'] for s in edited]}\n")
    result = graph.invoke(Command(resume=edited), config)

    print("trace:")
    for line in result.get("trace", []):
        print(" ", line)
    print()
    print("report:", result["report"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
