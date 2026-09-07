"""Chat-style UI for the LangGraph report agent.

Run with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from agent.graph import build_graph  # noqa: E402
from agent.schemas import Report  # noqa: E402

st.set_page_config(page_title="Report Agent", page_icon="🧭", layout="centered")

CONFIDENCE_BADGE = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}


@st.cache_resource
def get_graph():
    return build_graph()


def _escape_markdown_dollars(text: str) -> str:
    """Streamlit renders $...$ as LaTeX math, which mangles prices like '$12/month'."""
    return text.replace("$", "\\$")


def render_report(report: Report, tool_calls: list[str], loop_count: int) -> None:
    st.markdown(f"### {report.title}")
    st.caption(f"Confidence: {CONFIDENCE_BADGE.get(report.confidence, report.confidence)}")
    st.write(_escape_markdown_dollars(report.summary))

    for section in report.sections:
        with st.expander(section.heading, expanded=True):
            st.write(_escape_markdown_dollars(section.content))

    meta_cols = st.columns(2)
    with meta_cols[0]:
        if report.sources:
            st.caption("**Sources:** " + ", ".join(report.sources))
    with meta_cols[1]:
        tools_used = [t for t in tool_calls if t != "none"]
        if tools_used:
            st.caption(f"**Tools called:** {', '.join(tools_used)} · **loops:** {loop_count}")
        else:
            st.caption(f"**Tools called:** none · **loops:** {loop_count}")


st.title("🧭 Report Agent")
st.caption(
    "LangGraph agent: retrieves from Qdrant, grades sufficiency, calls web search or a "
    "calculator when needed, and writes a structured report."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            render_report(turn["report"], turn["tool_calls"], turn["loop_count"])
        else:
            st.write(turn["content"])

question = st.chat_input("Ask a question about Aurora Cloud Storage (or anything else)...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving, grading, and writing report..."):
                graph = get_graph()
                result = graph.invoke({"question": question})
        except Exception as exc:
            st.error(
                "The agent hit an error calling Gemini or Qdrant "
                f"(often a free-tier rate limit): {exc}"
            )
            st.stop()
        report = result["report"]
        tool_calls = result.get("tool_calls_made", [])
        loop_count = result.get("loop_count", 0)
        render_report(report, tool_calls, loop_count)

    st.session_state.history.append(
        {
            "role": "assistant",
            "report": report,
            "tool_calls": tool_calls,
            "loop_count": loop_count,
        }
    )
