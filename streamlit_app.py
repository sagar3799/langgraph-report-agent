"""Chat-style UI for the LangGraph report agent — ask questions, attach files to
grow the knowledge base, same way you'd attach a file in ChatGPT.

Run with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from agent.graph import build_graph  # noqa: E402
from agent.ingestion import (  # noqa: E402
    MAX_CHUNKS_PER_FILE,
    SUPPORTED_EXTENSIONS,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    ingest_uploaded_file,
)
from agent.retrieval.qdrant_store import list_sources  # noqa: E402
from agent.schemas import Report  # noqa: E402

st.set_page_config(page_title="Report Agent", page_icon="🧭", layout="centered")

CONFIDENCE_STYLE = {
    "high": ("High", "#16a34a", "rgba(34,197,94,0.15)"),
    "medium": ("Medium", "#ca8a04", "rgba(234,179,8,0.15)"),
    "low": ("Low", "#dc2626", "rgba(239,68,68,0.15)"),
}

st.markdown(
    """
    <style>
    .block-container { max-width: 840px; padding-top: 2.5rem; }
    h1 { font-weight: 800; letter-spacing: -0.02em; }
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .confidence-badge {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .upload-line { font-size: 0.92rem; margin: 2px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _escape_markdown_dollars(text: str) -> str:
    """Streamlit renders $...$ as LaTeX math, which mangles prices like '$12/month'."""
    return text.replace("$", "\\$")


def confidence_badge_html(confidence: str) -> str:
    default = (confidence, "#6b7280", "rgba(107,114,128,0.15)")
    label, color, bg = CONFIDENCE_STYLE.get(confidence, default)
    style = f"color:{color};background:{bg};"
    return f'<span class="confidence-badge" style="{style}">{label} confidence</span>'


def render_report(report: Report, tool_calls: list[str], loop_count: int) -> None:
    st.markdown(f"#### {report.title}")
    st.markdown(confidence_badge_html(report.confidence), unsafe_allow_html=True)
    st.write("")
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
        label = ", ".join(tools_used) if tools_used else "none"
        st.caption(f"**Tools called:** {label} · **loops:** {loop_count}")


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def ingest_files(files) -> list[str]:
    """Ingest each uploaded file, returning one status line per file."""
    lines = []
    for f in files:
        try:
            result = ingest_uploaded_file(f.name, f.getvalue())
            original = _format_bytes(result["original_bytes"])
            archived = _format_bytes(result["archived_bytes"])
            ratio = (
                result["original_bytes"] / result["archived_bytes"]
                if result["archived_bytes"]
                else 1
            )
            line = (
                f"✅ **{f.name}** — {result['chunks']} chunks added · "
                f"stored as {archived} (down from {original}, {ratio:.1f}x smaller)"
            )
            if result["truncated"]:
                line += (
                    f"  \n&nbsp;&nbsp;⚠️ only the first {MAX_CHUNKS_PER_FILE} of "
                    f"{result['total_chunks_found']} chunks were embedded (free-tier quota guard)"
                )
            lines.append(line)
        except (UnsupportedFileTypeError, EmptyDocumentError) as exc:
            lines.append(f"⚠️ **{f.name}** — {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any extraction/network failure to the user
            lines.append(f"❌ **{f.name}** — failed: {exc}")
    return lines


with st.sidebar:
    st.markdown("### 📚 Knowledge base")
    try:
        sources = list_sources()
    except Exception:
        sources = []
    if sources:
        st.caption(f"{len(sources)} document(s) indexed")
        for name in sources:
            st.markdown(f"- {name}")
    else:
        st.caption("No documents indexed yet.")
    st.divider()
    ext_list = ", ".join(SUPPORTED_EXTENSIONS)
    st.caption(f"Attach files in the chat box below to add more — {ext_list}")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.title("🧭 Report Agent")
st.caption(
    "LangGraph agent: retrieves from Qdrant, grades sufficiency, calls web search or a "
    "calculator when needed, and writes a structured report. Attach files to add them "
    "to the knowledge base."
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.write(turn["content"])
        else:
            if turn.get("upload_lines"):
                for line in turn["upload_lines"]:
                    st.markdown(f'<div class="upload-line">{line}</div>', unsafe_allow_html=True)
                if turn.get("report"):
                    st.write("")
            if turn.get("report"):
                render_report(turn["report"], turn["tool_calls"], turn["loop_count"])

submission = st.chat_input(
    "Ask a question, or attach files to add to the knowledge base...",
    accept_file="multiple",
    file_type=list(SUPPORTED_EXTENSIONS),
    max_upload_size=200,
)

if submission:
    question = submission.text.strip()
    uploaded_files = submission.files

    user_display = question
    if uploaded_files:
        attachment_line = "📎 " + ", ".join(f.name for f in uploaded_files)
        user_display = f"{attachment_line}\n\n{question}" if question else attachment_line

    st.session_state.history.append({"role": "user", "content": user_display})
    with st.chat_message("user"):
        st.write(user_display)

    with st.chat_message("assistant"):
        upload_lines = []
        if uploaded_files:
            with st.spinner(f"Reading {len(uploaded_files)} file(s)..."):
                upload_lines = ingest_files(uploaded_files)
            for line in upload_lines:
                st.markdown(f'<div class="upload-line">{line}</div>', unsafe_allow_html=True)
            if question:
                st.write("")

        report = tool_calls = loop_count = None
        if question:
            try:
                with st.spinner("Retrieving, grading, and writing report..."):
                    graph = build_graph()
                    result = graph.invoke({"question": question})
                report = result["report"]
                tool_calls = result.get("tool_calls_made", [])
                loop_count = result.get("loop_count", 0)
                render_report(report, tool_calls, loop_count)
            except Exception as exc:
                st.error(
                    "The agent hit an error calling Gemini or Qdrant "
                    f"(often a free-tier rate limit): {exc}"
                )

    st.session_state.history.append(
        {
            "role": "assistant",
            "upload_lines": upload_lines,
            "report": report,
            "tool_calls": tool_calls,
            "loop_count": loop_count,
        }
    )
