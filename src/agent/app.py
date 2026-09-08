"""FastAPI front end for the report agent."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import build_graph  # noqa: E402
from agent.logging_config import configure_logging  # noqa: E402
from agent.schemas import Report  # noqa: E402

configure_logging()

app = FastAPI(title="LangGraph Report Agent")
_graph = build_graph()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    report: Report
    tool_calls_made: list[str]
    loop_count: int
    trace: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = _graph.invoke({"question": request.question})
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Upstream Gemini/Qdrant call failed (often a free-tier rate limit): {exc}",
        ) from exc
    return AskResponse(
        report=result["report"],
        tool_calls_made=result.get("tool_calls_made", []),
        loop_count=result.get("loop_count", 0),
        trace=result.get("trace", []),
    )
