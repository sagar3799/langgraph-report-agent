"""Free web search tool via DuckDuckGo (no API key or signup needed)."""

from ddgs import DDGS


class WebSearchError(RuntimeError):
    pass


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Return [{"title", "url", "snippet"}, ...] for `query`."""
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # network/upstream failures shouldn't crash the graph
        raise WebSearchError(f"Web search failed for {query!r}: {exc}") from exc

    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results
    ]
