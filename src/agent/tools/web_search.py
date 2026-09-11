"""Free web search tool via DuckDuckGo (no API key or signup needed).

Fetches the actual page content of the top results in parallel, not just the search
snippet -- a two-line DuckDuckGo blurb often doesn't contain the specific fact/number a
question needs (e.g. a customs duty percentage stated on a government page but not in
its search-result summary). Falls back to the snippet per-result if a page can't be
fetched (blocked, times out, non-HTML), rather than failing the whole search.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import trafilatura
from ddgs import DDGS

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
FETCH_TIMEOUT = 8.0  # seconds per page; a slow site shouldn't stall the whole search
MAX_CONTENT_CHARS = 3000  # per page, keeps prompt size sane across MAX_RESULTS fetched pages
USER_AGENT = "Mozilla/5.0 (compatible; ReportAgent/1.0)"


class WebSearchError(RuntimeError):
    pass


def _fetch_page_text(url: str) -> str | None:
    """Best-effort fetch + extract a page's main text. None on any failure (caller falls
    back to the search snippet)."""
    try:
        response = httpx.get(
            url, timeout=FETCH_TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.info("  page fetch failed for %s: %s", url, exc)
        return None

    text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
    if not text:
        return None
    return text[:MAX_CONTENT_CHARS]


def web_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Return [{"title", "url", "snippet", "content"}, ...] for `query`.

    `content` is the extracted full-page text when the fetch succeeds, else None -- callers
    should fall back to `snippet` in that case.
    """
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as exc:  # network/upstream failures shouldn't crash the graph
        raise WebSearchError(f"Web search failed for {query!r}: {exc}") from exc

    urls = [r.get("href", "") for r in results]
    with ThreadPoolExecutor(max_workers=max(len(urls), 1)) as pool:
        contents = list(pool.map(lambda u: _fetch_page_text(u) if u else None, urls))

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
            "content": content,
        }
        for r, content in zip(results, contents, strict=True)
    ]
