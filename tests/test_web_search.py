import pytest

from agent.tools import web_search as web_search_module
from agent.tools.web_search import WebSearchError, web_search


class FakeDDGS:
    def __init__(self, results):
        self._results = results

    def text(self, query, max_results):
        return self._results[:max_results]


def test_web_search_uses_fetched_content_when_available(monkeypatch):
    monkeypatch.setattr(
        web_search_module,
        "DDGS",
        lambda: FakeDDGS([{"title": "T1", "href": "https://example.com", "body": "snippet"}]),
    )
    monkeypatch.setattr(web_search_module, "_fetch_page_text", lambda url: "full page text")

    results = web_search("query")

    assert results == [
        {
            "title": "T1",
            "url": "https://example.com",
            "snippet": "snippet",
            "content": "full page text",
        }
    ]


def test_web_search_falls_back_to_none_content_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        web_search_module,
        "DDGS",
        lambda: FakeDDGS([{"title": "T1", "href": "https://example.com", "body": "snippet"}]),
    )
    monkeypatch.setattr(web_search_module, "_fetch_page_text", lambda url: None)

    results = web_search("query")

    assert results[0]["content"] is None
    assert results[0]["snippet"] == "snippet"


def test_web_search_handles_result_with_no_url(monkeypatch):
    monkeypatch.setattr(
        web_search_module,
        "DDGS",
        lambda: FakeDDGS([{"title": "T1", "href": "", "body": "snippet"}]),
    )
    monkeypatch.setattr(
        web_search_module, "_fetch_page_text", lambda url: (_ for _ in ()).throw(AssertionError())
    )

    results = web_search("query")

    assert results[0]["content"] is None


def test_web_search_wraps_ddgs_exceptions(monkeypatch):
    class BrokenDDGS:
        def text(self, query, max_results):
            raise RuntimeError("network down")

    monkeypatch.setattr(web_search_module, "DDGS", BrokenDDGS)

    with pytest.raises(WebSearchError):
        web_search("query")
