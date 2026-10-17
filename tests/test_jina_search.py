import pytest
import httpx
from pydantic_temporal_example.tools.jina_search import JinaSearchTool, jina_search


def test_time_range_helpers():
    tool = JinaSearchTool(api_key="k")
    s = tool._build_time_range_filter("day")
    assert "after:" in s
    q = tool._append_time_range_to_prompt("hello", "m")
    assert "last month" in q


@pytest.mark.asyncio
async def test_basic_json_response(monkeypatch):
    class FakeResponse:
        def __init__(self, data):
            self._data = data
            self.text = ""
        def json(self):
            return self._data
        def raise_for_status(self):  # no-op
            return None
    class FakeAsyncClient:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def get(self, url, headers=None, params=None):
            assert "s.jina.ai" in url
            return FakeResponse({"data": [{"title": "t", "url": "u", "content": "c", "score": 0.9}]})
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient, raising=True)

    tool = JinaSearchTool(api_key="k")
    res = await tool("query")
    assert res and res[0]["title"] == "t"


@pytest.mark.asyncio
async def test_basic_text_fallback(monkeypatch):
    class FakeResponse:
        _ERROR_MSG = "not json"

        def __init__(self, text):
            self._text = text
        def json(self):
            raise ValueError(self._ERROR_MSG)
        @property
        def text(self):
            return self._text
        def raise_for_status(self):
            return None
    class FakeAsyncClient:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def get(self, url, headers=None, params=None):
            return FakeResponse("plain text body")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient, raising=True)

    tool = JinaSearchTool(api_key="k")
    res = await tool("query")
    assert res and res[0]["content"].startswith("plain text")


@pytest.mark.asyncio
async def test_jina_search_wrapper_truncates(monkeypatch):
    async def fake_call(_self, query, search_deep="basic", time_range=None):  # noqa: ARG001
        return [{"title": str(i), "url": "", "content": "", "score": 0.0} for i in range(10)]
    monkeypatch.setattr(JinaSearchTool, "__call__", fake_call, raising=True)
    out = await jina_search("q", max_results=5)
    assert len(out) == 5
