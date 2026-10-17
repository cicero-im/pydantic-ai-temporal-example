import pytest
import pydantic_temporal_example.temporal.github_activities as ga
from pydantic_temporal_example.agents.github_agent import GitHubResponse


class FakeAgentResult:
    def __init__(self, response: str):
        self.output = GitHubResponse(response=response)


@pytest.mark.asyncio
async def test_fetch_github_prs_monkeypatched(monkeypatch):
    async def fake_run(query: str, deps):  # noqa: ANN001
        return FakeAgentResult(response=f"Ran: {query}")

    monkeypatch.setattr(ga.github_agent, "run", fake_run, raising=True)
    res = await ga.fetch_github_prs("repo", "List all pull requests in the repository")
    assert isinstance(res, GitHubResponse)
    assert "List all pull requests" in res.response
