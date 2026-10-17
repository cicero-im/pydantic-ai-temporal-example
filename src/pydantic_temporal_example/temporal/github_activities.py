"""Temporal activities for GitHub operations."""

from __future__ import annotations

import logfire
from temporalio import activity

from pydantic_temporal_example.agents.github_agent import GitHubDependencies, GitHubResponse, github_agent
from pydantic_temporal_example.config import get_github_org


@activity.defn
async def fetch_github_prs(repo_name: str, query: str = "List all pull requests in the repository") -> GitHubResponse:
    """Fetch all pull requests from a GitHub repository.

    Args:
        repo_name: Repository name (without organization)
        query: The query/instruction to pass to the GitHub agent

    Returns:
        GitHubResponse with PR information
    """
    org = get_github_org()
    logfire.info("Fetching PRs from repository", repo_name=repo_name, organization=org, query=query)

    deps = GitHubDependencies(repo_name=repo_name)
    result = await github_agent.run(query, deps=deps)  # type: ignore[arg-type]

    logfire.info("Successfully fetched PRs", repo_name=repo_name, organization=org)
    output = result.output
    if isinstance(output, GitHubResponse):
        return output
    return GitHubResponse(response=str(output))


# All GitHub activities
ALL_GITHUB_ACTIVITIES = [fetch_github_prs]
