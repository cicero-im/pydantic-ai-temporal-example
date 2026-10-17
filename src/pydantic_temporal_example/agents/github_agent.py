"""GitHub-focused agent returning structured responses for repository and issue tasks."""

from __future__ import annotations

import logfire
import uvloop
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import AgentRunError, ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

from pydantic_temporal_example.tools import GitHubConn

provider = ClaudeCodeProvider({"use_sandbox_runtime": False, "model": "opus", "fallback_model": "sonnet"})
model_instance = ClaudeCodeModel("opus", provider=provider)


class GitHubDependencies(BaseModel):
    """Dependencies for the GitHub agent."""

    repo_name: str = Field(..., description="Repository name (without organization)")
    pr_number: int = Field(default=1, description="Pull request number")
    path: str = Field(default="", description="Path within the repository (default: root)")


class GitHubResponse(BaseModel):
    """Structured output produced by the GitHub agent."""

    response: str = Field(description="The formatted message to show to the user.")


# Define tool functions first (before agent creation)
async def view_repo_files(
    ctx: RunContext[GitHubDependencies],
    repo_name: str | None = None,
    path: str | None = None,
) -> str:
    """View files in the repository at the specified path.

    Args:
        ctx: Runtime context with dependencies
        repo_name: Optional repository name override (defaults to ctx.deps.repo_name)
        path: Optional path override (defaults to ctx.deps.path)

    Returns:
        Formatted string listing the files and directories
    """
    resolved_repo = repo_name or ctx.deps.repo_name
    resolved_path = path if path is not None else ctx.deps.path
    github = GitHubConn()
    files = github.get_repo_files(resolved_repo, resolved_path)
    result = [f"Files in {resolved_repo}/{resolved_path or 'root'}:"]
    for file in files:
        file_type = "📁" if file.type == "dir" else "📄"
        result.append(f"{file_type} {file.path}")
    return "\n".join(result)


async def view_pull_request(
    ctx: RunContext[GitHubDependencies],
    repo_name: str | None = None,
    pr_number: int | None = None,
) -> str:
    """View details of a specific pull request.

    Args:
        ctx: Runtime context with dependencies
        repo_name: Optional repository name override (defaults to ctx.deps.repo_name)
        pr_number: Optional PR number override (defaults to ctx.deps.pr_number)

    Returns:
        Formatted string with PR details
    """
    resolved_repo = repo_name or ctx.deps.repo_name
    resolved_pr_number = pr_number if pr_number is not None else ctx.deps.pr_number
    github = GitHubConn()
    pr = github.get_pull_request(resolved_repo, resolved_pr_number)
    return (
        f"Pull Request #{pr.number}: {pr.title}\n"
        f"State: {pr.state}\n"
        f"Author: {pr.user.login}\n"
        f"Created: {pr.created_at}\n"
        f"Description: {pr.body or 'No description'}\n"
        f"Changed Files: {pr.changed_files}\n"
        f"Additions: +{pr.additions} | Deletions: -{pr.deletions}"
    )


async def view_pr_comments(
    ctx: RunContext[GitHubDependencies],
    repo_name: str | None = None,
    pr_number: int | None = None,
) -> str:
    """View all comments on a pull request.

    Args:
        ctx: Runtime context with dependencies
        repo_name: Optional repository name override (defaults to ctx.deps.repo_name)
        pr_number: Optional PR number override (defaults to ctx.deps.pr_number)

    Returns:
        Formatted string with all comments
    """
    resolved_repo = repo_name or ctx.deps.repo_name
    resolved_pr_number = pr_number if pr_number is not None else ctx.deps.pr_number
    github = GitHubConn()
    comments = github.get_pr_comments(resolved_repo, resolved_pr_number)
    if not comments:
        return f"No comments found on PR #{resolved_pr_number}"

    result = [f"Comments on PR #{resolved_pr_number}:"]
    for comment in comments:
        comment_type = "💬" if comment["type"] == "issue_comment" else "📝"
        result.append(f"\n{comment_type} {comment['user']} ({comment['created_at']}):\n{comment['body']}")
        if "path" in comment:
            result.append(f"  File: {comment['path']}")

    return "\n".join(result)


async def view_branches(ctx: RunContext[GitHubDependencies], repo_name: str | None = None) -> str:
    """View all branches in the repository.

    Args:
        ctx: Runtime context with dependencies
        repo_name: Optional repository name override (defaults to ctx.deps.repo_name)

    Returns:
        Formatted string listing all branches
    """
    resolved_repo = repo_name or ctx.deps.repo_name
    github = GitHubConn()
    branches = github.get_branches(resolved_repo)
    logfire.info(f"Branches in {resolved_repo}: {branches}")
    result = [f"Branches in {resolved_repo}:"]
    for branch in branches:
        protected = "🔒" if branch["protected"] else "🔓"
        result.append(f"{protected} {branch['name']} ({branch['sha'][:7]})")
    logfire.info(f"Branches in {resolved_repo}: {result}")
    return "\n".join(result)


async def list_all_pull_requests(
    ctx: RunContext[GitHubDependencies],
    repo_name: str | None = None,
    state: str = "all",
) -> str:
    """List all pull requests in the repository.

    Args:
        ctx: Runtime context with dependencies
        repo_name: Optional repository name override (defaults to ctx.deps.repo_name)
        state: PR state filter ('open', 'closed', or 'all')

    Returns:
        Formatted string listing all PRs
    """
    resolved_repo = repo_name or ctx.deps.repo_name
    github = GitHubConn()
    prs = github.list_pull_requests(resolved_repo, state)
    if not prs:
        return f"No pull requests found in {resolved_repo}"

    result = [f"Pull Requests in {resolved_repo}:"]
    result.extend(
        f"\n#{pr['number']}: {pr['title']}"
        f"\n  State: {pr['state']} | Author: {pr['author']}"
        f"\n  Created: {pr['created_at']}"
        for pr in prs
    )
    logfire.info(f"Pull Requests in {resolved_repo}: {result}")
    return "\n".join(result)


# Create agent and register tools
github_agent = Agent(
    model=model_instance,
    deps_type=GitHubDependencies,
    output_type=str,
    system_prompt=(
        "You are a GitHub analysis agent that helps users understand "
        "repositories, pull requests, and code structure. "
        "Provide clear, informative responses based on the repository data. "
        "The repository name is provided via runtime context (deps). "
        "Use tools without guessing placeholders like 'current'. "
        "You should FOLLOW THE INSTRUCTIONS CAREFULLY, USE THE TOOLS AND THEN PROVIDE YOUR OUTPUT. "
        "Return the final answer as plain text markdown only."
    ),
)

# Register all tools
github_agent.tool(view_repo_files)
github_agent.tool(view_pull_request)
github_agent.tool(view_pr_comments)
github_agent.tool(view_branches)
github_agent.tool(list_all_pull_requests)


if __name__ == "__main__":
    # Example usage (requires GITHUB_PAT and GITHUB_ORG environment variables)

    async def main() -> None:
        """Run the GitHub agent example to demonstrate repository analysis."""
        try:
            logfire.info(f"Running GitHub agent with this: {GitHubConn()!s}")
            deps_instance = GitHubDependencies(repo_name="pydantic-ai-temporal-example", pr_number=1)
            logfire.info(f"Running GitHub agent with deps: {deps_instance}")
            result = await github_agent.run(
                "Show me the branches in this repository entitled pydantic-ai-temporal-example",
                deps=deps_instance,  # type: ignore[arg-type]
            )
            logfire.info(f"GitHub agent result: {result.output}")
        except AgentRunError as e:
            logfire.error(f"Agent run failed: {type(e).__name__} - {e!s}", exc_info=True)
        except UserError as e:
            logfire.warning(f"User error in GitHub agent: {e!s}", exc_info=True)
        except ApprovalRequired as e:
            logfire.info(f"Approval required for GitHub agent: {e!s}", exc_info=True)
        except CallDeferred as e:
            logfire.info(f"Call deferred in GitHub agent: {e!s}", exc_info=True)
        except ModelRetry as e:
            logfire.warning(f"Model retry requested in GitHub agent: {e!s}", exc_info=True)

    uvloop.run(main())
    """
    Example output:
    response='The repository has the following branches: main (protected), develop (not protected), ...'
    repo_analyzed='pydantic-ai-temporal-example'
    """
