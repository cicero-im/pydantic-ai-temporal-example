"""CLI entry point for the Temporal worker agent.

Provides a command-line interface to start the Temporal worker with configurable
host, port, and task queue options.
"""

import asyncio
import os
import signal
import sys
from typing import Annotated, Any

import httpx
import logfire
import typer
import uvloop

from pydantic_temporal_example.agents.blog_post_agent import BlogPostDependencies, blog_post_agent
from pydantic_temporal_example.agents.github_agent import GitHubDependencies, github_agent
from pydantic_temporal_example.config import get_settings
from pydantic_temporal_example.temporal.client import build_temporal_client
from pydantic_temporal_example.temporal.github_activities import fetch_github_prs
from pydantic_temporal_example.temporal.worker import temporal_worker
from pydantic_temporal_example.temporal.workflows import PeriodicGitHubPRCheckWorkflow
from pydantic_temporal_example.tools.jina_search import jina_search

app = typer.Typer()


# HTTP client functions for interacting with the API
async def send_workflow_request(
    prompt: str,
    *,
    app_host: str = "127.0.0.1",
    app_port: int = 4000,
    repeat: bool = False,
    repeat_interval: int = 30,
    repo_name: str = "default-repo",
    session_id: str | None = None,
    use_https: bool | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Send a workflow request to the API server.

    Args:
        prompt: The command/prompt to execute
        app_host: API server host
        app_port: API server port
        repeat: Whether to repeat the workflow
        repeat_interval: Interval in seconds for repetition
        repo_name: Repository name for GitHub operations
        session_id: Optional session identifier
        use_https: Whether to use HTTPS. If None, checks API_USE_HTTPS env var (default: False)
        max_retries: Maximum number of retries for transient failures (default: 3)

    Returns:
        JSON response from the API containing workflow_id and status

    Raises:
        httpx.HTTPStatusError: If the API returns an error status (4xx/5xx)
        httpx.RequestError: If the request fails due to network issues
        httpx.TimeoutException: If the request times out after all retries
    """
    # Determine protocol from parameter or environment variable
    if use_https is None:
        use_https = os.getenv("API_USE_HTTPS", "false").lower() == "true"
    protocol = "https" if use_https else "http"
    url = f"{protocol}://{app_host}:{app_port}/cli-workflow"

    payload = {
        "prompt": prompt,
        "repeat": repeat,
        "repeat_interval": repeat_interval,
        "repo_name": repo_name,
    }
    if session_id:
        payload["session_id"] = session_id

    # Configure retry logic for transient failures
    transport = httpx.AsyncHTTPTransport(retries=max_retries)

    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logfire.error(
                "API returned error status",
                status_code=e.response.status_code,
                url=url,
                error=str(e),
            )
            raise
        except httpx.RequestError as e:
            logfire.error(
                "Request failed due to network error",
                url=url,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise


async def check_workflow_response(
    workflow_id: str,
    *,
    app_host: str = "127.0.0.1",
    app_port: int = 4000,
    use_https: bool | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Check the response from a workflow.

    Handles both single workflow IDs and composite IDs (comma-separated).
    For composite IDs, queries the first workflow (one-shot execution).

    Args:
        workflow_id: The workflow ID to check (single or composite)
        app_host: API server host
        app_port: API server port
        use_https: Whether to use HTTPS. If None, checks API_USE_HTTPS env var (default: False)
        max_retries: Maximum number of retries for transient failures (default: 3)

    Returns:
        JSON response containing:
        - status: "pending" or "completed"
        - response: The workflow response (if completed)
        - workflow_id: The actual workflow ID queried (first ID if composite)

    Raises:
        httpx.HTTPStatusError: If the API returns an error status (4xx/5xx)
        httpx.RequestError: If the request fails due to network issues
        httpx.TimeoutException: If the request times out after all retries
    """
    # Determine protocol from parameter or environment variable
    if use_https is None:
        use_https = os.getenv("API_USE_HTTPS", "false").lower() == "true"
    protocol = "https" if use_https else "http"
    url = f"{protocol}://{app_host}:{app_port}/cli-workflow/{workflow_id}/response"

    # Configure retry logic for transient failures
    transport = httpx.AsyncHTTPTransport(retries=max_retries)

    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logfire.error(
                "API returned error status",
                status_code=e.response.status_code,
                url=url,
                workflow_id=workflow_id,
                error=str(e),
            )
            raise
        except httpx.RequestError as e:
            logfire.error(
                "Request failed due to network error",
                url=url,
                workflow_id=workflow_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise


@app.command()
def github_prs(
    repo: Annotated[str, typer.Option(help="Repository name")] = "potion",
    query: Annotated[
        str,
        typer.Option(help="Query for the GitHub agent"),
    ] = "List all pull requests in the repository",
) -> None:
    """Query GitHub agent for all PRs once using Temporal activity."""

    async def _run() -> None:
        logfire.info(f"Fetching all PRs from {repo}...")
        logfire.info(f"Query: {query}")

        try:
            await fetch_github_prs(repo, query)

            logfire.info("GitHub Agent Response:")
        except RuntimeError as e:
            logfire.error(f"Error fetching PRs: {e}")

    asyncio.run(_run())


@app.command()
def blog_post(
    repo: Annotated[str, typer.Option(help="Repository name")] = "pydantic-ai-temporal-example",
    title: Annotated[str, typer.Option(help="Blog post title")] = "Building Multi-Agent Workflows with Temporal",
    slug: Annotated[str, typer.Option(help="Slug for markdown filename (without .md)")] = "multi-agent-workflows",
    summary: Annotated[str, typer.Option(help="Short post summary")] = (
        "How to orchestrate specialized agents with Temporal workflows."
    ),
    branch: Annotated[str, typer.Option(help="Target branch to commit into")] = "main",
    blog_dir: Annotated[str, typer.Option(help="Directory where posts are stored")] = "docs/blog",
    enhance_prompt: Annotated[str, typer.Option(help="Instruction for enhancement pass")] = (
        "Improve clarity, examples, and actionable takeaways."
    ),
) -> None:
    """Start a blog post commit in GitHub and ask the blog-post agent to enhance it."""

    async def _run() -> None:
        deps = BlogPostDependencies(repo_name=repo, blog_dir=blog_dir, branch=branch)
        path = f"{blog_dir.rstrip('/')}/{slug}.md"

        start_query = (
            f"Start a new blog post.\n"
            f"Title: {title}\n"
            f"Slug: {slug}\n"
            f"Summary: {summary}\n"
            f"Use start_blog_post and commit to branch {branch}."
        )
        logfire.info("Starting blog post commit", repo=repo, path=path, branch=branch)
        start_result = await blog_post_agent.run(start_query, deps=deps)  # type: ignore[arg-type]
        logfire.info("Start result", result=str(start_result.output))

        enhance_query = (
            f"Enhance the blog post at {path}.\n"
            f"Goals: {enhance_prompt}\n"
            f"Steps:\n"
            f"1) Read the post\n"
            f"2) Improve it with concrete examples and structure\n"
            f"3) Commit the updated markdown with message 'feat(blog): enhance {slug}'"
        )
        logfire.info("Enhancing blog post with agent", repo=repo, path=path, branch=branch)
        enhance_result = await blog_post_agent.run(enhance_query, deps=deps)  # type: ignore[arg-type]
        logfire.info("Enhancement result", result=str(enhance_result.output))

        print("Start Result:")
        print(start_result.output)
        print("\nEnhancement Result:")
        print(enhance_result.output)

    asyncio.run(_run())


@app.command()
def github_prs_periodic(
    repo: Annotated[str, typer.Option(help="Repository name")] = "potion",
    interval: Annotated[int, typer.Option(help="Check interval in seconds")] = 30,
    query: Annotated[
        str,
        typer.Option(help="Query for the GitHub agent"),
    ] = "List all pull requests in the repository",
) -> None:
    """Run periodic GitHub PR checks using Temporal workflow."""

    async def _run() -> None:
        logfire.info(f"Starting periodic PR checks for {repo} every {interval}s...")
        logfire.info(f"Query: {query}")
        logfire.info("Make sure the Temporal worker is running in another terminal!")
        logfire.info("Run: python -m pydantic_temporal_example.cli main")

        try:
            # Connect to Temporal
            client = await build_temporal_client()

            # Start the workflow
            workflow_id = f"github-pr-check-{repo}"
            handle = await client.start_workflow(
                PeriodicGitHubPRCheckWorkflow.periodic_run,
                args=[repo, interval, query],
                id=workflow_id,
                task_queue="agent-task-queue",
            )

            logfire.info(f"Started workflow: {workflow_id}")

            # Wait for workflow (or Ctrl+C)
            try:
                await handle.result()
            except KeyboardInterrupt:
                logfire.info("Stopping workflow...")
                await handle.signal(PeriodicGitHubPRCheckWorkflow.stop)

        except RuntimeError as e:
            logfire.error(f"Error running periodic checks: {e}")

    asyncio.run(_run())


@app.command()
def jina_research(
    query: Annotated[str, typer.Option(help="Research query")] = "pydantic_ai",
    interval: Annotated[int, typer.Option(help="Interval in seconds")] = 30,
    iterations: Annotated[int, typer.Option(help="Number of iterations (0 = infinite)")] = 0,
) -> None:
    """Run periodic Jina research on a topic."""

    async def _run() -> None:
        count = 0
        while True:
            count += 1
            if iterations > 0 and count > iterations:
                break

            logfire.info(f"Research iteration {count}: Searching for '{query}'...")

            try:
                results = await jina_search(query, max_results=5)
                logfire.info(f"Found {len(results)} results")

                for i, result in enumerate(results, 1):
                    if "content" in result:
                        preview = result["content"][:200]
                        logfire.info(f"Result {i}: {preview}...")

            except RuntimeError as e:
                logfire.error(f"Error during research: {e}")

            if iterations == 0 or count < iterations:
                logfire.info(f"Waiting {interval} seconds before next iteration...")
                await asyncio.sleep(interval)

    asyncio.run(_run())


@app.command()
def combined_task(
    repo: Annotated[str, typer.Option(help="Repository name")] = "pydantic-ai-temporal-example",
    research_query: Annotated[str, typer.Option(help="Jina research query")] = "pydantic_ai",
    research_interval: Annotated[int, typer.Option(help="Research interval in seconds")] = 30,
) -> None:
    """Run both GitHub PR analysis and periodic Jina research concurrently."""

    async def github_task() -> None:
        deps = GitHubDependencies(repo_name=repo)

        logfire.info(f"Fetching all PRs from {repo}...")
        await github_agent.run(
            "List all pull requests in the repository and include their comments for each PR",
            deps=deps,  # type: ignore[arg-type]
        )

    async def jina_task() -> None:
        count = 0
        while True:
            count += 1
            logfire.info(f"Jina research iteration {count}: '{research_query}'...")

            try:
                results = await jina_search(research_query, max_results=5)
                logfire.info(f"Found {len(results)} research results")

                for i, result in enumerate(results, 1):
                    if "content" in result:
                        preview = result["content"][:200]
                        logfire.info(f"Research result {i}: {preview}...")

            except RuntimeError as e:
                logfire.error(f"Jina research error: {e}")

            await asyncio.sleep(research_interval)

    async def _run() -> None:
        # Run both tasks concurrently
        await asyncio.gather(
            github_task(),
            jina_task(),
        )

    asyncio.run(_run())


app_typer = app


@app.command()
def main(
    host: Annotated[
        str | None,
        typer.Option(
            help="The host for the temporal worker.",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            help="The port for the temporal worker.",
        ),
    ] = None,
    task_queue: Annotated[
        str | None,
        typer.Option(
            help="The task queue for the temporal worker.",
        ),
    ] = None,
) -> None:
    """Temporal Agent CLI."""
    # Resolve settings at runtime, not at import time
    settings = get_settings()
    host = host or settings.temporal_host
    port = port or settings.temporal_port
    task_queue = task_queue or settings.temporal_task_queue

    async def _main() -> None:
        try:
            async with temporal_worker(
                host=host,
                port=port,
                task_queue=task_queue,
            ):
                logfire.info(
                    "Temporal worker started",
                    host=host,
                    port=port,
                    task_queue=task_queue,
                )

                # Set up graceful shutdown
                shutdown_event = asyncio.Event()

                def signal_handler(_sig: int, _frame: object) -> None:
                    logfire.info("Received shutdown signal, stopping worker...")
                    shutdown_event.set()

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                await shutdown_event.wait()
                logfire.info("Worker stopped")
        except RuntimeError as e:
            logfire.exception(f"Worker failed: {e}")
            sys.exit(1)

    uvloop.run(_main())


if __name__ == "__main__":
    app()
