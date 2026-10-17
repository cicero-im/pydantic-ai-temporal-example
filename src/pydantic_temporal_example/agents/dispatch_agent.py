"""Dispatcher agent that routes to Slack, web research, or GitHub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import with_config
from pydantic_ai import Agent, WebSearchUserLocation
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai_claude_code import ClaudeCodeProvider

from pydantic_temporal_example.config import get_github_agent_model

provider = ClaudeCodeProvider({"use_sandbox_runtime": False})


@dataclass
@with_config(use_attribute_docstrings=True)
class NoResponse:
    """A marker that indicates that you do not currently need to reply to the thread.

    You should use this when the most recent messages in the thread are not directed toward you,
    e.g. if other users are discussing something between themselves.

    If you select this, no response to the latest messages in the thread will be sent.
    """

    type: Literal["no-response"]


@dataclass
@with_config(use_attribute_docstrings=True)
class SlackResponse:
    """A marker that indicates that you want to immediately send a response message.

    You can use this to request additional information from the user if they have not provided all the information
    necessary to make a WebResearchRequest.

    This tool provides a way to respond to the thread without triggering the web research agent.
    For example, if you ask the user a question, then they ask for clarification, you can use this to
    provide that clarification under the assumption that they will subsequently respond to your initial question.
    """

    type: Literal["slack-response"]
    response: str | list[dict[str, Any]]
    """
    The response to show to the user. This should either be a markdown text string, or valid Slack blockkit blocks
    """


@dataclass
@with_config(use_attribute_docstrings=True)
class WebResearchRequest:
    """A marker that indicates that you are ready to delegate to the web research agent.

    If you have not already provided suggestions in the thread, you should generally use this tool whenever you have the
    necessary information to do so, unless the user has made a different request that is better handled by a direct
    response. If you don't have the necessary information, consider using the SlackResponse tool to request that
    additional information.
    """

    type: Literal["web-research-request"]
    location: WebSearchUserLocation
    query: str
    extra_info: str | None
    thread_messages: list[dict[str, Any]] | None = None
    """Full Slack thread context for agent reference"""


@dataclass
@with_config(use_attribute_docstrings=True)
class GitHubRequest:
    """A marker that indicates that you are ready to delegate to the github agent."""

    type: Literal["github-request"]
    query: str
    extra_info: str | None
    thread_messages: list[dict[str, Any]] | None = None
    """Full Slack thread context for agent reference"""


@dataclass
@with_config(use_attribute_docstrings=True)
class BlogPostRequest:
    """A marker that indicates that you are ready to delegate to the blog post agent."""

    type: Literal["blog-post-request"]
    query: str
    extra_info: str | None
    thread_messages: list[dict[str, Any]] | None = None
    """Full Slack thread context for agent reference"""


@dataclass
@with_config(use_attribute_docstrings=True)
class WorkflowRequest:
    """Generic workflow request with agent routing and scheduling information.

    Use this when you need to specify:
    - Which agent type and role to use
    - Whether it should run once or repeatedly
    - How often it should repeat (if periodic)

    Examples:
    - "Review PRs every hour" → agent_type="github", agent_role="reviewer",
      workflow_type="periodic", interval_seconds=3600
    - "Implement auth feature" → agent_type="github", agent_role="implementer",
      workflow_type="oneshot"
    - "Analyze code quality every 30 mins" → agent_type="github",
      agent_role="analyzer", workflow_type="periodic", interval_seconds=1800
    """

    type: Literal["workflow-request"]
    agent_type: str
    """Type of agent: 'github', 'web_research', 'blog_post', 'slack'"""

    query: str
    """The actual query/instruction for the agent"""

    agent_role: str = "default"
    """Role specialization:
    - **GitHub Agents:** 'implementer', 'reviewer', 'fixer', 'verifier', 'analyzer', 'documenter', 'default'
    - **Web Research Agents:** 'default'
    - **Blog Post Agents:** 'default', 'editor', 'seo'
    - **Slack:** 'default'
    """

    workflow_type: Literal["oneshot", "periodic"] = "oneshot"
    """Whether to run once or repeatedly"""

    interval_seconds: int | None = None
    """Repeat interval in seconds (required if workflow_type='periodic')"""

    extra_info: str | None = None
    """Additional context for the agent"""

    thread_messages: list[dict[str, Any]] | None = None
    """Full conversation thread context"""

    def __post_init__(self) -> None:
        """Validate that interval_seconds is set and positive for periodic workflows."""
        if self.workflow_type == "periodic" and self.interval_seconds is None:
            msg = "interval_seconds is required when workflow_type='periodic'"
            raise ValueError(msg)
        if self.workflow_type == "periodic" and self.interval_seconds is not None and self.interval_seconds <= 0:
            msg = "interval_seconds must be a positive value when workflow_type='periodic'"
            raise ValueError(msg)


type DispatchResult = NoResponse | SlackResponse | WebResearchRequest | GitHubRequest | BlogPostRequest | WorkflowRequest

dispatch_agent = Agent(
    model=get_github_agent_model(),
    output_type=[NoResponse, SlackResponse, WebResearchRequest, GitHubRequest, BlogPostRequest, WorkflowRequest],
    instructions="""
    You are a dispatch agent that routes user requests to appropriate agents and workflows.

    Your primary tool is WorkflowRequest, which should be used for most requests. It allows you to specify:
    1. **agent_type**: github, web_research, blog_post, slack
    2. **agent_role**: Specialization (see below)
    3. **workflow_type**: oneshot (run once) or periodic (repeat)
    4. **interval_seconds**: How often to repeat (if periodic)

    ## Agent Types and Roles

    **GitHub Agents (All use same tools, different instructions):**
    - **implementer**: Implements features, writes code, creates new functionality
    - **reviewer**: Reviews PRs, checks for bugs, security, and code quality
    - **fixer**: Fixes bugs and issues identified in reviews or reports
    - **verifier**: Verifies that fixes work and features meet requirements
    - **analyzer**: Analyzes codebase metrics, patterns, tech debt
    - **documenter**: Generates documentation from code
    - **default**: General GitHub operations

    **Web Research:**
    - default: Research topics, gather information

    **Blog Post:**
    - default: Draft and improve blog posts in repository markdown files
    - editor: Improve clarity, flow, and structure
    - seo: Improve discoverability, titles, headings, and metadata

    **Slack:**
    - default: Direct response messages

    ## Examples

    User: "Review PRs for security issues every hour"
    → WorkflowRequest(agent_type="github", agent_role="reviewer", workflow_type="periodic", interval_seconds=3600)

    User: "Implement user authentication"
    → WorkflowRequest(agent_type="github", agent_role="implementer", workflow_type="oneshot")

    User: "Fix the bugs in PR #123"
    → WorkflowRequest(agent_type="github", agent_role="fixer", workflow_type="oneshot", query="Fix bugs in PR #123")

    User: "Verify that the authentication feature works"
    → WorkflowRequest(agent_type="github", agent_role="verifier", workflow_type="oneshot")

    User: "Analyze code quality every 30 minutes"
    → WorkflowRequest(agent_type="github", agent_role="analyzer", workflow_type="periodic", interval_seconds=1800)

    User: "Document the API"
    → WorkflowRequest(agent_type="github", agent_role="documenter", workflow_type="oneshot")

    User: "Draft a blog post about our multi-agent architecture"
    → WorkflowRequest(agent_type="blog_post", agent_role="default", workflow_type="oneshot")

    User: "Improve SEO for blog post docs/blog/multi-agent.md"
    → WorkflowRequest(agent_type="blog_post", agent_role="seo", workflow_type="oneshot")

    ## Multi-Step Workflows

    When user asks to "fix a PR", you should understand this as potentially multiple steps:
    1. First, use reviewer to identify issues
    2. Then fixer to fix them
    3. Finally verifier to confirm

    However, YOU only return ONE WorkflowRequest at a time for the CURRENT step.
    Multi-step orchestration happens externally.

    ## Backward Compatibility

    Legacy tools (GitHubRequest, BlogPostRequest, WebResearchRequest, SlackResponse, NoResponse) still work
    but prefer WorkflowRequest for its flexibility with roles and scheduling.

    If missing information, use SlackResponse to ask the user.
    """,
    tools=[duckduckgo_search_tool()],
)
