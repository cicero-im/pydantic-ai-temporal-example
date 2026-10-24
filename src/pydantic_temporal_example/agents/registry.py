"""Agent registry for plug-and-play agent architecture.

Instead of storing separate agent instances per role, this registry dynamically creates
agents with role-specific instructions. This avoids code duplication while maintaining
specialization through instruction templates.

Key Innovation: ONE agent implementation + DIFFERENT instructions = MULTIPLE specializations
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

from pydantic_temporal_example.agents.blog_post_agent import BlogPostDependencies, blog_post_agent
from pydantic_temporal_example.agents.github_agent import GitHubDependencies, github_agent
from pydantic_temporal_example.agents.instruction_templates import get_instructions_for_role, list_available_roles
from pydantic_temporal_example.agents.web_research_agent import build_web_research_agent
from pydantic_temporal_example.config import get_github_agent_model

# Cache for dynamically created agents to avoid recreating them
# Using Any for the agent value type to support agents with different output types
# (GitHubResponse, WebResearchResponse, etc.)
_AGENT_CACHE: dict[tuple[str, str], Any] = {}


def _create_github_agent_with_role(role: str) -> Agent[GitHubDependencies, str]:
    """Create a GitHub agent with role-specific instructions.

    Args:
        role: Role specialization (implementer, reviewer, fixer, verifier, etc.)

    Returns:
        Agent configured with role-specific instructions
    """
    # Get role-specific instructions
    instructions = get_instructions_for_role("github", role)

    # Extract toolsets from base github_agent using public API
    # The toolsets parameter accepts the full toolset list from another agent
    toolsets_list = None
    if hasattr(github_agent, "toolsets") and github_agent.toolsets:
        toolsets_list = github_agent.toolsets

    # Create new agent with same configuration but different instructions
    # Pass the toolsets from the base agent via public API
    return Agent(
        model=get_github_agent_model(),
        output_type=str,
        deps_type=GitHubDependencies,
        instructions=instructions,
        toolsets=toolsets_list,  # Use toolsets from base agent via public API
    )


def _create_blog_post_agent_with_role(role: str) -> Agent[BlogPostDependencies, str]:
    """Create a blog post agent with role-specific instructions."""
    instructions = get_instructions_for_role("blog_post", role)

    toolsets_list = None
    if hasattr(blog_post_agent, "toolsets") and blog_post_agent.toolsets:
        toolsets_list = blog_post_agent.toolsets

    return Agent(
        model=get_github_agent_model(),
        output_type=str,
        deps_type=BlogPostDependencies,
        instructions=instructions,
        toolsets=toolsets_list,
    )


def get_agent(agent_type: str, agent_role: str = "default") -> Any:
    """Get agent with role-specific instructions (dynamically created).

    Instead of pre-creating multiple agent instances, this function creates agents
    on-demand with role-specific instructions. The agent implementation (tools, model)
    stays the same, only instructions change based on role.

    Args:
        agent_type: Type of agent (e.g., "github", "web_research")
        agent_role: Role specialization (e.g., "implementer", "reviewer", "fixer", "verifier")

    Returns:
        Agent instance with role-specific instructions, or None for direct responses

    Raises:
        KeyError: If agent type is not supported

    Examples:
        >>> agent = get_agent("github", "reviewer")  # Creates reviewer with review instructions
        >>> agent = get_agent("github", "fixer")     # Creates fixer with fix instructions
        >>> agent = get_agent("web_research")        # Uses default role
    """
    # Check cache first
    key = (agent_type, agent_role)
    if key in _AGENT_CACHE:
        return _AGENT_CACHE[key]

    # Create agent based on type
    if agent_type == "github":
        # Dynamically create GitHub agent with role-specific instructions
        agent = _create_github_agent_with_role(agent_role)
        _AGENT_CACHE[key] = agent
        return agent

    if agent_type == "web_research":
        # Web research doesn't have multiple roles yet, use default
        if key not in _AGENT_CACHE:
            _AGENT_CACHE[key] = build_web_research_agent()
        return _AGENT_CACHE[key]

    if agent_type == "blog_post":
        agent = _create_blog_post_agent_with_role(agent_role)
        _AGENT_CACHE[key] = agent
        return agent

    if agent_type == "slack":
        # Slack is a direct response, no agent needed
        return None

    # Agent type not supported
    msg = f"Agent type '{agent_type}' not supported. Available types: github, web_research, blog_post, slack"
    raise KeyError(
        msg,
    )


def list_available_agent_roles() -> dict[str, list[str]]:
    """List all available agent types and their roles.

    Returns:
        Dictionary mapping agent_type to list of available roles

    Example:
        >>> list_available_agent_roles()
        {
            "github": ["default", "implementer", "reviewer", "fixer", "verifier", "analyzer", "documenter"],
            "web_research": ["default"],
            "blog_post": ["default", "editor", "seo"],
            "slack": ["default"]
        }
    """
    return {
        "github": list_available_roles("github"),
        "web_research": ["default"],
        "blog_post": list_available_roles("blog_post"),
        "slack": ["default"],
    }


def clear_agent_cache() -> None:
    """Clear the agent cache.

    Useful for testing or when you want to force agents to be recreated
    with updated instructions or configuration.
    """
    _AGENT_CACHE.clear()
