"""Agent specialized in creating and enhancing blog posts committed to GitHub."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai_claude_code import ClaudeCodeModel, ClaudeCodeProvider

from pydantic_temporal_example.tools import GitHubConn

provider = ClaudeCodeProvider({"use_sandbox_runtime": False, "model": "opus", "fallback_model": "sonnet"})
model_instance = ClaudeCodeModel("opus", provider=provider)


class BlogPostDependencies(BaseModel):
    """Dependencies for blog-post operations in a GitHub repository."""

    repo_name: str = Field(..., description="Repository name (without organization)")
    blog_dir: str = Field(default="docs/blog", description="Directory where blog posts are stored")
    branch: str = Field(default="main", description="Branch to commit changes into")


def _resolve_blog_path(deps: BlogPostDependencies, path_or_slug: str) -> str:
    """Resolve a blog path from a full path or slug-like input."""
    normalized = path_or_slug.strip()
    if not normalized:
        msg = "Blog path or slug cannot be empty"
        raise ValueError(msg)

    if "/" in normalized:
        resolved = normalized
    else:
        resolved = f"{deps.blog_dir.rstrip('/')}/{normalized}"

    if not resolved.endswith(".md"):
        resolved = f"{resolved}.md"
    return resolved


async def list_blog_posts(ctx: RunContext[BlogPostDependencies], blog_dir: str | None = None) -> str:
    """List markdown blog posts under the configured blog directory."""
    github = GitHubConn()
    target_dir = blog_dir or ctx.deps.blog_dir
    entries = github.get_repo_files(ctx.deps.repo_name, target_dir)
    posts = [entry.path for entry in entries if entry.type == "file" and entry.path.endswith(".md")]
    if not posts:
        return f"No blog posts found in {target_dir}"

    lines = [f"Blog posts in {target_dir}:"]
    lines.extend(f"- {post}" for post in posts)
    return "\n".join(lines)


async def read_blog_post(ctx: RunContext[BlogPostDependencies], path_or_slug: str) -> str:
    """Read an existing blog post from GitHub."""
    github = GitHubConn()
    path = _resolve_blog_path(ctx.deps, path_or_slug)
    return github.get_file_content(ctx.deps.repo_name, path, ref=ctx.deps.branch)


async def commit_blog_post(
    ctx: RunContext[BlogPostDependencies],
    path_or_slug: str,
    content: str,
    commit_message: str,
    branch: str | None = None,
) -> str:
    """Create or update a blog post file with a commit."""
    github = GitHubConn()
    path = _resolve_blog_path(ctx.deps, path_or_slug)
    target_branch = branch or ctx.deps.branch

    result = github.create_or_update_file(
        repo_name=ctx.deps.repo_name,
        path=path,
        content=content,
        commit_message=commit_message,
        branch=target_branch,
    )
    return (
        f"Blog post {result['action']} at {result['path']} on branch {result['branch']}. "
        f"Commit: {result['commit_sha']}"
    )


async def start_blog_post(
    ctx: RunContext[BlogPostDependencies],
    title: str,
    slug: str,
    summary: str = "",
    tags: str = "",
    branch: str | None = None,
) -> str:
    """Create an initial blog post draft with front matter and a first commit."""
    path = _resolve_blog_path(ctx.deps, slug)
    today = datetime.now(UTC).date().isoformat()
    tag_values = [tag.strip() for tag in tags.split(",") if tag.strip()]
    tags_yaml = ", ".join(f'"{tag}"' for tag in tag_values)

    starter_content = (
        f"---\n"
        f'title: "{title}"\n'
        f'date: "{today}"\n'
        f'summary: "{summary}"\n'
        f"tags: [{tags_yaml}]\n"
        f"draft: true\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"{summary or 'TBD'}\n\n"
        f"## Why this matters\n\n"
        f"- TBD\n\n"
        f"## Key ideas\n\n"
        f"- TBD\n\n"
        f"## Practical takeaways\n\n"
        f"- TBD\n"
    )

    github = GitHubConn()
    target_branch = branch or ctx.deps.branch
    result = github.create_or_update_file(
        repo_name=ctx.deps.repo_name,
        path=path,
        content=starter_content,
        commit_message=f"feat(blog): start {slug}",
        branch=target_branch,
    )
    return (
        f"Started blog post at {result['path']} on branch {result['branch']}. "
        f"Commit: {result['commit_sha']}"
    )


blog_post_agent = Agent(
    model=model_instance,
    deps_type=BlogPostDependencies,
    output_type=str,
    system_prompt=(
        "You are a blog post writing assistant that works directly in GitHub repositories. "
        "Use tools to inspect existing posts, start drafts, and commit improved content. "
        "When enhancing a post, read it first, produce improved markdown, and then commit it. "
        "Always return a concise summary of what changed and where."
    ),
)

blog_post_agent.tool(list_blog_posts)
blog_post_agent.tool(read_blog_post)
blog_post_agent.tool(commit_blog_post)
blog_post_agent.tool(start_blog_post)
