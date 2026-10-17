import pytest

import pydantic_temporal_example.agents.blog_post_agent as blog_mod


class FakeCtx:
    def __init__(self, deps):
        self.deps = deps


@pytest.mark.asyncio
async def test_start_blog_post_commits(monkeypatch):
    committed = {}

    class FakeGitHubConn:
        def create_or_update_file(self, repo_name, path, content, commit_message, branch):  # noqa: PLR0913
            committed.update(
                {
                    "repo_name": repo_name,
                    "path": path,
                    "content": content,
                    "commit_message": commit_message,
                    "branch": branch,
                }
            )
            return {"action": "created", "path": path, "branch": branch, "commit_sha": "abc123"}

    monkeypatch.setattr(blog_mod, "GitHubConn", FakeGitHubConn, raising=True)
    deps = blog_mod.BlogPostDependencies(repo_name="repo", blog_dir="docs/blog", branch="main")
    out = await blog_mod.start_blog_post(
        FakeCtx(deps),
        title="My Post",
        slug="my-post",
        summary="Summary",
        tags="python,ai",
    )

    assert "Started blog post" in out
    assert committed["path"] == "docs/blog/my-post.md"
    assert committed["commit_message"] == "feat(blog): start my-post"
    assert committed["branch"] == "main"
    assert "# My Post" in committed["content"]


@pytest.mark.asyncio
async def test_commit_blog_post_updates(monkeypatch):
    class FakeGitHubConn:
        def create_or_update_file(self, repo_name, path, content, commit_message, branch):  # noqa: PLR0913
            return {"action": "updated", "path": path, "branch": branch, "commit_sha": "def456"}

    monkeypatch.setattr(blog_mod, "GitHubConn", FakeGitHubConn, raising=True)
    deps = blog_mod.BlogPostDependencies(repo_name="repo", blog_dir="docs/blog", branch="main")
    out = await blog_mod.commit_blog_post(
        FakeCtx(deps),
        "my-post",
        "updated markdown",
        "feat(blog): enhance my-post",
    )
    assert "updated" in out
    assert "def456" in out
