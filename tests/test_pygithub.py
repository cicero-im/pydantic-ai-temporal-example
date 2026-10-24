from datetime import datetime

from github.GithubException import GithubException

from pydantic_temporal_example.tools.pygithub import GitHubConn
import pytest


def test_get_repo_empty_name_raises():
    c = GitHubConn(organization="o")
    with pytest.raises(ValueError):
        c.get_repo("")


def test_get_repo_files_and_branches(monkeypatch):
    class FakeContent:
        def __init__(self, t, p): self.type, self.path = t, p
    class FakeBranchCommit:
        def __init__(self, sha): self.sha = sha
    class FakeBranch:
        def __init__(self, name, sha, protected=False):
            self.name, self.commit, self.protected = name, FakeBranchCommit(sha), protected
    class FakeRepo:
        def get_contents(self, _path):
            return [FakeContent("file", "README.md"), FakeContent("dir", "src")]
        def get_branches(self):
            return [FakeBranch("main", "abcdef1", True), FakeBranch("dev", "1234567", False)]
    monkeypatch.setattr(GitHubConn, "get_repo", lambda _self, _repo_name: FakeRepo(), raising=True)
    c = GitHubConn(organization="o")
    files = c.get_repo_files("repo")
    assert any(x.type == "dir" for x in files)
    branches = c.get_branches("repo")
    assert branches[0]["protected"] is True


def test_get_pr_and_comments(monkeypatch):
    class User:
        def __init__(self, login): self.login = login
    class IssueComment:
        def __init__(self, user, body, created_at):
            self.user, self.body, self.created_at = user, body, created_at
    class ReviewComment(IssueComment):
        def __init__(self, user, body, created_at, path):
            super().__init__(user, body, created_at)
            self.path = path
    class PR:
        def get_issue_comments(self):
            return [IssueComment(User("alice"), "hi", datetime(2024,1,1))]
        def get_review_comments(self):
            return [ReviewComment(User("bob"), "nit", datetime(2024,1,2), "x.py")]
    class Repo:
        def get_pull(self, _n): return PR()
    monkeypatch.setattr(GitHubConn, "get_repo", lambda _self, _repo_name: Repo(), raising=True)
    out = GitHubConn().get_pr_comments("r", 1)
    kinds = {c["type"] for c in out}
    assert "issue_comment" in kinds and "review_comment" in kinds


def test_get_file_content(monkeypatch):
    class File:
        def __init__(self, text: str):
            self.decoded_content = text.encode("utf-8")

    class Repo:
        def get_contents(self, _path, ref=None):  # noqa: ARG002
            return File("hello markdown")

    monkeypatch.setattr(GitHubConn, "get_repo", lambda _self, _repo_name: Repo(), raising=True)
    out = GitHubConn(organization="o").get_file_content("repo", "docs/blog/test.md", ref="main")
    assert out == "hello markdown"


def test_create_or_update_file_updates(monkeypatch):
    class Existing:
        sha = "oldsha"

    class Commit:
        sha = "newsha"

    class Repo:
        def get_contents(self, _path, ref=None):  # noqa: ARG002
            return Existing()

        def update_file(self, **kwargs):
            assert kwargs["sha"] == "oldsha"
            return {"commit": Commit()}

    monkeypatch.setattr(GitHubConn, "get_repo", lambda _self, _repo_name: Repo(), raising=True)
    out = GitHubConn(organization="o").create_or_update_file(
        "repo",
        "docs/blog/test.md",
        "content",
        "feat(blog): update",
        branch="main",
    )
    assert out["action"] == "updated"
    assert out["commit_sha"] == "newsha"


def test_create_or_update_file_creates_when_missing(monkeypatch):
    class Commit:
        sha = "createdsha"

    class Repo:
        def get_contents(self, _path, ref=None):  # noqa: ARG002
            raise GithubException(404, {"message": "Not Found"}, {})

        def create_file(self, **kwargs):
            assert kwargs["branch"] == "main"
            return {"commit": Commit()}

    monkeypatch.setattr(GitHubConn, "get_repo", lambda _self, _repo_name: Repo(), raising=True)
    out = GitHubConn(organization="o").create_or_update_file(
        "repo",
        "docs/blog/new-post.md",
        "content",
        "feat(blog): start",
        branch="main",
    )
    assert out["action"] == "created"
    assert out["commit_sha"] == "createdsha"
