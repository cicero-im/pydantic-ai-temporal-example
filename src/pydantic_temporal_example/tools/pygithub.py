"""PyGithub wrapper for accessing GitHub repositories and pull requests."""

from typing import Any

import logfire
from github import Auth, Github
from github.ContentFile import ContentFile
from github.GithubException import GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

from pydantic_temporal_example.config import get_github_org, get_github_pat


class GitHubConn:
    """GitHub connection wrapper for accessing repository information.

    This class provides methods to interact with GitHub repositories
    using the PyGithub library.
    """

    def __init__(self, organization: str | None = None) -> None:
        """Initialize GitHub connection with authentication.

        Args:
            organization: GitHub organization name (defaults to GITHUB_ORG from config)

        Raises:
            ValueError: If organization is empty or not set in environment
        """
        auth = Auth.Token(get_github_pat())
        self.g = Github(auth=auth)
        self.organization = organization or get_github_org()

        if not self.organization or not self.organization.strip():
            msg = "GitHub organization must be set via GITHUB_ORG environment variable or constructor argument"
            raise ValueError(msg)

    def get_repo(self, repo_name: str) -> Repository:
        """Get a repository by name.

        Args:
            repo_name: Repository name (without organization)

        Returns:
            Repository object

        Raises:
            ValueError: If the repository name is invalid
            github.GithubException: If the repository cannot be found or accessed
        """
        if not repo_name or not repo_name.strip():
            msg = "Repository name cannot be empty"
            raise ValueError(msg)

        full_repo_name = f"{self.organization}/{repo_name}"
        try:
            return self.g.get_repo(full_repo_name)
        except Exception as e:
            logfire.error(
                "Error accessing repository",
                organization=self.organization,
                repo_name=repo_name,
                full_repo_name=full_repo_name,
                error=str(e),
            )
            raise  # Re-raise original exception

    def get_repo_files(self, repo_name: str, path: str = "") -> list[ContentFile]:
        """Get files from a repository at the specified path.

        Args:
            repo_name: Repository name (without organization)
            path: Path within the repository (default: root)

        Returns:
            List of ContentFile objects
        """
        try:
            repo = self.get_repo(repo_name)
            contents = repo.get_contents(path)
            return contents if isinstance(contents, list) else [contents]
        except Exception as e:
            logfire.error("Error getting files from repository", repo_name=repo_name, path=path, error=str(e))
            raise

    def get_pull_request(self, repo_name: str, pr_number: int) -> PullRequest:
        """Get a specific pull request.

        Args:
            repo_name: Repository name (without organization)
            pr_number: Pull request number

        Returns:
            PullRequest object
        """
        try:
            repo = self.get_repo(repo_name)
            return repo.get_pull(pr_number)
        except Exception as e:
            logfire.error(
                "Error getting pull request from repository",
                repo_name=repo_name,
                pr_number=pr_number,
                error=str(e),
            )
            raise

    def get_pr_comments(self, repo_name: str, pr_number: int) -> list[dict[str, Any]]:
        """Get comments from a pull request.

        Args:
            repo_name: Repository name (without organization)
            pr_number: Pull request number

        Returns:
            List of comment dictionaries with user, body, and created_at
        """
        try:
            pr = self.get_pull_request(repo_name, pr_number)

            # Get issue comments (general PR comments)
            issue_comments = [
                {
                    "user": comment.user.login,
                    "body": comment.body,
                    "created_at": comment.created_at.isoformat(),
                    "type": "issue_comment",
                }
                for comment in pr.get_issue_comments()
            ]

            # Get review comments (code-specific comments)
            review_comments = [
                {
                    "user": comment.user.login,
                    "body": comment.body,
                    "created_at": comment.created_at.isoformat(),
                    "path": comment.path,
                    "type": "review_comment",
                }
                for comment in pr.get_review_comments()
            ]

            return issue_comments + review_comments
        except Exception as e:
            logfire.error(
                "Error getting comments for pull request",
                repo_name=repo_name,
                pr_number=pr_number,
                error=str(e),
            )
            raise

    def get_branches(self, repo_name: str) -> list[dict[str, Any]]:
        """Get all branches from a repository.

        Args:
            repo_name: Repository name (without organization)

        Returns:
            List of branch dictionaries with name and sha
        """
        try:
            repo = self.get_repo(repo_name)
            return [
                {"name": branch.name, "sha": branch.commit.sha, "protected": branch.protected}
                for branch in repo.get_branches()
            ]
        except Exception as e:
            logfire.error("Error getting branches from repository", repo_name=repo_name, error=str(e))
            raise

    def list_pull_requests(self, repo_name: str, state: str = "all") -> list[dict[str, Any]]:
        """List all pull requests in a repository.

        Args:
            repo_name: Repository name (without organization)
            state: PR state filter ('open', 'closed', or 'all')

        Returns:
            List of PR dictionaries with number, title, state, and author
        """
        try:
            repo = self.get_repo(repo_name)
            return [
                {
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "author": pr.user.login,
                    "created_at": pr.created_at.isoformat() if pr.created_at else "",
                    "updated_at": pr.updated_at.isoformat() if pr.updated_at else "",
                }
                for pr in repo.get_pulls(state=state)
            ]
        except Exception as e:
            logfire.error("Error listing pull requests from repository", repo_name=repo_name, error=str(e))
            raise

    def get_file_content(self, repo_name: str, path: str, ref: str = "main") -> str:
        """Get text file content from a repository path.

        Args:
            repo_name: Repository name (without organization)
            path: Path to file in repository
            ref: Branch, tag, or commit SHA

        Returns:
            Decoded UTF-8 content of the file
        """
        try:
            repo = self.get_repo(repo_name)
            content_file = repo.get_contents(path, ref=ref)
            if isinstance(content_file, list):
                msg = f"Path '{path}' is a directory; expected a file"
                raise ValueError(msg)
            decoded = content_file.decoded_content.decode("utf-8")
            return decoded
        except Exception as e:
            logfire.error("Error reading file from repository", repo_name=repo_name, path=path, ref=ref, error=str(e))
            raise

    def create_or_update_file(
        self,
        repo_name: str,
        path: str,
        content: str,
        commit_message: str,
        branch: str = "main",
    ) -> dict[str, str]:
        """Create a file or update an existing file in a repository.

        Args:
            repo_name: Repository name (without organization)
            path: Path to the file in repository
            content: New file content
            commit_message: Commit message
            branch: Target branch name

        Returns:
            Metadata including action, path, and commit SHA
        """
        repo = self.get_repo(repo_name)
        try:
            existing = repo.get_contents(path, ref=branch)
            if isinstance(existing, list):
                msg = f"Path '{path}' is a directory; expected a file"
                raise ValueError(msg)

            result = repo.update_file(
                path=path,
                message=commit_message,
                content=content,
                sha=existing.sha,
                branch=branch,
            )
            commit = result["commit"]
            return {
                "action": "updated",
                "path": path,
                "branch": branch,
                "commit_sha": commit.sha,
            }
        except GithubException as e:
            if e.status != 404:
                logfire.error(
                    "Error updating file in repository",
                    repo_name=repo_name,
                    path=path,
                    branch=branch,
                    error=str(e),
                )
                raise

            result = repo.create_file(
                path=path,
                message=commit_message,
                content=content,
                branch=branch,
            )
            commit = result["commit"]
            return {
                "action": "created",
                "path": path,
                "branch": branch,
                "commit_sha": commit.sha,
            }
