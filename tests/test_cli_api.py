"""Tests for CLI to FastAPI communication."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from pydantic_temporal_example.api import CLIWorkflowRequest
from pydantic_temporal_example.app import app
from pydantic_temporal_example.dependencies import get_temporal_client
from pydantic_temporal_example.models import CLIResponse


@pytest.fixture
def api_client() -> TestClient:
    """Create a fresh TestClient for each test to avoid shared state."""
    return TestClient(app)


@pytest.fixture
def mock_temporal_client() -> Mock:
    """Override Temporal client dependency with an async mock."""
    app.dependency_overrides.clear()
    mock_client = Mock()
    mock_client.start_workflow = AsyncMock()

    async def _override_temporal_client() -> Mock:
        return mock_client

    app.dependency_overrides[get_temporal_client] = _override_temporal_client
    yield mock_client
    app.dependency_overrides.clear()


class TestCLIWorkflowAPI:
    """Test CLI workflow API endpoints."""

    def test_submit_cli_workflow_success(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test successful CLI workflow submission."""
        # Mock successful workflow start
        mock_temporal_client.start_workflow = AsyncMock()

        request_data = {
            "prompt": "Test prompt",
            "repeat": False,
            "repeat_interval": 30,
            "repo_name": "test-repo",
            "session_id": "test-session",
        }

        response = api_client.post("/cli-workflow", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "workflow_id" in data
        assert data["message"] == "Workflow assigned to a worker."
        assert data["is_repeating"] is False

    def test_submit_cli_workflow_with_repeat(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test CLI workflow submission with repeat enabled."""
        # Mock successful workflow start
        mock_temporal_client.start_workflow = AsyncMock()

        request_data = {
            "prompt": "Test repeating prompt",
            "repeat": True,
            "repeat_interval": 60,
            "repo_name": "test-repo",
        }

        response = api_client.post("/cli-workflow", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_repeating"] is True
        assert "Repeating mode enabled" in data["message"]
        # Should have two workflow IDs (main + periodic)
        assert "," in data["workflow_id"]

    def test_submit_cli_workflow_invalid_data(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test CLI workflow submission with invalid data."""
        request_data = {
            "prompt": "",  # Empty prompt should fail validation
            "repeat": False,
            "repeat_interval": 30,
            "repo_name": "test-repo",
        }

        response = api_client.post("/cli-workflow", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_get_workflow_response_success(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test successful workflow response retrieval."""
        # Mock workflow handle and query
        mock_handle = AsyncMock()
        mock_temporal_client.get_workflow_handle_for.return_value = mock_handle

        # Mock response
        mock_response = CLIResponse(content="Test response", metadata={"timestamp": "2024-10-29T14:30:22"})
        mock_handle.query.return_value = mock_response

        response = api_client.get("/cli-workflow/test-workflow-id/response")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["response"]["content"] == "Test response"

    def test_get_workflow_response_pending(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test workflow response when no response available."""
        # Mock workflow handle and query
        mock_handle = AsyncMock()
        mock_temporal_client.get_workflow_handle_for.return_value = mock_handle
        mock_handle.query.return_value = None

        response = api_client.get("/cli-workflow/test-workflow-id/response")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["response"] is None

    def test_get_workflow_response_not_found(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test workflow response when workflow doesn't exist."""
        # Mock TemporalError for workflow not found
        from temporalio.exceptions import TemporalError

        mock_temporal_client.get_workflow_handle_for.side_effect = TemporalError("Workflow not found")

        response = api_client.get("/cli-workflow/nonexistent-workflow/response")

        assert response.status_code == 404

    def test_stop_workflow_success(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test successful workflow stopping."""
        # Mock workflow handle and signal
        mock_handle = AsyncMock()
        mock_temporal_client.get_workflow_handle_for.return_value = mock_handle
        mock_handle.signal = AsyncMock()

        response = api_client.delete("/cli-workflow/test-workflow-id")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Successfully stopped 1 workflow(s)" == data["message"]

    def test_stop_workflow_not_found(self, mock_temporal_client: Mock, api_client: TestClient):
        """Test workflow stopping when workflow doesn't exist."""
        # Mock TemporalError for workflow not found
        from temporalio.exceptions import TemporalError

        mock_temporal_client.get_workflow_handle_for.side_effect = TemporalError("Workflow not found")
        mock_temporal_client.get_workflow_handle.side_effect = TemporalError("Workflow not found")

        response = api_client.delete("/cli-workflow/nonexistent-workflow")

        assert response.status_code == 404


class TestCLIWorkflowRequest:
    """Test CLI workflow request model validation."""

    def test_valid_request(self):
        """Test valid request model."""
        request = CLIWorkflowRequest(prompt="Test prompt", repeat=False, repeat_interval=30, repo_name="test-repo")
        assert request.prompt == "Test prompt"
        assert request.repeat is False
        assert request.repeat_interval == 30
        assert request.repo_name == "test-repo"

    def test_request_with_repeat(self):
        """Test request model with repeat enabled."""
        request = CLIWorkflowRequest(
            prompt="Test repeating prompt", repeat=True, repeat_interval=60, session_id="test-session"
        )
        assert request.repeat is True
        assert request.repeat_interval == 60
        assert request.session_id == "test-session"

    def test_invalid_repeat_interval(self):
        """Test request model with invalid repeat interval."""
        with pytest.raises(ValueError):
            CLIWorkflowRequest(
                prompt="Test prompt",
                repeat=True,
                repeat_interval=0,  # Should be >= 1
            )
