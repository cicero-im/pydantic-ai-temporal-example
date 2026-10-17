"""Tests for CLI HTTP client functions."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from pydantic_temporal_example.cli import check_workflow_response, send_workflow_request


class TestCLIHTTPClient:
    """Test CLI HTTP client functions."""

    @pytest.mark.asyncio
    async def test_send_workflow_request_success(self):
        """Test successful workflow request sending."""
        mock_response_data = {
            "success": True,
            "workflow_id": "test-workflow-id",
            "message": "Workflow assigned to a worker.",
            "is_repeating": False,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            result = await send_workflow_request(
                prompt="Test prompt",
                app_host="127.0.0.1",
                app_port=4000,
                repeat=False,
                repeat_interval=30,
                repo_name="test-repo",
            )

            assert result == mock_response_data
            mock_client.post.assert_called_once()

            # Verify the call arguments
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://127.0.0.1:4000/cli-workflow"
            assert call_args[1]["json"]["prompt"] == "Test prompt"
            assert call_args[1]["json"]["repeat"] is False
            assert call_args[1]["json"]["repo_name"] == "test-repo"

    @pytest.mark.asyncio
    async def test_send_workflow_request_with_repeat(self):
        """Test workflow request sending with repeat enabled."""
        mock_response_data = {
            "success": True,
            "workflow_id": "test-workflow-id,periodic-test-workflow-id",
            "message": "Workflow assigned to worker. (Repeating mode enabled)",
            "is_repeating": True,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            result = await send_workflow_request(
                prompt="Test repeating prompt",
                repeat=True,
                repeat_interval=60,
                repo_name="test-repo",
                session_id="test-session",
            )

            assert result == mock_response_data
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["repeat"] is True
            assert call_args[1]["json"]["repeat_interval"] == 60
            assert call_args[1]["json"]["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_send_workflow_request_connection_error(self):
        """Test workflow request sending with connection error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock connection error
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(httpx.ConnectError):
                await send_workflow_request(prompt="Test prompt", app_host="invalid-host", app_port=4000)

    @pytest.mark.asyncio
    async def test_check_workflow_response_success(self):
        """Test successful workflow response checking."""
        mock_response_data = {
            "status": "completed",
            "response": {"content": "Test response content", "metadata": {"timestamp": "2024-10-29T14:30:22"}},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response

            result = await check_workflow_response(workflow_id="test-workflow-id", app_host="127.0.0.1", app_port=4000)

            assert result == mock_response_data
            mock_client.get.assert_called_once_with(
                "http://127.0.0.1:4000/cli-workflow/test-workflow-id/response", timeout=10.0
            )

    @pytest.mark.asyncio
    async def test_check_workflow_response_pending(self):
        """Test workflow response checking when pending."""
        mock_response_data = {"status": "pending", "response": None}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response

            result = await check_workflow_response(
                workflow_id="pending-workflow-id", app_host="127.0.0.1", app_port=4000
            )

            assert result["status"] == "pending"
            assert result["response"] is None

    @pytest.mark.asyncio
    async def test_check_workflow_response_not_found(self):
        """Test workflow response checking when workflow not found."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock 404 response with a real request/response pair so raise_for_status behaves normally
            request = httpx.Request("GET", "http://127.0.0.1:4000/cli-workflow/nonexistent-workflow-id/response")
            mock_client.get.return_value = httpx.Response(status_code=404, request=request)

            with pytest.raises(httpx.HTTPStatusError):
                await check_workflow_response(
                    workflow_id="nonexistent-workflow-id", app_host="127.0.0.1", app_port=4000
                )
