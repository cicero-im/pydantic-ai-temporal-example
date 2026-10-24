"""Generic workflows that work with any agent from the registry.

These workflows separate scheduling logic from business logic, enabling:
- Plug-and-play agent architecture
- Reusable workflow patterns (oneshot, periodic)
- No workflow changes needed when adding new agents
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from pydantic_ai.durable_exec.temporal import TemporalAgent
from temporalio import workflow

from pydantic_temporal_example.agents.blog_post_agent import BlogPostDependencies
from pydantic_temporal_example.agents.github_agent import GitHubDependencies
from pydantic_temporal_example.agents.registry import get_agent
from pydantic_temporal_example.agents.web_research_agent import WebResearchResponse
from pydantic_temporal_example.models import CLIResponse

if TYPE_CHECKING:
    from temporalio.workflow import ActivityConfig

# Activity config with 5-minute timeout for agent operations
_agent_activity_config: ActivityConfig = {"start_to_close_timeout": timedelta(minutes=5)}


@workflow.defn
class GenericOneShotWorkflow:
    """Generic one-shot workflow that works with any agent from the registry.

    Executes an agent once and returns the result. The dispatcher determines
    which agent to use based on the user's request.
    """

    def __init__(self) -> None:
        """Initialize workflow state."""
        self._latest_response: CLIResponse | None = None
        self._repo_name: str = "default-repo"

    @workflow.run
    async def run(
        self,
        agent_type: str,
        agent_role: str,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Execute agent once and return result.

        Args:
            agent_type: Type of agent from registry (e.g., "github", "web_research", "blog_post")
            agent_role: Role specialization (e.g., "implementer", "reviewer", "default")
            query: The query/instruction for the agent
            context: Additional context (repo_name, etc.)

        Returns:
            Agent response as string
        """
        workflow.logger.info(f"OneShot execution: {agent_type}/{agent_role}")
        workflow.logger.info(f"Query: {query}")

        # Extract context
        if context:
            self._repo_name = context.get("repo_name", "default-repo")

        # Get agent from registry
        try:
            agent = get_agent(agent_type, agent_role)
            if agent is None:
                # Direct response without agent (e.g., Slack)
                workflow.logger.info("Direct response without agent")
                response = query
            else:
                # Execute agent via Temporal
                temporal_agent = TemporalAgent(
                    agent,
                    name=f"{agent_type}_{agent_role}",
                    activity_config=_agent_activity_config,
                )

                # Prepare dependencies based on agent type
                if agent_type == "github":
                    deps = GitHubDependencies(repo_name=self._repo_name)
                    result = await temporal_agent.run(query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
                    response = str(result.output)
                elif agent_type == "blog_post":
                    deps = BlogPostDependencies(repo_name=self._repo_name)
                    result = await temporal_agent.run(query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
                    response = str(result.output)
                elif agent_type == "web_research":
                    result = await temporal_agent.run(query, output_type=WebResearchResponse)
                    response = result.output.response
                else:
                    # Generic execution
                    result = await temporal_agent.run(query, output_type=str)
                    response = str(result.output)

                workflow.logger.info(f"Agent executed successfully: {agent_type}/{agent_role}")

        except KeyError as e:
            workflow.logger.error(f"Agent not found: {e}")
            response = f"Error: Agent {agent_type}/{agent_role} not found in registry"
        except (ValueError, RuntimeError, TypeError) as e:
            workflow.logger.error(f"Agent execution failed: {e}")
            response = f"Error executing agent: {e!s}"

        # Store response
        self._latest_response = CLIResponse(
            content=response,
            metadata={
                "agent_type": agent_type,
                "agent_role": agent_role,
                "timestamp": workflow.now().isoformat(),
            },
        )

        return response

    @workflow.query
    def get_latest_response(self) -> CLIResponse | None:
        """Query to retrieve the most recent response."""
        return self._latest_response


@workflow.defn
class GenericPeriodicWorkflow:
    """Generic periodic workflow that works with any agent from the registry.

    Repeatedly executes an agent at a specified interval. The dispatcher determines
    which agent to use and how often to run it.
    """

    def __init__(self) -> None:
        """Initialize workflow state."""
        self._should_continue = True
        self._execution_count = 0
        self._repo_name: str = "default-repo"
        self._conversation_messages: list[dict[str, Any]] = []

    @workflow.run
    async def periodic_run(
        self,
        agent_type: str,
        agent_role: str,
        query: str,
        interval_seconds: int = 30,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Execute agent periodically at specified interval.

        Args:
            agent_type: Type of agent from registry (e.g., "github", "web_research", "blog_post")
            agent_role: Role specialization (e.g., "implementer", "reviewer", "default")
            query: The query/instruction for the agent
            interval_seconds: How often to execute (in seconds)
            context: Additional context (repo_name, etc.)

        Note:
            Temporal handles the repetition logic. The agent is just executed on each iteration.
        """
        # Extract context
        if context:
            self._repo_name = context.get("repo_name", "default-repo")

        workflow.logger.info(
            f"Starting periodic execution: {agent_type}/{agent_role}, interval: {interval_seconds}s",
        )
        workflow.logger.info(f"Query: {query}")

        # Get agent from registry once (reuse across iterations)
        try:
            agent = get_agent(agent_type, agent_role)
            if agent is None:
                workflow.logger.error("Cannot run periodic workflow with direct response agent")
                return

            temporal_agent = TemporalAgent(
                agent,
                name=f"{agent_type}_{agent_role}",
                activity_config=_agent_activity_config,
            )

        except KeyError as e:
            workflow.logger.error(f"Agent not found: {e}")
            return

        # Periodic execution loop
        while self._should_continue:
            self._execution_count += 1
            execution_num = self._execution_count

            workflow.logger.info(f"Execution #{execution_num} - Running {agent_type}/{agent_role}")

            try:
                # Build conversation context
                user_message = {
                    "role": "user",
                    "content": f"Repository: {self._repo_name}. {query}",
                    "timestamp": workflow.now().isoformat(),
                }
                self._conversation_messages.append(user_message)

                # Execute agent based on type
                if agent_type == "github":
                    deps = GitHubDependencies(repo_name=self._repo_name)
                    result = await temporal_agent.run(query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
                    response = str(result.output)
                elif agent_type == "blog_post":
                    deps = BlogPostDependencies(repo_name=self._repo_name)
                    result = await temporal_agent.run(query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
                    response = str(result.output)
                elif agent_type == "web_research":
                    result = await temporal_agent.run(query, output_type=WebResearchResponse)
                    response = result.output.response
                else:
                    result = await temporal_agent.run(query, output_type=str)
                    response = str(result.output)

                # Store response
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": workflow.now().isoformat(),
                }
                self._conversation_messages.append(assistant_message)

                workflow.logger.info(f"Execution #{execution_num} - Completed successfully")

            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                workflow.logger.error(f"Execution #{execution_num} - Failed: {e}")
                # Continue to next iteration even if this one failed

            # Wait before next execution
            workflow.logger.info(f"Waiting {interval_seconds}s before next execution...")
            await workflow.sleep(interval_seconds)

    @workflow.signal
    async def stop(self) -> None:
        """Signal to stop the periodic executions."""
        workflow.logger.info("Received stop signal")
        self._should_continue = False

    @workflow.query
    def get_execution_count(self) -> int:
        """Query to get the current execution count."""
        return self._execution_count

    @workflow.query
    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Query to retrieve the full conversation history."""
        return self._conversation_messages
