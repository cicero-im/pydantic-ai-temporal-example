"""Temporal workflows orchestrating Slack threads and agent dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any, assert_never

from pydantic_ai.durable_exec.temporal import TemporalAgent
from temporalio import workflow

from pydantic_temporal_example.agents.blog_post_agent import BlogPostDependencies, blog_post_agent
from pydantic_temporal_example.agents.dispatch_agent import (
    BlogPostRequest,
    DispatchResult,
    GitHubRequest,
    NoResponse,
    SlackResponse,
    WebResearchRequest,
    dispatch_agent,
)
from pydantic_temporal_example.agents.github_agent import (
    GitHubDependencies,
    github_agent,
)
from pydantic_temporal_example.agents.web_research_agent import (
    WebResearchResponse,
    build_web_research_agent,
)
from pydantic_temporal_example.models import (
    AppMentionEvent,
    CLIPromptEvent,
    CLIResponse,
    MessageChannelsEvent,
    SlackConversationsRepliesRequest,
    SlackMessageID,
    SlackReaction,
    SlackReply,
)
from pydantic_temporal_example.temporal.slack_activities import (
    slack_chat_post_message,
    slack_conversations_replies,
    slack_reactions_add,
    slack_reactions_remove,
)

if TYPE_CHECKING:
    from temporalio.workflow import ActivityConfig

# Activity config with 5-minute timeout for agent operations
_agent_activity_config: ActivityConfig = {"start_to_close_timeout": timedelta(minutes=5)}

temporal_dispatch_agent = TemporalAgent(
    dispatch_agent,
    name="dispatch_agent",
    activity_config=_agent_activity_config,
)
temporal_github_agent = TemporalAgent(github_agent, name="github_agent", activity_config=_agent_activity_config)
temporal_blog_post_agent = TemporalAgent(blog_post_agent, name="blog_post_agent", activity_config=_agent_activity_config)

# Build web research agent only if JINA_API_KEY is configured
_web_research_agent = build_web_research_agent()
if _web_research_agent is not None:
    temporal_web_research_agent = TemporalAgent(
        _web_research_agent,
        name="web_research_agent",
        activity_config=_agent_activity_config,
    )
else:
    temporal_web_research_agent = None


@workflow.defn
class SlackThreadWorkflow:
    """Orchestrates a Slack thread: collects messages, dispatches, and runs agents."""

    def __init__(self) -> None:
        """Initialize pending event queue and thread message store."""
        self._pending_events: asyncio.Queue[AppMentionEvent | MessageChannelsEvent] = asyncio.Queue()
        self._thread_messages: list[dict[str, Any]] = []

    @property
    def _most_recent_ts(self) -> str | None:
        if not self._thread_messages:
            return None
        # assume _thread_messages is always sorted by ts
        return self._thread_messages[-1]["ts"]

    @workflow.run
    async def run(self) -> None:
        """Main workflow loop: waits for queued events and handles each one."""
        while True:
            await workflow.wait_condition(lambda: not self._pending_events.empty())
            while not self._pending_events.empty():
                event = self._pending_events.get_nowait()
                await self.handle_event(event)

    @workflow.signal
    async def submit_message_channels_event(self, event: MessageChannelsEvent) -> None:
        """Signal to enqueue a message channels event for processing."""
        await self._pending_events.put(event)

    @workflow.signal
    async def submit_app_mention_event(self, event: AppMentionEvent) -> None:
        """Signal to enqueue an app mention event for processing."""
        await self._pending_events.put(event)

    async def handle_event(self, event: AppMentionEvent | MessageChannelsEvent) -> None:
        """Process a Slack event: fetch updates, dispatch to agents, and post reply."""
        # add thinking reaction immediately
        most_recent_ts = self._most_recent_ts or event.event_ts
        event_message = SlackMessageID(channel=event.channel, ts=most_recent_ts)

        await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            slack_reactions_add,
            SlackReaction(message=event_message, name="spin"),
            start_to_close_timeout=timedelta(seconds=10),
        )

        # Get new messages in the thread
        request = SlackConversationsRepliesRequest(
            channel=event.channel,
            ts=event.reply_thread_ts,
            oldest=most_recent_ts,
        )
        new_messages: list[dict[str, Any]] = await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            slack_conversations_replies,
            request,
            start_to_close_timeout=timedelta(seconds=10),
        )
        self._thread_messages.extend(new_messages)

        # Get directive from the dispatch agent
        # Pass thread messages as JSON string to dispatch agent
        stringified_thread = json.dumps(self._thread_messages, indent=2)
        dispatcher_result = await temporal_dispatch_agent.run(stringified_thread, output_type=DispatchResult)  # type: ignore[call-arg]  # pyright: ignore[reportUnknownVariableType]

        if isinstance(dispatcher_result.output, NoResponse):
            return

        # remove thinking reaction before posting a message
        await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            slack_reactions_remove,
            SlackReaction(message=event_message, name="spin"),
            start_to_close_timeout=timedelta(seconds=10),
        )

        response: str | list[dict[str, Any]]
        if isinstance(dispatcher_result.output, SlackResponse):
            response = dispatcher_result.output.response
        elif isinstance(dispatcher_result.output, GitHubRequest):
            # Extract query and create dependencies
            request = dispatcher_result.output
            # Default repo used when not specified in thread context
            deps = GitHubDependencies(repo_name="default-repo")
            result = await temporal_github_agent.run(request.query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
            response = str(result.output)
        elif isinstance(dispatcher_result.output, BlogPostRequest):
            request = dispatcher_result.output
            deps = BlogPostDependencies(repo_name="pydantic-ai-temporal-example")
            result = await temporal_blog_post_agent.run(request.query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
            response = str(result.output)
        elif isinstance(dispatcher_result.output, WebResearchRequest):
            # Populate thread context and pass structured request
            if temporal_web_research_agent is None:
                response = "Web research is not available. Please configure JINA_API_KEY."
            else:
                request = dispatcher_result.output
                # Pass the query string to the agent
                result = await temporal_web_research_agent.run(request.query, output_type=WebResearchResponse)
                response = result.output.response
        else:
            assert_never(dispatcher_result.output)  # type: ignore[arg-type]

        # Post response
        await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            slack_chat_post_message,
            SlackReply(thread=event_message, content=response),
            start_to_close_timeout=timedelta(seconds=10),
        )


@workflow.defn
class PeriodicGitHubPRCheckWorkflow:
    """Periodically executes queries using the dispatcher for plug-and-play agent support."""

    def __init__(self) -> None:
        """Initialize workflow state."""
        self._should_continue = True
        self._check_count = 0
        self._repo_name: str = "default-repo"
        self._conversation_messages: list[dict[str, Any]] = []

    @workflow.run
    async def periodic_run(
        self,
        repo_name: str,
        check_interval_seconds: int = 30,
        query: str = "List all pull requests in the repository",
    ) -> None:
        """Main workflow loop: periodically executes queries via dispatcher.

        Args:
            repo_name: Repository name to check (without organization)
            check_interval_seconds: How often to check for PRs (default: 30 seconds)
            query: The query/instruction to pass to the dispatch agent

        Note:
            Uses the dispatcher agent to route queries to appropriate agents (GitHub, WebResearch, etc.)
            This enables plug-and-play agent architecture.
        """
        self._repo_name = repo_name
        workflow.logger.info(
            f"Starting periodic query execution for repository: {repo_name}, interval: {check_interval_seconds}s",
        )
        workflow.logger.info(f"Query: {query}")

        while self._should_continue:
            self._check_count += 1
            check_num = self._check_count
            workflow.logger.info(f"Check #{check_num} - Executing query via dispatcher")

            # Build conversation context for dispatcher
            user_message = {
                "role": "user",
                "content": f"Repository: {repo_name}. {query}",
                "timestamp": workflow.now().isoformat(),
            }
            self._conversation_messages.append(user_message)

            # Use dispatcher to determine which agent to use (GitHub, WebResearch, etc.)
            stringified_conversation = json.dumps(self._conversation_messages, indent=2)
            dispatcher_result = await temporal_dispatch_agent.run(stringified_conversation, output_type=DispatchResult)  # type: ignore[call-arg]

            # Handle dispatcher result
            response: str | list[dict[str, Any]]
            if isinstance(dispatcher_result.output, NoResponse):
                workflow.logger.info(f"Check #{check_num} - Dispatcher determined no response needed")
                response = "(No response needed)"
            elif isinstance(dispatcher_result.output, SlackResponse):
                response = dispatcher_result.output.response
                workflow.logger.info(f"Check #{check_num} - Processed via Slack response")
            elif isinstance(dispatcher_result.output, GitHubRequest):
                # Route to GitHub agent based on dispatcher decision
                request = dispatcher_result.output
                deps = GitHubDependencies(repo_name=self._repo_name)
                result = await temporal_github_agent.run(request.query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
                response = str(result.output)
                workflow.logger.info(f"Check #{check_num} - Processed via GitHub agent")
            elif isinstance(dispatcher_result.output, BlogPostRequest):
                request = dispatcher_result.output
                deps = BlogPostDependencies(repo_name=self._repo_name)
                result = await temporal_blog_post_agent.run(request.query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
                response = str(result.output)
                workflow.logger.info(f"Check #{check_num} - Processed via Blog Post agent")
            elif isinstance(dispatcher_result.output, WebResearchRequest):
                # Route to Web Research agent based on dispatcher decision
                if temporal_web_research_agent is None:
                    response = "Web research is not available. Please configure JINA_API_KEY."
                    workflow.logger.warning(f"Check #{check_num} - Web research requested but not configured")
                else:
                    request = dispatcher_result.output
                    result = await temporal_web_research_agent.run(request.query, output_type=WebResearchResponse)
                    response = result.output.response
                    workflow.logger.info(f"Check #{check_num} - Processed via Web Research agent")
            else:
                assert_never(dispatcher_result.output)  # type: ignore[arg-type]

            # Store response in conversation history
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": workflow.now().isoformat(),
            }
            self._conversation_messages.append(assistant_message)

            # Log completion
            workflow.logger.info(f"Check #{check_num} - Completed successfully")

            # Wait before the next check (this allows concurrent execution)
            workflow.logger.info(f"Waiting {check_interval_seconds}s before next check...")
            await workflow.sleep(check_interval_seconds)

    @workflow.signal
    async def stop(self) -> None:
        """Signal to stop the periodic checks."""
        workflow.logger.info("Received stop signal")
        self._should_continue = False

    @workflow.query
    def get_check_count(self) -> int:
        """Query to get the current check count."""
        return self._check_count


@workflow.defn
class CLIConversationWorkflow:
    """Orchestrates a CLI conversation: collects prompts, dispatches, and runs agents."""

    def __init__(self) -> None:
        """Initialize pending event queue and conversation message store."""
        self._pending_events: asyncio.Queue[CLIPromptEvent] = asyncio.Queue()
        self._conversation_messages: list[dict[str, Any]] = []
        self._response_ready: asyncio.Event = asyncio.Event()
        self._latest_response: CLIResponse | None = None
        self._repo_name: str = "default-repo"

    @workflow.run
    async def run(self) -> None:
        """Main workflow loop: waits for queued prompts and handles each one."""
        while True:
            await workflow.wait_condition(lambda: not self._pending_events.empty())
            while not self._pending_events.empty():
                event = self._pending_events.get_nowait()
                await self.handle_prompt(event)

    @workflow.signal
    async def submit_prompt(self, event: CLIPromptEvent) -> None:
        """Signal to enqueue a CLI prompt for processing."""
        await self._pending_events.put(event)

    @workflow.query
    def get_latest_response(self) -> CLIResponse | None:
        """Query to retrieve the most recent response."""
        return self._latest_response

    @workflow.query
    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Query to retrieve the full conversation history."""
        return self._conversation_messages

    async def handle_prompt(self, event: CLIPromptEvent) -> None:
        """Process a CLI prompt: dispatch to agents and prepare response."""
        # Add user message to conversation history
        user_message = {
            "role": "user",
            "content": event.prompt,
            "timestamp": event.timestamp,
        }
        self._conversation_messages.append(user_message)

        # Get directive from the dispatch agent
        # Pass conversation messages as JSON string to dispatch agent
        stringified_conversation = json.dumps(self._conversation_messages, indent=2)
        dispatcher_result = await temporal_dispatch_agent.run(stringified_conversation, output_type=DispatchResult)  # type: ignore[call-arg]

        if isinstance(dispatcher_result.output, NoResponse):
            # Store empty response
            self._latest_response = CLIResponse(content="(No response needed)")
            return

        response: str | list[dict[str, Any]]
        if isinstance(dispatcher_result.output, SlackResponse):
            response = dispatcher_result.output.response
        elif isinstance(dispatcher_result.output, GitHubRequest):
            # Extract query and create dependencies
            request = dispatcher_result.output
            # Use configured repo name from workflow instance
            deps = GitHubDependencies(repo_name=self._repo_name)
            result = await temporal_github_agent.run(request.query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
            response = str(result.output)
        elif isinstance(dispatcher_result.output, BlogPostRequest):
            request = dispatcher_result.output
            deps = BlogPostDependencies(repo_name=self._repo_name)
            result = await temporal_blog_post_agent.run(request.query, output_type=str, deps=deps)  # type: ignore[call-arg, arg-type]
            response = str(result.output)
        elif isinstance(dispatcher_result.output, WebResearchRequest):
            # Delegate to web research agent
            if temporal_web_research_agent is None:
                response = "Web research is not available. Please configure JINA_API_KEY."
            else:
                request = dispatcher_result.output
                result = await temporal_web_research_agent.run(request.query, output_type=WebResearchResponse)
                response = result.output.response
        else:
            assert_never(dispatcher_result.output)  # type: ignore[arg-type]

        # Store response in conversation history
        assistant_message = {
            "role": "assistant",
            "content": response,
            "timestamp": workflow.now().isoformat(),
        }
        self._conversation_messages.append(assistant_message)

        # Set latest response for query
        self._latest_response = CLIResponse(
            content=response,
            metadata={
                "timestamp": assistant_message["timestamp"],
                "message_count": len(self._conversation_messages),
            },
        )
        self._response_ready.set()
