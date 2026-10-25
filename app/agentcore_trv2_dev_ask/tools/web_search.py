"""Web search through the AgentCore Gateway's managed web-search connector (MCP, IAM-signed)."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Optional

from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.tools.mcp import MCPAgentTool, MCPClient

_AWS_SERVICE = "bedrock-agentcore"
_PREFIX = "AGENTCORE_GATEWAY_"
_SUFFIX = "_URL"
# The consumer this process holds on the MCP client, so the client outlives each question's agent.
KEEPER = "trv2-process"

log = logging.getLogger("trv2.agent")


def find_gateway_url(env: Mapping[str, str] = os.environ) -> Optional[str]:
    """The gateway endpoint: GATEWAY_URL for local runs, else the CLI-injected AGENTCORE_GATEWAY_<NAME>_URL."""
    if env.get("GATEWAY_URL"):
        return env["GATEWAY_URL"]
    for name in sorted(env):
        if name.startswith(_PREFIX) and name.endswith(_SUFFIX):
            return env[name]
    return None


def load_web_search() -> Optional[MCPClient]:
    """An MCP client for the gateway, or None so the agent still answers without search."""
    endpoint = find_gateway_url()
    if not endpoint:
        return None
    region = os.environ.get("GATEWAY_REGION", "us-east-1")
    # continue_on_error: if the gateway cannot be reached, the agent gets no search tools
    # and still answers, instead of every question failing.
    return MCPClient(
        lambda: aws_iam_streamablehttp_client(endpoint=endpoint, aws_region=region, aws_service=_AWS_SERVICE),
        continue_on_error=True,
    )


class WebSearchConnection:
    """Keeps the gateway's MCP connection across questions (Agent #2).

    Each question builds a fresh agent, which becomes a consumer of the MCP client. When the agent is collected,
    Strands removes it, and a client with no consumer left stops and starts again, listing its tools, on the next
    question. Holding one consumer for the process keeps the connection. It is started again before a question
    when the last start failed (Strands keeps a swallowed failure, and its stop() drops every consumer), when a
    started connection is no longer active, when the last question's web search failed (the gateway may drop the
    MCP session while the client still looks alive), or when it has not been used for max_idle_seconds."""

    def __init__(self, client: Any, max_idle_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._client = client
        self._max_idle_seconds = max_idle_seconds
        self._clock = clock
        self._last_used: Optional[float] = None
        self._broken = False

    def mark_broken(self) -> None:
        """A web search call failed; the next question starts the connection again."""
        self._broken = True

    def failure_watch(self) -> HookProvider:
        """A hook for each question's agent that reports a failed web search back here."""
        return _FailureWatch(self)

    def _stopped_working(self) -> bool:
        started = bool(getattr(self._client, "_tool_provider_started", False))
        is_active = getattr(self._client, "_is_session_active", None)
        return started and callable(is_active) and not is_active()

    def for_question(self) -> Any:
        now = self._clock()
        failed = bool(getattr(self._client, "_connection_failed", False))
        idle = self._last_used is not None and now - self._last_used > self._max_idle_seconds
        if failed or idle or self._broken or self._stopped_working():
            try:
                # With no other consumer, this stops the client and clears the kept failure.
                self._client.remove_consumer(KEEPER)
            except Exception as error:  # noqa: BLE001 - stop() has reset the client's state; the question goes on
                log.warning("mcp reset failed: %s", type(error).__name__)
            self._broken = False
        self._client.add_consumer(KEEPER)
        self._last_used = now
        return self._client


class _FailureWatch(HookProvider):
    """Marks the connection broken when a tool from the MCP client (the web search) returns an error."""

    def __init__(self, connection: WebSearchConnection) -> None:
        self._connection = connection

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self.on_after_tool_call)

    def on_after_tool_call(self, event: Any) -> None:
        result = event.result if isinstance(event.result, dict) else {}
        if isinstance(event.selected_tool, MCPAgentTool) and result.get("status") == "error":
            self._connection.mark_broken()
