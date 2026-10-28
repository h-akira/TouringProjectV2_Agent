"""AgentCore entrypoint for the touring agent. Wiring only; the logic lives in handler.py."""
from __future__ import annotations

import logging
import threading

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models.bedrock import BedrockModel

from config import load_settings
from handler import Deps, handle
from history import SessionStore
from hooks import ToolCallLimit, ToolResultTrimmer
from model_config import ModelIdSource
from prompt import SYSTEM_PROMPT
from tools.places import find_place, nearby_places
from tools.web_search import WebSearchConnection, load_web_search

logging.getLogger("trv2.agent").setLevel(logging.INFO)

app = BedrockAgentCoreApp()
SETTINGS = load_settings()
_WEB_SEARCH_CLIENT = load_web_search()
# Held for the process, so the gateway connection is not made again for every question (Agent #2).
WEB_SEARCH = WebSearchConnection(_WEB_SEARCH_CLIENT, SETTINGS.mcp_max_idle_seconds) if _WEB_SEARCH_CLIENT else None
if WEB_SEARCH is None:
    app.logger.warning("no gateway url in env - answering without web search")
# Agents are built in worker threads (Agent #4). Building one may start the MCP client; two at once would both start
# it, and the second start fails, leaving that question without web search (Agent #10).
_BUILD_LOCK = threading.Lock()


def build_agent(model_id: str, messages: list[dict]) -> Agent:
    """A fresh agent per question: the model id may change, and the hooks count per question."""
    with _BUILD_LOCK:
        return Agent(
            model=BedrockModel(model_id=model_id, region_name=SETTINGS.region, max_tokens=SETTINGS.max_tokens),
            messages=messages,
            system_prompt=SYSTEM_PROMPT,
            tools=[nearby_places, find_place] + ([WEB_SEARCH.for_question()] if WEB_SEARCH else []),
            hooks=[ToolCallLimit(SETTINGS.max_tool_calls), ToolResultTrimmer(SETTINGS.max_tool_result_chars)]
            + ([WEB_SEARCH.failure_watch()] if WEB_SEARCH else []),
            # History is pruned by handler.py after each successful turn.
            conversation_manager=NullConversationManager(),
            callback_handler=None,
        )


DEPS = Deps(
    settings=SETTINGS,
    model_ids=ModelIdSource(
        SETTINGS.model_id_parameter,
        boto3.client("ssm", region_name=SETTINGS.region),
        SETTINGS.model_cache_seconds,
    ),
    sessions=SessionStore(),
    agent_factory=build_agent,
)


@app.entrypoint
async def invoke(payload: dict, context):
    # runtimeSessionId is the conversation id; the same id continues the conversation.
    session_id = getattr(context, "session_id", None) or "default-session"
    async for event in handle(payload, session_id, DEPS):
        yield event


if __name__ == "__main__":
    app.run()
