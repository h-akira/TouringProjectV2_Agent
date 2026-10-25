"""One invocation: read the model id, run the agent, stream events, and keep history on success."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from config import Settings
from history import SessionStore, prune_history
from model_config import ModelIdError
from stream import EventShaper, done_event, error_event, final_answer
from tools.places import LOCATION_KEY

log = logging.getLogger("trv2.agent")

# Said when the model ran into a limit (turns, tokens) before answering: the rider hears that it could not be
# looked up instead of the failure tone (Agent #3, US-A-05.003). No second model call, so no extra input tokens.
LIMIT_ANSWER = "調べきれませんでした。"
_LIMIT_PREFIX = "limit_"


@dataclass
class Deps:
    settings: Settings
    model_ids: Any
    sessions: SessionStore
    agent_factory: Callable[[str, list[dict]], Any]


def _prompt(payload: dict) -> str:
    value = payload.get("prompt")
    return value.strip() if isinstance(value, str) else ""


def _log(record: dict) -> None:
    # One JSON line per invocation; never the prompt or the position (personal data).
    log.info(json.dumps(record, ensure_ascii=False))


async def handle(payload: dict, session_id: str, deps: Deps) -> AsyncIterator[dict]:
    prompt = _prompt(payload)
    if not prompt:
        yield error_event("bad_request", "prompt is required")
        return
    try:
        model_id = deps.model_ids.get()
    except ModelIdError as error:
        _log({"event": "invoke_failed", "session": session_id, "reason": str(error)})
        yield error_event("agent_failed", str(error))
        return

    shaper = EventShaper()
    tools: list[str] = []
    result = None
    try:
        # Inside the try: building the agent loads MCP tools and can fail on its own.
        agent = deps.agent_factory(model_id, deps.sessions.get(session_id))
        async for event in agent.stream_async(
            prompt,
            invocation_state={LOCATION_KEY: payload.get("location")},
            limits={"turns": deps.settings.max_turns},
        ):
            if "result" in event:
                result = event["result"]
                continue
            for shaped in shaper.shape(event):
                if shaped["type"] == "tool":
                    tools.append(shaped["name"])
                yield shaped
    except Exception as error:  # noqa: BLE001 - reported to the Backend as a failure
        _log({"event": "invoke_failed", "session": session_id, "modelId": model_id, "reason": type(error).__name__})
        yield error_event("agent_failed", type(error).__name__)
        return

    answer = None
    if result is not None and not final_answer(result.message) and str(result.stop_reason).startswith(_LIMIT_PREFIX):
        answer = LIMIT_ANSWER
        # Kept as the turn's answer, so the history still alternates question and answer.
        agent.messages.append({"role": "assistant", "content": [{"text": LIMIT_ANSWER}]})
    elif result is None or not final_answer(result.message):
        _log({"event": "invoke_failed", "session": session_id, "modelId": model_id, "reason": "empty answer"})
        yield error_event("agent_failed", "empty answer")
        return

    deps.sessions.put(session_id, prune_history(agent.messages, deps.settings.max_history_messages))
    done = done_event(result, model_id, answer)
    _log({"event": "invoke_done", "session": session_id, "modelId": model_id, "tools": tools,
          "usage": done["usage"], "stopReason": done["stopReason"]})
    yield done
