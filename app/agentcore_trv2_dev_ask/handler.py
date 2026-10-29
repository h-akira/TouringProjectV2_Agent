"""One invocation: read the model id, run the agent, stream events, and keep history on success."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from strands.types.exceptions import MaxTokensReachedException

from config import Settings
from history import SessionLocks, SessionStore, prune_history
from model_config import ModelIdError
from stream import EventShaper, done_event, error_event, final_answer
from tools.places import LOCATION_KEY

log = logging.getLogger("trv2.agent")

# Said when the model ran into the turn cap, or its output token cap, before answering: the rider hears that it
# could not be looked up instead of the failure tone (Agent #3, #9, US-A-05.003). No second model call, so no extra
# input tokens.
LIMIT_ANSWER = "調べきれませんでした。"
_LIMIT_PREFIX = "limit_"
# What Strands puts in place of a tool call cut off at the output token cap (recover_message_on_max_tokens_reached).
_CUT_OFF_TOOL = re.compile(r"The selected tool .*'s tool use was incomplete due to maximum token limits being reached\.")
# The longest a call may be told to take; a longer number is taken as a mistake.
MAX_DEADLINE_SECONDS = 600.0


@dataclass
class Deps:
    settings: Settings
    model_ids: Any
    sessions: SessionStore
    agent_factory: Callable[[str, list[dict]], Any]
    locks: SessionLocks = field(default_factory=SessionLocks)


def _prompt(payload: dict) -> str:
    value = payload.get("prompt")
    return value.strip() if isinstance(value, str) else ""


def _deadline_seconds(payload: dict, settings: Settings) -> float:
    """How long this call may take: what the Backend sent (adr/0014), or the default when it sent nothing sensible."""
    value = payload.get("deadlineSeconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= MAX_DEADLINE_SECONDS:
        return float(value)
    return settings.default_deadline_seconds


def _settle_cut_off(messages: list[dict]) -> str:
    """The answer to a reply cut off at the output token cap: its text after the last tool call so far (a preface is
    not an answer, 07 UC-8), or LIMIT_ANSWER. Strands has already added the cut-off reply to the history; it is
    replaced by the answer, so the history still alternates."""
    last = messages[-1] if messages and messages[-1].get("role") == "assistant" else None
    texts = [block["text"] for block in (last.get("content", []) if last else []) if isinstance(block.get("text"), str)]
    cut = [index for index, value in enumerate(texts) if _CUT_OFF_TOOL.fullmatch(value)]
    text = "".join(texts[cut[-1] + 1:] if cut else texts).strip()
    answer = text or LIMIT_ANSWER
    messages[len(messages) - (1 if last else 0):] = [{"role": "assistant", "content": [{"text": answer}]}]
    return answer


def _log(record: dict) -> None:
    # One JSON line per invocation; never the prompt or the position (personal data).
    log.info(json.dumps(record, ensure_ascii=False))


async def handle(payload: dict, session_id: str, deps: Deps) -> AsyncIterator[dict]:
    prompt = _prompt(payload)
    if not prompt:
        yield error_event("bad_request", "prompt is required")
        return
    try:
        # The wait for the session's lock counts against the deadline too. A cut call keeps no history (adr/0014).
        async with asyncio.timeout(_deadline_seconds(payload, deps.settings)):
            async with deps.locks.get(session_id):
                async for event in _run(payload, prompt, session_id, deps):
                    yield event
    except TimeoutError:
        _log({"event": "invoke_failed", "session": session_id, "reason": "timeout"})
        yield error_event("timeout", "deadline passed")


async def _run(payload: dict, prompt: str, session_id: str, deps: Deps) -> AsyncIterator[dict]:
    try:
        # Blocking (SSM); a worker thread keeps the event loop, and /ping, answering (Agent #4).
        model_id = await asyncio.to_thread(deps.model_ids.get)
    except ModelIdError as error:
        _log({"event": "invoke_failed", "session": session_id, "reason": str(error)})
        yield error_event("agent_failed", str(error))
        return

    shaper = EventShaper()
    tools: list[str] = []
    result = None
    cut_off = False
    try:
        # Inside the try: building the agent loads MCP tools and can fail on its own. Blocking too, so off the loop.
        agent = await asyncio.to_thread(deps.agent_factory, model_id, deps.sessions.get(session_id))
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
    except MaxTokensReachedException:
        # Raised, not returned, at the output token cap (Agent #9).
        cut_off = True
    except Exception as error:  # noqa: BLE001 - reported to the Backend as a failure
        _log({"event": "invoke_failed", "session": session_id, "modelId": model_id, "reason": type(error).__name__})
        yield error_event("agent_failed", type(error).__name__)
        return

    answer = None
    if cut_off:
        answer = _settle_cut_off(agent.messages)
        result = SimpleNamespace(message=agent.messages[-1], stop_reason="max_tokens", metrics=agent.event_loop_metrics)
    elif result is not None and not final_answer(result.message) and str(result.stop_reason).startswith(_LIMIT_PREFIX):
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
