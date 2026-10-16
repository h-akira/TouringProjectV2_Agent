"""Shape Strands stream events into the small set of events the Backend consumes."""
from __future__ import annotations

from typing import Any


def final_answer(message: dict) -> str:
    """Text of the last assistant message: the answer after the last tool call, without preface."""
    return "".join(block["text"] for block in message.get("content", []) if isinstance(block.get("text"), str)).strip()


def error_event(code: str, message: str) -> dict:
    return {"type": "error", "code": code, "message": message}


def done_event(result: Any, model_id: str) -> dict:
    usage = result.metrics.accumulated_usage
    return {
        "type": "done",
        "answer": final_answer(result.message),
        "stopReason": str(result.stop_reason),
        "usage": {
            "inputTokens": usage.get("inputTokens", 0),
            "outputTokens": usage.get("outputTokens", 0),
            "modelCalls": result.metrics.cycle_count,
        },
        "modelId": model_id,
    }


class EventShaper:
    """Emit text deltas and one event per tool call; drop Strands bookkeeping events."""

    def __init__(self) -> None:
        self._seen_tool_ids: set[str] = set()

    def shape(self, event: dict) -> list[dict]:
        text = event.get("data")
        if isinstance(text, str) and text:
            return [{"type": "text", "text": text}]
        tool = event.get("current_tool_use")
        if isinstance(tool, dict) and tool.get("toolUseId") and tool["toolUseId"] not in self._seen_tool_ids:
            self._seen_tool_ids.add(tool["toolUseId"])
            return [{"type": "tool", "name": tool.get("name", "")}]
        return []
