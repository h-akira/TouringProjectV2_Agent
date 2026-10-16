"""Hooks that bound the cost of one question: a tool-call cap and tool-result trimming."""
from __future__ import annotations

import json
from typing import Any

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

LIMIT_MESSAGE = "ツールの呼び出し回数の上限に達しました。これ以上は調べずに、手持ちの情報で答えてください。"
TRUNCATED_SUFFIX = "…（以下省略）"


class ToolCallLimit(HookProvider):
    """Cancel tool calls past max_calls so the model answers with what it has."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.on_before_tool_call)

    def on_before_tool_call(self, event: Any) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            event.cancel_tool = LIMIT_MESSAGE


def _block_text(block: dict) -> str | None:
    if isinstance(block.get("text"), str):
        return block["text"]
    if "json" in block:
        return json.dumps(block["json"], ensure_ascii=False)
    return None


def trim_tool_result(result: dict, max_chars: int) -> dict:
    """Return the result with its text cut to max_chars in total; unchanged if it already fits."""
    texts = [text for text in (_block_text(block) for block in result.get("content", [])) if text is not None]
    if sum(len(text) for text in texts) <= max_chars:
        return result
    remaining = max_chars
    blocks: list[dict] = []
    for text in texts:
        if remaining <= 0:
            break
        if len(text) > remaining:
            blocks.append({"text": text[:remaining] + TRUNCATED_SUFFIX})
            remaining = 0
        else:
            blocks.append({"text": text})
            remaining -= len(text)
    return {**result, "content": blocks}


class ToolResultTrimmer(HookProvider):
    """Trim every tool result before it reaches the model (search results dominate input tokens)."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self.on_after_tool_call)

    def on_after_tool_call(self, event: Any) -> None:
        if isinstance(event.result, dict):
            event.result = trim_tool_result(event.result, self.max_chars)
