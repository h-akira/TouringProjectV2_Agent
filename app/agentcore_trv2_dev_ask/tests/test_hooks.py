"""Tests for the tool-call limit and the tool-result trimming hooks."""
from types import SimpleNamespace

from strands.hooks import HookRegistry

from hooks import LIMIT_MESSAGE, TRUNCATED_SUFFIX, ToolCallLimit, ToolResultTrimmer, trim_tool_result


def before_event():
    return SimpleNamespace(cancel_tool=False, tool_use={"name": "web", "toolUseId": "t"})


def test_calls_within_limit_run():
    limit = ToolCallLimit(max_calls=2)
    events = [before_event() for _ in range(2)]
    for event in events:
        limit.on_before_tool_call(event)
    assert [e.cancel_tool for e in events] == [False, False]


def test_calls_over_limit_are_cancelled_with_message():
    limit = ToolCallLimit(max_calls=1)
    first, second, third = before_event(), before_event(), before_event()
    for event in (first, second, third):
        limit.on_before_tool_call(event)
    assert first.cancel_tool is False
    assert second.cancel_tool == LIMIT_MESSAGE
    assert third.cancel_tool == LIMIT_MESSAGE


def test_hooks_register_callbacks():
    registry = HookRegistry()
    ToolCallLimit(1).register_hooks(registry)
    ToolResultTrimmer(10).register_hooks(registry)
    assert registry.has_callbacks()


def test_short_result_is_unchanged():
    result = {"toolUseId": "t", "status": "success", "content": [{"text": "abc"}, {"json": {"k": 1}}]}
    assert trim_tool_result(result, 100) == result


def test_long_text_is_cut_at_budget():
    result = {"toolUseId": "t", "status": "success", "content": [{"text": "a" * 10}, {"text": "b" * 10}]}
    trimmed = trim_tool_result(result, 15)
    assert trimmed["content"] == [{"text": "a" * 10}, {"text": "b" * 5 + TRUNCATED_SUFFIX}]
    assert trimmed["toolUseId"] == "t" and trimmed["status"] == "success"


def test_json_blocks_are_counted_and_trimmed():
    result = {"toolUseId": "t", "status": "success", "content": [{"json": {"results": ["x" * 50]}}]}
    trimmed = trim_tool_result(result, 20)
    assert len(trimmed["content"]) == 1
    assert trimmed["content"][0]["text"].endswith(TRUNCATED_SUFFIX)
    assert len(trimmed["content"][0]["text"]) == 20 + len(TRUNCATED_SUFFIX)


def test_trimmer_replaces_event_result():
    event = SimpleNamespace(result={"toolUseId": "t", "status": "success", "content": [{"text": "a" * 30}]})
    ToolResultTrimmer(max_chars=10).on_after_tool_call(event)
    assert event.result["content"] == [{"text": "a" * 10 + TRUNCATED_SUFFIX}]


def test_trimmer_ignores_non_dict_results():
    error = RuntimeError("x")
    event = SimpleNamespace(result=error)
    ToolResultTrimmer(max_chars=10).on_after_tool_call(event)
    assert event.result is error
