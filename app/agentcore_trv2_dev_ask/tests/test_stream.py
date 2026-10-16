"""Tests for shaping Strands events into the events sent to the Backend."""
from types import SimpleNamespace

from stream import EventShaper, done_event, error_event, final_answer


def test_text_delta_becomes_text_event():
    assert EventShaper().shape({"data": "こん", "delta": {}}) == [{"type": "text", "text": "こん"}]


def test_tool_use_is_reported_once_per_call():
    shaper = EventShaper()
    event = {"type": "tool_use_stream", "current_tool_use": {"toolUseId": "t1", "name": "nearby_places", "input": ""}}
    assert shaper.shape(event) == [{"type": "tool", "name": "nearby_places"}]
    assert shaper.shape(event) == []


def test_other_events_are_dropped():
    assert EventShaper().shape({"init_event_loop": True}) == []
    assert EventShaper().shape({"data": ""}) == []


def test_final_answer_joins_text_blocks():
    message = {"role": "assistant", "content": [{"text": "A"}, {"toolUse": {}}, {"text": "B "}]}
    assert final_answer(message) == "AB"


def test_done_event_carries_answer_usage_and_model():
    result = SimpleNamespace(
        message={"role": "assistant", "content": [{"text": "愛鷹山です。"}]},
        stop_reason="end_turn",
        metrics=SimpleNamespace(accumulated_usage={"inputTokens": 1200, "outputTokens": 80, "totalTokens": 1280}, cycle_count=2),
    )
    assert done_event(result, "model-a") == {
        "type": "done",
        "answer": "愛鷹山です。",
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1200, "outputTokens": 80, "modelCalls": 2},
        "modelId": "model-a",
    }


def test_error_event_shape():
    assert error_event("agent_failed", "x") == {"type": "error", "code": "agent_failed", "message": "x"}
