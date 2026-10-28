"""Tests for one invocation: payload checks, model id, streaming, history and errors."""
import asyncio
import threading
from types import SimpleNamespace

from strands import Agent
from strands.models import Model

from config import load_settings
from handler import LIMIT_ANSWER, Deps, handle
from history import SessionStore
from model_config import ModelIdError


def user(text):
    return {"role": "user", "content": [{"text": text}]}


def assistant(text):
    return {"role": "assistant", "content": [{"text": text}]}


def make_result(text, usage=None):
    return SimpleNamespace(
        message=assistant(text),
        stop_reason="end_turn",
        metrics=SimpleNamespace(accumulated_usage=usage or {"inputTokens": 10, "outputTokens": 5}, cycle_count=1),
    )


class FakeAgent:
    def __init__(self, messages, events, final_text, raise_after=None):
        self.messages = list(messages)
        self._events = events
        self._final_text = final_text
        self._raise_after = raise_after
        self.calls = []

    async def stream_async(self, prompt, invocation_state=None, limits=None):
        self.calls.append({"prompt": prompt, "invocation_state": invocation_state, "limits": limits})
        self.messages.append(user(prompt))
        for index, event in enumerate(self._events):
            if self._raise_after is not None and index == self._raise_after:
                raise RuntimeError("model failed")
            yield event
        self.messages.append(assistant(self._final_text))
        yield {"result": make_result(self._final_text)}


class FixedModelIds:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def get(self):
        if self.error:
            raise self.error
        return self.value


def run(payload, deps, session_id="s1"):
    async def collect():
        return [event async for event in handle(payload, session_id, deps)]

    return asyncio.run(collect())


def make_deps(agent=None, model_ids=None, sessions=None):
    created = []

    def factory(model_id, messages):
        created.append((model_id, messages))
        return agent

    deps = Deps(
        settings=load_settings({}),
        model_ids=model_ids or FixedModelIds("model-a"),
        sessions=sessions or SessionStore(),
        agent_factory=factory,
    )
    return deps, created


def test_missing_prompt_is_bad_request():
    deps, created = make_deps()
    assert run({"location": None}, deps) == [{"type": "error", "code": "bad_request", "message": "prompt is required"}]
    assert created == []


def test_model_id_error_is_agent_failed():
    deps, created = make_deps(model_ids=FixedModelIds(error=ModelIdError("failed to read /p: RuntimeError")))
    events = run({"prompt": "Q"}, deps)
    assert events == [{"type": "error", "code": "agent_failed", "message": "failed to read /p: RuntimeError"}]
    assert created == []


def test_normal_flow_streams_and_ends_with_done():
    agent = FakeAgent([], [{"data": "愛鷹"}, {"data": "山です。"}], "愛鷹山です。")
    deps, created = make_deps(agent=agent)
    location = {"latitude": 35.1, "longitude": 138.8, "headingDegrees": 90}
    events = run({"prompt": "右手の山は？", "location": location}, deps)
    assert events[:2] == [{"type": "text", "text": "愛鷹"}, {"type": "text", "text": "山です。"}]
    assert events[-1]["type"] == "done" and events[-1]["answer"] == "愛鷹山です。" and events[-1]["modelId"] == "model-a"
    assert created == [("model-a", [])]
    assert agent.calls[0]["invocation_state"] == {"location": location}
    assert agent.calls[0]["limits"] == {"turns": 8}


def test_answer_excludes_preface_before_tool_call():
    agent = FakeAgent(
        [],
        [{"data": "調べます。"}, {"current_tool_use": {"toolUseId": "t1", "name": "web_search"}}, {"data": "晴れです。"}],
        "晴れです。",
    )
    deps, _ = make_deps(agent=agent)
    events = run({"prompt": "天気は？"}, deps)
    assert {"type": "tool", "name": "web_search"} in events
    assert events[-1]["answer"] == "晴れです。"


def test_history_is_stored_pruned_and_reused():
    sessions = SessionStore()
    sessions.put("s1", [user("Q0"), assistant("A0")])
    agent = FakeAgent([user("Q0"), assistant("A0")], [], "A1")
    deps, created = make_deps(agent=agent, sessions=sessions)
    run({"prompt": "Q1"}, deps)
    assert created[0][1] == [user("Q0"), assistant("A0")]
    assert sessions.get("s1") == [user("Q0"), assistant("A0"), user("Q1"), assistant("A1")]


def test_error_does_not_store_history():
    sessions = SessionStore()
    sessions.put("s1", [user("Q0"), assistant("A0")])
    agent = FakeAgent([user("Q0"), assistant("A0")], [{"data": "x"}, {"data": "y"}], "never", raise_after=1)
    deps, _ = make_deps(agent=agent, sessions=sessions)
    events = run({"prompt": "Q1"}, deps)
    assert events[-1] == {"type": "error", "code": "agent_failed", "message": "RuntimeError"}
    assert sessions.get("s1") == [user("Q0"), assistant("A0")]


def test_empty_answer_is_agent_failed():
    agent = FakeAgent([], [], "   ")
    deps, _ = make_deps(agent=agent)
    assert run({"prompt": "Q"}, deps)[-1] == {"type": "error", "code": "agent_failed", "message": "empty answer"}


class LimitedAgent(FakeAgent):
    """The model kept calling tools until the turn cap; the last message is a tool result, not an answer."""

    def __init__(self, messages, stop_reason="limit_turns"):
        super().__init__(messages, [], "")
        self._stop_reason = stop_reason

    async def stream_async(self, prompt, invocation_state=None, limits=None):
        self.calls.append({"prompt": prompt, "invocation_state": invocation_state, "limits": limits})
        self.messages.append(user(prompt))
        self.messages.append({"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "web_search", "input": {}}}]})
        tool_result = {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "上限"}]}}]}
        self.messages.append(tool_result)
        yield {"current_tool_use": {"toolUseId": "t1", "name": "web_search"}}
        yield {"result": SimpleNamespace(message=tool_result, stop_reason=self._stop_reason,
                                         metrics=SimpleNamespace(accumulated_usage={"inputTokens": 10}, cycle_count=8))}


def test_a_question_cut_off_at_the_turn_cap_is_answered_with_a_fixed_line():
    # The rider hears that it could not be looked up, not the failure tone (Agent #3).
    sessions = SessionStore()
    deps, _ = make_deps(agent=LimitedAgent([]), sessions=sessions)
    events = run({"prompt": "Q1"}, deps)
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == LIMIT_ANSWER and events[-1]["stopReason"] == "limit_turns"
    assert sessions.get("s1") == [user("Q1"), assistant(LIMIT_ANSWER)]


def test_an_empty_answer_that_did_not_hit_a_limit_is_still_a_failure():
    deps, _ = make_deps(agent=LimitedAgent([], stop_reason="end_turn"))
    assert run({"prompt": "Q"}, deps)[-1] == {"type": "error", "code": "agent_failed", "message": "empty answer"}


def test_reading_the_model_id_and_building_the_agent_do_not_block_the_event_loop():
    # SSM and the MCP connection are blocking calls; they run in a worker thread so /ping keeps answering (Agent #4).
    threads = {}

    class RecordingModelIds:
        def get(self):
            threads["model_id"] = threading.get_ident()
            return "model-a"

    agent = FakeAgent([], [], "A")
    deps, _ = make_deps(model_ids=RecordingModelIds())

    def factory(model_id, messages):
        threads["factory"] = threading.get_ident()
        return agent

    deps.agent_factory = factory
    assert run({"prompt": "Q"}, deps)[-1]["type"] == "done"
    loop_thread = threading.get_ident()
    assert threads["model_id"] != loop_thread and threads["factory"] != loop_thread


def test_agent_creation_error_is_agent_failed():
    def failing_factory(model_id, messages):
        raise RuntimeError("gateway unreachable")

    deps, _ = make_deps()
    deps.agent_factory = failing_factory
    assert run({"prompt": "Q"}, deps) == [{"type": "error", "code": "agent_failed", "message": "RuntimeError"}]


class CutOffModel(Model):
    """A model whose reply stops at its output token cap (Agent #9), after the given content blocks."""

    def __init__(self, blocks):
        self._blocks = blocks

    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {}

    def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        yield {"messageStart": {"role": "assistant"}}
        for block in self._blocks:
            if "text" in block:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
            else:
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": block["tool"]}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"query": "'}}}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "max_tokens"}}
        yield {"metadata": {"usage": {"inputTokens": 30, "outputTokens": 800, "totalTokens": 830},
                            "metrics": {"latencyMs": 1}}}


def cut_off_deps(blocks, sessions):
    deps, _ = make_deps(sessions=sessions)
    deps.agent_factory = lambda model_id, messages: Agent(model=CutOffModel(blocks), messages=messages,
                                                          callback_handler=None)
    return deps


def test_an_answer_cut_off_at_the_output_token_cap_is_read_as_far_as_it_goes():
    # Strands raises MaxTokensReachedException instead of returning; the partial answer is still an answer (Agent #9).
    sessions = SessionStore()
    events = run({"prompt": "Q1"}, cut_off_deps([{"text": "右手の山は筑波山で、"}], sessions))
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "右手の山は筑波山で、" and events[-1]["stopReason"] == "max_tokens"
    assert events[-1]["usage"]["outputTokens"] == 800
    assert sessions.get("s1") == [user("Q1"), assistant("右手の山は筑波山で、")]


def test_a_reply_cut_off_in_a_tool_call_is_answered_with_the_fixed_line():
    # Strands puts an English notice where the cut-off tool call was; the rider must not hear it.
    sessions = SessionStore()
    events = run({"prompt": "Q1"}, cut_off_deps([{"tool": "web_search"}], sessions))
    assert events[-1]["type"] == "done" and events[-1]["answer"] == LIMIT_ANSWER
    assert sessions.get("s1") == [user("Q1"), assistant(LIMIT_ANSWER)]


def test_a_preface_before_a_cut_off_tool_call_is_not_the_answer():
    # Only text after the last tool call is an answer (07 UC-8), so a preface is not read out.
    sessions = SessionStore()
    events = run({"prompt": "Q1"}, cut_off_deps([{"text": "ウェブで調べます。"}, {"tool": "web_search"}], sessions))
    assert events[-1]["type"] == "done" and events[-1]["answer"] == LIMIT_ANSWER
    assert sessions.get("s1") == [user("Q1"), assistant(LIMIT_ANSWER)]
