"""Smoke test that main wires the agent without calling AWS."""
import main


def test_build_agent_wires_model_tools_hooks_and_history():
    history = [{"role": "user", "content": [{"text": "Q"}]}, {"role": "assistant", "content": [{"text": "A"}]}]
    agent = main.build_agent("us.anthropic.claude-sonnet-5-5", history)
    assert agent.model.config["model_id"] == "us.anthropic.claude-sonnet-5-5"
    assert agent.messages == history
    assert {"nearby_places", "find_place"} <= set(agent.tool_names)
    assert "ライダー" in agent.system_prompt


def test_each_agent_takes_the_web_search_from_the_held_connection(monkeypatch):
    # The connection, not a bare client, hands the MCP client to each question's agent (Agent #2).
    from strands.tools.mcp import MCPClient

    def unreachable():
        raise ConnectionError("no gateway")

    from strands.hooks import HookProvider

    taken, watched = [], []

    class Watch(HookProvider):
        def register_hooks(self, registry, **kwargs):
            watched.append(True)

    class Connection:
        def for_question(self):
            taken.append(True)
            return MCPClient(unreachable, continue_on_error=True)

        def failure_watch(self):
            return Watch()

    monkeypatch.setattr(main, "WEB_SEARCH", Connection())
    main.build_agent("model-a", [])
    main.build_agent("model-a", [])
    # Each agent also reports a failed web search back to the connection.
    assert taken == [True, True] and watched == [True, True]


def test_agents_built_at_once_load_the_web_search_one_at_a_time(monkeypatch):
    # Two first questions at once would both start the MCP client, and the second start fails (Agent #10).
    import threading
    import time

    from strands.hooks import HookProvider
    from strands.tools import ToolProvider

    inside, most, built = [0], [0], []

    class SlowProvider(ToolProvider):
        async def load_tools(self, **kwargs):
            inside[0] += 1
            most[0] = max(most[0], inside[0])
            time.sleep(0.05)
            inside[0] -= 1
            return []

        def add_consumer(self, consumer_id, **kwargs):
            pass

        def remove_consumer(self, consumer_id, **kwargs):
            pass

    class Connection:
        def for_question(self):
            return SlowProvider()

        def failure_watch(self):
            return NoWatch()

    class NoWatch(HookProvider):
        def register_hooks(self, registry, **kwargs):
            pass

    monkeypatch.setattr(main, "WEB_SEARCH", Connection())
    threads = [threading.Thread(target=lambda: built.append(main.build_agent("model-a", []))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 2 and most[0] == 1
