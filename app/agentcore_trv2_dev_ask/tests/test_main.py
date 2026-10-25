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

    taken = []

    class Connection:
        def for_question(self):
            taken.append(True)
            return MCPClient(unreachable, continue_on_error=True)

    monkeypatch.setattr(main, "WEB_SEARCH", Connection())
    main.build_agent("model-a", [])
    main.build_agent("model-a", [])
    assert taken == [True, True]
