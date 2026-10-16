"""Smoke test that main wires the agent without calling AWS."""
import main


def test_build_agent_wires_model_tools_hooks_and_history():
    history = [{"role": "user", "content": [{"text": "Q"}]}, {"role": "assistant", "content": [{"text": "A"}]}]
    agent = main.build_agent("us.anthropic.claude-sonnet-5-5", history)
    assert agent.model.config["model_id"] == "us.anthropic.claude-sonnet-5-5"
    assert agent.messages == history
    assert {"nearby_places", "find_place"} <= set(agent.tool_names)
    assert "ライダー" in agent.system_prompt
