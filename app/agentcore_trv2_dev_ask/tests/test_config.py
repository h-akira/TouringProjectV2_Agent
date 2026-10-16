"""Tests for config.load_settings."""
from config import load_settings


def test_defaults():
    settings = load_settings({})
    assert settings.model_id_parameter == "/trv2/dev/agent-model-id"
    assert settings.region == "us-east-1"
    assert settings.max_tokens == 800
    assert settings.max_tool_calls == 4
    assert settings.max_turns == 8
    assert settings.max_history_messages == 10
    assert settings.max_tool_result_chars == 3000
    assert settings.model_cache_seconds == 60.0


def test_environment_overrides():
    settings = load_settings({
        "MODEL_ID_PARAMETER": "/trv2/prod/agent-model-id",
        "AWS_REGION": "us-west-2",
        "BEDROCK_MAX_TOKENS": "500",
        "AGENT_MAX_TOOL_CALLS": "2",
        "AGENT_MAX_TURNS": "5",
        "AGENT_MAX_HISTORY_MESSAGES": "6",
        "AGENT_MAX_TOOL_RESULT_CHARS": "1000",
        "MODEL_ID_CACHE_SECONDS": "5",
    })
    assert settings.model_id_parameter == "/trv2/prod/agent-model-id"
    assert settings.region == "us-west-2"
    assert (settings.max_tokens, settings.max_tool_calls, settings.max_turns) == (500, 2, 5)
    assert (settings.max_history_messages, settings.max_tool_result_chars) == (6, 1000)
    assert settings.model_cache_seconds == 5.0
