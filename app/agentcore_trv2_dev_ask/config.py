"""Runtime settings, read from environment variables set in agentcore.json."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id_parameter: str
    region: str
    max_tokens: int
    max_tool_calls: int
    max_turns: int
    max_history_messages: int
    max_tool_result_chars: int
    model_cache_seconds: float


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    """Build settings from the environment, falling back to the slice-1 defaults."""
    return Settings(
        model_id_parameter=env.get("MODEL_ID_PARAMETER", "/trv2/dev/agent-model-id"),
        region=env.get("AWS_REGION", "us-east-1"),
        max_tokens=int(env.get("BEDROCK_MAX_TOKENS", "800")),
        max_tool_calls=int(env.get("AGENT_MAX_TOOL_CALLS", "4")),
        max_turns=int(env.get("AGENT_MAX_TURNS", "8")),
        max_history_messages=int(env.get("AGENT_MAX_HISTORY_MESSAGES", "10")),
        max_tool_result_chars=int(env.get("AGENT_MAX_TOOL_RESULT_CHARS", "3000")),
        model_cache_seconds=float(env.get("MODEL_ID_CACHE_SECONDS", "60")),
    )
