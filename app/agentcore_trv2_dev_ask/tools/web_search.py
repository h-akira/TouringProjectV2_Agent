"""Web search through the AgentCore Gateway's managed web-search connector (MCP, IAM-signed)."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Optional

from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands.tools.mcp import MCPClient

_AWS_SERVICE = "bedrock-agentcore"
_PREFIX = "AGENTCORE_GATEWAY_"
_SUFFIX = "_URL"


def find_gateway_url(env: Mapping[str, str] = os.environ) -> Optional[str]:
    """The gateway endpoint: GATEWAY_URL for local runs, else the CLI-injected AGENTCORE_GATEWAY_<NAME>_URL."""
    if env.get("GATEWAY_URL"):
        return env["GATEWAY_URL"]
    for name in sorted(env):
        if name.startswith(_PREFIX) and name.endswith(_SUFFIX):
            return env[name]
    return None


def load_web_search() -> Optional[MCPClient]:
    """An MCP client for the gateway, or None so the agent still answers without search."""
    endpoint = find_gateway_url()
    if not endpoint:
        return None
    region = os.environ.get("GATEWAY_REGION", "us-east-1")
    return MCPClient(lambda: aws_iam_streamablehttp_client(endpoint=endpoint, aws_region=region, aws_service=_AWS_SERVICE))
