"""Tests for finding the gateway url injected by the AgentCore CLI."""
from tools.web_search import find_gateway_url


def test_override_wins():
    env = {"GATEWAY_URL": "https://override", "AGENTCORE_GATEWAY_GWTRV2DEVMAIN_URL": "https://injected"}
    assert find_gateway_url(env) == "https://override"


def test_injected_url_is_found_by_prefix():
    assert find_gateway_url({"AGENTCORE_GATEWAY_GWTRV2DEVMAIN_URL": "https://injected"}) == "https://injected"


def test_no_gateway():
    assert find_gateway_url({"PATH": "/bin"}) is None


def test_gateway_failure_degrades_to_no_tools(monkeypatch):
    from tools import web_search

    monkeypatch.setenv("GATEWAY_URL", "https://example.invalid/mcp")
    client = web_search.load_web_search()
    assert client is not None
    assert client._continue_on_error is True
