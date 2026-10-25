"""Tests for finding the gateway url injected by the AgentCore CLI, and for keeping its connection."""
import asyncio

from strands.tools.mcp import MCPClient

from tools.web_search import KEEPER, WebSearchConnection, find_gateway_url


class FakeClient:
    def __init__(self):
        self.consumers = set()
        self.removed = []
        self._connection_failed = False

    def add_consumer(self, consumer_id, **kwargs):
        self.consumers.add(consumer_id)

    def remove_consumer(self, consumer_id, **kwargs):
        self.removed.append(consumer_id)
        self.consumers.discard(consumer_id)
        if not self.consumers:
            self._connection_failed = False


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_connection_is_held_across_questions():
    # Each question's agent is a consumer of the client; with one held for the process, the client is not
    # stopped and started again for every question (Agent #2).
    client = FakeClient()
    connection = WebSearchConnection(client, max_idle_seconds=600, clock=Clock())
    assert connection.for_question() is client and KEEPER in client.consumers
    connection.for_question()
    assert client.removed == []


def test_a_failed_connection_is_tried_again_on_the_next_question():
    client, clock = FakeClient(), Clock()
    connection = WebSearchConnection(client, max_idle_seconds=600, clock=clock)
    connection.for_question()
    client._connection_failed = True
    connection.for_question()
    assert client.removed == [KEEPER] and KEEPER in client.consumers and client._connection_failed is False


def test_a_connection_left_idle_is_started_again():
    # The gateway's session may have gone while nobody asked.
    client, clock = FakeClient(), Clock()
    connection = WebSearchConnection(client, max_idle_seconds=600, clock=clock)
    connection.for_question()
    clock.now += 600
    connection.for_question()
    assert client.removed == []
    clock.now += 601
    connection.for_question()
    assert client.removed == [KEEPER] and KEEPER in client.consumers


def test_a_real_client_that_failed_to_connect_connects_again():
    # Strands keeps a swallowed start failure, and its stop() empties the consumers; both are undone here.
    def unreachable():
        raise ConnectionError("no gateway")

    client = MCPClient(unreachable, continue_on_error=True)
    connection = WebSearchConnection(client, max_idle_seconds=600)
    connection.for_question()
    assert asyncio.run(client.load_tools()) == [] and client._connection_failed is True
    connection.for_question()
    assert client._connection_failed is False and KEEPER in client._consumers


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
