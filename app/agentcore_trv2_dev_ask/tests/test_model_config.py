"""Tests for reading the model id from SSM with a short cache."""
import pytest

from model_config import ModelIdError, ModelIdSource


class FakeSsm:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def get_parameter(self, Name):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return {"Parameter": {"Name": Name, "Value": value}}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_reads_once_within_cache_window():
    ssm, clock = FakeSsm(["model-a"]), Clock()
    source = ModelIdSource("/p", ssm, cache_seconds=60, clock=clock)
    assert source.get() == "model-a"
    clock.now = 59
    assert source.get() == "model-a"
    assert ssm.calls == 1


def test_rereads_after_cache_expires():
    ssm, clock = FakeSsm(["model-a", "model-b"]), Clock()
    source = ModelIdSource("/p", ssm, cache_seconds=60, clock=clock)
    source.get()
    clock.now = 61
    assert source.get() == "model-b"


def test_value_is_stripped_and_empty_is_an_error():
    assert ModelIdSource("/p", FakeSsm([" model-a \n"]), 60, Clock()).get() == "model-a"
    with pytest.raises(ModelIdError, match="/p is empty"):
        ModelIdSource("/p", FakeSsm(["  "]), 60, Clock()).get()


def test_client_error_is_wrapped():
    with pytest.raises(ModelIdError, match="failed to read /p: RuntimeError"):
        ModelIdSource("/p", FakeSsm([RuntimeError("denied")]), 60, Clock()).get()


def test_failure_is_not_cached():
    ssm = FakeSsm([RuntimeError("denied"), "model-a"])
    source = ModelIdSource("/p", ssm, 60, Clock())
    with pytest.raises(ModelIdError):
        source.get()
    assert source.get() == "model-a"
