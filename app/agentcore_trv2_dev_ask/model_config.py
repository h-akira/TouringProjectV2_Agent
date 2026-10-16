"""The model id lives in SSM so an admin can switch models without a deploy."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class ModelIdError(RuntimeError):
    """The model id could not be read; the question fails instead of guessing a model."""


class ModelIdSource:
    def __init__(
        self,
        parameter_name: str,
        ssm_client: Any,
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = parameter_name
        self._ssm = ssm_client
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._cached: tuple[str, float] | None = None

    def get(self) -> str:
        now = self._clock()
        if self._cached is not None and now - self._cached[1] < self._cache_seconds:
            return self._cached[0]
        try:
            value = self._ssm.get_parameter(Name=self._name)["Parameter"]["Value"].strip()
        except Exception as error:
            raise ModelIdError(f"failed to read {self._name}: {type(error).__name__}") from error
        if not value:
            raise ModelIdError(f"{self._name} is empty")
        self._cached = (value, now)
        return value
