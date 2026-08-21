from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def retry(
    func: Callable[..., T],
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Retry a function call with exponential backoff."""
    last_error: Exception | None = None
    sleep_time = delay

    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - behavior is simple retry logic
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(sleep_time)
            sleep_time *= backoff

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry function ended without a result.")


async def retry_async(
    func: Callable[..., T],
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Async version of retry with exponential backoff."""
    last_error: Exception | None = None
    sleep_time = delay

    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - behavior is simple retry logic
            last_error = exc
            if attempt >= retries:
                raise
            await asyncio.sleep(sleep_time)
            sleep_time *= backoff

    if last_error is not None:
        raise last_error
    raise RuntimeError("Async retry function ended without a result.")
