"""
agent/llm/_retry.py — shared retry for Gemini calls, covering two distinct
transient conditions any real caller (not just tests) needs to survive:
  - 429 RESOURCE_EXHAUSTED (rate/quota limit) -- retry after the API's own
    suggested retryDelay, when given.
  - 5xx (e.g. 503 UNAVAILABLE, "high demand... usually temporary") --
    exponential backoff, no server-supplied delay to key off.
"""
from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

from google.genai import errors as genai_errors

T = TypeVar("T")

MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_SECONDS = 20.0


def _suggested_delay(exc: Exception) -> float:
    message = str(exc)
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", message)
    if match:
        return float(match.group(1)) + 1.0
    return DEFAULT_BACKOFF_SECONDS


def call_with_retry(fn: Callable[[], T]) -> T:
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn()
        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) != 429:
                raise
            last_exc = exc
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(_suggested_delay(exc))
        except genai_errors.ServerError as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(DEFAULT_BACKOFF_SECONDS * (2 ** attempt))
    raise last_exc  # pragma: no cover -- unreachable, satisfies type checkers
