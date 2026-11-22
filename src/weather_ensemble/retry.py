from __future__ import annotations

import time
from typing import Any

import requests

# 429 (rate-limited) and the transient 5xx codes are worth retrying - the
# request itself was fine, the server just couldn't handle it *right now*.
# Other HTTP errors (bad request, missing/invalid API key, not found) are
# raised immediately: a retry can never fix them.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 15,
    max_retries: int = 4,
    backoff_seconds: float = 2.0,
) -> requests.Response:
    """requests.get with exponential backoff on transient failures.

    A nightly run sends bursts of requests (e.g. 30 locations x 4 Open-Meteo
    models) that can intermittently hit rate limits or transient server errors,
    especially on free tiers. Retried: a RETRYABLE_STATUS_CODES response, or a
    request error with no HTTP response at all (connection error, timeout).
    Any other HTTP error is raised at once. Only this one request is retried.

    Makes up to max_retries + 1 attempts (5 by default), sleeping
    `backoff_seconds * 2**attempt` between them (2s, 4s, 8s, 16s by default),
    then re-raises the last error.
    """
    last_exc: requests.exceptions.RequestException | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            status = getattr(exc.response, "status_code", None)
            if status is not None and status not in RETRYABLE_STATUS_CODES:
                raise
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_seconds * (2**attempt))
    raise last_exc
