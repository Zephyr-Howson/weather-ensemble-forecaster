from __future__ import annotations

import time
from typing import Any

import requests

# 429 (rate-limited) and 5xx (server-side) are worth retrying - the request
# itself was fine, the server just couldn't handle it *right now*. 4xx client
# errors (bad request, missing/invalid API key, not found) are not: retrying
# an invalid request just wastes time and delays failing for a reason a retry
# can never fix.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 15,
    max_retries: int = 4,
    backoff_seconds: float = 2.0,
) -> requests.Response:
    """requests.get with exponential backoff on transient failures.

    Built after a real incident: a burst of ~120 requests (30 locations x 4
    Open-Meteo models) in quick succession intermittently hit rate-limits/
    transient server errors against Open-Meteo's free tier - previously that
    meant silently losing that source's data for that location for the day,
    with no way to distinguish "genuinely unavailable" from "server hiccuped
    once." A connection/timeout error (no HTTP response at all) is treated the
    same as a retryable status - it's the same class of transient failure.

    Retries are scoped to exactly this one request - a rate limit on one
    location's Open-Meteo call only delays and re-attempts *that* call, not
    anything else already fetched or still pending; nothing beyond this single
    URL+params is touched or redone.

    Sleeps `backoff_seconds * 2**attempt` between attempts (2s, 4s, 8s, 16s by
    default - a rate limit is more likely to have cleared after a longer wait
    than a network blip is) before giving up and re-raising the last error.
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
