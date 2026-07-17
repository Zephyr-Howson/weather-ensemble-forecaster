from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from weather_ensemble.retry import get_with_retry


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = MagicMock()
    response.status_code = status_code
    error = requests.exceptions.HTTPError(f"{status_code} error")
    error.response = response
    return error


def test_succeeds_on_first_try_without_sleeping():
    ok_response = MagicMock()
    ok_response.raise_for_status.return_value = None

    with patch("weather_ensemble.retry.requests.get", return_value=ok_response) as mock_get, patch(
        "weather_ensemble.retry.time.sleep"
    ) as mock_sleep:
        result = get_with_retry("https://example.com")

    assert result is ok_response
    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


def test_retries_only_the_failed_request_then_succeeds():
    ok_response = MagicMock()
    ok_response.raise_for_status.return_value = None
    failing_response = MagicMock()
    failing_response.raise_for_status.side_effect = _http_error(503)

    with patch("weather_ensemble.retry.requests.get", side_effect=[failing_response, failing_response, ok_response]) as mock_get, patch(
        "weather_ensemble.retry.time.sleep"
    ) as mock_sleep:
        result = get_with_retry("https://example.com", max_retries=4, backoff_seconds=2.0)

    assert result is ok_response
    assert mock_get.call_count == 3
    # Two failures before success - only those two are retried, backing off
    # 2s then 4s (backoff_seconds * 2**attempt), nothing beyond this request touched.
    assert mock_sleep.call_args_list == [((2.0,),), ((4.0,),)]


def test_does_not_retry_non_retryable_client_error():
    failing_response = MagicMock()
    failing_response.raise_for_status.side_effect = _http_error(404)

    with patch("weather_ensemble.retry.requests.get", return_value=failing_response) as mock_get, patch(
        "weather_ensemble.retry.time.sleep"
    ) as mock_sleep:
        try:
            get_with_retry("https://example.com")
            raised = False
        except requests.exceptions.HTTPError:
            raised = True

    assert raised
    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


def test_gives_up_after_max_retries():
    failing_response = MagicMock()
    failing_response.raise_for_status.side_effect = _http_error(429)

    with patch("weather_ensemble.retry.requests.get", return_value=failing_response) as mock_get, patch(
        "weather_ensemble.retry.time.sleep"
    ):
        try:
            get_with_retry("https://example.com", max_retries=2)
            raised = False
        except requests.exceptions.HTTPError:
            raised = True

    assert raised
    assert mock_get.call_count == 3  # the initial attempt plus 2 retries
