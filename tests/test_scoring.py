import pandas as pd

from weather_ensemble.scoring import rolling_error_over_time
from weather_ensemble.service import classify_condition, compute_mae_scores


def test_classify_condition():
    assert classify_condition(5) == "clear"
    assert classify_condition(40) == "cloudy"
    assert classify_condition(80) == "rainy"
    assert classify_condition(None) == "unknown"


def test_compute_mae_scores_uses_true_observed_targets():
    df = pd.DataFrame(
        {
            "source": ["a", "a", "b"],
            "max_temp": [20, 24, 19],
            "actual_max_temp": [22, 22, 22],
            "min_temp": [10, 11, 9],
            "actual_min_temp": [10, 10, 10],
            "precipitation_sum": [0.0, 2.0, 5.0],
            "actual_precipitation_sum": [1.0, 1.0, 3.0],
            "wind_gusts": [40, 50, 35],
            "actual_wind_gusts": [45, 45, 30],
        }
    )
    scores = compute_mae_scores(df)
    assert scores["a"]["max_temp"] == 2.0
    assert scores["b"]["max_temp"] == 3.0
    assert scores["a"]["precipitation_sum"] == 1.0
    assert scores["a"]["wind_gusts"] == 5.0


def test_rolling_error_over_time_is_centered_not_trailing():
    """A centered window lets a date "see" errors from days after it, not just
    before - that's the whole point of switching off pandas' default trailing
    rolling window. 15 days of zero error except a single spike on day 8
    (index 7); day 5 (index 4, three days before the spike) should reflect
    it - a trailing window never would, since day 5 has no visibility into
    day 8 yet.
    """
    dates = pd.date_range("2026-06-01", periods=15, freq="D")
    abs_error = [0.0] * 15
    abs_error[7] = 7.0
    df = pd.DataFrame(
        {
            "target": ["max_temp"] * 15,
            "model": ["ensemble"] * 15,
            "forecast_date": dates,
            "abs_error": abs_error,
        }
    )

    trend = rolling_error_over_time(df, window=7)

    day5_mae = trend[trend["forecast_date"] == dates[4]]["rolling_mae"].iloc[0]
    assert day5_mae == 1.0  # mean of indices 1..7 (7 values, one of them the spike)


def test_rain_probability_is_not_scored_against_fake_actual():
    df = pd.DataFrame(
        {
            "source": ["a"],
            "rain_probability": [80.0],
            "actual_did_rain": [1],
            "actual_precipitation_sum": [2.0],
        }
    )
    scores = compute_mae_scores(df)
    assert "rain_probability" not in scores["a"]
