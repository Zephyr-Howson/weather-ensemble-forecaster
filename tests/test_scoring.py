import pandas as pd

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
