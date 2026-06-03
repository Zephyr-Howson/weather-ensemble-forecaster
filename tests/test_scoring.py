import pandas as pd

from weather_ensemble.service import classify_condition, compute_mae_scores
from weather_ensemble.sources.open_meteo import precipitation_to_rain_probability


def test_classify_condition():
    assert classify_condition(5) == "clear"
    assert classify_condition(40) == "cloudy"
    assert classify_condition(80) == "rainy"


def test_precipitation_to_rain_probability():
    assert precipitation_to_rain_probability(0) == 5.0
    assert precipitation_to_rain_probability(0.2) == 30.0
    assert precipitation_to_rain_probability(2.0) == 70.0
    assert precipitation_to_rain_probability(8.0) == 90.0


def test_compute_mae_scores():
    df = pd.DataFrame(
        {
            "source": ["a", "a", "b"],
            "max_temp": [20, 24, 19],
            "actual_max_temp": [22, 22, 22],
            "min_temp": [10, 11, 9],
            "actual_min_temp": [10, 10, 10],
            "rain_probability": [20, 80, 30],
            "actual_rain_probability": [30, 60, 30],
            "uv_index": [5, 6, 6],
            "actual_uv_index": [5, 5, 5],
            "wind_speed": [20, 25, 30],
            "actual_wind_speed": [22, 22, 22],
        }
    )
    scores = compute_mae_scores(df)
    assert scores["a"]["max_temp"] == 2.0
    assert scores["b"]["max_temp"] == 3.0
