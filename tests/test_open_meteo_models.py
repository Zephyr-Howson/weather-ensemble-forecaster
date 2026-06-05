from weather_ensemble.config import OPEN_METEO_MODELS
from weather_ensemble.sources import OPEN_METEO_FORECAST_SOURCES


def test_open_meteo_model_registry_has_multiple_models():
    assert len(OPEN_METEO_MODELS) >= 4
    assert "best_match" in OPEN_METEO_MODELS
    assert "ecmwf_ifs025" in OPEN_METEO_MODELS
    assert "gfs_global" in OPEN_METEO_MODELS


def test_open_meteo_sources_match_model_registry():
    expected = {f"open_meteo_{model}" for model in OPEN_METEO_MODELS}
    assert set(OPEN_METEO_FORECAST_SOURCES) == expected
