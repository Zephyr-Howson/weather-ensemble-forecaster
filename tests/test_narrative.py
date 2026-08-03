from __future__ import annotations

from datetime import date

from weather_ensemble import narrative
from weather_ensemble.config import Location, get_periods_db_path
from weather_ensemble.db import connect, connect_periods

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")
TARGET_DATE = date(2026, 6, 21)


def _insert_best_prediction(db_path, **overrides):
    values = {
        "location_name": LOCATION.name,
        "lat": LOCATION.lat,
        "lon": LOCATION.lon,
        "forecast_date": TARGET_DATE.isoformat(),
        "generated_at": "2026-06-20T21:00:00",
        "window_days": 30,
        "min_days": 14,
        "max_temp": 24.3,
        "min_temp": 14.1,
        "precipitation_sum": 2.1,
        "did_rain_probability": 0.39,
        "wind_speed": 14.0,
        "wind_gusts": 35.0,
        "cloud_cover": 25.0,
        "humidity": 55.0,
        "pressure_msl": 1015.0,
    }
    values.update(overrides)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO best_predictions (
                location_name, lat, lon, forecast_date, generated_at, window_days, min_days,
                max_temp, min_temp, precipitation_sum, did_rain_probability,
                wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl
            ) VALUES (
                :location_name, :lat, :lon, :forecast_date, :generated_at, :window_days, :min_days,
                :max_temp, :min_temp, :precipitation_sum, :did_rain_probability,
                :wind_speed, :wind_gusts, :cloud_cover, :humidity, :pressure_msl
            )
            """,
            values,
        )
        conn.commit()


def test_generate_and_store_best_narrative_skips_when_no_best_prediction(tmp_path):
    """No best_predictions row for this date at all - nothing to narrate,
    so this comes back as a clean "skipped" result rather than an error.
    """
    db_path = tmp_path / "weather.db"
    connect(db_path).close()  # create the schema, but insert nothing

    result = narrative.generate_and_store_best_narrative(db_path, LOCATION, target_date=TARGET_DATE)

    assert result == {"skipped": "no_best_prediction", "forecast_date": TARGET_DATE.isoformat()}


def test_generate_and_store_best_narrative_skips_when_llm_unavailable(tmp_path, monkeypatch):
    """A Best prediction exists, but no ANTHROPIC_API_KEY is configured - the
    whole feature must degrade to a silent skip, never an exception, and
    never a partial/broken row written to best_narratives.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db_path = tmp_path / "weather.db"
    _insert_best_prediction(db_path)

    result = narrative.generate_and_store_best_narrative(db_path, LOCATION, target_date=TARGET_DATE)

    assert result == {"skipped": "llm_unavailable", "forecast_date": TARGET_DATE.isoformat()}
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM best_narratives").fetchall()
    assert rows == []


class _RaisingMessages:
    def create(self, **kwargs):
        raise RuntimeError("simulated API failure (auth/network/rate-limit/whatever)")


class _RaisingClient:
    def __init__(self, *args, **kwargs):
        self.messages = _RaisingMessages()


def test_call_llm_returns_none_when_the_api_call_itself_raises(monkeypatch):
    """Any failure from the API call (auth, network, rate limit, a malformed
    response) must be swallowed by _call_llm's own try/except, not just the
    "no key configured" early-out - the caller never sees an exception.
    """
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
    monkeypatch.setattr(anthropic, "Anthropic", _RaisingClient)

    assert narrative._call_llm("anything") is None


def test_generate_and_store_best_narrative_skips_when_llm_raises(tmp_path, monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
    monkeypatch.setattr(anthropic, "Anthropic", _RaisingClient)
    db_path = tmp_path / "weather.db"
    _insert_best_prediction(db_path)

    result = narrative.generate_and_store_best_narrative(db_path, LOCATION, target_date=TARGET_DATE)

    assert result == {"skipped": "llm_unavailable", "forecast_date": TARGET_DATE.isoformat()}
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM best_narratives").fetchall()
    assert rows == []


def test_generate_and_store_best_narrative_persists_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
    db_path = tmp_path / "weather.db"
    _insert_best_prediction(db_path)

    with connect_periods(get_periods_db_path(db_path)) as pconn:
        pconn.execute(
            "INSERT INTO best_predictions_periods "
            "(location_name, lat, lon, forecast_date, period, generated_at, window_days, min_days, precipitation_sum) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (LOCATION.name, LOCATION.lat, LOCATION.lon, TARGET_DATE.isoformat(), "afternoon", "2026-06-20T21:00:00", 30, 14, 1.8),
        )
        pconn.commit()

    monkeypatch.setattr(narrative, "_call_llm", lambda prompt: "A mild day, mostly sunny with a chance of afternoon showers.")

    result = narrative.generate_and_store_best_narrative(db_path, LOCATION, target_date=TARGET_DATE)

    assert result["narrative"] == "A mild day, mostly sunny with a chance of afternoon showers."
    assert result["model"] == narrative.MODEL

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT forecast_date, model, narrative FROM best_narratives WHERE location_name = ?", (LOCATION.name,)
        ).fetchone()
    assert row is not None
    assert row["forecast_date"] == TARGET_DATE.isoformat()
    assert row["model"] == narrative.MODEL
    assert row["narrative"] == "A mild day, mostly sunny with a chance of afternoon showers."


def test_build_user_prompt_includes_key_fields():
    prediction = {
        "max_temp": 24.3,
        "min_temp": 14.1,
        "precipitation_sum": 2.1,
        "did_rain_probability": 0.39,
        "cloud_cover": 25.0,
        "wind_speed": 14.0,
        "wind_gusts": 35.0,
    }
    period_precip = {"overnight": 0.0, "morning": 0.0, "afternoon": 1.3, "evening": 0.8}

    prompt = narrative._build_user_prompt(LOCATION, TARGET_DATE, prediction, period_precip)

    assert "Melbourne" in prompt
    assert "24.3" in prompt and "14.1" in prompt
    assert "39%" in prompt
    assert "2.1mm" in prompt
    assert "afternoon 1.3mm" in prompt
    assert "25%" in prompt
    assert "14 km/h" in prompt and "35 km/h" in prompt


def test_call_llm_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert narrative._call_llm("anything") is None
