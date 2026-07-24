from __future__ import annotations

from weather_ensemble import retry
from weather_ensemble.config import Location
from weather_ensemble.sources import (
    accuweather,
    bom,
    openweathermap,
    silo,
    visual_crossing,
    weatherapi,
    weatherbit,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_weatherapi_preserves_zero_humidity(monkeypatch):
    monkeypatch.setattr(weatherapi.os, "getenv", lambda key: "test-key")
    monkeypatch.setattr(
        retry.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            {
                "forecast": {
                    "forecastday": [
                        {"date": "2026-06-18"},
                        {
                            "date": "2026-06-19",
                            "day": {
                                "maxtemp_c": 21,
                                "mintemp_c": 11,
                                "daily_chance_of_rain": 0,
                                "totalprecip_mm": 0,
                                "uv": 0,
                                "maxwind_kph": 0,
                                "avghumidity": 0,
                                "condition": {"code": 1000},
                            },
                            "hour": [{"gust_kph": 0, "cloud": 0, "humidity": 80, "pressure_mb": 1008}],
                        },
                    ]
                }
            }
        ),
    )

    record = weatherapi.fetch_forecast(Location(name="Melbourne", lat=-37.8, lon=144.9))

    assert record.humidity == 0


def test_visual_crossing_preserves_zero_values(monkeypatch):
    monkeypatch.setattr(visual_crossing.os, "getenv", lambda key: "test-key")
    monkeypatch.setattr(
        retry.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            {
                "days": [
                    {
                        "datetime": "2026-06-19",
                        "tempmax": 20,
                        "tempmin": 10,
                        "precipprob": 0,
                        "precip": 0,
                        "uvindex": 0,
                        "windspeed": 0,
                        "windgust": 0,
                        "cloudcover": 0,
                        "humidity": 0,
                        "pressure": 0,
                        "hours": [{"windgust": 60, "cloudcover": 70, "humidity": 80, "pressure": 1010}],
                    }
                ]
            }
        ),
    )

    record = visual_crossing.fetch_forecast(Location(name="Melbourne", lat=-37.8, lon=144.9))

    assert record.wind_gusts == 0
    assert record.cloud_cover == 0
    assert record.humidity == 0
    assert record.pressure_msl == 0


def test_openweathermap_aggregates_3hourly_into_daily(monkeypatch):
    from datetime import datetime, time, timedelta
    from zoneinfo import ZoneInfo

    from weather_ensemble.config import local_today

    location = Location(name="Melbourne", lat=-37.8, lon=144.9, timezone="Australia/Melbourne")
    target_date = local_today(location) + timedelta(days=1)
    tz = ZoneInfo(location.timezone)

    def entry_at(hour: int, day_offset: int = 0, **fields) -> dict:
        day = target_date + timedelta(days=day_offset)
        ts = int(datetime.combine(day, time(hour, 0), tzinfo=tz).timestamp())
        return {"dt": ts, "main": {}, "wind": {}, "clouds": {}, "weather": [{"id": 800}], **fields}

    monkeypatch.setattr(openweathermap.os, "getenv", lambda key: "test-key")
    monkeypatch.setattr(
        retry.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            {
                "list": [
                    # A block from the previous day should be excluded from the aggregate.
                    entry_at(21, day_offset=-1, main={"temp_max": 99.0, "temp_min": 99.0, "humidity": 99}),
                    entry_at(0, main={"temp_max": 12.0, "temp_min": 8.0, "humidity": 0, "sea_level": 1010}, pop=0, wind={"speed": 0, "gust": 0}, clouds={"all": 0}),
                    entry_at(12, main={"temp_max": 20.0, "temp_min": 15.0, "humidity": 50, "sea_level": 1010}, pop=0.25, rain={"3h": 1.5}, wind={"speed": 10.0, "gust": 20.0}, clouds={"all": 40}),
                ]
            }
        ),
    )

    record = openweathermap.fetch_forecast(location)

    assert record.forecast_date == target_date
    assert record.max_temp == 20.0
    assert record.min_temp == 8.0
    assert record.wind_speed == 36.0  # max 10 m/s -> 36 km/h
    assert record.wind_gusts == 72.0  # max 20 m/s -> 72 km/h
    assert record.rain_probability == 25.0
    assert record.precipitation_sum == 1.5
    assert record.cloud_cover == 20  # mean of 0 and 40
    assert record.humidity == 25  # mean of 0 and 50 (the 99 from the other day excluded)


def test_weatherbit_converts_wind_to_kmh_and_prefers_slp(monkeypatch):
    monkeypatch.setattr(weatherbit.os, "getenv", lambda key: "test-key")
    monkeypatch.setattr(
        retry.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            {
                "data": [
                    {"valid_date": "2026-06-18"},
                    {
                        "valid_date": "2026-06-19",
                        "max_temp": 20.0,
                        "min_temp": 10.0,
                        "pop": 0,
                        "precip": 0,
                        "uv": 0,
                        "wind_spd": 10.0,
                        "wind_gust_spd": 20.0,
                        "clouds": 0,
                        "rh": 0,
                        "pres": 1005.0,
                        "slp": 1013.0,
                        "weather": {"code": 800},
                    },
                ]
            }
        ),
    )

    record = weatherbit.fetch_forecast(Location(name="Melbourne", lat=-37.8, lon=144.9))

    assert record.wind_speed == 36.0  # 10 m/s -> 36 km/h
    assert record.wind_gusts == 72.0  # 20 m/s -> 72 km/h
    assert record.pressure_msl == 1013.0  # prefers slp over pres
    assert record.rain_probability == 0
    assert record.humidity == 0
    assert record.cloud_cover == 0


def test_accuweather_two_step_lookup_preserves_zero_values(monkeypatch):
    monkeypatch.setattr(accuweather.os, "getenv", lambda key: "test-key")

    def fake_get(url, params=None, timeout=None):
        if "geoposition" in url:
            return _FakeResponse({"Key": "12345"})
        return _FakeResponse(
            {
                "DailyForecasts": [
                    {"Date": "2026-06-18T07:00:00+10:00"},
                    {
                        "Date": "2026-06-19T07:00:00+10:00",
                        "Temperature": {"Maximum": {"Value": 20.0}, "Minimum": {"Value": 10.0}},
                        "Day": {
                            "PrecipitationProbability": 0,
                            "TotalLiquid": {"Value": 0},
                            "Wind": {"Speed": {"Value": 15.0}},
                            "WindGust": {"Speed": {"Value": 25.0}},
                            "CloudCover": 0,
                            "RelativeHumidity": {"Average": 0},
                            "Icon": 1,
                        },
                    },
                ]
            }
        )

    monkeypatch.setattr(retry.requests, "get", fake_get)

    record = accuweather.fetch_forecast(Location(name="Melbourne", lat=-37.8, lon=144.9))

    assert record.forecast_date.isoformat() == "2026-06-19"
    assert record.wind_speed == 15.0  # AccuWeather's metric wind speed is already km/h
    assert record.wind_gusts == 25.0
    assert record.precipitation_sum == 0
    assert record.cloud_cover == 0
    assert record.humidity == 0


def test_silo_parses_variable_codes(monkeypatch):
    monkeypatch.setattr(silo.os, "getenv", lambda key: "test@example.com")
    monkeypatch.setattr(
        retry.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            {
                "data": [
                    {
                        "date": "2026-06-19",
                        "variables": [
                            {"variable_code": "max_temp", "value": 20.0},
                            {"variable_code": "min_temp", "value": 10.0},
                            {"variable_code": "daily_rain", "value": 0},
                        ],
                    }
                ]
            }
        ),
    )

    from datetime import date

    record = silo.fetch_actual(Location(name="Melbourne", lat=-37.8, lon=144.9), date(2026, 6, 19))

    assert record.max_temp == 20.0
    assert record.min_temp == 10.0
    assert record.precipitation_sum == 0
    assert record.did_rain == 0


def test_bom_maps_daily_forecast_and_preserves_zero_values(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url.endswith("/locations"):
            return _FakeResponse({"data": [{"geohash": "r1r0fsn", "name": "Melbourne"}]})
        return _FakeResponse(
            {
                "data": [
                    {
                        "date": "2026-07-14T14:00:00Z",
                        "temp_max": 15,
                        "temp_min": 7,
                        "rain": {"chance": 0, "amount": {"min": 0, "max": 0}},
                        "uv": {"max_index": 0},
                    },
                    {
                        "date": "2026-07-15T14:00:00Z",
                        "temp_max": 15,
                        "temp_min": 10,
                        "rain": {"chance": 0, "amount": {"min": 0, "max": 0}},
                        "uv": {"max_index": 0},
                    },
                ]
            }
        )

    monkeypatch.setattr(retry.requests, "get", fake_get)

    record = bom.fetch_forecast(Location(name="Melbourne", lat=-37.8, lon=144.9, timezone="Australia/Melbourne"))

    assert record.forecast_date.isoformat() == "2026-07-16"
    assert record.rain_probability == 0
    assert record.precipitation_sum == 0
