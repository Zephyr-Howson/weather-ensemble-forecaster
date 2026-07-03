from __future__ import annotations

from weather_ensemble.config import Location
from weather_ensemble.sources import visual_crossing, weatherapi


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
        weatherapi.requests,
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
        visual_crossing.requests,
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
