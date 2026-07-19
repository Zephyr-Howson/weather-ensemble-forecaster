from __future__ import annotations

from datetime import date, datetime

from weather_ensemble.config import Location
from weather_ensemble.models import ActualRecord
from weather_ensemble.service import _fetch_blended_actual
from weather_ensemble.sources import open_meteo, silo

LOCATION = Location(name="Melbourne", lat=-37.8, lon=144.9)
TARGET_DATE = date(2026, 6, 19)


def _base_record(**overrides) -> ActualRecord:
    fields = dict(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=TARGET_DATE,
        collected_at=datetime(2026, 6, 20, 9, 0),
        max_temp=20.0,
        min_temp=10.0,
        precipitation_sum=0.0,
        did_rain=0,
        uv_index=4.0,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={"open_meteo": True},
    )
    fields.update(overrides)
    return ActualRecord(**fields)


def test_silo_overrides_only_its_own_fields(monkeypatch):
    monkeypatch.setattr(open_meteo, "fetch_actual", lambda location, target_date: _base_record())
    monkeypatch.setattr(
        silo,
        "fetch_actual",
        lambda location, target_date: ActualRecord(
            source="silo",
            location_name=LOCATION.name,
            lat=LOCATION.lat,
            lon=LOCATION.lon,
            actual_date=TARGET_DATE,
            collected_at=datetime(2026, 6, 20, 9, 0),
            max_temp=21.5,
            min_temp=11.5,
            precipitation_sum=5.0,
            did_rain=1,
            raw_json={"silo": True},
        ),
    )

    result = _fetch_blended_actual(LOCATION, TARGET_DATE)

    # SILO's rainfall/max/min temp win...
    assert result.max_temp == 21.5
    assert result.min_temp == 11.5
    assert result.precipitation_sum == 5.0
    assert result.did_rain == 1
    # ...but everything SILO doesn't cover stays Open-Meteo's.
    assert result.uv_index == 4.0
    assert result.wind_speed == 15.0
    assert result.cloud_cover == 40.0
    assert result.humidity == 60.0
    assert result.pressure_msl == 1015.0
    # source must stay exactly what Open-Meteo reported - db.upsert_actual's
    # ON CONFLICT key is (source, location_name, actual_date), so changing it
    # here would insert a duplicate row instead of updating the existing one.
    assert result.source == "open_meteo_archive"
    assert result.raw_json["silo_overridden_fields"] == ["did_rain", "max_temp", "min_temp", "precipitation_sum"]


def test_silo_failure_falls_back_to_pure_open_meteo(monkeypatch):
    base = _base_record()
    monkeypatch.setattr(open_meteo, "fetch_actual", lambda location, target_date: base)
    monkeypatch.setattr(
        silo,
        "fetch_actual",
        lambda location, target_date: (_ for _ in ()).throw(RuntimeError("SILO_EMAIL is not set.")),
    )

    result = _fetch_blended_actual(LOCATION, TARGET_DATE)

    assert result is base


def test_silo_with_no_data_for_date_falls_back_to_pure_open_meteo(monkeypatch):
    base = _base_record()
    monkeypatch.setattr(open_meteo, "fetch_actual", lambda location, target_date: base)
    monkeypatch.setattr(
        silo,
        "fetch_actual",
        lambda location, target_date: ActualRecord(
            source="silo",
            location_name=LOCATION.name,
            lat=LOCATION.lat,
            lon=LOCATION.lon,
            actual_date=TARGET_DATE,
            collected_at=datetime(2026, 6, 20, 9, 0),
        ),
    )

    result = _fetch_blended_actual(LOCATION, TARGET_DATE)

    assert result is base
