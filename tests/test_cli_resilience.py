from __future__ import annotations

import argparse
from pathlib import Path

from weather_ensemble import cli
from weather_ensemble.config import Location

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        db=Path("unused.db"),
        window=30,
        train_window=90,
        model_dir=Path("models"),
        deploy_phases=None,
        backfill=None,
        collect_open_meteo=False,
        collect=False,
        all=False,
        record_actual=False,
        forecast=False,
        export=None,
        build_dataset=None,
        train=False,
        predict_ml=False,
        backtest_days=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_guarded_returns_true_on_success():
    calls = []
    assert cli._guarded(LOCATION, "step", lambda: calls.append(1)) is True
    assert calls == [1]


def test_guarded_catches_exception_and_returns_false(capsys):
    def _boom():
        raise RuntimeError("simulated transient failure")

    result = cli._guarded(LOCATION, "record_actual", _boom)

    assert result is False
    out = capsys.readouterr().out
    assert "WARN: record_actual failed for Melbourne" in out
    assert "simulated transient failure" in out


def test_run_for_location_survives_one_failed_step_and_still_runs_the_rest(monkeypatch, capsys):
    """A single failing step (e.g. a transient API error) must not crash the
    whole location - and must not be silently swallowed either, or a real
    systemic failure could go unnoticed."""
    monkeypatch.setattr(cli, "record_actual", lambda db, location: (_ for _ in ()).throw(RuntimeError("network blip")))
    forecast_calls = []
    monkeypatch.setattr(cli, "blend_forecast", lambda db, location, window: forecast_calls.append(location.name) or {"ok": True})

    args = _base_args(record_actual=True, forecast=True)
    ok = cli._run_for_location(args, LOCATION)

    assert ok is False
    assert forecast_calls == ["Melbourne"]  # ran despite record_actual failing
    assert "WARN: record_actual failed for Melbourne" in capsys.readouterr().out


def test_run_for_location_returns_true_when_every_step_succeeds(monkeypatch):
    monkeypatch.setattr(cli, "record_actual", lambda db, location: None)
    args = _base_args(record_actual=True)
    assert cli._run_for_location(args, LOCATION) is True
