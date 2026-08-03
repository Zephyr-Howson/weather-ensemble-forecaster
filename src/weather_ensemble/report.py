from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from weather_ensemble import db
from weather_ensemble.config import (
    AUSTRALIAN_LOCATIONS,
    PERIODS,
    TARGETS,
    Location,
    get_periods_db_path,
)
from weather_ensemble.data_quality import compute_data_quality
from weather_ensemble.scoring import (
    BASELINE_CLIMATOLOGY,
    BASELINE_PERSISTENCE,
    MODEL_BEST,
    MODEL_ENSEMBLE,
    MODEL_ML,
    leaderboard,
    rolling_error_over_time,
)
from weather_ensemble.sources import FORECAST_SOURCES


def _js_object_assignment(var_name: str, data: dict) -> str:
    """`window.VAR = {...}` as one assignment per top-level key instead of a
    single json.dumps of the whole dict on one line.

    A generated report with 30 locations' worth of MAE/trend history embeds
    megabytes of JSON - as one line, that's long enough that most diff tools
    (VS Code's included) give up computing a real line-by-line comparison and
    fall back to "whole line changed", making every regenerated report look
    like a total rewrite. Splitting into one line per key keeps each line's
    content bounded by a single location/target's data - typically tens of KB,
    not megabytes - so a diff can actually localize what changed. The runtime
    result is identical either way; only how it's written to the file differs.
    """
    lines = [f"window.{var_name} = {{}};"]
    for key, value in data.items():
        lines.append(f"window.{var_name}[{json.dumps(key)}] = {json.dumps(value)};")
    return "\n".join(lines)


def _location_data_script(location_data: dict) -> str:
    """Same one-assignment-per-line idea as _js_object_assignment, but one
    level deeper: location_data is target -> {"locations": {location: {...}},
    ...other small keys}, and "locations" (30 locations' full MAE/trend
    history) is what actually makes a single target's line hundreds of KB.
    Splitting that inner dict too bounds every line to roughly one location's
    worth of data - tens of KB, comfortably below where diff tools give up.
    """
    lines = ["window.__LOCATION_DATA = {};"]
    for target, spec in location_data.items():
        target_json = json.dumps(target)
        shallow = {k: v for k, v in spec.items() if k != "locations"}
        lines.append(f"window.__LOCATION_DATA[{target_json}] = {json.dumps(shallow)};")
        lines.append(f'window.__LOCATION_DATA[{target_json}]["locations"] = {{}};')
        for location_name, location_spec in spec["locations"].items():
            location_json = json.dumps(location_name)
            lines.append(f'window.__LOCATION_DATA[{target_json}]["locations"][{location_json}] = {json.dumps(location_spec)};')
    return "\n".join(lines)


TARGET_LABELS = {
    "max_temp": "Max temperature",
    "min_temp": "Min temperature",
    "precipitation_sum": "Precipitation",
    "did_rain": "Rain",
    "wind_speed": "Wind speed",
    "wind_gusts": "Wind gusts",
    "cloud_cover": "Cloud cover",
    "humidity": "Humidity",
    "pressure_msl": "Pressure (MSL)",
}

# Small-slice sub-daily rain prediction - synthetic per-period target names
# (see scoring.build_period_predictions_long's docstring for why these are
# just extra "target" values rather than a separate rendering path).
PERIOD_TARGETS = [f"precipitation_sum_{period}" for period in PERIODS]
TARGET_LABELS.update({f"precipitation_sum_{period}": f"Precipitation — {period.capitalize()}" for period in PERIODS})

# Single source of truth for target order, used by both the "recent
# forecasts" table and the historical-performance cards - requested display
# order rather than TARGETS' declaration order (did_rain moves before
# precipitation_sum, and the sub-daily periods slot in right after it).
TARGET_ORDER = [
    "max_temp",
    "min_temp",
    "did_rain",
    "precipitation_sum",
    *PERIOD_TARGETS,
    "wind_speed",
    "wind_gusts",
    "cloud_cover",
    "humidity",
    "pressure_msl",
]

RECENT_DAYS_COUNT = 5
RECENT_TARGETS = TARGET_ORDER
RECENT_UNITS = {
    "max_temp": "°C",
    "min_temp": "°C",
    "precipitation_sum": "mm",
    "wind_speed": "km/h",
    "wind_gusts": "km/h",
    "cloud_cover": "%",
    "humidity": "%",
    "pressure_msl": "hPa",
}
RECENT_UNITS.update({f"precipitation_sum_{period}": "mm" for period in PERIODS})


def _format_recent_value(target: str, value, field: str) -> str:
    if value is None:
        return "—"
    if target == "did_rain":
        if field == "actual":
            return "Yes" if value else "No"
        return f"{float(value) * 100:.0f}%"
    return f"{float(value):.1f}{RECENT_UNITS.get(target, '')}"


def _recent_forecast_data(db_path: Path, locations: list[Location]) -> dict[str, list[dict]]:
    """Per location, the most recent RECENT_DAYS_COUNT dates that have any
    ensemble/ML prediction or actual at all - not a fixed calendar-day anchor
    (today, yesterday, ...), which could land on a date with nothing recorded
    yet. Newest first. A specific field missing for one of those dates (a
    prediction that failed to generate, an actual that hasn't arrived) is left
    as None per-target rather than dropping the row, so the table renders a
    blank cell instead.
    """
    data: dict[str, list[dict]] = {}
    with db.connect(db_path) as conn, db.connect_periods(get_periods_db_path(db_path)) as period_conn:
        for location in locations:
            date_rows = conn.execute(
                """
                SELECT forecast_date AS d FROM ensemble_predictions WHERE location_name = ?
                UNION
                SELECT forecast_date AS d FROM ml_predictions WHERE location_name = ?
                UNION
                SELECT actual_date AS d FROM actuals WHERE location_name = ?
                ORDER BY d DESC LIMIT ?
                """,
                (location.name, location.name, location.name, RECENT_DAYS_COUNT),
            ).fetchall()

            days = []
            for row in date_rows:
                d_iso = row["d"]
                ens = conn.execute(
                    "SELECT * FROM ensemble_predictions WHERE location_name = ? AND forecast_date = ? "
                    "ORDER BY generated_at DESC LIMIT 1",
                    (location.name, d_iso),
                ).fetchone()
                ml = conn.execute(
                    "SELECT * FROM ml_predictions WHERE location_name = ? AND forecast_date = ? "
                    "ORDER BY generated_at DESC LIMIT 1",
                    (location.name, d_iso),
                ).fetchone()
                best = conn.execute(
                    "SELECT * FROM best_predictions WHERE location_name = ? AND forecast_date = ? "
                    "ORDER BY generated_at DESC LIMIT 1",
                    (location.name, d_iso),
                ).fetchone()
                actual = conn.execute(
                    "SELECT * FROM actuals WHERE location_name = ? AND actual_date = ? "
                    "ORDER BY collected_at DESC LIMIT 1",
                    (location.name, d_iso),
                ).fetchone()

                ensemble_values = {t: (ens[t] if ens is not None else None) for t in RECENT_TARGETS if t in TARGETS}
                ml_values = {t: (ml[t] if ml is not None else None) for t in RECENT_TARGETS if t in TARGETS}
                # best_predictions has no did_rain column at all (only
                # did_rain_probability, see db.py) - excluded from this generic
                # pull rather than raising, and set explicitly below like the
                # other two models' rain field.
                best_values = {t: (best[t] if best is not None else None) for t in RECENT_TARGETS if t in TARGETS and t != "did_rain"}
                actual_values = {t: (actual[t] if actual is not None else None) for t in RECENT_TARGETS if t in TARGETS}

                # "Rain" is a % chance forecast now (still scored against the
                # binary did_rain actual - see scoring.py), so the two forecast
                # columns show each model's underlying probability field
                # instead of its thresholded 0/1 classification. Actual stays
                # the true observed outcome, untouched.
                if ens is not None and ens["rain_probability"] is not None:
                    ensemble_values["did_rain"] = round(ens["rain_probability"] / 100.0, 3)
                if ml is not None and ml["did_rain_probability"] is not None:
                    ml_values["did_rain"] = ml["did_rain_probability"]
                if best is not None and best["did_rain_probability"] is not None:
                    best_values["did_rain"] = best["did_rain_probability"]

                # Sub-daily rain periods live in a separate DB/tables (see
                # get_periods_db_path) keyed by (location, date, period)
                # rather than a column per period, so each is a small extra
                # lookup merged in under its synthetic "precipitation_sum_
                # {period}" target name.
                for period in PERIODS:
                    key = f"precipitation_sum_{period}"
                    ens_p = period_conn.execute(
                        "SELECT precipitation_sum FROM ensemble_predictions_periods "
                        "WHERE location_name = ? AND forecast_date = ? AND period = ? "
                        "ORDER BY generated_at DESC LIMIT 1",
                        (location.name, d_iso, period),
                    ).fetchone()
                    ml_p = period_conn.execute(
                        "SELECT precipitation_sum FROM ml_predictions_periods "
                        "WHERE location_name = ? AND forecast_date = ? AND period = ? "
                        "ORDER BY generated_at DESC LIMIT 1",
                        (location.name, d_iso, period),
                    ).fetchone()
                    best_p = period_conn.execute(
                        "SELECT precipitation_sum FROM best_predictions_periods "
                        "WHERE location_name = ? AND forecast_date = ? AND period = ? "
                        "ORDER BY generated_at DESC LIMIT 1",
                        (location.name, d_iso, period),
                    ).fetchone()
                    actual_p = period_conn.execute(
                        "SELECT precipitation_sum FROM actual_periods "
                        "WHERE location_name = ? AND actual_date = ? AND period = ? "
                        "ORDER BY collected_at DESC LIMIT 1",
                        (location.name, d_iso, period),
                    ).fetchone()
                    ensemble_values[key] = ens_p["precipitation_sum"] if ens_p is not None else None
                    ml_values[key] = ml_p["precipitation_sum"] if ml_p is not None else None
                    best_values[key] = best_p["precipitation_sum"] if best_p is not None else None
                    actual_values[key] = actual_p["precipitation_sum"] if actual_p is not None else None

                days.append(
                    {
                        "date": d_iso,
                        "ensemble": ensemble_values,
                        "ml": ml_values,
                        "best": best_values,
                        "actual": actual_values,
                    }
                )
            data[location.name] = days
    return data


def _narrative_data(db_path: Path, locations: list[Location]) -> dict[str, dict | None]:
    """Per location, the latest stored best_narratives row (see narrative.py)
    - None when no narrative exists yet for that location (no
    ANTHROPIC_API_KEY configured, the LLM call failed for any reason, or
    --narrate-best just hasn't run yet). The section stays hidden entirely
    for a location with no entry, exactly like a source with no data at all
    - never a broken or stale-looking segment.
    """
    data: dict[str, dict | None] = {}
    with db.connect(db_path) as conn:
        for location in locations:
            row = conn.execute(
                "SELECT forecast_date, narrative FROM best_narratives WHERE location_name = ? "
                "ORDER BY forecast_date DESC, generated_at DESC LIMIT 1",
                (location.name,),
            ).fetchone()
            data[location.name] = {"date": row["forecast_date"], "narrative": row["narrative"]} if row is not None else None
    return data


def _narrative_html(narrative_data: dict[str, dict | None], sample_location: str) -> str:
    """Rendered once for a sample location purely as the pre-JS document
    structure, matching _recent_forecast_html's approach below - starts
    hidden since the default location selection is "All locations" (pooled),
    and updateNarrative (see _narrative_script) fills in the real selected
    location's narrative as soon as it resolves the current dropdown value.
    """
    sample = narrative_data.get(sample_location)
    date_text = escape(sample["date"]) if sample else ""
    text = escape(sample["narrative"]) if sample else ""
    return f"""<section class="narrative-section" id="narrative-section" style="display:none">
  <p class="narrative-date" id="narrative-date">{date_text}</p>
  <p class="narrative-text" id="narrative-text">{text}</p>
</section>"""


def _narrative_script(narrative_data: dict[str, dict | None]) -> str:
    return f"""
<script>
{_js_object_assignment("__NARRATIVE_DATA", narrative_data)}
function updateNarrative(loc) {{
  var section = document.getElementById("narrative-section");
  if (!section) return;
  var entry = loc ? window.__NARRATIVE_DATA[loc] : null;
  if (!loc || loc === "__ALL__" || !entry) {{
    section.style.display = "none";
    return;
  }}
  section.style.display = "";
  var dateEl = document.getElementById("narrative-date");
  if (dateEl) dateEl.textContent = entry.date;
  var textEl = document.getElementById("narrative-text");
  if (textEl) textEl.textContent = entry.narrative;
}}
</script>
"""


def _recent_forecast_html(recent_data: dict[str, list[dict]], sample_location: str) -> str:
    """Rendered once for a sample location purely as the pre-JS document
    structure - the section starts hidden (style="display:none") since the
    default location selection is "All locations" (pooled), and JS shows it
    with the real selected location's data as soon as it resolves the current
    dropdown value (which may be a remembered real location, not "__ALL__").

    One panel is shown at a time via a day-tab strip, rather than stacking all
    RECENT_DAYS_COUNT panels - on a phone that's the difference between one
    table and a scroll past ~5x as many rows before reaching the first chart.
    updateRecentForecast (see _recent_forecast_script) still refreshes every
    panel's cells regardless of which one is visible, so switching tabs is a
    pure CSS/attribute toggle with no data re-fetch.
    """
    sample_days = recent_data.get(sample_location, [])

    tabs = []
    day_cards = []
    for day_idx, day in enumerate(sample_days):
        is_first = day_idx == 0
        tabs.append(
            f'<button type="button" class="day-tab{" active" if is_first else ""}" '
            f'data-day-tab="{day_idx}">{escape(day["date"])}</button>'
        )
        rows = "".join(
            f"<tr><td>{escape(TARGET_LABELS.get(t, t))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='ensemble'>{escape(_format_recent_value(t, day['ensemble'].get(t), 'ensemble'))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='ml'>{escape(_format_recent_value(t, day['ml'].get(t), 'ml'))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='best'>{escape(_format_recent_value(t, day['best'].get(t), 'best'))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='actual'>{escape(_format_recent_value(t, day['actual'].get(t), 'actual'))}</td>"
            "</tr>"
            for t in RECENT_TARGETS
        )
        hidden_attr = "" if is_first else " hidden"
        day_cards.append(
            f"""<div class="recent-day-card" data-day-panel="{day_idx}"{hidden_attr}>
  <h3 data-day-label="{day_idx}">{escape(day["date"])}</h3>
  <div class="recent-day-table-scroll">
    <table>
      <colgroup><col class="col-metric"><col class="col-num"><col class="col-num"><col class="col-num"><col class="col-num"></colgroup>
      <thead><tr><th>Metric</th><th class="num">Weighted</th><th class="num">ML</th><th class="num">Best</th><th class="num">Actual</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""
        )

    return f"""<section class="recent-forecast-section" id="recent-forecast-section" style="display:none">
  <h2>Recent forecasts</h2>
  <div class="day-tabs" id="day-tabs">{"".join(tabs)}</div>
  <div class="recent-days" id="recent-days">{"".join(day_cards)}</div>
</section>"""


def _recent_forecast_script(recent_data: dict) -> str:
    return f"""
<script>
{_js_object_assignment("__RECENT_DATA", recent_data)}
window.__RECENT_UNITS = {json.dumps(RECENT_UNITS)};
window.__formatRecentValue = function (target, value, field) {{
  if (value === null || value === undefined) return "—";
  if (target === "did_rain") {{
    if (field === "actual") return value ? "Yes" : "No";
    return (Number(value) * 100).toFixed(0) + "%";
  }}
  var unit = window.__RECENT_UNITS[target] || "";
  return Number(value).toFixed(1) + unit;
}};
function updateRecentForecast(loc) {{
  var section = document.getElementById("recent-forecast-section");
  if (!section) return;
  var days = loc ? window.__RECENT_DATA[loc] : null;
  if (!loc || loc === "__ALL__" || !days) {{
    section.style.display = "none";
    return;
  }}
  section.style.display = "";
  // The section (and every day panel inside it) starts hidden, so its
  // scrollable table has zero measurable width until now - re-check the
  // fade right as it actually becomes visible, not just on load/resize.
  var visiblePanel = document.querySelector('#recent-days [data-day-panel]:not([hidden])');
  if (visiblePanel && typeof updateScrollFade === "function") {{
    updateScrollFade(visiblePanel.querySelector(".recent-day-table-scroll"));
  }}
  days.forEach(function (day, i) {{
    var label = document.querySelector('[data-day-label="' + i + '"]');
    if (label) label.textContent = day.date;
    var tab = document.querySelector('.day-tab[data-day-tab="' + i + '"]');
    if (tab) tab.textContent = day.date;
    document.querySelectorAll('[data-day="' + i + '"]').forEach(function (cell) {{
      var target = cell.dataset.target, field = cell.dataset.field;
      var value = day[field] ? day[field][target] : null;
      cell.textContent = window.__formatRecentValue(target, value, field);
    }});
  }});
}}
document.addEventListener("DOMContentLoaded", function () {{
  var tabsEl = document.getElementById("day-tabs");
  if (!tabsEl) return;
  tabsEl.addEventListener("click", function (e) {{
    var tab = e.target.closest(".day-tab");
    if (!tab) return;
    tabsEl.querySelectorAll(".day-tab").forEach(function (t) {{ t.classList.remove("active"); }});
    tab.classList.add("active");
    var idx = tab.dataset.dayTab;
    document.querySelectorAll('#recent-days [data-day-panel]').forEach(function (panel) {{
      panel.hidden = panel.dataset.dayPanel !== idx;
    }});
    var newlyVisible = document.querySelector('#recent-days [data-day-panel="' + idx + '"]');
    if (newlyVisible && typeof updateScrollFade === "function") {{
      updateScrollFade(newlyVisible.querySelector(".recent-day-table-scroll"));
    }}
  }});
}});
</script>
"""

# Reference palette (see dataviz skill, references/palette.md). Ensemble/ML -
# the two models this report is actually about - get the palette's blue/green
# categorical slots, painted solid and on top. Baselines are grey, dashed
# reference lines rather than competing series. Individual raw provider
# sources are deliberately NOT flat gray: each gets its own shade along a
# fixed orange->red gradient (the palette's own orange/red categorical steps
# as endpoints) so they stay visually distinguishable while still reading as
# one de-emphasized "family" behind the hero lines. The gradient position is
# assigned by each source's fixed position in FORECAST_SOURCES (identity, not
# by current error rank) so a source never changes shade because it started
# winning or losing - see the "recolor-on-filter" anti-pattern.
HERO_STYLE = {
    MODEL_ENSEMBLE: {
        "legend": "Weighted",
        "light": "#2a78d6",
        "dark": "#3987e5",
        "dash": "solid",
        "width": 2.5,
        "opacity": 1.0,
    },
    MODEL_ML: {
        "legend": "ML model",
        "light": "#008300",
        "dark": "#008300",
        "dash": "solid",
        "width": 2.5,
        "opacity": 1.0,
    },
    # Purple, per-mode-shifted (not a straight light/dark pair like the other
    # two heroes) - a manual OKLab check against this exact blue found that a
    # single mid-tone purple loses CVD separation from Weighted's blue almost
    # entirely under protanopia simulation on a dark surface (ΔE ~2, well
    # under the ≥8 target). Going deep/saturated in light mode and light/
    # pastel in dark mode instead uses lightness - a channel CVD doesn't
    # compromise - as the separating signal from blue, clearing the ≥15
    # normal-vision floor and the ≥8 CVD target against both blue and green
    # in both modes (the dataviz skill's own validator script wasn't runnable
    # in this environment - no node - so this was checked with an equivalent
    # manual OKLab + Brettel-matrix CVD simulation instead).
    MODEL_BEST: {
        "legend": "Best",
        "light": "#68166b",
        "dark": "#e0c5f5",
        "dash": "solid",
        "width": 2.5,
        "opacity": 1.0,
    },
}

BASELINE_STYLE = {
    BASELINE_PERSISTENCE: {
        "legend": "Baseline: persistence",
        "light": "#52514e",
        "dark": "#c3c2b7",
        "dash": "dash",
        "width": 1.75,
        "opacity": 0.95,
    },
    BASELINE_CLIMATOLOGY: {
        "legend": "Baseline: 30d trailing average",
        "light": "#52514e",
        "dark": "#c3c2b7",
        "dash": "dot",
        "width": 1.75,
        "opacity": 0.95,
    },
}

RAW_SOURCE_LEGEND = "Individual forecast sources"
RAW_SOURCE_BASE_STYLE = {"dash": "solid", "width": 1.5, "opacity": 0.8}
RAW_SOURCE_ORDER = list(FORECAST_SOURCES.keys())
# Yellow -> dark red: wide enough in both hue and lightness that adjacent
# sources stay visually distinct even after the 0.8 opacity blend that keeps
# them de-emphasized behind the ensemble/ML hero lines. The palette's own
# orange/red categorical steps span too narrow a range for this (both read as
# "orange-ish red", barely different once alpha-blended) - this ramp is
# deliberately wider than the validated 8-hue categorical set since it's
# ordinal-by-position (a fixed source order), not identity-by-hue-family.
RAW_SOURCE_GRADIENT_LIGHT = ("#d1a300", "#7a1220")
RAW_SOURCE_GRADIENT_DARK = ("#f2c14e", "#b23a4a")

# Paint order (traces added in this order so the models the report is about
# render on top of the de-emphasized context lines behind them).
_Z_ORDER = {MODEL_ENSEMBLE: 3, MODEL_ML: 4, MODEL_BEST: 5, BASELINE_PERSISTENCE: 1, BASELINE_CLIMATOLOGY: 1}


def _z_key(model: str) -> tuple[int, str]:
    return (_Z_ORDER.get(model, 0), model)


CHROME = {
    "light": {"font": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"font": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835"},
}


def _hex_lerp(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _raw_source_colors(models: set[str]) -> dict[str, dict[str, str]]:
    order = [m for m in RAW_SOURCE_ORDER if m in models]
    order += sorted(m for m in models if m not in RAW_SOURCE_ORDER)
    n = len(order)
    colors = {}
    for i, name in enumerate(order):
        t = i / (n - 1) if n > 1 else 0.0
        colors[name] = {
            "light": _hex_lerp(*RAW_SOURCE_GRADIENT_LIGHT, t),
            "dark": _hex_lerp(*RAW_SOURCE_GRADIENT_DARK, t),
        }
    return colors


def _style_for(model: str, raw_colors: dict[str, dict[str, str]]) -> dict:
    if model in HERO_STYLE:
        return HERO_STYLE[model]
    if model in BASELINE_STYLE:
        return BASELINE_STYLE[model]
    colors = raw_colors.get(model, {"light": RAW_SOURCE_GRADIENT_LIGHT[0], "dark": RAW_SOURCE_GRADIENT_DARK[0]})
    return {"legend": RAW_SOURCE_LEGEND, **RAW_SOURCE_BASE_STYLE, **colors}


def _display_name(model: str) -> str:
    if model.startswith("open_meteo_"):
        return "Open-Meteo: " + model.removeprefix("open_meteo_").replace("_", " ")
    if model in HERO_STYLE:
        return HERO_STYLE[model]["legend"]
    if model in BASELINE_STYLE:
        return BASELINE_STYLE[model]["legend"]
    return model.replace("_", " ")


def _bar_display_name(model: str) -> str:
    """Same as _display_name, except the leaderboard bars (and their table-view
    twin) spell out "Weighted model"/"Best model" rather than just "Weighted"/
    "Best" - the trend chart's legend/hover and the recent-forecasts table
    keep the shorter form.
    """
    if model == MODEL_ENSEMBLE:
        return "Weighted model"
    if model == MODEL_BEST:
        return "Best model"
    return _display_name(model)


def _axis_layout(fig: go.Figure) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": CHROME["light"]["font"], "size": 13},
        margin={"l": 8, "r": 48, "t": 8, "b": 36},
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=CHROME["light"]["grid"], linecolor=CHROME["light"]["axis"], zeroline=False, showgrid=True)
    fig.update_yaxes(gridcolor=CHROME["light"]["grid"], linecolor=CHROME["light"]["axis"], zeroline=False, showgrid=False)


def _board_height(n_bars: int) -> int:
    return 34 * max(n_bars, 1) + 60


def _leaderboard_figure(
    board_t: pd.DataFrame, raw_colors: dict[str, dict[str, str]]
) -> tuple[go.Figure, list[str], list[str], list[str], int]:
    board_t = board_t.sort_values("mae", ascending=True)
    model_order = board_t["model"].tolist()
    colors_light = [_style_for(m, raw_colors)["light"] for m in model_order]
    colors_dark = [_style_for(m, raw_colors)["dark"] for m in model_order]

    fig = go.Figure(
        go.Bar(
            x=board_t["mae"],
            y=[_bar_display_name(m) for m in model_order],
            orientation="h",
            marker_color=colors_light,
            text=[f"{v:.2f}" for v in board_t["mae"]],
            textposition="outside",
            cliponaxis=False,
            customdata=board_t["n"],
            hovertemplate="<b>%{x:.3f}</b> MAE  (n=%{customdata})<extra>%{y}</extra>",
        )
    )
    _axis_layout(fig)
    fig.update_xaxes(title_text="MAE", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    height = _board_height(len(board_t))
    fig.update_layout(height=height)
    return fig, colors_light, colors_dark, model_order, height


def _trend_figure(
    date_index: list[str],
    model_order: list[str],
    series_by_model: list[list],
    raw_colors: dict[str, dict[str, str]],
    height: int,
) -> tuple[go.Figure, list[str], list[str]]:
    """Build the trend figure with every trace's x fixed to the full `date_index`.

    Every trace (even one with real data on only a handful of those dates)
    gets the same-length x/y from the start. This matters beyond the initial
    render: the location dropdown's client-side Plotly.restyle only ever
    updates `y` (x never changes when switching locations), so if a trace's x
    were shorter than the other locations' y-arrays, the restyle would
    silently misalign the real values against the wrong dates instead of
    erroring - exactly the bug that made a partial-history line (like the
    ensemble/ML backtest, or a single-sample source) vanish or scramble.

    No on-chart legend: the leaderboard bar chart right beside this one
    already labels every model/source by name using the exact same color
    mapping, so a second legend here would just repeat it. Trace `name` is
    kept (not decorative) - it's what the unified hover tooltip uses to label
    each line's value.
    """
    fig = go.Figure()
    colors_light: list[str] = []
    colors_dark: list[str] = []
    x = pd.to_datetime(date_index)

    for model, y in zip(model_order, series_by_model):
        style = _style_for(model, raw_colors)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=style["legend"],
                opacity=style["opacity"],
                line={"color": style["light"], "width": style["width"], "dash": style["dash"]},
                hovertemplate=f"{escape(_display_name(model))}: %{{y:.3f}}<extra></extra>",
            )
        )
        colors_light.append(style["light"])
        colors_dark.append(style["dark"])

    _axis_layout(fig)
    fig.update_layout(hovermode="x unified", height=height, margin={"l": 48, "r": 16, "t": 16, "b": 36})
    fig.update_yaxes(title_text="rolling MAE", showgrid=True, rangemode="tozero")
    return fig, colors_light, colors_dark


def _board_series(subset_t: pd.DataFrame, model_order: list[str], recent_days: int) -> tuple[list, list]:
    """Per-model MAE/n for one location, reindexed onto a shared, fixed model order."""
    if subset_t.empty:
        return [None] * len(model_order), [0] * len(model_order)
    rows = {row["model"]: (row["mae"], int(row["n"])) for _, row in leaderboard(subset_t, recent_days=recent_days).iterrows()}
    mae = [round(rows[m][0], 4) if m in rows else None for m in model_order]
    n = [rows[m][1] if m in rows else 0 for m in model_order]
    return mae, n


def _trend_series(subset_t: pd.DataFrame, model_order: list[str], date_index: list[str], window: int) -> list[list]:
    """Per-model rolling MAE for one location, reindexed onto a shared date axis and model order."""
    series = {m: [None] * len(date_index) for m in model_order}
    if not subset_t.empty:
        date_pos = {d: i for i, d in enumerate(date_index)}
        for model, group in rolling_error_over_time(subset_t, window=window).groupby("model"):
            if model not in series:
                continue
            for _, r in group.iterrows():
                pos = date_pos.get(r["forecast_date"].date().isoformat())
                if pos is not None:
                    series[model][pos] = round(float(r["rolling_mae"]), 4)
    return [series[m] for m in model_order]


def _table_view(board_t: pd.DataFrame) -> str:
    rows = []
    for _, r in board_t.sort_values("mae").iterrows():
        rows.append(
            f"<tr><td>{escape(_bar_display_name(r['model']))}</td>"
            f"<td class='num'>{r['mae']:.3f}</td><td class='num'>{int(r['n'])}</td></tr>"
        )
    return (
        "<details class='table-view'><summary>Table view</summary>"
        "<table><thead><tr><th>Model</th><th class='num'>MAE</th><th class='num'>n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></details>"
    )


def _data_quality_for_js(data_quality: dict) -> dict:
    """data_quality.compute_data_quality returns raw entity identifiers
    (source names, "ensemble"/"ml"/"best") - presentation-agnostic, like the
    rest of that module. This is the one place that translates them to the
    same friendly labels (_display_name) everywhere else in the report uses,
    done once here rather than needing the embedded JS to re-derive them.
    """
    return {
        key: {
            "missing_pct": payload["missing_pct"],
            "null_pct": payload["null_pct"],
            "entities": [
                {"label": _display_name(e["name"]), "missing_pct": e["missing_pct"], "null_pct": e["null_pct"]}
                for e in payload["entities"]
            ],
        }
        for key, payload in data_quality.items()
    }


def _data_quality_html(data_quality_js: dict, window_days: int) -> str:
    """Renders the pooled ("__ALL__") view as the initial static HTML -
    updateDataQuality (see _data_quality_script) swaps in per-location
    numbers on location change, exactly like every other location-reactive
    element in this report.
    """
    pooled = data_quality_js["__ALL__"]
    rows = "".join(
        f"<tr><td>{escape(e['label'])}</td>"
        f"<td class='num'>{e['missing_pct']:.2f}%</td>"
        f"<td class='num'>{e['null_pct']:.2f}%</td></tr>"
        for e in pooled["entities"]
    )
    return f"""<section class="data-quality-section" id="data-quality-section">
  <h2>Data quality</h2>
  <p class="section-sub">Over the last {window_days}d, scoped to <span id="dq-location-label">all locations pooled</span>.</p>
  <div class="dq-summary">
    <div class="dq-stat">
      <span class="dq-stat-value" id="dq-missing-pct">{pooled["missing_pct"]:.2f}%</span>
      <span class="dq-stat-label">of expected forecasts are missing entirely</span>
    </div>
    <div class="dq-stat">
      <span class="dq-stat-value" id="dq-null-pct">{pooled["null_pct"]:.2f}%</span>
      <span class="dq-stat-label">of expected fields in existing forecasts are missing</span>
    </div>
  </div>
  <details class="table-view dq-breakdown">
    <summary>Breakdown by source/model</summary>
    <table>
      <thead><tr><th>Source</th><th class="num">Missing rows</th><th class="num">Unexpected nulls</th></tr></thead>
      <tbody id="dq-breakdown-body">{rows}</tbody>
    </table>
  </details>
</section>"""


def _data_quality_script(data_quality_js: dict) -> str:
    return f"""
<script>
{_js_object_assignment("__DATA_QUALITY", data_quality_js)}
function updateDataQuality(loc) {{
  var dq = window.__DATA_QUALITY[loc] || window.__DATA_QUALITY["__ALL__"];
  if (!dq) return;
  var missingEl = document.getElementById("dq-missing-pct");
  if (missingEl) missingEl.textContent = dq.missing_pct.toFixed(2) + "%";
  var nullEl = document.getElementById("dq-null-pct");
  if (nullEl) nullEl.textContent = dq.null_pct.toFixed(2) + "%";
  var label = document.getElementById("dq-location-label");
  if (label) label.textContent = (!loc || loc === "__ALL__") ? "all locations pooled" : loc;
  var tbody = document.getElementById("dq-breakdown-body");
  if (tbody) {{
    tbody.innerHTML = dq.entities.map(function (e) {{
      return "<tr><td>" + e.label + "</td><td class='num'>" + e.missing_pct.toFixed(2) + "%</td>"
        + "<td class='num'>" + e.null_pct.toFixed(2) + "%</td></tr>";
    }}).join("");
  }}
}}
</script>
"""


def _legend_gradient_css() -> str:
    """CSS custom properties for the legend's raw-source gradient swatch,
    generated from the same RAW_SOURCE_GRADIENT_LIGHT/DARK constants the
    charts themselves use - kept as a small standalone snippet (rather than
    folding into the static _PAGE_CSS string) so those two hex pairs stay a
    single source of truth instead of a second hardcoded copy drifting out
    of sync with the chart gradient.
    """
    return f"""
:root {{ --raw-grad-a: {RAW_SOURCE_GRADIENT_LIGHT[0]}; --raw-grad-b: {RAW_SOURCE_GRADIENT_LIGHT[1]}; }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{ --raw-grad-a: {RAW_SOURCE_GRADIENT_DARK[0]}; --raw-grad-b: {RAW_SOURCE_GRADIENT_DARK[1]}; }}
}}
:root[data-theme="dark"] {{ --raw-grad-a: {RAW_SOURCE_GRADIENT_DARK[0]}; --raw-grad-b: {RAW_SOURCE_GRADIENT_DARK[1]}; }}
"""


_PAGE_CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page-plane: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page-plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --border: rgba(255,255,255,0.10);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --page-plane: #0d0d0d;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --border: rgba(255,255,255,0.10);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body {
  margin: 0;
  background: var(--page-plane);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 24px 80px; }
@media (max-width: 640px) { .wrap { padding: 18px 14px 56px; } }

header.top { margin-bottom: 18px; }
header.top h1 { font-size: 24px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.01em; }
header.top p { margin: 0; color: var(--text-secondary); font-size: 13.5px; }
@media (max-width: 640px) { header.top h1 { font-size: 20px; } header.top p { font-size: 13px; } }
.chip-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.chip {
  font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 999px; padding: 4px 10px;
}

.section-sub { font-size: 12.5px; color: var(--text-secondary); margin: 0 0 14px; }

/* Sticky filter bar: Location/baselines/theme scope every chart on the page
   (Recent forecasts included, via the location dropdown), so it stays right
   under the header - reachable without hunting for it - rather than living
   inside the Historical accuracy section it doesn't exclusively belong to. */
.subnav {
  position: sticky;
  top: 0;
  z-index: 20;
  background: var(--surface-1);
  border-bottom: 1px solid var(--border);
  margin: 0 0 20px;
  padding: 10px 0;
}
/* The jump-nav is scoped entirely to Historical accuracy's cards, so it lives
   there instead - its own sticky bar, pinned just below the filter bar above
   (top: var(--subnav-height), kept in sync by updateToolbarHeight) so it's
   still reachable while scrolling through 13+ cards without following you
   around the Recent forecasts section it has nothing to do with. */
.section-nav {
  position: sticky;
  top: var(--subnav-height, 56px);
  z-index: 15;
  background: var(--page-plane);
  border-bottom: 1px solid var(--border);
  margin: 0 0 16px;
  padding: 10px 0;
}
.jump-nav {
  display: flex;
  gap: 6px;
  min-width: 0;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
  padding-bottom: 2px;
}
.jump-nav a {
  flex: 0 0 auto;
  font-size: 12.5px;
  color: var(--text-secondary);
  background: var(--page-plane);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 12px;
  white-space: nowrap;
  text-decoration: none;
}
.jump-nav a:hover, .jump-nav a.active { color: var(--text-primary); border-color: var(--text-secondary); }
/* Only fades when there's actually more to scroll to (see updateScrollFade) -
   an unconditional fade would clip the last pill's readability even when
   every pill already fits on screen. */
.jump-nav.has-overflow, .day-tabs.has-overflow {
  mask-image: linear-gradient(to right, black calc(100% - 28px), transparent 100%);
  -webkit-mask-image: linear-gradient(to right, black calc(100% - 28px), transparent 100%);
}

.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.theme-toggle {
  border: 1px solid var(--border); background: var(--page-plane); color: var(--text-secondary);
  border-radius: 8px; padding: 8px 12px; font-size: 12.5px; cursor: pointer; white-space: nowrap;
}
.theme-toggle:hover { color: var(--text-primary); }
/* The one control that changes what every chart on the page shows, so it
   gets its own labeled, bordered group (not just a bare unlabeled select
   blending in among the other pills) plus a visible accent - is-filtered
   is toggled by renderCharts whenever a specific location (not "All
   locations") is selected, so "am I looking at one place or everything
   pooled" is answerable at a glance instead of only from the header chip. */
.location-field {
  display: flex; align-items: center; gap: 8px;
  border: 1px solid var(--border); background: var(--page-plane);
  border-radius: 8px; padding: 8px 12px; cursor: pointer;
}
.location-field:focus-within { border-color: var(--text-secondary); }
.location-field.is-filtered { border-color: #2a78d6; }
.location-field-label { font-size: 12.5px; font-weight: 600; color: var(--text-secondary); white-space: nowrap; }
.location-field.is-filtered .location-field-label { color: #2a78d6; }
.location-select {
  /* A transparent background here doesn't just blend into .location-field
     for the closed control - the browser reuses the exact same
     background/color pair to paint the native options popup. Transparent
     falls back to the popup's own opaque default (often white) while
     `color` stays var(--text-primary) - white-on-white in dark mode, so the
     option list is unreadable. Giving it the same solid --page-plane as its
     wrapper keeps the closed look identical while making the popup itself
     theme-correct. */
  border: none; background: var(--page-plane); color: var(--text-primary);
  font-family: inherit; font-size: 12.5px; cursor: pointer; padding: 0;
  flex: 1 1 auto; min-width: 130px;
}
.location-select:focus { outline: none; }
.location-select option { background: var(--page-plane); color: var(--text-primary); }
.baseline-toggle {
  display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-secondary);
  border: 1px solid var(--border); background: var(--page-plane); border-radius: 8px;
  padding: 8px 12px; cursor: pointer; white-space: nowrap;
}
.baseline-toggle input { margin: 0; cursor: pointer; width: 16px; height: 16px; }

/* Below ~700px each control becomes its own full-width row with a
   44px-tall touch target, instead of a wrapped cluster of small pills. */
@media (max-width: 700px) {
  .controls > * { flex: 1 1 auto; min-height: 44px; }
}

.legend-key { display: flex; gap: 14px 18px; flex-wrap: wrap; align-items: center; margin: 0 0 22px; font-size: 12.5px; color: var(--text-secondary); }
.legend-key span.swatch { display: inline-block; width: 14px; height: 2px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }
/* A gradient patch (not a line) for "individual sources" - it stands for a
   whole family of distinguishable colors, not one line's color, so it reads
   differently on purpose from the solid hero/baseline swatches beside it. */
.legend-key span.swatch.swatch-gradient { height: 8px; background: linear-gradient(90deg, var(--raw-grad-a), var(--raw-grad-b)); }
#legend-baselines { display: flex; gap: 14px 18px; flex-wrap: wrap; }

.narrative-section {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 22px;
}
.narrative-date { font-size: 12px; font-weight: 650; color: var(--text-muted); margin: 0 0 6px; text-transform: uppercase; letter-spacing: 0.02em; }
.narrative-text { font-size: 14.5px; line-height: 1.5; color: var(--text-primary); margin: 0; }

.recent-forecast-section { margin-bottom: 28px; }
.recent-forecast-section h2 { font-size: 16px; font-weight: 650; margin: 0 0 12px; }

/* Day picker: one panel visible at a time instead of stacking all 5 days -
   a phone screen would otherwise need to scroll past ~65 table rows before
   reaching a single chart. */
.day-tabs {
  display: flex; gap: 6px; overflow-x: auto; -webkit-overflow-scrolling: touch;
  scrollbar-width: thin; margin-bottom: 10px; padding-bottom: 2px;
}
.day-tabs button {
  flex: 0 0 auto; font: inherit; font-size: 12.5px; color: var(--text-secondary);
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 999px;
  padding: 8px 14px; cursor: pointer; white-space: nowrap; min-height: 40px;
}
.day-tabs button.active { color: var(--text-primary); border-color: var(--text-secondary); font-weight: 600; }

.recent-day-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
}
.recent-day-card[hidden] { display: none; }
.recent-day-card h3 { font-size: 13.5px; font-weight: 600; margin: 0 0 8px; color: var(--text-secondary); }
/* On a narrow phone, 5 columns squeezed to fit the screen read as cramped
   no matter how small the font gets - min-width keeps every column at a
   comfortable size and lets the table scroll horizontally instead (the same
   overflow-x + fade-when-truncated pattern as .jump-nav/.day-tabs), rather
   than shrinking cell padding and text until values collide. */
.recent-day-table-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
}
.recent-day-table-scroll.has-overflow {
  mask-image: linear-gradient(to right, black calc(100% - 28px), transparent 100%);
  -webkit-mask-image: linear-gradient(to right, black calc(100% - 28px), transparent 100%);
}
/* table-layout: fixed + a shared colgroup keeps the Metric/Weighted/ML/Best/
   Actual columns at identical widths across every day panel, regardless of
   how long any one panel's values happen to be - without it each table sizes
   its columns independently and they drift out of alignment panel to panel. */
.recent-day-card table { width: 100%; min-width: 520px; border-collapse: collapse; font-size: 13px; table-layout: fixed; }
.recent-day-card col.col-metric { width: 28%; }
.recent-day-card col.col-num { width: 18%; }
.recent-day-card th, .recent-day-card td { padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: right; }
.recent-day-card th:first-child, .recent-day-card td:first-child { text-align: left; color: var(--text-secondary); }
.recent-day-card td.num, .recent-day-card th.num { font-variant-numeric: tabular-nums; }
@media (max-width: 480px) {
  .recent-day-card { padding: 12px 12px; }
  .recent-day-card table { font-size: 12.5px; }
}

.historical-accuracy-heading { font-size: 16px; font-weight: 650; margin: 0 0 12px; }

.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px 8px;
  margin-bottom: 18px;
  /* Anchor-jump lands below the sticky toolbar, not underneath it. A fixed
     px guess broke on mobile, where the toolbar stacks onto 2-3 rows and is
     2-3x taller than the single-row desktop layout - --toolbar-height is
     measured from the real, current toolbar (see the script at the bottom
     of the page) so this stays correct at any width or wrapped row count. */
  scroll-margin-top: var(--toolbar-height, 72px);
}
.card h2 { font-size: 15px; font-weight: 600; margin: 0 0 14px; }
.card .panels { display: grid; grid-template-columns: minmax(220px, 0.85fr) minmax(320px, 1.6fr); gap: 8px 20px; }
@media (max-width: 860px) { .card .panels { grid-template-columns: 1fr; } }
@media (max-width: 640px) { .card { padding: 14px 14px 6px; } .card h2 { font-size: 14px; } }

details.table-view { margin: 4px 0 14px; }
details.table-view summary { cursor: pointer; font-size: 12.5px; color: var(--text-secondary); padding: 4px 0; }
details.table-view table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12.5px; }
details.table-view th, details.table-view td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
details.table-view td.num, details.table-view th.num { text-align: right; font-variant-numeric: tabular-nums; }

footer { color: var(--text-muted); font-size: 12px; margin-top: 20px; }
.empty { color: var(--text-secondary); padding: 40px 0; text-align: center; }

.data-quality-section { margin: 28px 0 0; }
.data-quality-section h2 { font-size: 16px; font-weight: 650; margin: 0 0 4px; }
.dq-summary { display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0; }
.dq-stat {
  flex: 1 1 220px;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 18px;
}
.dq-stat-value { display: block; font-size: 24px; font-weight: 650; letter-spacing: -0.01em; }
.dq-stat-label { display: block; font-size: 12.5px; color: var(--text-secondary); margin-top: 2px; }
.dq-breakdown table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12.5px; }
.dq-breakdown th, .dq-breakdown td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
.dq-breakdown td.num, .dq-breakdown th.num { text-align: right; font-variant-numeric: tabular-nums; }
"""


def _theme_script(theme_traces: dict) -> str:
    return f"""
<script>
window.__THEME_TRACES = {json.dumps(theme_traces)};
function applyPlotlyTheme(dark) {{
  var chrome = dark
    ? {{font: "{CHROME['dark']['font']}", grid: "{CHROME['dark']['grid']}", axis: "{CHROME['dark']['axis']}"}}
    : {{font: "{CHROME['light']['font']}", grid: "{CHROME['light']['grid']}", axis: "{CHROME['light']['axis']}"}};
  document.querySelectorAll(".js-plotly-plot").forEach(function (div) {{
    var spec = window.__THEME_TRACES[div.id];
    // Line-chart trace count never changes (only visibility toggles), so its
    // color array is always the right length to restyle directly. The bar
    // chart's baseline entries can be filtered OUT of x/y/text/customdata by
    // the baseline toggle - restyling its (always-full-length) color array
    // here independently of that filter would misalign colors against bars
    // (see renderCharts). Bar coloring is handled there instead, called below.
    if (spec && spec.kind !== "bar") {{
      Plotly.restyle(div, {{"line.color": dark ? spec.dark : spec.light}});
    }}
    Plotly.relayout(div, {{
      "font.color": chrome.font,
      "xaxis.gridcolor": chrome.grid, "yaxis.gridcolor": chrome.grid,
      "xaxis.linecolor": chrome.axis, "yaxis.linecolor": chrome.axis,
    }});
  }});
  if (typeof renderCharts === "function") renderCharts();
}}
function currentTheme() {{
  var stored = localStorage.getItem("weather-report-theme");
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}}
function setTheme(theme) {{
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("weather-report-theme", theme);
  applyPlotlyTheme(theme === "dark");
  var btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = theme === "dark" ? "Light mode" : "Dark mode";
}}
document.addEventListener("DOMContentLoaded", function () {{
  setTheme(currentTheme());
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", function (e) {{
    if (!localStorage.getItem("weather-report-theme")) setTheme(e.matches ? "dark" : "light");
  }});
  var btn = document.getElementById("theme-toggle");
  if (btn) {{
    btn.addEventListener("click", function () {{
      setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
    }});
  }}
}});
</script>
"""


def _toolbar_height_script() -> str:
    """Measure the two stacked sticky bars - .subnav (filters, pinned at the
    very top) and .section-nav (the jump-nav, pinned just below it once
    scrolled that far) - into custom properties: --subnav-height positions
    .section-nav directly under .subnav with no gap or overlap, and
    --toolbar-height (their combined height) is what .card's scroll-margin-top
    needs so a jump-nav anchor click clears BOTH bars, not just one. A fixed
    px guess for either broke on mobile, where .subnav wraps onto 2-3 rows
    and is 2-3x taller than the single-row desktop layout. Re-measured on
    resize/orientation change, since crossing the 700px breakpoint changes
    the row count.
    """
    return """
<script>
function updateToolbarHeight() {
  var subnav = document.querySelector(".subnav");
  var sectionNav = document.querySelector(".section-nav");
  if (!subnav) return;
  var subnavHeight = subnav.getBoundingClientRect().height;
  var sectionNavHeight = sectionNav ? sectionNav.getBoundingClientRect().height : 0;
  document.documentElement.style.setProperty("--subnav-height", subnavHeight + "px");
  document.documentElement.style.setProperty("--toolbar-height", (subnavHeight + sectionNavHeight + 16) + "px");
}
// Only fades a horizontally-scrolling strip's trailing edge when it
// actually has more content past the visible area - an unconditional fade
// would clip the last pill/tab's readability even when everything already
// fits on screen (e.g. a wide desktop viewport with few jump-nav targets).
function updateScrollFade(el) {
  if (!el) return;
  el.classList.toggle("has-overflow", el.scrollWidth > el.clientWidth + 1);
}
function updateAllScrollFades() {
  updateScrollFade(document.querySelector(".jump-nav"));
  updateScrollFade(document.getElementById("day-tabs"));
  var visiblePanel = document.querySelector('#recent-days [data-day-panel]:not([hidden])');
  if (visiblePanel) updateScrollFade(visiblePanel.querySelector(".recent-day-table-scroll"));
}
document.addEventListener("DOMContentLoaded", function () {
  updateToolbarHeight();
  updateAllScrollFades();
});
window.addEventListener("resize", function () {
  updateToolbarHeight();
  updateAllScrollFades();
});
</script>
"""


def _controls_script(location_data: dict) -> str:
    return f"""
<script>
{_location_data_script(location_data)}
function renderCharts() {{
  var select = document.getElementById("location-select");
  var baselineToggle = document.getElementById("baseline-toggle");
  var loc = select ? select.value : "__ALL__";
  var showBaselines = baselineToggle ? baselineToggle.checked : true;

  Object.keys(window.__LOCATION_DATA).forEach(function (target) {{
    var spec = window.__LOCATION_DATA[target];
    var locData = spec.locations[loc] || spec.locations["__ALL__"];
    if (!locData) return;
    var mask = spec.baseline_mask_board;

    var boardId = "board-" + target;
    var boardDiv = document.getElementById(boardId);
    if (boardDiv) {{
      var mae = locData.mae, n = locData.n, categories = spec.board_categories;
      // Colors must be filtered by the exact same mask, in the exact same
      // restyle call, as x/y/text/customdata - a bar chart's baseline entries
      // live inside these arrays (not separate traces), so filtering the
      // values without also filtering marker.color shifts every color that
      // came after a removed baseline one slot out of place (this is what
      // made weatherbit/gfs_global sometimes render grey - the baseline's
      // color - when baselines were hidden).
      var themeSpec = window.__THEME_TRACES[boardId];
      var isDark = document.documentElement.getAttribute("data-theme") === "dark";
      var colors = themeSpec ? (isDark ? themeSpec.dark : themeSpec.light) : null;
      if (!showBaselines) {{
        mae = mae.filter(function (_, i) {{ return !mask[i]; }});
        n = n.filter(function (_, i) {{ return !mask[i]; }});
        categories = categories.filter(function (_, i) {{ return !mask[i]; }});
        if (colors) colors = colors.filter(function (_, i) {{ return !mask[i]; }});
      }}
      var update = {{
        x: [mae],
        y: [categories],
        text: [mae.map(function (v) {{ return v === null ? "" : v.toFixed(2); }})],
        customdata: [n],
      }};
      if (colors) update["marker.color"] = [colors];
      Plotly.restyle(boardDiv, update);
      Plotly.relayout(boardDiv, {{"xaxis.autorange": true}});
    }}

    var trendDiv = document.getElementById("trend-" + target);
    if (trendDiv) {{
      Plotly.restyle(trendDiv, {{y: locData.trend}});
      var baselineIdx = spec.baseline_trace_indices_trend;
      if (baselineIdx && baselineIdx.length) {{
        Plotly.restyle(trendDiv, {{visible: showBaselines}}, baselineIdx);
      }}
      Plotly.relayout(trendDiv, {{"yaxis.autorange": true}});
    }}
  }});

  var legendBaselines = document.getElementById("legend-baselines");
  if (legendBaselines) legendBaselines.style.display = showBaselines ? "" : "none";

  var chip = document.getElementById("location-chip");
  if (chip) chip.textContent = loc === "__ALL__" ? "all locations pooled" : "viewing " + loc;
  var locationField = select ? select.closest(".location-field") : null;
  if (locationField) locationField.classList.toggle("is-filtered", loc !== "__ALL__");
  localStorage.setItem("weather-report-location", loc);
  localStorage.setItem("weather-report-show-baselines", showBaselines ? "1" : "0");
  if (typeof updateNarrative === "function") updateNarrative(loc);
  if (typeof updateRecentForecast === "function") updateRecentForecast(loc);
  if (typeof updateDataQuality === "function") updateDataQuality(loc);
}}
document.addEventListener("DOMContentLoaded", function () {{
  var select = document.getElementById("location-select");
  var baselineToggle = document.getElementById("baseline-toggle");
  if (!select || !baselineToggle) return;

  var storedLoc = localStorage.getItem("weather-report-location");
  var hasOption = false;
  for (var i = 0; i < select.options.length; i++) {{
    if (select.options[i].value === storedLoc) {{ hasOption = true; break; }}
  }}
  if (hasOption) select.value = storedLoc;

  var storedBaselines = localStorage.getItem("weather-report-show-baselines");
  if (storedBaselines !== null) baselineToggle.checked = storedBaselines === "1";

  renderCharts();
  select.addEventListener("change", renderCharts);
  baselineToggle.addEventListener("change", renderCharts);
}});
</script>
"""


def build_html_report(
    long_df: pd.DataFrame,
    output_path: Path,
    db_path: Path,
    rolling_window: int = 7,
    recent_days: int = 30,
    history_days: int = 90,
    title: str = "Weather forecast accuracy",
) -> Path:
    """Render a self-contained interactive HTML report.

    Leads with today's ensemble/ML forecast plus the last few days' predictions
    alongside their actuals (queried directly from the DB via `db_path`, since
    today's forecast has no actual yet and would never appear in the scored
    `long_df`). Below that, every chart is scoped to the last `history_days`
    (default ~3 months). One card per target variable: a leaderboard (mean
    absolute error over the last `recent_days`) and a rolling `rolling_window`-
    day MAE-over-time line chart, plus a plain-HTML table twin of the
    leaderboard. The ensemble, ML, and adaptive "Best" models are drawn in
    their own colors (blue/green/purple) and painted on top; every raw
    provider source gets its own shade along a fixed orange->red gradient
    (de-emphasized via opacity/width) so they stay distinguishable without
    competing with the three models this report is actually about, and the
    two baselines are grey, dashed/dotted reference lines rather than
    competing series.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    narrative_data = _narrative_data(db_path, AUSTRALIAN_LOCATIONS)
    sample_narrative_location = next(iter(narrative_data), "")
    narrative_html = _narrative_html(narrative_data, sample_narrative_location)
    narrative_script = _narrative_script(narrative_data)

    recent_data = _recent_forecast_data(db_path, AUSTRALIAN_LOCATIONS)
    # Purely the pre-JS document structure - the section starts hidden and JS
    # fills it with whichever location the shared dropdown actually resolves
    # to (see _recent_forecast_html's docstring), so which location's data
    # gets baked into the static markup here doesn't matter.
    sample_recent_location = next(iter(recent_data), "")
    recent_forecast_html = _recent_forecast_html(recent_data, sample_recent_location)
    recent_forecast_script = _recent_forecast_script(recent_data)

    if not long_df.empty:
        cutoff = long_df["forecast_date"].max() - pd.Timedelta(days=history_days)
        long_df = long_df[long_df["forecast_date"] > cutoff]

    if long_df.empty:
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>{_PAGE_CSS}</style></head><body><div class="wrap">
<header class="top"><div><h1>{escape(title)}</h1><p>No scored predictions yet.</p></div></header>
{recent_forecast_html}
<div class="empty">Run the pipeline for a few days to accumulate forecasts and actuals, then regenerate this report.</div>
</div>
{recent_forecast_script}
</body></html>"""
        output_path.write_text(html, encoding="utf-8")
        return output_path

    recent_days = min(recent_days, history_days)
    board = leaderboard(long_df, recent_days=recent_days)
    trend = rolling_error_over_time(long_df, window=rolling_window)
    targets = [t for t in TARGET_ORDER if t in long_df["target"].unique()]
    raw_colors = _raw_source_colors(
        {m for m in long_df["model"].unique() if m not in HERO_STYLE and m not in BASELINE_STYLE}
    )

    # Every chart defaults to "All locations" (pooled). The dropdown swaps in
    # per-location data client-side via Plotly.restyle rather than building a
    # separate figure per location - so bar categories / line trace order (and
    # their colors) are frozen to the pooled ranking and reused unchanged
    # across every location (identity follows the entity, not a per-location
    # re-sort - see the "recolor/reorder-on-filter" anti-pattern).
    location_names = sorted(long_df["location_name"].unique())
    target_slices = {t: long_df[long_df["target"] == t] for t in targets}
    combo_slices = {key: df for key, df in long_df.groupby(["target", "location_name"])}

    # Same recent_days window as the leaderboard above (default 30) - reusing
    # that existing, already-configurable parameter rather than a second
    # hardcoded "last N days" concept for what's fundamentally the same idea.
    data_quality_js = _data_quality_for_js(compute_data_quality(db_path, location_names, window_days=recent_days))
    data_quality_html = _data_quality_html(data_quality_js, recent_days)
    data_quality_script = _data_quality_script(data_quality_js)

    theme_traces: dict[str, dict] = {}
    location_data: dict[str, dict] = {}
    cards = []
    rendered_targets: list[str] = []
    for target in targets:
        board_t = board[board["target"] == target]
        trend_t = trend[trend["target"] == target]
        if board_t.empty or trend_t.empty:
            continue
        rendered_targets.append(target)

        board_id = f"board-{target}"
        trend_id = f"trend-{target}"
        board_fig, board_light, board_dark, board_order, panel_height = _leaderboard_figure(board_t, raw_colors)

        trend_order = sorted(trend_t["model"].unique(), key=_z_key)
        date_index = sorted({d.date().isoformat() for d in trend_t["forecast_date"]})
        # Computed once and reused as both the initial ("All locations") render
        # and the "__ALL__" entry in the location cube below, so the two can
        # never drift apart into different numbers for the same view.
        all_trend_series = _trend_series(target_slices[target], trend_order, date_index, rolling_window)
        all_mae, all_n = _board_series(target_slices[target], board_order, recent_days)

        # Same height as the bar chart next to it, not a fixed 300px - the
        # bar chart's height already flexes with its row count (34px/bar), so
        # matching it here keeps both panels the same height in the layout
        # instead of leaving the (usually taller) bar chart's extra space unused.
        trend_fig, trend_light, trend_dark = _trend_figure(date_index, trend_order, all_trend_series, raw_colors, panel_height)
        theme_traces[board_id] = {"kind": "bar", "light": board_light, "dark": board_dark}
        theme_traces[trend_id] = {"kind": "line", "light": trend_light, "dark": trend_dark}

        locations_payload = {"__ALL__": {"mae": all_mae, "n": all_n, "trend": all_trend_series}}
        for loc in location_names:
            subset_t = combo_slices.get((target, loc), target_slices[target].iloc[0:0])
            mae, n = _board_series(subset_t, board_order, recent_days)
            trend_series = _trend_series(subset_t, trend_order, date_index, rolling_window)
            locations_payload[loc] = {"mae": mae, "n": n, "trend": trend_series}
        location_data[target] = {
            "locations": locations_payload,
            # What the "hide baselines" toggle needs: the bar chart's baseline
            # entries live inside one trace's arrays (not separate traces), so
            # hiding them means re-filtering x/y/text/customdata together (never
            # just x) - the same array-length-mismatch trap the location
            # dropdown hit with the trend chart. board_categories is the full
            # label list so JS can filter it by the same mask it applies to the
            # values, instead of leaving a label with no bar next to it.
            "board_categories": [_bar_display_name(m) for m in board_order],
            "baseline_mask_board": [m in BASELINE_STYLE for m in board_order],
            # The trend chart's baselines are separate traces, so hiding them
            # is just a per-trace visibility toggle by index.
            "baseline_trace_indices_trend": [i for i, m in enumerate(trend_order) if m in BASELINE_STYLE],
        }

        board_div = board_fig.to_html(full_html=False, include_plotlyjs=False, div_id=board_id, config={"displayModeBar": False, "responsive": True})
        trend_div = trend_fig.to_html(full_html=False, include_plotlyjs=False, div_id=trend_id, config={"displayModeBar": False, "responsive": True})

        cards.append(
            f"""<section class="card" id="card-{escape(target)}">
  <h2>{escape(TARGET_LABELS.get(target, target))}</h2>
  <div class="panels">
    <div>{board_div}{_table_view(board_t)}</div>
    <div>{trend_div}</div>
  </div>
</section>"""
        )

    n_locations = long_df["location_name"].nunique()
    date_min = long_df["forecast_date"].min().date()
    date_max = long_df["forecast_date"].max().date()
    generated = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="minutes")

    location_options = "".join(
        f'<option value="{escape(loc)}">{escape(loc)}</option>' for loc in location_names
    )

    jump_nav = "".join(
        f'<a href="#card-{escape(target)}">{escape(TARGET_LABELS.get(target, target))}</a>' for target in rendered_targets
    )

    # Hero (Weighted/ML) and raw-source colors are always visible so the
    # trend line chart - which has no on-chart legend of its own, see
    # _trend_figure's docstring - is self-explanatory without hovering.
    # Baselines are wrapped in their own #legend-baselines span and hidden
    # by default: dashed-vs-dotted isn't something a bar chart can show, so
    # they're the one pairing that needs a legend at all, and only while the
    # baseline toggle is actually on - see renderCharts.
    legend_key_hero = "".join(
        f"<span><span class='swatch' style='background:{style['light']}'></span>{escape(style['legend'])}</span>"
        for style in HERO_STYLE.values()
    )
    legend_key_raw = f"<span><span class='swatch swatch-gradient'></span>{escape(RAW_SOURCE_LEGEND)}</span>"
    legend_key_baselines = "".join(
        f"<span><span class='swatch' style='background:{style['light']}'></span>{escape(style['legend'])}</span>"
        for style in BASELINE_STYLE.values()
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_PAGE_CSS}{_legend_gradient_css()}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>{escape(title)}</h1>
    <div class="chip-row">
      <span class="chip" id="location-chip">{n_locations} location(s) pooled</span>
      <span class="chip">last {history_days}d &middot; {date_min} &rarr; {date_max}</span>
      <span class="chip">generated {generated}</span>
    </div>
  </header>
  <div class="subnav">
    <div class="controls">
      <label class="location-field">
        <span class="location-field-label">Location</span>
        <select id="location-select" class="location-select">
          <option value="__ALL__">All locations (pooled)</option>
          {location_options}
        </select>
      </label>
      <label class="baseline-toggle">
        <input type="checkbox" id="baseline-toggle">
        Show baselines
      </label>
      <button id="theme-toggle" class="theme-toggle" type="button">Dark mode</button>
    </div>
  </div>
  {narrative_html}
  {recent_forecast_html}
  <h2 class="historical-accuracy-heading">Historical accuracy</h2>
  <p class="section-sub">Last {recent_days}d leaderboard &middot; {rolling_window}d rolling MAE over time &middot; bar/line order stays fixed to the all-locations ranking when you switch locations.</p>
  <div class="section-nav">
    <nav class="jump-nav" aria-label="Jump to metric">{jump_nav}</nav>
  </div>
  <div class="legend-key" id="legend-key">{legend_key_hero}{legend_key_raw}<span id="legend-baselines" style="display:none">{legend_key_baselines}</span></div>
  {''.join(cards)}
  <footer>Lower is better for every metric shown. Rain is scored as % chance of rain against the binary rain/no-rain outcome (mean absolute error between the forecast probability and the 0/1 actual); every other metric is mean absolute error in its native unit.</footer>
  {data_quality_html}
</div>
{_theme_script(theme_traces)}
{_controls_script(location_data)}
{narrative_script}
{recent_forecast_script}
{data_quality_script}
{_toolbar_height_script()}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
