from __future__ import annotations

import json
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from weather_ensemble import db
from weather_ensemble.config import AUSTRALIAN_LOCATIONS, TARGETS, Location, local_today
from weather_ensemble.scoring import (
    BASELINE_CLIMATOLOGY,
    BASELINE_PERSISTENCE,
    MODEL_ENSEMBLE,
    MODEL_ML,
    leaderboard,
    rolling_error_over_time,
)
from weather_ensemble.sources import FORECAST_SOURCES

TARGET_LABELS = {
    "max_temp": "Max temperature",
    "min_temp": "Min temperature",
    "precipitation_sum": "Precipitation",
    "did_rain": "Did it rain (0/1 error)",
    "uv_index": "UV index",
    "wind_speed": "Wind speed",
    "wind_gusts": "Wind gusts",
}

RECENT_DAYS_COUNT = 5
RECENT_UNITS = {
    "max_temp": "°C",
    "min_temp": "°C",
    "precipitation_sum": "mm",
    "wind_speed": "km/h",
    "wind_gusts": "km/h",
}


def _format_recent_value(target: str, value) -> str:
    if value is None:
        return "—"
    if target == "did_rain":
        return "Yes" if value else "No"
    return f"{float(value):.1f}{RECENT_UNITS.get(target, '')}"


def _recent_forecast_data(db_path: Path, locations: list[Location]) -> dict[str, list[dict]]:
    """Per location, the last RECENT_DAYS_COUNT days (today first): whatever
    ensemble/ML prediction and actual exist for that date. A prediction that
    hasn't landed yet (today has no actual - it hasn't happened; a forecast
    that failed to generate) is left as None per-target rather than dropping
    the row or the whole day, so the table renders a blank cell instead.
    """
    data: dict[str, list[dict]] = {}
    with db.connect(db_path) as conn:
        for location in locations:
            today = local_today(location)
            days = []
            for i in range(RECENT_DAYS_COUNT):
                d = today - timedelta(days=i)
                d_iso = d.isoformat()
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
                actual = conn.execute(
                    "SELECT * FROM actuals WHERE location_name = ? AND actual_date = ? "
                    "ORDER BY collected_at DESC LIMIT 1",
                    (location.name, d_iso),
                ).fetchone()
                days.append(
                    {
                        "label": {0: "Today", 1: "Yesterday"}.get(i, d_iso),
                        "date": d_iso,
                        "ensemble": {t: (ens[t] if ens is not None else None) for t in TARGETS},
                        "ml": {t: (ml[t] if ml is not None else None) for t in TARGETS},
                        "actual": {t: (actual[t] if actual is not None else None) for t in TARGETS},
                    }
                )
            data[location.name] = days
    return data


def _recent_forecast_html(recent_data: dict[str, list[dict]], default_location: str, location_names: list[str]) -> str:
    options = "".join(
        f'<option value="{escape(loc)}"{" selected" if loc == default_location else ""}>{escape(loc)}</option>'
        for loc in location_names
    )
    default_days = recent_data.get(default_location, [])

    day_cards = []
    for day_idx, day in enumerate(default_days):
        rows = "".join(
            f"<tr><td>{escape(TARGET_LABELS.get(t, t))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='ensemble'>{escape(_format_recent_value(t, day['ensemble'].get(t)))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='ml'>{escape(_format_recent_value(t, day['ml'].get(t)))}</td>"
            f"<td class='num' data-day='{day_idx}' data-target='{t}' data-field='actual'>{escape(_format_recent_value(t, day['actual'].get(t)))}</td>"
            "</tr>"
            for t in TARGETS
        )
        header_text = day["date"] if day["label"] == day["date"] else f'{day["label"]} · {day["date"]}'
        day_cards.append(
            f"""<div class="recent-day-card">
  <h3 data-day-label="{day_idx}">{escape(header_text)}</h3>
  <table>
    <thead><tr><th>Metric</th><th class="num">Ensemble</th><th class="num">ML</th><th class="num">Actual</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""
        )

    return f"""<section class="recent-forecast-section">
  <div class="recent-forecast-header">
    <h2>Recent forecast vs actual</h2>
    <select id="recent-location-select" class="location-select">{options}</select>
  </div>
  <div class="recent-days">{"".join(day_cards)}</div>
</section>"""


def _recent_forecast_script(recent_data: dict) -> str:
    return f"""
<script>
window.__RECENT_DATA = {json.dumps(recent_data)};
window.__RECENT_UNITS = {json.dumps(RECENT_UNITS)};
window.__formatRecentValue = function (target, value) {{
  if (value === null || value === undefined) return "—";
  if (target === "did_rain") return value ? "Yes" : "No";
  var unit = window.__RECENT_UNITS[target] || "";
  return Number(value).toFixed(1) + unit;
}};
function renderRecentForecast() {{
  var select = document.getElementById("recent-location-select");
  if (!select) return;
  var days = window.__RECENT_DATA[select.value] || [];
  days.forEach(function (day, i) {{
    var label = document.querySelector('[data-day-label="' + i + '"]');
    if (label) label.textContent = day.label === day.date ? day.date : (day.label + " · " + day.date);
    document.querySelectorAll('[data-day="' + i + '"]').forEach(function (cell) {{
      var target = cell.dataset.target, field = cell.dataset.field;
      var value = day[field] ? day[field][target] : null;
      cell.textContent = window.__formatRecentValue(target, value);
    }});
  }});
  localStorage.setItem("weather-report-recent-location", select.value);
}}
document.addEventListener("DOMContentLoaded", function () {{
  var select = document.getElementById("recent-location-select");
  if (!select) return;
  var stored = localStorage.getItem("weather-report-recent-location");
  if (stored && window.__RECENT_DATA[stored]) select.value = stored;
  renderRecentForecast();
  select.addEventListener("change", renderRecentForecast);
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
        "legend": "Ensemble (weighted blend)",
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
_Z_ORDER = {MODEL_ENSEMBLE: 3, MODEL_ML: 4, BASELINE_PERSISTENCE: 1, BASELINE_CLIMATOLOGY: 1}


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


def _axis_layout(fig: go.Figure) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=CHROME["light"]["font"], size=13),
        margin=dict(l=8, r=48, t=8, b=36),
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
            y=[_display_name(m) for m in model_order],
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
    """
    fig = go.Figure()
    colors_light: list[str] = []
    colors_dark: list[str] = []
    legend_seen: set[str] = set()
    x = pd.to_datetime(date_index)

    for model, y in zip(model_order, series_by_model):
        style = _style_for(model, raw_colors)
        show = style["legend"] not in legend_seen
        legend_seen.add(style["legend"])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=style["legend"],
                legendgroup=style["legend"],
                showlegend=show,
                opacity=style["opacity"],
                line=dict(color=style["light"], width=style["width"], dash=style["dash"]),
                hovertemplate=f"{escape(_display_name(model))}: %{{y:.3f}}<extra></extra>",
            )
        )
        colors_light.append(style["light"])
        colors_dark.append(style["dark"])

    _axis_layout(fig)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)))
    fig.update_layout(hovermode="x unified", height=height, margin=dict(l=48, r=16, t=48, b=36))
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
            f"<tr><td>{escape(_display_name(r['model']))}</td>"
            f"<td class='num'>{r['mae']:.3f}</td><td class='num'>{int(r['n'])}</td></tr>"
        )
    return (
        "<details class='table-view'><summary>Table view</summary>"
        "<table><thead><tr><th>Model</th><th class='num'>MAE</th><th class='num'>n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></details>"
    )


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
body {
  margin: 0;
  background: var(--page-plane);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }

header.top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 28px; }
header.top h1 { font-size: 22px; font-weight: 650; margin: 0 0 6px; letter-spacing: -0.01em; }
header.top p { margin: 0; color: var(--text-secondary); font-size: 13.5px; }
.chip-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.chip {
  font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 999px; padding: 4px 10px;
}

.controls { display: flex; gap: 8px; align-items: center; }
.theme-toggle, .location-select {
  border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
  border-radius: 8px; padding: 7px 12px; font-size: 12.5px; cursor: pointer; white-space: nowrap;
}
.theme-toggle:hover { color: var(--text-primary); }
.location-select { font-family: inherit; }
.baseline-toggle {
  display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-secondary);
  border: 1px solid var(--border); background: var(--surface-1); border-radius: 8px;
  padding: 7px 12px; cursor: pointer; white-space: nowrap;
}
.baseline-toggle input { margin: 0; cursor: pointer; }

.legend-key { display: flex; gap: 18px; flex-wrap: wrap; margin: 0 0 22px; font-size: 12.5px; color: var(--text-secondary); }
.legend-key span.swatch { display: inline-block; width: 14px; height: 2px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }

.recent-forecast-section { margin-bottom: 28px; }
.recent-forecast-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.recent-forecast-header h2 { font-size: 16px; font-weight: 650; margin: 0; }
.recent-days { display: flex; flex-direction: column; gap: 10px; }
.recent-day-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
}
.recent-day-card h3 { font-size: 13.5px; font-weight: 600; margin: 0 0 8px; color: var(--text-secondary); }
.recent-day-card table { width: 100%; border-collapse: collapse; font-size: 13px; }
.recent-day-card th, .recent-day-card td { padding: 5px 10px; border-bottom: 1px solid var(--border); text-align: right; }
.recent-day-card th:first-child, .recent-day-card td:first-child { text-align: left; color: var(--text-secondary); }
.recent-day-card td.num, .recent-day-card th.num { font-variant-numeric: tabular-nums; }

.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px 8px;
  margin-bottom: 18px;
}
.card h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.card .sub { font-size: 12px; color: var(--text-muted); margin: 0 0 14px; }
.card .panels { display: grid; grid-template-columns: minmax(220px, 0.85fr) minmax(320px, 1.6fr); gap: 8px 20px; }
@media (max-width: 860px) { .card .panels { grid-template-columns: 1fr; } }

details.table-view { margin: 4px 0 14px; }
details.table-view summary { cursor: pointer; font-size: 12.5px; color: var(--text-secondary); }
details.table-view table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12.5px; }
details.table-view th, details.table-view td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }
details.table-view td.num, details.table-view th.num { text-align: right; font-variant-numeric: tabular-nums; }

footer { color: var(--text-muted); font-size: 12px; margin-top: 20px; }
.empty { color: var(--text-secondary); padding: 40px 0; text-align: center; }
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


def _controls_script(location_data: dict) -> str:
    return f"""
<script>
window.__LOCATION_DATA = {json.dumps(location_data)};
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

  var chip = document.getElementById("location-chip");
  if (chip) chip.textContent = loc === "__ALL__" ? "all locations pooled" : "viewing " + loc;
  localStorage.setItem("weather-report-location", loc);
  localStorage.setItem("weather-report-show-baselines", showBaselines ? "1" : "0");
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
    leaderboard. The ensemble and ML model are drawn in their own colors and
    painted on top; every raw provider source gets its own shade along a fixed
    orange->red gradient (de-emphasized via opacity/width) so they stay
    distinguishable without competing with the two models this report is
    actually about, and the two baselines are grey, dashed/dotted reference
    lines rather than competing series.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recent_data = _recent_forecast_data(db_path, AUSTRALIAN_LOCATIONS)
    recent_location_names = [loc.name for loc in AUSTRALIAN_LOCATIONS]
    default_recent_location = "Melbourne" if "Melbourne" in recent_data else next(iter(recent_data), "")
    recent_forecast_html = _recent_forecast_html(recent_data, default_recent_location, recent_location_names)
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
    targets = [t for t in TARGETS if t in long_df["target"].unique()]
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

    theme_traces: dict[str, dict] = {}
    location_data: dict[str, dict] = {}
    cards = []
    for target in targets:
        board_t = board[board["target"] == target]
        trend_t = trend[trend["target"] == target]
        if board_t.empty or trend_t.empty:
            continue

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
            "board_categories": [_display_name(m) for m in board_order],
            "baseline_mask_board": [m in BASELINE_STYLE for m in board_order],
            # The trend chart's baselines are separate traces, so hiding them
            # is just a per-trace visibility toggle by index.
            "baseline_trace_indices_trend": [i for i, m in enumerate(trend_order) if m in BASELINE_STYLE],
        }

        board_div = board_fig.to_html(full_html=False, include_plotlyjs=False, div_id=board_id, config={"displayModeBar": False, "responsive": True})
        trend_div = trend_fig.to_html(full_html=False, include_plotlyjs=False, div_id=trend_id, config={"displayModeBar": False, "responsive": True})

        cards.append(
            f"""<section class="card">
  <h2>{escape(TARGET_LABELS.get(target, target))}</h2>
  <p class="sub">Last {recent_days}d leaderboard &middot; {rolling_window}d rolling MAE over time &middot; bar/line order is fixed to the all-locations ranking</p>
  <div class="panels">
    <div>{board_div}{_table_view(board_t)}</div>
    <div>{trend_div}</div>
  </div>
</section>"""
        )

    n_locations = long_df["location_name"].nunique()
    date_min = long_df["forecast_date"].min().date()
    date_max = long_df["forecast_date"].max().date()
    generated = datetime.now().isoformat(timespec="minutes")

    location_options = "".join(
        f'<option value="{escape(loc)}">{escape(loc)}</option>' for loc in location_names
    )

    legend_key_entries = [
        *HERO_STYLE.values(),
        *BASELINE_STYLE.values(),
        {"legend": RAW_SOURCE_LEGEND, "light": _hex_lerp(*RAW_SOURCE_GRADIENT_LIGHT, 0.5)},
    ]
    legend_key = "".join(
        f"<span><span class='swatch' style='background:{style['light']}'></span>{escape(style['legend'])}</span>"
        for style in legend_key_entries
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_PAGE_CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>{escape(title)}</h1>
      <p>Ensemble and ML predictions scored against observed weather, alongside every individual forecast source and two naive baselines.</p>
      <div class="chip-row">
        <span class="chip" id="location-chip">{n_locations} location(s) pooled</span>
        <span class="chip">last {history_days}d &middot; {date_min} &rarr; {date_max}</span>
        <span class="chip">generated {generated}</span>
      </div>
    </div>
    <div class="controls">
      <label class="baseline-toggle">
        <input type="checkbox" id="baseline-toggle" checked>
        Show baselines
      </label>
      <select id="location-select" class="location-select">
        <option value="__ALL__">All locations (pooled)</option>
        {location_options}
      </select>
      <button id="theme-toggle" class="theme-toggle" type="button">Dark mode</button>
    </div>
  </header>
  {recent_forecast_html}
  <div class="legend-key">{legend_key}</div>
  {''.join(cards)}
  <footer>Lower is better for every metric shown, including did_rain (mean absolute error against the 0/1 outcome). Bar/line order stays fixed to the all-locations ranking when you switch locations, so series don't jump around.</footer>
</div>
{_theme_script(theme_traces)}
{_controls_script(location_data)}
{recent_forecast_script}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
