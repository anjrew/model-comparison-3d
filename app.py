import html
import json
import math
import os
import re

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import models_api as api

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".app_state.json")
PROFILES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".app_profiles.json")


def _load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_state():
    data = {k: st.session_state[k] for k in STATE_KEYS if k in st.session_state}
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

webgl_check = components.declare_component("webgl_check", path="frontend")

AXES = {
    "cost": "Cost ($/1M input tokens)",
    "speed": "Speed",
    "intelligence": "Intelligence",
    "context": "Context (tokens)",
}

STATE_KEYS = [
    "chart_mode", "ball_size", "size_scale", "log_exp", "ball_max", "color_mode", "show_field", "log_x",
    "field_surfaces", "field_res", "field_opacity", "field_curve", "field_density",
    "x_axis", "y_axis", "z_axis",
    "w_cost", "w_speed", "w_intel",
    "search", "show_all", "prov_search", "sel_providers",
    "reasoning_only", "open_weights", "sel_sources", "min_context", "max_context",
    "sel_continents", "sel_countries",
    "hl_search", "hl_names", "table_search",
    "effort_filter", "effort_ladder_only", "show_effort_variants", "effort_color_mode",
    "effort_models", "effort_metric", "effort_color_by",
] + [f"rng_{m}_{b}" for m in AXES for b in ("min", "max")]

EFFORT_ORDER = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_RANK = {e: i for i, e in enumerate(EFFORT_ORDER)}
EFFORT_COST_MULT = {
    "off": 1.0, "minimal": 1.2, "low": 1.5, "medium": 2.2,
    "high": 3.5, "xhigh": 5.5, "max": 8.0, "reasoning": 2.2,
}
EFFORT_COLOR_MAP = {
    "off": "#64748B", "minimal": "#22C55E", "low": "#84CC16", "medium": "#EAB308",
    "high": "#F97316", "xhigh": "#EF4444", "max": "#A21CAF", "reasoning": "#0EA5E9",
}
_EFFORT_CAPTION = (
    "Effort = how much a reasoning model 'thinks' before answering (off → max). "
    "Higher effort is usually smarter but slower and more expensive. "
    "'≈' and amber mark an **estimated** cost; green marks an AA-**measured** value."
)


def _load_profiles():
    try:
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_profiles(profiles):
    try:
        with open(PROFILES_FILE, "w") as f:
            json.dump(profiles, f, indent=2)
    except Exception:
        pass


def _capture_state():
    return {k: st.session_state[k] for k in STATE_KEYS if k in st.session_state}


def _apply_state(data):
    st.session_state["_pending_state"] = data


def _reset_state():
    st.session_state["_pending_reset"] = True

DEFAULT_AXES = ["cost", "speed", "intelligence"]

VALUE_SCALE = [
    (0.00, "rgb(110,0,0)"),
    (0.30, "rgb(200,30,40)"),
    (0.55, "rgb(255,150,30)"),
    (0.80, "rgb(250,230,0)"),
    (1.00, "rgb(70,200,80)"),
]

_BALL_MIN = 5.0

_SRC_STYLE = {
    "Live (AA)": "background-color:#d8f3dc;color:#14532d",
    "User-defined": "background-color:#e7e5e4;color:#44403c",
    "Estimated (heuristic)": "background-color:#fef3c7;color:#92400e",
    "Estimated (effort)": "background-color:#ffedd5;color:#9a3412",
}


def _score_source(df):
    has_aa = df.get("aa_intelligence_index")
    live_aa = (df["scores_live"] & has_aa.notna()) if has_aa is not None else pd.Series(False, index=df.index)
    return pd.Series(
        np.where(df["scores_live"] & ~live_aa, "User-defined",
                 np.where(live_aa, "Live (AA)", "Estimated (heuristic)")),
        index=df.index,
    )


def _cell_style(v):
    return _SRC_STYLE.get(v, "")


_LIVE_OUTLINE_COLOR = "#22d3ee"


def _apply_live_outline(fig, color=_LIVE_OUTLINE_COLOR, width=2.5):
    for tr in fig.data:
        if tr.type not in ("scatter", "scatter3d") or tr.customdata is None:
            continue
        flags = [bool(row[0]) for row in tr.customdata]
        if not any(flags):
            continue
        tr.marker.line.color = [color if f else "rgba(0,0,0,0)" for f in flags]
        tr.marker.line.width = width


def _cost_transform(c, log_cost):
    return math.log10(max(c, 1e-6)) if log_cost else c


def _score(cost_v, intel_v, speed_v, cb, w_cost, w_speed, w_intel, log_cost, curve=1.0):
    c_lo, c_hi, s_lo, s_hi, i_lo, i_hi = cb
    cspan = (c_hi - c_lo) or 1
    cheap = 1 - (_cost_transform(cost_v, log_cost) - c_lo) / cspan
    sspan = (s_hi - s_lo) or 1
    sn = (speed_v - s_lo) / sspan
    ispan = (i_hi - i_lo) or 1
    inn = (intel_v - i_lo) / ispan
    wsum = max(w_cost + w_speed + w_intel, 1e-9)
    if curve <= 1.001:
        return (w_cost * cheap + w_speed * sn + w_intel * inn) / wsum
    cc = min(max(cheap, 0.0), 1.0)
    ss = min(max(sn, 0.0), 1.0)
    ii = min(max(inn, 0.0), 1.0)
    d = (w_cost * (1.0 - cc) ** curve
         + w_speed * (1.0 - ss) ** curve
         + w_intel * (1.0 - ii) ** curve) ** (1.0 / curve)
    return 1.0 - d / (wsum ** (1.0 / curve))


def _effort_ordered(levels):
    return sorted({e for e in (levels or []) if e in EFFORT_RANK}, key=EFFORT_RANK.get)


def _effort_cost_est(row, level, ref):
    base = float(row.get("cost") or 0.0)
    out = row.get("cost_out")
    try:
        out = float(out)
        if out != out:
            out = base
    except (TypeError, ValueError):
        out = base
    mult = EFFORT_COST_MULT.get(level, 1.5)
    ref_mult = EFFORT_COST_MULT.get(ref, 1.0) or 1.0
    return base + max(out, 0.0) * (mult / ref_mult)


def _measured_levels(row):
    ladder = [v for v in (row.get("effort_ladder") or []) if v.get("effort") in EFFORT_RANK]
    best = {}
    for v in ladder:
        e = v["effort"]
        if e not in best or (v.get("intelligence") or -1) > (best[e].get("intelligence") or -1):
            best[e] = v
    return best


def _has_effort_ladder(row):
    return len(_measured_levels(row)) >= 2


def _effort_ladder_rows(row, w_cost, w_speed, w_intel, log_x, curve=1.0):
    measured = _measured_levels(row)
    levels = _effort_ordered(list(measured.keys()))
    if not levels:
        return [], None
    ref = levels[0]
    all_measured = all(measured[e].get("cost_per_task") is not None for e in levels)
    rows = []
    for e in levels:
        v = measured[e]
        intel = v.get("intelligence")
        speed = v.get("speed")
        if speed is None:
            speed = row.get("speed")
        aa_cost = v.get("cost_per_task")
        if all_measured:
            cost, cost_src = aa_cost, "Live (AA)"
        else:
            cost, cost_src = _effort_cost_est(row, e, ref), "Estimated (effort)"
        rows.append({"effort": e, "intelligence": intel, "speed": speed,
                     "cost": cost, "cost_source": cost_src, "aa_cost": aa_cost})
    cost_vals = [_cost_transform(r["cost"], log_x) for r in rows]
    c_lo, c_hi = min(cost_vals), max(cost_vals)
    if c_hi <= c_lo:
        c_hi = c_lo + 1e-6
    intel_vals = [r["intelligence"] for r in rows if r["intelligence"] is not None]
    speed_vals = [r["speed"] for r in rows if r["speed"] is not None]
    i_lo, i_hi = (min(intel_vals), max(intel_vals)) if intel_vals else (0.0, 1.0)
    s_lo, s_hi = (min(speed_vals), max(speed_vals)) if speed_vals else (0.0, 1.0)
    if i_hi <= i_lo:
        i_hi = i_lo + 1e-6
    if s_hi <= s_lo:
        s_hi = s_lo + 1e-6
    cb = (c_lo, c_hi, s_lo, s_hi, i_lo, i_hi)
    for r in rows:
        r["value"] = _score(r["cost"], r["intelligence"] or 0.0, r["speed"] or 0.0,
                            cb, w_cost, w_speed, w_intel, log_x, curve=curve)
    best = max(rows, key=lambda r: r["value"])["effort"]
    return rows, best


def _best_effort_meta(row, w_cost, w_speed, w_intel, log_x, curve=1.0):
    rows, best = _effort_ladder_rows(row, w_cost, w_speed, w_intel, log_x, curve)
    if not best:
        return None, None, ""
    for r in rows:
        if r["effort"] == best:
            return best, r["cost"], r["cost_source"]
    return best, None, ""


def _expand_effort(visible):
    rows = []
    for r in visible.to_dict("records"):
        measured = _measured_levels(r)
        levels = _effort_ordered(list(measured.keys()))
        if len(levels) < 2:
            d = dict(r)
            ae = r.get("aa_effort")
            d["effort"] = ae if ae in EFFORT_RANK else None
            rows.append(d)
            continue
        for e in levels:
            v = measured[e]
            d = dict(r)
            d["effort"] = e
            if v.get("intelligence") is not None:
                d["intelligence"] = v["intelligence"]
            if v.get("speed") is not None:
                d["speed"] = v["speed"]
            rows.append(d)
    return pd.DataFrame(rows).reset_index(drop=True)


def _effort_cost_str(cost, source):
    if cost is None or (isinstance(cost, float) and cost != cost):
        return "n/a"
    pre = "≈" if source == "Estimated (effort)" else ""
    return f"{pre}${float(cost):.3f}"


def _hnum(v, fmt="{:.1f}"):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)


def _ball_sizes(raw, size_scale, ball_max, log_exp):
    mapped = pd.to_numeric(raw, errors="coerce")
    s_lo, s_hi = _BALL_MIN, float(ball_max)
    good = mapped.dropna()
    good = good[good > 0] if size_scale in ("Log", "Logᵖ") else good
    if not len(good):
        return pd.Series([8] * len(mapped), index=mapped.index)
    vmax = float(good.max())
    if size_scale == "Log":
        fmax = math.log10(max(vmax, 1.0))

        def ratio(v):
            return math.log10(max(float(v), 1.0)) / fmax
    elif size_scale == "Logᵖ":
        p = float(log_exp)
        fmax = math.log10(max(vmax, 1.0))

        def ratio(v):
            return (math.log10(max(float(v), 1.0)) / fmax) ** p
    else:
        def ratio(v):
            return float(v) / vmax if vmax > 0 else 0.0

    return mapped.map(
        lambda v: min(s_hi, max(s_lo, s_hi * ratio(v))) if v == v and v > 0 else s_lo
    )


def _color_chips(items):
    spans = []
    for label, color in items:
        spans.append(
            "<span style='display:inline-flex;align-items:center;margin:0 14px 4px 0'>"
            f"<span style='width:12px;height:12px;border-radius:3px;background:{color};"
            "display:inline-block;margin-right:5px;border:1px solid rgba(0,0,0,.25)'></span>"
            f"<span>{html.escape(str(label))}</span></span>"
        )
    return " ".join(spans)


def build_effort_figure(per_model, pts, sizes, cfg, color_by, log_x):
    """3D (or 2D fallback) effort ladders: nodes per level, edges between levels."""
    x_axis, y_axis, z_axis = cfg["x_axis"], cfg["y_axis"], cfg["z_axis"]
    is_3d = cfg["chart_type"] == "3D (WebGL)"

    def axis_val(r, row, metric):
        if metric == "cost":
            return r["cost"]
        if metric == "speed":
            return r["speed"]
        if metric == "intelligence":
            return r["intelligence"]
        return row["context"]

    fig = go.Figure()
    for name, row, rows, best in per_model:
        pr = sorted(rows, key=lambda r: EFFORT_RANK.get(r["effort"], 99))
        idx = [pts.index[(pts["name"] == name) & (pts["effort"] == r["effort"])][0] for r in pr]
        psize = [float(sizes.loc[i]) for i in idx]
        pc = api.provider_color(row["provider"])
        mcolors = ([EFFORT_COLOR_MAP.get(r["effort"], "#94A3B8") for r in pr]
                   if color_by == "Effort level" else pc)
        cdata = [[r["effort"], _hnum(r["intelligence"], "{:.1f}"), _hnum(r["speed"], "{:.1f}"),
                  _effort_cost_str(r["cost"], r["cost_source"]), r["cost_source"],
                  "n/a" if r["aa_cost"] is None else f"{r['aa_cost']:.3f}",
                  round(r["value"], 3)] for r in pr]
        hover = (f"<b>{name}</b><br>Effort: %{{customdata[0]}}"
                 "<br>Intelligence: %{customdata[1]}<br>Speed: %{customdata[2]}"
                 "<br>Cost: %{customdata[3]} (%{customdata[4]})"
                 "<br>AA $/task: %{customdata[5]}<br>Value: %{customdata[6]}<extra></extra>")
        marker = dict(size=psize, color=mcolors, sizemode="diameter", sizeref=1, sizemin=1,
                      line=dict(width=1, color="#111827"))
        node_kwargs = dict(
            x=[axis_val(r, row, x_axis) for r in pr],
            y=[axis_val(r, row, y_axis) for r in pr],
            mode="markers", name=name, legendgroup=name,
            marker=marker, customdata=cdata, hovertemplate=hover,
        )
        if is_3d:
            node_kwargs["z"] = [axis_val(r, row, z_axis) for r in pr]
            fig.add_trace(go.Scatter3d(**node_kwargs))
        else:
            fig.add_trace(go.Scatter(**node_kwargs))
        for a, b in zip(pr, pr[1:]):
            ecolor = (EFFORT_COLOR_MAP.get(b["effort"], "#94A3B8")
                      if color_by == "Effort level" else pc)
            d_i = (b["intelligence"] or 0) - (a["intelligence"] or 0)
            d_s = (b["speed"] or 0) - (a["speed"] or 0)
            seg = (f"<b>{name}</b><br>{a['effort']} → {b['effort']}"
                   f"<br>Δ Intelligence: {d_i:+.1f}<br>Δ Speed: {d_s:+.1f}"
                   f"<br>Cost: {_effort_cost_str(a['cost'], a['cost_source'])} → "
                   f"{_effort_cost_str(b['cost'], b['cost_source'])}")
            edge_kwargs = dict(
                x=[axis_val(a, row, x_axis), axis_val(b, row, x_axis)],
                y=[axis_val(a, row, y_axis), axis_val(b, row, y_axis)],
                mode="lines", showlegend=False, legendgroup=name,
                line=dict(width=6 if is_3d else 4, color=ecolor),
                hovertemplate=seg + "<extra></extra>",
            )
            if is_3d:
                edge_kwargs["z"] = [axis_val(a, row, z_axis), axis_val(b, row, z_axis)]
                fig.add_trace(go.Scatter3d(**edge_kwargs))
            else:
                fig.add_trace(go.Scatter(**edge_kwargs))
    use_log = log_x and x_axis == "cost" and bool((pts["cost"] > 0).all())
    if is_3d:
        fig.update_layout(height=680, margin=dict(l=0, r=0, t=30, b=0),
                          scene=dict(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis],
                                     zaxis_title=AXES[z_axis],
                                     xaxis=dict(type="log") if use_log else {}))
    else:
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis],
                          legend_title="Model")
        if use_log:
            fig.update_xaxes(type="log")
    return fig


def render_effort_panel(df, hl_names, w_cost, w_speed, w_intel, log_x, curve, cfg):
    st.subheader("🧠 Effort tradeoff")
    st.caption(_EFFORT_CAPTION)
    cand = df[df.apply(_has_effort_ladder, axis=1)].copy()
    if cand.empty:
        st.info("No models with a **measured** effort ladder in the current filters. "
                "Add an Artificial Analysis key (sidebar) or clear filters to see effort recommendations. "
                "Models that only list *supported* effort levels appear in the main chart's hover/table.")
        return
    names = sorted(cand["name"].unique())
    hl_set = set(hl_names or [])
    defaults = [n for n in names if n in hl_set][:6]
    if not defaults:
        defaults = cand.sort_values("value", ascending=False)["name"].drop_duplicates().head(5).tolist()
    sel = st.multiselect("Models to compare across effort levels", names,
                         default=defaults, max_selections=8, key="effort_models",
                         help="Only models where Artificial Analysis measured ≥2 effort levels.")
    if not sel:
        return
    ctl1, ctl2 = st.columns(2)
    with ctl1:
        color_by = st.radio("Color by", ["Effort level", "Provider"], horizontal=True,
                            key="effort_color_by")
    with ctl2:
        st.caption(f"Axes follow the sidebar: {AXES[cfg['x_axis']]} / {AXES[cfg['y_axis']]} / "
                   f"{AXES[cfg['z_axis']]} · ball size: {cfg['ball_size']}")

    per_model = []
    points = []
    for name in sel:
        subset = cand[cand["name"] == name].copy()
        subset["_n"] = subset.apply(lambda r: len(_measured_levels(r)), axis=1)
        pool = subset[subset["_n"] == subset["_n"].max()]
        priced = pool[pool["cost"] > 0]
        if not priced.empty:
            pool = priced
        row = pool.sort_values("value", ascending=False).iloc[0]
        rows, best = _effort_ladder_rows(row, w_cost, w_speed, w_intel, log_x, curve)
        if not rows:
            continue
        per_model.append((name, row, rows, best))
        for r in rows:
            points.append({
                "name": name, "provider": row["provider"], "effort": r["effort"],
                "cost": r["cost"], "speed": r["speed"], "intelligence": r["intelligence"],
                "context": row["context"], "params": row["params"],
                "cost_source": r["cost_source"], "aa_cost": r["aa_cost"], "value": r["value"],
            })
    if not points:
        return
    pts = pd.DataFrame(points)

    raw = None
    if cfg["ball_size"] == "Parameters":
        raw = pts["params"]
    elif cfg["ball_size"] == "Context":
        raw = pts["context"]
    elif cfg["ball_size"] == "Z-axis value":
        raw = pts[cfg["z_axis"]]
    if raw is None:
        sizes = pd.Series([12] * len(pts), index=pts.index)
    else:
        sizes = _ball_sizes(raw, cfg["size_scale"], cfg["ball_max"], cfg["log_exp"])

    fig = build_effort_figure(per_model, pts, sizes, cfg, color_by, log_x)
    st.plotly_chart(fig, use_container_width=True)
    levels_present = [e for e in EFFORT_ORDER if e in set(pts["effort"])]
    if color_by == "Effort level":
        st.markdown("**Color key — effort level**", unsafe_allow_html=True)
        st.markdown(_color_chips([(e, EFFORT_COLOR_MAP.get(e, "#94A3B8"))
                                  for e in levels_present]), unsafe_allow_html=True)
    else:
        provs = sorted(pts["provider"].unique())
        st.markdown("**Color key — provider**", unsafe_allow_html=True)
        st.markdown(_color_chips([(p, api.provider_color(p)) for p in provs]),
                    unsafe_allow_html=True)
    st.caption("Edges connect each model's effort levels (off → max), colored by the level reached; "
               "hover an edge to see that step's Δintelligence and Δspeed. "
               "Hover a ball for its measured/estimated cost and the weighted value. "
               "≈ and 'Estimated (effort)' mark approximated cost.")

    with st.expander("Per-model effort tables", expanded=False):
        for name, row, rows, best in per_model:
            st.markdown(f"**{name}** · {row['provider']} · supported: "
                        f"{', '.join(_effort_ordered(row['effort_levels'])) or 'n/a'} "
                        f"({row['effort_style']}) · ★ best: **{best}**")
            trows = []
            for r in rows:
                trows.append({
                    "Effort": ("★ " if r["effort"] == best else "") + r["effort"],
                    "Intel": r["intelligence"],
                    "Speed": r["speed"],
                    "Cost": _effort_cost_str(r["cost"], r["cost_source"]),
                    "Cost source": r["cost_source"],
                    "AA $/task": "n/a" if r["aa_cost"] is None else f"{r['aa_cost']:.3f}",
                    "Value": round(r["value"], 3),
                })
            st.dataframe(pd.DataFrame(trows), hide_index=True, width="stretch")
    st.caption("★ = best with the current cost/speed/intelligence weights. "
               "≈ and 'Estimated (effort)' = approximated from output-token scaling, not measured.")


def _metric_axis_range(visible, metric):
    lo = st.session_state.get(f"rng_{metric}_min")
    hi = st.session_state.get(f"rng_{metric}_max")
    if metric == "context":
        dlo, dhi = float(visible["context"].min()), float(visible["context"].max())
        lo = float(lo) * 1000 if lo is not None else dlo
        hi = float(hi) * 1000 if hi is not None else dhi
        if hi <= lo:
            hi = lo + 1
        return lo, hi
    if metric == "cost":
        dlo, dhi = float(visible["cost"].min()), float(visible["cost"].max())
        lo = max(lo if lo is not None else dlo, 1e-6)
        hi = max(hi if hi is not None else dhi, lo)
        return lo, hi
    dlo, dhi = float(visible[metric].min()), float(visible[metric].max())
    lo = float(lo) if lo is not None else dlo
    hi = float(hi) if hi is not None else dhi
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _compute_bounds(visible, log_cost):
    c_lo, c_hi = 0.01, 10.0
    lo_ov = st.session_state.get("rng_cost_min")
    hi_ov = st.session_state.get("rng_cost_max")
    if lo_ov is not None:
        c_lo = float(lo_ov)
    if hi_ov is not None:
        c_hi = float(hi_ov)
    if c_hi <= c_lo:
        c_hi = c_lo + 1e-6
    s_lo, s_hi = _axis_render_range(visible, "speed")
    i_lo, i_hi = _axis_render_range(visible, "intelligence")
    c_lo = _cost_transform(c_lo, log_cost)
    c_hi = _cost_transform(c_hi, log_cost)
    return (c_lo, c_hi, s_lo, s_hi, i_lo, i_hi)


def _axis_ticks(lo, hi, log=False, steps=6):
    if hi <= lo:
        hi = lo + (abs(lo) or 1) * 0.01 + 1e-9
    if log:
        lo, hi = max(lo, 1e-6), max(hi, 1e-6)
        return np.logspace(np.log10(lo), np.log10(hi), steps)
    return np.linspace(lo, hi, steps)


def _field_ticks(visible, metric, lo, hi, log, steps, density):
    lin = _axis_ticks(lo, hi, log=log, steps=steps)
    if density <= 0.0 or metric not in visible.columns:
        return lin
    vals = visible[metric].dropna()
    if log:
        vals = vals[vals > 0].map(lambda v: math.log10(float(v)))
    if len(vals) < steps:
        return lin
    qt = np.quantile(vals.to_numpy(dtype=float), np.linspace(0.0, 1.0, steps))
    if log:
        qt = 10.0 ** qt
    return (1.0 - density) * lin + density * qt


def _axis_step(axis):
    return {"cost": 1e-6, "speed": 1e-6, "intelligence": 1e-6, "context": 1000}.get(axis, 1e-6)


def _axis_format(axis):
    return {"cost": "%.6f", "speed": "%.6f", "intelligence": "%.6f", "context": "%.0f"}.get(axis, "%.6f")


def _axis_render_range(visible, metric):
    lo, hi = _metric_axis_range(visible, metric)
    if metric == "cost":
        lo = 10 ** (math.log10(lo) - 0.4)
        hi = 10 ** (math.log10(hi) + 0.4)
    else:
        span = (hi - lo) or 1
        lo = lo - 0.05 * span
        hi = hi + 0.05 * span
    return lo, hi


def _field_grid_range(visible, metric):
    lo, hi = _metric_axis_range(visible, metric)
    if metric == "cost":
        lo = 10 ** (math.log10(lo) - 0.9)
        hi = 10 ** (math.log10(hi) + 0.9)
    else:
        span = (hi - lo) or 1
        lo = lo - 0.2 * span
        hi = hi + 0.2 * span
    return lo, hi


def _adaptive_surfaces(visible, full, x_axis, y_axis, z_axis, log_x, surfaces):
    ratios = []
    for ax in (x_axis, y_axis, z_axis):
        clo, chi = _metric_axis_range(visible, ax)
        flo, fhi = _metric_axis_range(full, ax)
        if ax == "cost":
            clo, chi = math.log10(max(clo, 1e-9)), math.log10(max(chi, 1e-9))
            flo, fhi = math.log10(max(flo, 1e-9)), math.log10(max(fhi, 1e-9))
        cspan = (chi - clo) or 1e-9
        fspan = (fhi - flo) or 1e-9
        ratios.append(min(cspan / fspan, 1.0))
    geomean = (ratios[0] * ratios[1] * ratios[2]) ** (1.0 / 3.0)
    scale = 0.3 + 0.7 * geomean
    return max(2, round(surfaces * scale))


def _apply_axis_ranges(fig, chart_type, x_axis, y_axis, z_axis, log_x, visible):
    for dim, ax in (("x", x_axis), ("y", y_axis), ("z", z_axis)):
        lo, hi = _axis_render_range(visible, ax)
        is_log = ax == "cost" and log_x
        lv = math.log10(lo) if is_log else lo
        hv = math.log10(hi) if is_log else hi
        rng = [lv, hv]
        if chart_type == "3D (WebGL)":
            fig.update_layout(scene={f"{dim}axis": dict(range=rng)})
        elif dim != "z":
            getattr(fig, f"update_{dim}axes")(range=rng)


def build_value_field(visible, x_axis, y_axis, z_axis, log_x, w_cost, w_speed, w_intel, cb,
                      steps=14, opacity=0.14, surfaces=22, curve=1.0, density=0.0):
    med = visible[["cost", "intelligence", "speed"]].median()
    xr = _field_grid_range(visible, x_axis)
    yr = _field_grid_range(visible, y_axis)
    zr = _field_grid_range(visible, z_axis)
    ticks = {}
    for ax, rng in ((x_axis, xr), (y_axis, yr), (z_axis, zr)):
        ticks[ax] = _field_ticks(visible, ax, rng[0], rng[1], log_x and ax == "cost", steps, density)
    grids = {
        "cost": ticks.get("cost", np.full(steps, med["cost"])),
        "intelligence": ticks.get("intelligence", np.full(steps, med["intelligence"])),
        "speed": ticks.get("speed", np.full(steps, med["speed"])),
    }
    slot_for = {x_axis: 0, y_axis: 1, z_axis: 2}

    xx, yy, zz = np.meshgrid(ticks[x_axis], ticks[y_axis], ticks[z_axis], indexing="ij")
    vv = np.empty_like(xx, dtype=float)
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                idxs = (i, j, k)
                c = grids["cost"][idxs[slot_for.get("cost", 0)]]
                a = grids["intelligence"][idxs[slot_for.get("intelligence", 0)]]
                s = grids["speed"][idxs[slot_for.get("speed", 0)]]
                vv[i, j, k] = _score(c, a, s, cb, w_cost, w_speed, w_intel, log_x, curve=curve)

    vv = vv.ravel()
    cmin, cmax = 0.0, 1.0

    return go.Volume(
        x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
        value=vv.ravel(),
        cmin=cmin, cmax=cmax,
        isomin=float(vv.min()), isomax=float(vv.max()),
        opacity=opacity,
        surface_count=surfaces,
        colorscale=VALUE_SCALE,
        showscale=False,
        showlegend=False,
        hoverinfo="skip",
        caps=dict(x_show=False, y_show=False, z_show=False),
    )


@st.cache_data(ttl=3600, show_spinner="Loading model catalog from models.dev…")
def get_catalog(force=False):
    return api.load_catalog(force=force)


@st.cache_data(ttl=86400, show_spinner="Fetching Artificial Analysis scores…")
def get_aa(key):
    try:
        return api.fetch_aa(key)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_scored(catalog, custom_models, aa):
    return api.apply_scores(list(catalog) + list(custom_models), aa=aa)


def main():
    st.set_page_config(page_title="LLM 3D Model Compare", page_icon="📊", layout="wide")

    if st.session_state.pop("_pending_reset", False):
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass
        for k in [k for k in st.session_state if k in STATE_KEYS]:
            del st.session_state[k]
    pending = st.session_state.pop("_pending_state", None)
    if pending:
        for k, v in pending.items():
            st.session_state[k] = v

    if "custom_models" not in st.session_state:
        st.session_state.custom_models = []
    if "aa_key_input" not in st.session_state:
        st.session_state.aa_key_input = api.load_aa_key() or ""
    if "state_seeded" not in st.session_state:
        for _k, _v in _load_state().items():
            st.session_state[_k] = _v
        st.session_state.state_seeded = True

    def _aa_key_cb():
        key = st.session_state.aa_key_input.strip()
        api.save_aa_key(key)

    catalog = get_catalog()
    with st.sidebar:
        with st.expander("🔑 API key & data", expanded=False):
            st.text_input(
                "Artificial Analysis API key (free tier)",
                type="password",
                key="aa_key_input",
                on_change=_aa_key_cb,
                help="Get a free key: sign up at artificialanalysis.ai, then open "
                     "https://artificialanalysis.ai/orgs/<your-username>/api-access, create a key, "
                     "and paste it here. Saved to ~/.config/model-compare/aa_key.",
            )
            def _clear_key():
                st.session_state.aa_key_input = ""
                api.save_aa_key("")

            st.button("Clear saved key", on_click=_clear_key)

    aa_key = st.session_state.aa_key_input.strip() or None
    aa = get_aa(aa_key) if aa_key else None

    scored = get_scored(catalog, st.session_state.custom_models, aa)
    df = pd.DataFrame(scored)

    no_ctx = df["context"].isna() | (df["context"] <= 0)
    unrendered = df.loc[no_ctx].copy()
    unrendered["reason"] = np.where(
        unrendered["context"].isna(), "Context length missing", "Context length = 0"
    )
    df = df.loc[~no_ctx].reset_index(drop=True)

    src = _score_source(df)
    n_live_aa = int((src == "Live (AA)").sum())
    n_est = int((src == "Estimated (heuristic)").sum())
    n_user = int((src == "User-defined").sum())

    with st.sidebar:
        st.caption(f"{len(df):,} models · {df['provider'].nunique():,} providers")
        if n_live_aa == 0 and n_user == 0:
            st.caption("All scores estimated (heuristic) — add an AA API key for live intelligence/speed")
        elif n_live_aa == 0:
            st.caption(f"{n_est:,} estimated (heuristic) + {n_user:,} user-defined — add an AA API key for live data")
        else:
            extra = f" · {n_user:,} user-defined" if n_user else ""
            st.caption(f"{n_live_aa:,} live (AA) · {n_est:,} estimated (heuristic){extra}")

        with st.expander("📈 Chart", expanded=True):
            chart_mode = st.radio(
                "3D needs WebGL. Auto picks the best option.",
                ["Auto", "3D (WebGL)", "2D (fallback)"],
                horizontal=True,
                key="chart_mode",
            )
            detected = webgl_check()
            st.caption(f"WebGL: {'detecting…' if detected is None else detected}")
            if chart_mode == "Auto":
                chart_type = "2D (fallback)" if detected == "no" else "3D (WebGL)"
            else:
                chart_type = chart_mode
            if detected == "no" and chart_type == "3D (WebGL)":
                st.warning("⚠️ WebGL is disabled in this browser — 3D will not render.")

            x_axis = st.selectbox("X axis", list(AXES), index=0, key="x_axis")
            y_axis = st.selectbox("Y axis", list(AXES), index=1, key="y_axis")
            z_axis = st.selectbox("Z axis", list(AXES), index=2, key="z_axis")
            log_x = st.checkbox("Log scale for cost", value=True, key="log_x")

        with st.expander("🏀 Ball", expanded=True):
            ball_size = st.radio("Ball size", ["Parameters", "Context", "Z-axis value", "Uniform"], horizontal=True, key="ball_size")
            if ball_size != "Uniform":
                st.radio("Size scale", ["Log", "Logᵖ", "Linear"], horizontal=True, key="size_scale", index=0,
                         help="The largest model in view always renders at the max size; everything else "
                              "scales by its true ratio to it. Log: log-ratio (compresses). "
                              "Logᵖ: log-ratio with a tunable exponent. Linear: proportional to the raw value.")
                if st.session_state.get("size_scale", "Log") == "Logᵖ":
                    st.slider("Log exponent", 0.25, 4.0, 1.0, 0.25, key="log_exp",
                              help="Higher = more extreme: small values spread apart, large ones compress.")
                st.slider("Max ball size (px)", 20, 400, 60, key="ball_max",
                          help="Caps the largest node; the smallest stays 5px. Tune this if big/small contrast is too strong or weak.")

        with st.expander("🎨 Field", expanded=False):
            color_mode = st.radio("Color by", ["Value score", "Provider"], horizontal=True, index=1, key="color_mode")
            field_curve = st.slider("Field curvature", 1.0, 4.0, 1.0, 0.1, key="field_curve",
                                    help="1 = flat planar isosurfaces (linear score). Higher curves them into "
                                         "shells around the ideal corner (cheap + fast + smart). Also shapes "
                                         "the value-score colors.")
            show_field = st.checkbox("Show value field (3D gradient)", value=True, key="show_field")
            if show_field:
                field_surfaces = st.slider("Field surfaces", 5, 60, 22, key="field_surfaces")
                field_res = st.slider("Field density", 6, 20, 14, key="field_res")
                field_opacity = st.slider("Field opacity", 1, 40, 14, key="field_opacity", format="%d%%") / 100
                st.slider("Grid follows model density", 0, 100, 50, key="field_density",
                          format="%d%%",
                          help="Shifts grid samples toward where models actually live (quantile spacing), "
                               "concentrating the isosurface detail in the crowded region.")

        with st.expander("⚖️ Weights", expanded=False):
            st.caption("Tilt the value gradient toward what matters")
            w_cost = st.slider("Cheapness (cost) weight", 0, 100, 33, key="w_cost")
            w_speed = st.slider("Speed weight", 0, 100, 33, key="w_speed")
            w_intel = st.slider("Intelligence weight", 0, 100, 34, key="w_intel")
            wsum = max(w_cost + w_speed + w_intel, 1)
            st.caption(f"Normalized: {round(100 * w_cost / wsum)}% / {round(100 * w_speed / wsum)}% / {round(100 * w_intel / wsum)}%")

        with st.expander("🧠 Effort", expanded=False):
            st.caption("Reasoning effort levels. Recommendations use the weights above.")
            effort_filter = st.multiselect(
                "Supported effort levels (any of)",
                EFFORT_ORDER,
                key="effort_filter",
                help="Only show models that support at least one of these reasoning-effort levels "
                     "(from models.dev). Empty = no effort filtering.",
            )
            effort_ladder_only = st.checkbox(
                "Only models with a measured effort ladder", value=False, key="effort_ladder_only",
                help="Artificial Analysis has measured ≥2 effort levels for these models, so a "
                     "best level can actually be recommended.")
            show_effort_variants = st.checkbox(
                "Show effort variants on chart", value=False, key="show_effort_variants",
                help="Expand measured models into one ball per effort level (intelligence & speed "
                     "change per level; cost stays the model's base input price).")
            st.radio("Effort color (variants)", ["Provider", "Effort level"], horizontal=True,
                     index=0, key="effort_color_mode")

        with st.expander("🎚️ Axis ranges", expanded=False):
            st.caption("Leave blank to auto-scale.")
            seen_metrics = set()
            for dim, ax in (("X", x_axis), ("Y", y_axis), ("Z", z_axis)):
                if ax in seen_metrics:
                    continue
                seen_metrics.add(ax)
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(f"{dim} min ({ax})", value=None, step=_axis_step(ax), format=_axis_format(ax), key=f"rng_{ax}_min")
                with c2:
                    st.number_input(f"{dim} max ({ax})", value=None, step=_axis_step(ax), format=_axis_format(ax), key=f"rng_{ax}_max")

        with st.expander("🔎 Filter", expanded=True):
            search = st.text_input("Search model name", key="search")
            providers = sorted(df["provider"].unique())

            show_all = st.checkbox("Show all providers", value=True, key="show_all")
            sel_providers = []
            if not show_all:
                with st.expander(f"Choose providers ({len(providers):,})"):
                    prov_search = st.text_input("Search providers", key="prov_search")
                    opts = [p for p in providers if prov_search.lower() in p.lower()]
                    sel_providers = st.multiselect("Providers to show (empty = all)", opts, key="sel_providers")
                    st.caption(f"Found {len(opts):,} matching providers.")

            reasoning_only = st.checkbox("Reasoning models only", value=False, key="reasoning_only")
            open_weights = st.checkbox("Open-weights only", value=False, key="open_weights")
            sel_sources = st.multiselect(
                "Score source",
                ["Live (AA)", "Estimated (heuristic)", "User-defined"],
                key="sel_sources",
                help="Where intelligence/speed come from. Empty = show all. "
                     "Live = measured by Artificial Analysis; Estimated = local heuristic; "
                     "User-defined = numbers you typed in yourself.",
            )
            st.caption("Context length (K tokens)")
            ctx_c1, ctx_c2 = st.columns(2)
            with ctx_c1:
                min_context = st.number_input("Min", min_value=0, step=16, value=0, key="min_context")
            with ctx_c2:
                max_context = st.number_input("Max", min_value=0, step=16, value=0, key="max_context")

            st.caption("Origin")
            continent_opts = sorted([c for c in df["continent"].dropna().unique()])
            country_opts = sorted([c for c in df["country"].dropna().unique()])
            sel_continents = st.multiselect("Continent", continent_opts, key="sel_continents")
            sel_countries = st.multiselect("Country", country_opts, key="sel_countries")

            with st.expander(f"Browse all {len(providers):,} providers"):
                counts = df.groupby("provider").size().sort_values(ascending=False).rename("models")
                st.dataframe(counts, width="stretch")

        with st.expander("⭐ Highlight", expanded=False):
            hl_search = st.text_input("Search model to highlight", key="hl_search")
            all_names = sorted(df["name"].unique())

            def _hl_norm(s):
                return re.sub(r"[^a-z0-9]+", "", s.lower())

            q = hl_search.strip().lower()
            if q:
                tokens = [_hl_norm(t) for t in q.split()]
                tokens = [t for t in tokens if t]
                if tokens:
                    norms = {n: _hl_norm(n) for n in all_names}
                    hl_opts = [n for n in all_names if all(t in norms[n] for t in tokens)]
                else:
                    hl_opts = []
            else:
                hl_opts = all_names
            hl_names = st.multiselect("Models to highlight", hl_opts, max_selections=10, key="hl_names")
            st.caption(
                f"{len(hl_opts):,} matching · large gold markers" if q
                else "Type to search · highlighted render as large gold markers"
            )

        with st.expander("➕ Custom models", expanded=False):
            with st.form("add_model", clear_on_submit=True):
                name = st.text_input("Model name")
                provider = st.text_input("Provider")
                cost = st.number_input("Cost ($/1M input)", min_value=0.0, step=0.01)
                speed = st.slider("Speed (1-10)", 1, 10, 7)
                intelligence = st.slider("Intelligence (1-10)", 1.0, 10.0, 7.0, 0.1)
                context = st.number_input("Context (tokens)", min_value=0, step=1000, value=0)
                if st.form_submit_button("Add"):
                    if name and provider:
                        st.session_state.custom_models.append(
                            {"id": f"custom/{name}", "name": name, "provider": provider, "cost": cost,
                             "speed": speed, "intelligence": intelligence, "context": context,
                             "reasoning": False, "open_weights": False, "scores_live": True}
                        )
                        st.rerun()

            if st.session_state.custom_models:
                to_remove = st.selectbox("Remove custom model", [m["name"] for m in st.session_state.custom_models])
                if st.button("Remove"):
                    st.session_state.custom_models = [m for m in st.session_state.custom_models if m["name"] != to_remove]
                    st.rerun()

        if st.button("Clear cache & reload"):
            get_catalog.clear()
            st.rerun()

        with st.expander("⚙️ Configurations", expanded=False):
            st.caption("Save and restore the full sidebar setup. The API key is not included.")
            profiles = _load_profiles()
            cfg_name = st.text_input("Configuration name", key="cfg_name")
            b1, b2 = st.columns(2)
            with b1:
                save_clicked = st.button("💾 Save current", disabled=not cfg_name.strip(),
                                         use_container_width=True)
            with b2:
                reset_clicked = st.button("↺ Reset to defaults", use_container_width=True)
            if save_clicked:
                profiles = _load_profiles()
                profiles[cfg_name.strip()] = _capture_state()
                _save_profiles(profiles)
                st.toast(f"Saved configuration '{cfg_name.strip()}'")
                st.rerun()
            if reset_clicked:
                _reset_state()
                st.rerun()
            if profiles:
                pick = st.selectbox("Saved configurations", [""] + sorted(profiles.keys()))
                l1, l2 = st.columns(2)
                with l1:
                    load_clicked = st.button("📥 Load", disabled=not pick, use_container_width=True)
                with l2:
                    del_clicked = st.button("🗑 Delete", disabled=not pick, use_container_width=True)
                if load_clicked:
                    _apply_state(profiles[pick])
                    st.rerun()
                if del_clicked:
                    profiles.pop(pick, None)
                    _save_profiles(profiles)
                    st.rerun()

    visible = df
    if search:
        visible = visible[visible["name"].str.contains(search, case=False, na=False)]
    if not show_all and sel_providers:
        visible = visible[visible["provider"].isin(sel_providers)]
    if reasoning_only:
        visible = visible[visible["reasoning"]]
    if open_weights:
        visible = visible[visible["open_weights"]]
    if sel_sources:
        visible = visible[_score_source(visible).isin(sel_sources)]
    if min_context:
        visible = visible[visible["context"] >= min_context * 1000]
    if max_context:
        visible = visible[visible["context"] <= max_context * 1000]
    if sel_continents:
        visible = visible[visible["continent"].isin(sel_continents)]
    if sel_countries:
        visible = visible[visible["country"].isin(sel_countries)]
    if effort_filter:
        visible = visible[visible["effort_levels"].map(
            lambda ls: any(e in (ls or []) for e in effort_filter))]
    if effort_ladder_only:
        visible = visible[visible.apply(_has_effort_ladder, axis=1)]
    visible = visible.sort_values("intelligence", ascending=False).reset_index(drop=True)

    if visible.empty:
        st.warning("No models match the current filters.")
        _save_state()
        return

    src = _score_source(visible)
    ring_mask = (src == "Live (AA)").to_numpy()

    st.title("📊 3D LLM Model Comparison")
    st.caption(f"{len(visible):,} models shown. Hover for details; drag to rotate.")
    if len(unrendered):
        st.markdown(f"🚫 **{len(unrendered):,} models excluded** (no context) — [view list](#excluded-models)")

    color_map = {p: api.provider_color(p) for p in df["provider"].unique()}

    cb = _compute_bounds(visible, log_x)
    c_lo, c_hi, s_lo, s_hi, i_lo, i_hi = cb
    visible = visible.copy()
    wsum = max(w_cost + w_speed + w_intel, 1)
    visible["value"] = visible.apply(
        lambda r: _score(r["cost"], r["intelligence"], r["speed"], cb,
                         w_cost, w_speed, w_intel, log_x, curve=field_curve),
        axis=1,
    )
    meta = visible.apply(
        lambda r: _best_effort_meta(r, w_cost, w_speed, w_intel, log_x, field_curve),
        axis=1,
    )
    visible["best_effort"] = [m[0] for m in meta]
    visible["effort_cost"] = [m[1] for m in meta]
    visible["effort_cost_src"] = [m[2] for m in meta]
    value_range = (0.0, 1.0)

    plot_df = _expand_effort(visible) if show_effort_variants else visible.copy()
    if show_effort_variants:
        plot_df["value"] = plot_df.apply(
            lambda r: _score(r["cost"], r["intelligence"], r["speed"], cb,
                             w_cost, w_speed, w_intel, log_x, curve=field_curve),
            axis=1,
        )
        plot_df["_effort_disp"] = plot_df["effort"]
    else:
        plot_df["_effort_disp"] = plot_df["best_effort"]
    plot_df = plot_df.reset_index(drop=True)
    plot_src = _score_source(plot_df)
    ring_mask = (plot_src == "Live (AA)").to_numpy()

    size_scale = st.session_state.get("size_scale", "Log")
    if ball_size == "Parameters":
        raw, size_label = plot_df["params"], "Ball size = parameters (B)"
    elif ball_size == "Context":
        raw, size_label = plot_df["context"], "Ball size = context (tokens)"
    elif ball_size == "Z-axis value":
        raw, size_label = plot_df[z_axis], f"Ball size = {AXES[z_axis]}"
    else:
        raw, size_label = None, "Uniform ball size"

    if raw is None:
        sizes = pd.Series([10] * len(plot_df), index=plot_df.index)
    else:
        sizes = _ball_sizes(raw, size_scale, st.session_state.get("ball_max", 60),
                            st.session_state.get("log_exp", 1.0))

    print(f"[ball-size] {size_label} · scale={size_scale} · min={float(sizes.min()):.1f}px "
          f"max={float(sizes.max()):.1f}px · distinct={int(sizes.nunique())}", flush=True)

    hover_cols = ["Provider", "Cost ($/1M in)", "Speed (1-10)", "Intelligence (1-10)", "Score source",
                  "Effort", "Effort levels", "Best effort", "Effort cost", "Effort cost source",
                  "Context", "Params (B)", "Country", "Reasoning", "AA Intell. Index", "AA tokens/s",
                  "Ball size (px)"]

    hdata = plot_df.copy()
    hdata["_bsize"] = sizes
    hdata["_ring"] = ring_mask
    hdata["Provider"] = hdata["provider"].fillna("n/a")
    hdata["Cost ($/1M in)"] = hdata["cost"].map(lambda v: _hnum(v, "{:.2f}"))
    hdata["Speed (1-10)"] = hdata["speed"].map(lambda v: _hnum(v, "{:.1f}"))
    hdata["Intelligence (1-10)"] = hdata["intelligence"].map(lambda v: _hnum(v, "{:.1f}"))
    hdata["Score source"] = plot_src.to_numpy()
    hdata["Effort"] = hdata["_effort_disp"].map(lambda v: v if isinstance(v, str) else "n/a")
    hdata["Effort levels"] = hdata["effort_levels"].map(
        lambda ls: ", ".join(_effort_ordered(ls)) or "n/a")
    hdata["Best effort"] = hdata["best_effort"].map(lambda v: v if isinstance(v, str) else "n/a")
    hdata["Effort cost"] = [_effort_cost_str(c, s)
                            for c, s in zip(hdata["effort_cost"], hdata["effort_cost_src"])]
    hdata["Effort cost source"] = hdata["effort_cost_src"].map(lambda s: s or "n/a")
    hdata["Context"] = hdata["context"].map(lambda v: _hnum(v, "{:,.0f}"))
    hdata["Params (B)"] = hdata["params"].map(lambda v: _hnum(v, "{:.1f}"))
    hdata["Country"] = hdata["country"].fillna("n/a")
    hdata["Reasoning"] = hdata["reasoning"].map(lambda v: "Yes" if v else "No")
    hdata["AA Intell. Index"] = hdata["aa_intelligence_index"].map(lambda v: _hnum(v, "{:.1f}")) if "aa_intelligence_index" in hdata.columns else "n/a"
    hdata["AA tokens/s"] = hdata["aa_tokens_per_sec"].map(lambda v: _hnum(v, "{:.1f}")) if "aa_tokens_per_sec" in hdata.columns else "n/a"
    hdata["Ball size (px)"] = sizes.map(lambda v: _hnum(v, "{:.0f}"))
    hover_data = {c: True for c in hover_cols}

    def _hl_hover(hdf, dims):
        cols = ["name"] + hover_cols
        cdata = hdf[cols].to_numpy()
        lines = ["<b>%{customdata[0]}</b>"]
        for i, c in enumerate(hover_cols, start=1):
            lines.append(f"{c}: %{{customdata[{i}]}}")
        for ch, ax in dims:
            lines.append(f"{AXES[ax]}: %{{{ch}}}")
        return "<br>".join(lines) + "<extra>Highlighted</extra>", cdata

    use_continuous = color_mode == "Value score"
    hl_mask = plot_df["name"].isin(hl_names) if hl_names else pd.Series(False, index=plot_df.index)
    base_opac = 0.18 if len(hl_names) else 0.45
    use_effort_color = show_effort_variants and st.session_state.get("effort_color_mode") == "Effort level"
    if chart_type == "3D (WebGL)":
        if use_continuous:
            fig = px.scatter_3d(hdata, x=x_axis, y=y_axis, z=z_axis,
                                color="value", color_continuous_scale=VALUE_SCALE,
                                range_color=value_range,
                                size="_bsize", size_max=200,
                                hover_name="name", hover_data=hover_data,
                                custom_data=["_ring"], text=None, title=None)
        elif use_effort_color:
            fig = px.scatter_3d(hdata, x=x_axis, y=y_axis, z=z_axis,
                                color="effort", color_discrete_map=EFFORT_COLOR_MAP,
                                size="_bsize", size_max=200,
                                hover_name="name", hover_data=hover_data,
                                custom_data=["_ring"], text=None, title=None)
        else:
            fig = px.scatter_3d(hdata, x=x_axis, y=y_axis, z=z_axis,
                                color="provider", color_discrete_map=color_map,
                                size="_bsize", size_max=200,
                                hover_name="name", hover_data=hover_data,
                                custom_data=["_ring"], text=None, title=None)
        if show_field and len(visible) >= 4:
            surfs = _adaptive_surfaces(visible, df, x_axis, y_axis, z_axis, log_x, field_surfaces)
            fig.add_trace(build_value_field(visible, x_axis, y_axis, z_axis, log_x,
                                            w_cost, w_speed, w_intel, cb,
                                            steps=field_res, opacity=field_opacity,
                                            surfaces=surfs, curve=field_curve,
                                            density=st.session_state.get("field_density", 50) / 100.0))
        fig.update_traces(marker=dict(sizemode="diameter", sizeref=1, sizemin=1, opacity=base_opac),
                          selector=dict(type="scatter3d"))
        _apply_live_outline(fig)
        if len(hl_names):
            hdf = plot_df[hl_mask]
            htemplate, hcustom = _hl_hover(hdata[hl_mask], (("x", x_axis), ("y", y_axis), ("z", z_axis)))
            fig.add_trace(go.Scatter3d(
                x=hdf[x_axis], y=hdf[y_axis], z=hdf[z_axis],
                mode="markers",
                name=f"Highlighted ({len(hdf)})",
                marker=dict(size=(sizes[hl_mask] * 1.6 + 4).clip(upper=260), color="#FFD700",
                            opacity=1.0, line=dict(width=2, color="#000000")),
                hovertemplate=htemplate,
                customdata=hcustom,
            ))
        fig.update_layout(scene=dict(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis], zaxis_title=AXES[z_axis]),
                          height=750, margin=dict(l=0, r=0, t=30, b=0))
        if use_continuous:
            fig.update_coloraxes(colorbar=dict(title="Value", thickness=15))
        else:
            fig.update_layout(legend_title="Effort" if use_effort_color else "Provider")
        if log_x and x_axis == "cost":
            fig.update_layout(scene=dict(xaxis=dict(type="log")))
    else:
        if use_continuous:
            fig = px.scatter(hdata, x=x_axis, y=y_axis, color="value",
                             color_continuous_scale=VALUE_SCALE, range_color=value_range,
                             size="_bsize", size_max=200,
                             hover_name="name", hover_data=hover_data,
                             custom_data=["_ring"], title=None)
        elif use_effort_color:
            fig = px.scatter(hdata, x=x_axis, y=y_axis, color="effort",
                             color_discrete_map=EFFORT_COLOR_MAP,
                             size="_bsize", size_max=200,
                             hover_name="name", hover_data=hover_data,
                             custom_data=["_ring"], title=None)
        else:
            fig = px.scatter(hdata, x=x_axis, y=y_axis, color="provider", color_discrete_map=color_map,
                             size="_bsize", size_max=200,
                             hover_name="name", hover_data=hover_data,
                             custom_data=["_ring"], title=None)
        fig.update_traces(marker=dict(sizemode="diameter", sizeref=1, sizemin=1, opacity=base_opac))
        _apply_live_outline(fig)
        if len(hl_names):
            hdf = plot_df[hl_mask]
            htemplate, hcustom = _hl_hover(hdata[hl_mask], (("x", x_axis), ("y", y_axis)))
            fig.add_trace(go.Scatter(
                x=hdf[x_axis], y=hdf[y_axis],
                mode="markers",
                name=f"Highlighted ({len(hdf)})",
                marker=dict(size=(sizes[hl_mask] * 1.6 + 4).clip(upper=260), color="#FFD700",
                            opacity=1.0, line=dict(width=2, color="#000000")),
                hovertemplate=htemplate,
                customdata=hcustom,
            ))
        fig.update_layout(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis],
                          height=750, margin=dict(l=0, r=0, t=30, b=0))
        if use_continuous:
            fig.update_coloraxes(colorbar=dict(title="Value", thickness=15))
        else:
            fig.update_layout(legend_title="Effort" if use_effort_color else "Provider")
        fig.add_annotation(text=size_label, xref="paper", yref="paper",
                           x=0, y=1.08, showarrow=False, font=dict(size=12), xanchor="left")
        if log_x and x_axis == "cost":
            fig.update_xaxes(type="log")

    _apply_axis_ranges(fig, chart_type, x_axis, y_axis, z_axis, log_x, plot_df)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Ball size: {size_label} · {float(sizes.min()):.0f}–{float(sizes.max()):.0f} px · {int(sizes.nunique()):,} distinct")
    if ring_mask.any():
        st.caption(f"Balls with a cyan outline = intelligence & speed measured by Artificial Analysis ({int(ring_mask.sum()):,}); plain balls = heuristic estimates.")
    if chart_type == "3D (WebGL)" and show_field and len(visible) >= 4:
        st.caption(f"Field surfaces: {surfs} (adapted from {field_surfaces} to the current axis ranges)")
    if show_effort_variants:
        st.caption(f"Effort variants on: {len(plot_df):,} points ({len(plot_df) - len(visible):,} extra balls). "
                   "Variant cost uses the model's base input price; per-level cost is in the Effort tradeoff panel below.")

    render_effort_panel(
        visible, hl_names, w_cost, w_speed, w_intel, log_x, field_curve,
        {
            "x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis,
            "chart_type": chart_type, "ball_size": ball_size, "size_scale": size_scale,
            "ball_max": st.session_state.get("ball_max", 60),
            "log_exp": st.session_state.get("log_exp", 1.0),
        },
    )

    st.subheader("Table")
    table_search = st.text_input("Search table (matches any column)", key="table_search")
    tbl = visible.copy()
    tbl["Score source"] = src.reindex(tbl.index).to_numpy()
    tbl["Effort levels"] = tbl["effort_levels"].map(lambda ls: ", ".join(_effort_ordered(ls)) or "n/a")
    tbl["Best effort"] = tbl["best_effort"].map(lambda v: v if isinstance(v, str) else "n/a")
    _cols = list(visible.columns)
    _pos = _cols.index("name") + 1
    _cols[_pos:_pos] = ["Score source", "Effort levels", "Best effort"]
    tbl = tbl[_cols]
    if table_search:
        mask = tbl.astype(str).apply(lambda col: col.str.contains(table_search, case=False, na=False)).any(axis=1)
        tbl = tbl[mask]
    st.dataframe(tbl.style.map(_cell_style, subset=["Score source"]), width="stretch", hide_index=True)
    st.caption(f"Showing {len(tbl):,} of {len(visible):,} filtered models. "
               "Score source colors: green = live (Artificial Analysis), amber = heuristic estimate, "
               "orange = effort-estimated cost, grey = user-defined. "
               "'Best effort' is the level with the highest weighted value (★ in the Effort panel).")

    if len(unrendered):
        st.markdown('<a id="excluded-models"></a>', unsafe_allow_html=True)
        st.subheader(f"🚫 Excluded models — no context ({len(unrendered):,})")
        st.caption("Hidden because a context length is required to judge the model.")
        st.dataframe(unrendered[["name", "provider", "country", "reason"]],
                     width="stretch", hide_index=True)

    _save_state()


if __name__ == "__main__":
    main()
