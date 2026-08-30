import json
import math
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import models_api as api

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".app_state.json")

STATE_KEYS = [
    "chart_mode", "ball_size", "color_mode", "show_field", "log_x",
    "field_surfaces", "field_res", "field_opacity",
    "x_axis", "y_axis", "z_axis",
    "w_cost", "w_speed", "w_intel",
    "search", "show_all", "prov_search", "sel_providers",
    "reasoning_only", "open_weights", "min_context", "max_context",
    "hl_search", "hl_names", "table_search",
]


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

DEFAULT_AXES = ["cost", "speed", "intelligence"]

VALUE_SCALE = [
    (0.00, "rgb(110,0,0)"),
    (0.30, "rgb(200,30,40)"),
    (0.55, "rgb(255,150,30)"),
    (0.80, "rgb(250,230,0)"),
    (1.00, "rgb(70,200,80)"),
]


def _cost_transform(c, log_cost):
    return math.log10(max(c, 1e-6)) if log_cost else c


def _score(cost_v, intel_v, speed_v, cb, w_cost, w_speed, w_intel, log_cost):
    c_lo, c_hi, s_lo, s_hi, i_lo, i_hi = cb
    cspan = (c_hi - c_lo) or 1
    cheap = 1 - (_cost_transform(cost_v, log_cost) - c_lo) / cspan
    sspan = (s_hi - s_lo) or 1
    sn = (speed_v - s_lo) / sspan
    ispan = (i_hi - i_lo) or 1
    inn = (intel_v - i_lo) / ispan
    wsum = max(w_cost + w_speed + w_intel, 1e-9)
    return (w_cost * cheap + w_speed * sn + w_intel * inn) / wsum


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
                      steps=14, opacity=0.14, surfaces=22):
    med = visible[["cost", "intelligence", "speed"]].median()
    xr = _field_grid_range(visible, x_axis)
    yr = _field_grid_range(visible, y_axis)
    zr = _field_grid_range(visible, z_axis)
    ticks = {
        x_axis: _axis_ticks(xr[0], xr[1], log=log_x and x_axis == "cost", steps=steps),
        y_axis: _axis_ticks(yr[0], yr[1], log=log_x and y_axis == "cost", steps=steps),
        z_axis: _axis_ticks(zr[0], zr[1], log=log_x and z_axis == "cost", steps=steps),
    }
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
                vv[i, j, k] = _score(c, a, s, cb, w_cost, w_speed, w_intel, log_x)

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


def main():
    st.set_page_config(page_title="LLM 3D Model Compare", page_icon="📊", layout="wide")

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
            if st.button("Clear saved key"):
                st.session_state.aa_key_input = ""
                api.save_aa_key("")
                st.rerun()

    aa_key = st.session_state.aa_key_input.strip() or None
    aa = get_aa(aa_key) if aa_key else None

    models = catalog + st.session_state.custom_models
    scored = api.apply_scores(models, aa=aa)
    df = pd.DataFrame(scored)

    with st.sidebar:
        live = sum(1 for m in scored if m["scores_live"])
        st.caption(f"{len(df):,} models · {df['provider'].nunique():,} providers")
        st.caption(f"{live:,} live AA scores" if live else "AA key optional (live speed/intelligence)")

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

        with st.expander("🎨 Field & weights", expanded=True):
            ball_size = st.radio("Ball size", ["Parameters", "Z-axis value", "Uniform"], horizontal=True, key="ball_size")
            color_mode = st.radio("Color by", ["Value score", "Provider"], horizontal=True, index=1, key="color_mode")
            show_field = st.checkbox("Show value field (3D gradient)", value=True, key="show_field")
            if show_field:
                field_surfaces = st.slider("Field surfaces", 5, 60, 22, key="field_surfaces")
                field_res = st.slider("Field density", 6, 20, 14, key="field_res")
                field_opacity = st.slider("Field opacity", 1, 40, 14, key="field_opacity", format="%d%%") / 100

            st.caption("⚖️ Value weights — tilt the gradient toward what matters")
            w_cost = st.slider("Cheapness (cost) weight", 0, 100, 33, key="w_cost")
            w_speed = st.slider("Speed weight", 0, 100, 33, key="w_speed")
            w_intel = st.slider("Intelligence weight", 0, 100, 34, key="w_intel")
            wsum = max(w_cost + w_speed + w_intel, 1)
            st.caption(f"Normalized: {round(100 * w_cost / wsum)}% / {round(100 * w_speed / wsum)}% / {round(100 * w_intel / wsum)}%")

        with st.expander("🎚️ Axis ranges", expanded=False):
            st.caption("Leave blank to auto-scale.")
            for dim, ax in (("X", x_axis), ("Y", y_axis), ("Z", z_axis)):
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
            st.caption("Context length (K tokens)")
            ctx_c1, ctx_c2 = st.columns(2)
            with ctx_c1:
                min_context = st.number_input("Min", min_value=0, step=16, value=0, key="min_context")
            with ctx_c2:
                max_context = st.number_input("Max", min_value=0, step=16, value=0, key="max_context")

            with st.expander(f"Browse all {len(providers):,} providers"):
                counts = df.groupby("provider").size().sort_values(ascending=False).rename("models")
                st.dataframe(counts, width="stretch")

        with st.expander("⭐ Highlight", expanded=False):
            hl_search = st.text_input("Search model to highlight", key="hl_search")
            hl_opts = [n for n in sorted(df["name"].unique()) if hl_search.lower() in n.lower()]
            hl_names = st.multiselect("Models to highlight", hl_opts, max_selections=10, key="hl_names")
            st.caption("Highlighted models render as large gold markers.")

        with st.expander("➕ Custom models", expanded=False):
            with st.form("add_model", clear_on_submit=True):
                name = st.text_input("Model name")
                provider = st.text_input("Provider")
                cost = st.number_input("Cost ($/1M input)", min_value=0.0, step=0.01)
                speed = st.slider("Speed (1-10)", 1, 10, 7)
                intelligence = st.slider("Intelligence (1-10)", 1.0, 10.0, 7.0, 0.1)
                if st.form_submit_button("Add"):
                    if name and provider:
                        st.session_state.custom_models.append(
                            {"id": f"custom/{name}", "name": name, "provider": provider, "cost": cost,
                             "speed": speed, "intelligence": intelligence, "context": 0,
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

    visible = df
    if search:
        visible = visible[visible["name"].str.contains(search, case=False, na=False)]
    if not show_all and sel_providers:
        visible = visible[visible["provider"].isin(sel_providers)]
    if reasoning_only:
        visible = visible[visible["reasoning"]]
    if open_weights:
        visible = visible[visible["open_weights"]]
    if min_context:
        visible = visible[visible["context"] >= min_context * 1000]
    if max_context:
        visible = visible[visible["context"] <= max_context * 1000]
    visible = visible.sort_values("intelligence", ascending=False).reset_index(drop=True)

    if visible.empty:
        st.warning("No models match the current filters.")
        return

    st.title("📊 3D LLM Model Comparison")
    st.caption(f"{len(visible):,} models shown. Hover for details; drag to rotate.")

    color_map = {p: api.provider_color(p) for p in df["provider"].unique()}

    cb = _compute_bounds(visible, log_x)
    c_lo, c_hi, s_lo, s_hi, i_lo, i_hi = cb
    cspan = (c_hi - c_lo) or 1
    cheap = visible["cost"].apply(lambda c: 1 - (_cost_transform(c, log_x) - c_lo) / cspan)
    visible = visible.copy()
    wsum = max(w_cost + w_speed + w_intel, 1)
    s_norm = (visible["speed"] - s_lo) / (s_hi - s_lo)
    i_norm = (visible["intelligence"] - i_lo) / (i_hi - i_lo)
    visible["value"] = (w_cost * cheap + w_speed * s_norm + w_intel * i_norm) / wsum
    value_range = (0.0, 1.0)

    if ball_size == "Parameters":
        pvals = visible["params"].dropna()
        if len(pvals):
            lo, hi = math.log10(max(pvals.min(), 0.1)), math.log10(max(pvals.max(), 0.1))
            span = (hi - lo) or 1
            sizes = visible["params"].apply(
                lambda p: 5 + 30 * (math.log10(max(p, 0.1)) - lo) / span if p and p == p else 8
            )
        else:
            sizes = pd.Series([8] * len(visible), index=visible.index)
        size_label = "Ball size = parameters (B)"
    elif ball_size == "Z-axis value":
        vmin, vmax = visible[z_axis].min(), visible[z_axis].max()
        span = (vmax - vmin) or 1
        sizes = 5 + 25 * (visible[z_axis] - vmin) / span
        size_label = f"Ball size = {AXES[z_axis]}"
    else:
        sizes = pd.Series([8] * len(visible), index=visible.index)
        size_label = "Uniform ball size"

    hover_cols = ["Provider", "Cost ($/1M in)", "Speed (1-10)", "Intelligence (1-10)",
                  "Context", "Params (B)", "Reasoning", "AA Intell. Index", "AA tokens/s"]

    def _hnum(v, fmt="{:.1f}"):
        return "n/a" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)

    hdata = visible.copy()
    hdata["Provider"] = hdata["provider"].fillna("n/a")
    hdata["Cost ($/1M in)"] = hdata["cost"].map(lambda v: _hnum(v, "{:.2f}"))
    hdata["Speed (1-10)"] = hdata["speed"].map(lambda v: _hnum(v, "{:.1f}"))
    hdata["Intelligence (1-10)"] = hdata["intelligence"].map(lambda v: _hnum(v, "{:.1f}"))
    hdata["Context"] = hdata["context"].map(lambda v: _hnum(v, "{:,.0f}"))
    hdata["Params (B)"] = hdata["params"].map(lambda v: _hnum(v, "{:.1f}"))
    hdata["Reasoning"] = hdata["reasoning"].map(lambda v: "Yes" if v else "No")
    hdata["AA Intell. Index"] = hdata["aa_intelligence_index"].map(lambda v: _hnum(v, "{:.1f}")) if "aa_intelligence_index" in hdata.columns else "n/a"
    hdata["AA tokens/s"] = hdata["aa_tokens_per_sec"].map(lambda v: _hnum(v, "{:.1f}")) if "aa_tokens_per_sec" in hdata.columns else "n/a"
    hover_data = {c: True for c in hover_cols}

    use_continuous = color_mode == "Value score"
    hl_mask = visible["name"].isin(hl_names) if hl_names else pd.Series(False, index=visible.index)
    base_opac = 0.18 if len(hl_names) else 0.45
    if chart_type == "3D (WebGL)":
        if use_continuous:
            fig = px.scatter_3d(hdata, x=x_axis, y=y_axis, z=z_axis,
                                color="value", color_continuous_scale=VALUE_SCALE,
                                range_color=value_range,
                                hover_name="name", hover_data=hover_data,
                                text=None, title=None)
        else:
            fig = px.scatter_3d(hdata, x=x_axis, y=y_axis, z=z_axis,
                                color="provider", color_discrete_map=color_map,
                                hover_name="name", hover_data=hover_data,
                                text=None, title=None)
        if show_field and len(visible) >= 4:
            fig.add_trace(build_value_field(visible, x_axis, y_axis, z_axis, log_x,
                                            w_cost, w_speed, w_intel, cb,
                                            steps=field_res, opacity=field_opacity,
                                            surfaces=field_surfaces))
        fig.update_traces(marker=dict(size=sizes, opacity=base_opac), selector=dict(type="scatter3d"))
        if len(hl_names):
            hdf = visible[hl_mask]
            fig.add_trace(go.Scatter3d(
                x=hdf[x_axis], y=hdf[y_axis], z=hdf[z_axis],
                mode="markers",
                name=f"Highlighted ({len(hdf)})",
                marker=dict(size=(sizes[hl_mask] * 1.6 + 4).clip(upper=45), color="#FFD700",
                            opacity=1.0, line=dict(width=2, color="#000000")),
                hovertemplate="%{customdata}<extra>Highlighted</extra>",
                customdata=hdf["name"],
            ))
        fig.update_layout(scene=dict(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis], zaxis_title=AXES[z_axis]),
                          height=750, margin=dict(l=0, r=0, t=30, b=0))
        if use_continuous:
            fig.update_coloraxes(colorbar=dict(title="Value", thickness=15))
        else:
            fig.update_layout(legend_title="Provider")
        if log_x and x_axis == "cost":
            fig.update_layout(scene=dict(xaxis=dict(type="log")))
    else:
        if use_continuous:
            fig = px.scatter(hdata, x=x_axis, y=y_axis, color="value",
                             color_continuous_scale=VALUE_SCALE, range_color=value_range,
                             hover_name="name", hover_data=hover_data, title=None)
        else:
            fig = px.scatter(hdata, x=x_axis, y=y_axis, color="provider", color_discrete_map=color_map,
                             hover_name="name", hover_data=hover_data, title=None)
        fig.update_traces(marker=dict(size=sizes, opacity=base_opac))
        if len(hl_names):
            hdf = visible[hl_mask]
            fig.add_trace(go.Scatter(
                x=hdf[x_axis], y=hdf[y_axis],
                mode="markers",
                name=f"Highlighted ({len(hdf)})",
                marker=dict(size=(sizes[hl_mask] * 1.6 + 4).clip(upper=45), color="#FFD700",
                            opacity=1.0, line=dict(width=2, color="#000000")),
                hovertemplate="%{customdata}<extra>Highlighted</extra>",
                customdata=hdf["name"],
            ))
        fig.update_layout(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis],
                          height=750, margin=dict(l=0, r=0, t=30, b=0))
        if use_continuous:
            fig.update_coloraxes(colorbar=dict(title="Value", thickness=15))
        else:
            fig.update_layout(legend_title="Provider")
        fig.add_annotation(text=size_label, xref="paper", yref="paper",
                           x=0, y=1.08, showarrow=False, font=dict(size=12), xanchor="left")
        if log_x and x_axis == "cost":
            fig.update_xaxes(type="log")

    _apply_axis_ranges(fig, chart_type, x_axis, y_axis, z_axis, log_x, visible)

    st.plotly_chart(fig, width="stretch")

    st.subheader("Table")
    table_search = st.text_input("Search table (matches any column)", key="table_search")
    tbl = visible
    if table_search:
        mask = tbl.astype(str).apply(lambda col: col.str.contains(table_search, case=False, na=False)).any(axis=1)
        tbl = tbl[mask]
    st.dataframe(tbl, width="stretch", hide_index=True)
    st.caption(f"Showing {len(tbl):,} of {len(visible):,} filtered models.")

    _save_state()


if __name__ == "__main__":
    main()
