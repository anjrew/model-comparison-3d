import math

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import models_api as api

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


def _score(cost_v, intel_v, speed_v, clow_log, chi_log, w_cost, w_speed, w_intel):
    cspan = (chi_log - clow_log) or 1
    cheap = 1 - min(1, max(0, (math.log10(max(cost_v, 1e-6)) - clow_log) / cspan))
    wsum = max(w_cost + w_speed + w_intel, 1e-9)
    return (w_cost * cheap + w_speed * (speed_v - 1) / 9 + w_intel * (intel_v - 1) / 9) / wsum


def _axis_ticks(vals, log=False, steps=6):
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    if log:
        lo, hi = max(lo, 1e-6), max(hi, 1e-6)
        return np.logspace(np.log10(lo), np.log10(hi), steps)
    return np.linspace(lo, hi, steps)


def build_value_volume(visible, x_axis, y_axis, z_axis, log_x, w_cost, w_speed, w_intel, steps=6):
    med = visible[["cost", "intelligence", "speed"]].median()
    ticks = {
        x_axis: _axis_ticks(visible[x_axis], log=log_x and x_axis == "cost", steps=steps),
        y_axis: _axis_ticks(visible[y_axis], log=False, steps=steps),
        z_axis: _axis_ticks(visible[z_axis], log=False, steps=steps),
    }
    grids = {
        "cost": ticks.get("cost", np.full(steps, med["cost"])),
        "intelligence": ticks.get("intelligence", np.full(steps, med["intelligence"])),
        "speed": ticks.get("speed", np.full(steps, med["speed"])),
    }
    slot_for = {x_axis: 0, y_axis: 1, z_axis: 2}

    xx, yy, zz = np.meshgrid(ticks[x_axis], ticks[y_axis], ticks[z_axis], indexing="ij")
    clow_log = math.log10(max(float(visible["cost"].min()), 1e-6))
    chi_log = math.log10(max(float(visible["cost"].max()), 1e-6))
    vv = np.empty_like(xx, dtype=float)
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                idxs = (i, j, k)
                c = grids["cost"][idxs[slot_for.get("cost", 0)]]
                a = grids["intelligence"][idxs[slot_for.get("intelligence", 0)]]
                s = grids["speed"][idxs[slot_for.get("speed", 0)]]
                vv[i, j, k] = max(0.0, min(1.0, _score(c, a, s, clow_log, chi_log, w_cost, w_speed, w_intel)))

    return go.Volume(
        x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
        value=vv.ravel(),
        cmin=float(vv.min()), cmax=float(vv.max()),
        isomin=float(vv.min()), isomax=float(vv.max()),
        opacity=0.12,
        surface_count=45,
        colorscale=VALUE_SCALE,
        showscale=False,
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
    st.set_page_config(page_title="Model Compare 3D", page_icon="📊", layout="wide")

    if "custom_models" not in st.session_state:
        st.session_state.custom_models = []
    if "aa_key_input" not in st.session_state:
        st.session_state.aa_key_input = api.load_aa_key() or ""

    def _aa_key_cb():
        key = st.session_state.aa_key_input.strip()
        api.save_aa_key(key)

    catalog = get_catalog()
    st.sidebar.text_input(
        "Artificial Analysis API key (free tier works)",
        type="password",
        key="aa_key_input",
        on_change=_aa_key_cb,
        help="Get a free key: sign up at artificialanalysis.ai, then open "
             "https://artificialanalysis.ai/orgs/<your-username>/api-access, create a key, "
             "and paste it here. Saved to ~/.config/model-compare/aa_key.",
    )
    if st.sidebar.button("Clear saved key"):
        st.session_state.aa_key_input = ""
        api.save_aa_key("")
        st.rerun()

    aa_key = st.session_state.aa_key_input.strip() or None
    aa = get_aa(aa_key) if aa_key else None

    models = catalog + st.session_state.custom_models
    scored = api.apply_scores(models, aa=aa)
    df = pd.DataFrame(scored)

    with st.sidebar:
        st.header("⚙️ Controls")

        live = sum(1 for m in scored if m["scores_live"])
        st.caption(f"Source: models.dev — {len(df):,} models across {df['provider'].nunique():,} providers.")
        st.caption(f"Speed/Intelligence: {live:,} live from Artificial Analysis" if live else "Speed/Intelligence: estimates (add AA API key for live values)")

        st.subheader("📈 Chart type")
        chart_mode = st.radio(
            "3D needs WebGL. Auto picks the best option.",
            ["Auto", "3D (WebGL)", "2D (fallback)"],
            horizontal=True,
        )
        detected = webgl_check()
        st.caption(f"WebGL status: {'detecting…' if detected is None else detected}")
        if chart_mode == "Auto":
            chart_type = "2D (fallback)" if detected == "no" else "3D (WebGL)"
        else:
            chart_type = chart_mode
        if detected == "no" and chart_type == "3D (WebGL)":
            st.warning("⚠️ WebGL is disabled in this browser — 3D will not render.")

        st.divider()

        st.subheader("🎚️ Axes")
        x_axis = st.selectbox("X axis", list(AXES), index=0)
        y_axis = st.selectbox("Y axis", list(AXES), index=1)
        z_axis = st.selectbox("Z axis", list(AXES), index=2)
        log_x = st.checkbox("Log scale for cost", value=True)
        ball_size = st.radio("Ball size", ["Parameters", "Z-axis value", "Uniform"], horizontal=True)
        color_mode = st.radio("Color by", ["Value score", "Provider"], horizontal=True, index=1)
        show_field = st.checkbox("Show value field (3D gradient)", value=True)

        st.caption("⚖️ Value weights — tilt the gradient toward what matters")
        _TOTAL_W = 100
        for _k, _d in (("w_cost", 33), ("w_speed", 33), ("w_intel", 34)):
            if _k not in st.session_state:
                st.session_state[_k] = _d

        def _w_cb(kc):
            others = [k for k in ("w_cost", "w_speed", "w_intel") if k != kc]
            v = max(0, min(100, int(st.session_state[kc])))
            remaining = _TOTAL_W - v
            other_old = sum(st.session_state[o] for o in others)
            if other_old <= 0:
                for i, o in enumerate(others):
                    st.session_state[o] = remaining // len(others) + (1 if i < remaining % len(others) else 0)
            else:
                for o in others:
                    st.session_state[o] = round(st.session_state[o] / other_old * remaining)
                diff = _TOTAL_W - (v + sum(st.session_state[o] for o in others))
                biggest = max(others, key=lambda o: st.session_state[o])
                st.session_state[biggest] = max(0, st.session_state[biggest] + diff)
            st.rerun()

        st.slider("Cheapness (cost) weight", 0, _TOTAL_W, key="w_cost",
                  on_change=_w_cb, kwargs={"kc": "w_cost"})
        st.slider("Speed weight", 0, _TOTAL_W, key="w_speed",
                  on_change=_w_cb, kwargs={"kc": "w_speed"})
        st.slider("Intelligence weight", 0, _TOTAL_W, key="w_intel",
                  on_change=_w_cb, kwargs={"kc": "w_intel"})
        st.caption(f"Total: {st.session_state.w_cost + st.session_state.w_speed + st.session_state.w_intel} / 100")
        w_cost, w_speed, w_intel = st.session_state.w_cost, st.session_state.w_speed, st.session_state.w_intel

        st.divider()

        st.subheader("🔎 Filter")
        search = st.text_input("Search model name")
        providers = sorted(df["provider"].unique())

        show_all = st.checkbox("Show all providers", value=True)
        sel_providers = []
        if not show_all:
            with st.expander(f"Choose providers ({len(providers):,})"):
                prov_search = st.text_input("Search providers")
                opts = [p for p in providers if prov_search.lower() in p.lower()]
                sel_providers = st.multiselect("Providers to show (empty = all)", opts)
                st.caption(f"Found {len(opts):,} matching providers.")

        with st.expander(f"Browse all {len(providers):,} providers"):
            counts = df.groupby("provider").size().sort_values(ascending=False).rename("models")
            st.dataframe(counts, width="stretch")

        reasoning_only = st.checkbox("Reasoning models only", value=False)
        open_weights = st.checkbox("Open-weights only", value=False)
        min_context = st.slider("Min context (K tokens)", 0, 1024, 0, 16)

        st.divider()

        st.subheader("⭐ Highlight")
        hl_search = st.text_input("Search model to highlight")
        hl_opts = [n for n in sorted(df["name"].unique()) if hl_search.lower() in n.lower()]
        hl_names = st.multiselect("Models to highlight", hl_opts, max_selections=10)
        st.caption("Highlighted models render as large gold markers.")

        st.divider()

        st.subheader("➕ Add custom model")
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
    visible = visible.sort_values("intelligence", ascending=False).reset_index(drop=True)

    if visible.empty:
        st.warning("No models match the current filters.")
        return

    st.title("📊 3D Model Comparison")
    st.caption(f"{len(visible):,} models shown. Hover for details; drag to rotate.")

    color_map = {p: api.provider_color(p) for p in df["provider"].unique()}

    clow, chigh = float(visible["cost"].min()), float(visible["cost"].max())
    lo, hi = math.log10(max(clow, 1e-6)), math.log10(max(chigh, 1e-6))
    span = (hi - lo) or 1
    cheap = visible["cost"].apply(lambda c: 1 - min(1, max(0, (math.log10(max(c, 1e-6)) - lo) / span)))
    visible = visible.copy()
    wsum = max(w_cost + w_speed + w_intel, 1)
    visible["value"] = (w_cost * cheap + w_speed * (visible["speed"] - 1) / 9 + w_intel * (visible["intelligence"] - 1) / 9) / wsum
    visible["value"] = visible["value"].clip(0, 1)

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

    hover_data = {"provider": True, "cost": ":.2f", "speed": True, "intelligence": True,
                  "context": True, "reasoning": True, "params": ":.1f"}
    if live:
        hover_data["aa_intelligence_index"] = True
        hover_data["aa_tokens_per_sec"] = ":.1f"

    use_continuous = color_mode == "Value score"
    hl_mask = visible["name"].isin(hl_names) if hl_names else pd.Series(False, index=visible.index)
    base_opac = 0.18 if len(hl_names) else 0.45
    if chart_type == "3D (WebGL)":
        if use_continuous:
            fig = px.scatter_3d(visible, x=x_axis, y=y_axis, z=z_axis,
                                color="value", color_continuous_scale=VALUE_SCALE,
                                range_color=(0, 1),
                                hover_name="name", hover_data=hover_data,
                                text=None, title=None)
        else:
            fig = px.scatter_3d(visible, x=x_axis, y=y_axis, z=z_axis,
                                color="provider", color_discrete_map=color_map,
                                hover_name="name", hover_data=hover_data,
                                text=None, title=None)
        if show_field and len(visible) >= 4:
            fig.add_trace(build_value_volume(visible, x_axis, y_axis, z_axis, log_x, w_cost, w_speed, w_intel))
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
            fig = px.scatter(visible, x=x_axis, y=y_axis, color="value",
                             color_continuous_scale=VALUE_SCALE, range_color=(0, 1),
                             hover_name="name", hover_data=hover_data, title=None)
        else:
            fig = px.scatter(visible, x=x_axis, y=y_axis, color="provider", color_discrete_map=color_map,
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

    st.plotly_chart(fig, width="stretch")

    st.subheader("Table")
    st.dataframe(visible, width="stretch", hide_index=True)

    st.caption(f"Showing all {len(visible):,} filtered models.")


if __name__ == "__main__":
    main()
