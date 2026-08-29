import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

import models_api as api

webgl_check = components.declare_component("webgl_check", path="frontend")

AXES = {
    "cost": "Cost ($/1M input tokens)",
    "speed": "Speed",
    "intelligence": "Intelligence",
    "context": "Context (tokens)",
}

DEFAULT_AXES = ["cost", "speed", "intelligence"]


@st.cache_data(ttl=3600, show_spinner="Loading model catalog from models.dev…")
def get_catalog(force=False):
    return api.load_catalog(force=force)


@st.cache_data(ttl=1800, show_spinner="Fetching Artificial Analysis scores…")
def get_aa(key):
    try:
        return api.fetch_aa(key)
    except Exception:
        return None


def main():
    st.set_page_config(page_title="Model Compare 3D", page_icon="📊", layout="wide")

    if "custom_models" not in st.session_state:
        st.session_state.custom_models = []

    catalog = get_catalog()
    aa_key = st.sidebar.text_input("Artificial Analysis API key (for live speed & intelligence)", type="password")
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
            chart_type = "3D (WebGL)" if detected == "yes" else "2D (fallback)"
        else:
            chart_type = chart_mode
        if detected == "no" and chart_type == "3D (WebGL)":
            st.warning("⚠️ WebGL is disabled in this browser — 3D will not render.")

        st.divider()

        st.subheader("🎚️ Axes")
        x_axis = st.selectbox("X axis", list(AXES), index=0)
        y_axis = st.selectbox("Y axis", list(AXES), index=DEFAULT_AXES.index("speed"))
        z_axis = st.selectbox("Z axis", list(AXES), index=DEFAULT_AXES.index("intelligence"))
        log_x = st.checkbox("Log scale for cost", value=False)

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

    hover_data = {"cost": ":.2f", "speed": True, "intelligence": True, "context": True, "reasoning": True}
    if chart_type == "3D (WebGL)":
        fig = px.scatter_3d(visible, x=x_axis, y=y_axis, z=z_axis,
                            color="provider", color_discrete_map=color_map,
                            hover_name="name", hover_data=hover_data,
                            text=None, title=None)
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis], zaxis_title=AXES[z_axis]),
                          legend_title="Provider", height=750, margin=dict(l=0, r=0, t=30, b=0))
        if log_x and x_axis == "cost":
            fig.update_layout(scene=dict(xaxis=dict(type="log")))
    else:
        fig = px.scatter(visible, x=x_axis, y=y_axis, color="provider", color_discrete_map=color_map,
                         hover_name="name", hover_data=hover_data, title=None)
        fig.update_traces(marker=dict(size=6), textposition="top center")
        fig.update_layout(xaxis_title=AXES[x_axis], yaxis_title=AXES[y_axis],
                          legend_title="Provider", height=750, margin=dict(l=0, r=0, t=30, b=0))
        if log_x and x_axis == "cost":
            fig.update_xaxes(type="log")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Table")
    st.dataframe(visible.head(500), width="stretch", hide_index=True)

    st.caption(f"Showing {min(len(visible), 500)} of {len(visible):,} models.")


if __name__ == "__main__":
    main()
