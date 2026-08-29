import streamlit as st
import pandas as pd
import plotly.express as px

from models_data import DEFAULT_MODELS, PROVIDER_COLORS

st.set_page_config(page_title="Model Compare 3D", page_icon="📊", layout="wide")


@st.cache_data
def color_for(provider):
    return PROVIDER_COLORS.get(provider, PROVIDER_COLORS["Other"])


def ensure_state():
    if "models" not in st.session_state:
        st.session_state.models = [dict(m) for m in DEFAULT_MODELS]
    if "hidden_providers" not in st.session_state:
        st.session_state.hidden_providers = set()


def main():
    ensure_state()
    models = st.session_state.models
    df = pd.DataFrame(models)

    st.title("📊 3D Model Comparison")
    st.caption("Compare the best current LLMs across providers on **cost, speed, and intelligence**.")

    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("Providers")
        providers = sorted(df["provider"].unique())
        all_on = st.checkbox("Select all providers", value=True)
        checked = {}
        for p in providers:
            default = p not in st.session_state.hidden_providers and all_on
            checked[p] = st.checkbox(p, value=default)

        hidden = {p for p, v in checked.items() if not v}
        st.session_state.hidden_providers = hidden
        visible = df[~df["provider"].isin(hidden)]

        st.divider()

        st.subheader("Axes")
        axis_labels = {"cost": "Cost ($/1M input tokens)", "speed": "Speed (relative)", "intelligence": "Intelligence (1-10)"}
        x_axis = st.selectbox("X axis", ["cost", "speed", "intelligence"], index=0)
        y_axis = st.selectbox("Y axis", ["speed", "cost", "intelligence"], index=0)
        z_axis = st.selectbox("Z axis", ["intelligence", "cost", "speed"], index=0)
        log_x = st.checkbox("Log scale for cost", value=False)

        st.divider()

        st.subheader("📈 Chart type")
        chart_type = st.radio(
            "3D needs WebGL. If the chart fails to render, pick 2D.",
            ["3D (WebGL)", "2D (fallback)"],
            horizontal=True,
        )

        st.divider()

        st.subheader("➕ Add model")
        with st.form("add_model", clear_on_submit=True):
            name = st.text_input("Model name")
            provider = st.text_input("Provider")
            cost = st.number_input("Cost ($/1M input tokens)", min_value=0.0, step=0.01)
            speed = st.slider("Speed (1-10)", 1, 10, 7)
            intelligence = st.slider("Intelligence (1-10)", 1.0, 10.0, 7.0, 0.1)
            submitted = st.form_submit_button("Add model")
            if submitted:
                if name and provider:
                    st.session_state.models.append(
                        {"name": name, "provider": provider,
                         "cost": cost, "speed": speed, "intelligence": intelligence}
                    )
                    st.success(f"Added {name}")
                    st.rerun()

        st.divider()

        st.subheader("🗑️ Remove model")
        to_remove = st.selectbox("Pick a model", options=[""] + [m["name"] for m in models])
        if st.button("Remove model", disabled=not to_remove):
            st.session_state.models = [m for m in st.session_state.models if m["name"] != to_remove]
            st.rerun()

        if st.button("Reset to defaults"):
            st.session_state.models = [dict(m) for m in DEFAULT_MODELS]
            st.session_state.hidden_providers = set()
            st.rerun()

    if visible.empty:
        st.warning("No models to show. Enable a provider or add a model.")
        return

    if chart_type == "3D (WebGL)":
        fig = px.scatter_3d(
            visible,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color="provider",
            color_discrete_map=PROVIDER_COLORS,
            hover_name="name",
            hover_data={"cost": ":.2f", "speed": True, "intelligence": True},
            text="name",
            title=None,
        )

        fig.update_traces(marker=dict(size=8), textposition="top center")
        fig.update_layout(
            scene=dict(
                xaxis_title=axis_labels[x_axis],
                yaxis_title=axis_labels[y_axis],
                zaxis_title=axis_labels[z_axis],
            ),
            legend_title="Provider",
            height=750,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        if log_x and x_axis == "cost":
            fig.update_layout(scene=dict(xaxis=dict(type="log")))
    else:
        fig = px.scatter(
            visible,
            x=x_axis,
            y=y_axis,
            color="provider",
            color_discrete_map=PROVIDER_COLORS,
            hover_name="name",
            hover_data={"cost": ":.2f", "speed": True, "intelligence": True},
            text="name",
            title=None,
            height=750,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title=axis_labels[x_axis],
            yaxis_title=axis_labels[y_axis],
            legend_title="Provider",
            height=750,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig.add_annotation(
            text=f"{axis_labels[z_axis]} shown by bubble size",
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
            font=dict(size=12), xanchor="left",
        )
        fig.update_traces(marker=dict(size=[m[z_axis] * 14 for _, m in visible.iterrows()]))
        if log_x and x_axis == "cost":
            fig.update_xaxes(type="log")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Table")
    st.dataframe(visible.sort_values("intelligence", ascending=False), width="stretch", hide_index=True)

    st.caption(f"Showing {len(visible)} models across {visible['provider'].nunique()} providers.")


if __name__ == "__main__":
    main()
