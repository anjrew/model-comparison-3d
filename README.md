# 3D Model Comparison

An interactive Streamlit app that plots the best current LLMs across providers on a 3D chart comparing **cost**, **speed**, and **intelligence**.

## Features

- 3D scatter chart (Plotly) of models colored by provider
- **Auto WebGL detection**: falls back to a 2D bubble chart automatically when the browser has WebGL disabled (e.g. VMs, remote desktops)
- Swap any metric onto any axis (cost, speed, intelligence)
- Log scale toggle for cost
- Add and remove models and providers from the sidebar
- Reset to the default model catalog

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Metrics

- **Cost**: USD per 1M input tokens (approx.)
- **Speed**: relative output speed, 1-10
- **Intelligence**: coding/reasoning capability, 1-10
