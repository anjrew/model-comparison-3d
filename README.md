# 3D Model Comparison

Interactive Streamlit app plotting **7,000+ LLMs across 200+ providers** from the
[models.dev](https://models.dev) API on a 3D chart — **cost, speed, intelligence**.

## Features

- **Live data**: full model catalog + real pricing/context fetched from the models.dev API (keyless, cached 24h)
- **Live speed & intelligence**: optionally provide an Artificial Analysis API key for real benchmark values; otherwise transparently-labeled estimates
- 3D scatter (Plotly) with auto WebGL detection and a 2D fallback for browsers/VMs without WebGL
- Swap any metric onto any axis (cost, speed, intelligence, context), log-scale for cost
- Filter by search, provider, reasoning-only, open-weights, min context
- Add/remove your own custom models

## Getting an Artificial Analysis API key (free)

Live speed & intelligence values come from [Artificial Analysis](https://artificialanalysis.ai).
The **free tier** includes the intelligence index, speed, and pricing — 100 requests per day is plenty.

1. Create an account / sign in at [artificialanalysis.ai](https://artificialanalysis.ai)
2. Open your API access page: `https://artificialanalysis.ai/orgs/<your-username>/api-access`
3. Create a key and copy it
4. Paste it into the app's sidebar once — it is saved to `~/.config/model-compare/aa_key` and auto-loaded next launch

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Data sources

| Metric | Source | Auth |
| --- | --- | --- |
| Catalog, pricing, context | [models.dev](https://models.dev) API | keyless |
| Speed, intelligence | [Artificial Analysis](https://artificialanalysis.ai) API | optional key |
| Speed, intelligence (fallback) | local estimates | — |
