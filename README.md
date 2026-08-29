# 3D Model Comparison

Interactive Streamlit app comparing **7,000+ LLMs across 200+ providers** on a 3D chart.

Plot any of these on the axes: **cost, speed, intelligence, context**.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Data

| Metric | Source | Auth |
| --- | --- | --- |
| Catalog, pricing, context | [models.dev](https://models.dev) API | keyless, cached 24h |
| Speed & intelligence | [Artificial Analysis](https://artificialanalysis.ai) API | free key (optional) |
| Speed & intelligence (fallback) | local estimates | — |

Models without published parameter counts get an estimate based on intelligence (shown as such in the table).

## Free Artificial Analysis key (optional, recommended)

Makes speed & intelligence live instead of estimated. Free tier: 100 requests/day.

1. Sign up at [artificialanalysis.ai](https://artificialanalysis.ai)
2. Open `https://artificialanalysis.ai/orgs/<your-username>/api-access`
3. Create a key and paste it in the sidebar once — it's saved to `~/.config/model-compare/aa_key` and auto-loaded next launch

## Chart controls

- **Color by** — `Value score` (green = cheap + smart + fast, red = expensive + dumb + slow) or `Provider`
- **Ball size** — parameters, z-axis value, or uniform
- **Chart type** — Auto picks 3D when WebGL works, otherwise falls back to 2D
- **Filters** — search, provider, reasoning-only, open-weights, min context
- **Axes** — swap cost/speed/intelligence/context, log-scale cost
