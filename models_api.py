import json
import os
import time
import urllib.request

MODELS_DEV_URL = "https://models.dev/api.json"
AA_URL = "https://artificialanalysis.ai/api/v1/models"
CACHE_FILE = os.path.join(os.path.expanduser("~"), ".cache", "model_compare_modelsdev.json")
CACHE_TTL = 24 * 60 * 60

PALETTE = [
    "#4D6BFE", "#D97757", "#10A37F", "#F43F5E", "#F59E0B", "#7C3AED",
    "#0EA5E9", "#84CC16", "#DB2777", "#14B8A6", "#E11D48", "#2563EB",
    "#CA8A04", "#059669", "#7F1D1D", "#4338CA", "#A21CAF", "#0F766E",
    "#B45309", "#1D4ED8", "#64748B", "#15803D", "#9A3412", "#4D4D4D",
]

_FRONTIER = ("opus", "sonnet", "gpt-5", "gpt 5", "codex", "claude 4", "gemini 3",
             "k2", "kimi", "deepseek", "v4", "v3.1", "qwen3", "glm-4.6", "grok 4")
_FAST = ("flash", "mini", "small", "haiku", "lite", "air", "nano", "turbo", "fast")


def _get_json(url, key=None, timeout=30):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0 Safari/537.36")
    req.add_header("Accept", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_models_dev(force=False):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if not force and os.path.exists(CACHE_FILE) and time.time() - os.path.getmtime(CACHE_FILE) < CACHE_TTL:
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    data = _get_json(MODELS_DEV_URL)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)
    return data


def load_catalog(force=False):
    data = fetch_models_dev(force=force)
    models = []
    for prov_id, prov in data.items():
        provider = prov.get("name") or prov_id
        for mid, m in prov.get("models", {}).items():
            cost = m.get("cost") or {}
            inp = cost.get("input")
            if inp is None:
                continue
            ctx = (m.get("limit") or {}).get("context")
            models.append({
                "id": f"{prov_id}/{mid}",
                "name": m.get("name") or mid,
                "provider": provider,
                "cost": inp,
                "cost_out": cost.get("output"),
                "context": ctx or 0,
                "reasoning": bool(m.get("reasoning")),
                "open_weights": bool(m.get("open_weights")),
                "release_date": m.get("release_date"),
                "last_updated": m.get("last_updated"),
            })
    return models


def fetch_aa(key):
    data = _get_json(AA_URL, key=key)
    rows = data.get("models") or data.get("data") or []
    out = {}
    for r in rows:
        mid = r.get("id") or r.get("model") or ""
        if not mid:
            continue
        out[mid] = {
            "intelligence": r.get("intelligence_index") or r.get("intelligence") or r.get("intelligence_index_percent"),
            "speed": r.get("output_speed") or r.get("speed"),
        }
    return out


def estimate_intelligence(m):
    s = 6.2
    if m["reasoning"]:
        s += 1.0
    name = m["name"].lower()
    if any(k in name for k in _FRONTIER):
        s += 0.8
    if m["cost"] >= 2.5:
        s += 0.5
    s = min(10.0, s)
    return round(s, 1)


def estimate_speed(m):
    s = 7.0
    if m["reasoning"]:
        s -= 1.5
    name = m["name"].lower()
    if any(k in name for k in _FAST):
        s += 1.5
    if m["context"] and m["context"] > 200000:
        s -= 0.5
    s = max(1.0, min(10.0, s))
    return round(s, 1)


def apply_scores(models, aa=None):
    out = []
    for m in models:
        row = dict(m)
        if aa:
            hit = aa.get(m["id"]) or aa.get(m["id"].split("/", 1)[-1])
            if hit and hit.get("intelligence") is not None and hit.get("speed") is not None:
                row["intelligence"] = round(float(hit["intelligence"]), 1)
                row["speed"] = round(float(hit["speed"]), 1)
                row["scores_live"] = True
                out.append(row)
                continue
        row["intelligence"] = estimate_intelligence(m)
        row["speed"] = estimate_speed(m)
        row["scores_live"] = False
        out.append(row)
    return out


def provider_color(provider, index=0):
    h = 0
    for ch in provider:
        h = (h * 31 + ord(ch)) % 1_000_000
    return PALETTE[h % len(PALETTE)]
