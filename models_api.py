import json
import os
import re
import time
import urllib.request

MODELS_DEV_URL = "https://models.dev/api.json"
AA_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
CACHE_FILE = os.path.join(os.path.expanduser("~"), ".cache", "model_compare_modelsdev.json")
AA_CACHE_FILE = os.path.join(os.path.expanduser("~"), ".cache", "model_compare_aa.json")
AA_KEY_FILE = os.path.join(os.path.expanduser("~"), ".config", "model-compare", "aa_key")
CACHE_TTL = 24 * 60 * 60


def load_aa_key():
    try:
        if os.path.exists(AA_KEY_FILE):
            with open(AA_KEY_FILE) as f:
                return f.read().strip() or None
    except Exception:
        pass
    return None


def save_aa_key(key):
    key = (key or "").strip()
    try:
        os.makedirs(os.path.dirname(AA_KEY_FILE), exist_ok=True)
        with open(AA_KEY_FILE, "w") as f:
            f.write(key)
    except Exception:
        pass

PALETTE = [
    "#4D6BFE", "#D97757", "#10A37F", "#F43F5E", "#F59E0B", "#7C3AED",
    "#0EA5E9", "#84CC16", "#DB2777", "#14B8A6", "#E11D48", "#2563EB",
    "#CA8A04", "#059669", "#7F1D1D", "#4338CA", "#A21CAF", "#0F766E",
    "#B45309", "#1D4ED8", "#64748B", "#15803D", "#9A3412", "#4D4D4D",
]

_FRONTIER = ("opus", "sonnet", "gpt-5", "gpt 5", "codex", "claude 4", "gemini 3",
             "k2", "kimi", "deepseek", "v4", "v3.1", "qwen3", "glm-4.6", "grok 4")
_FAST = ("flash", "mini", "small", "haiku", "lite", "air", "nano", "turbo", "fast")


def _get_json(url, key=None, timeout=30, aa_key=False):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0 Safari/537.36")
    req.add_header("Accept", "application/json")
    if key:
        if aa_key:
            req.add_header("x-api-key", key)
        else:
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


_T_PARAM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*t(?:[^a-z]|$)")
_B_PARAM_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?:[^a-z]|$)")


def parse_params(name, mid):
    s = f"{name} {mid}".lower()
    ts = [float(x) for x in _T_PARAM_RE.findall(s)]
    bs = [float(x) for x in _B_PARAM_RE.findall(s)]
    vals = [t * 1000 for t in ts] + bs
    return max(vals) if vals else None


_FAMILY_COUNTRY = {
    "gpt": "US", "gpt-mini": "US", "gpt-nano": "US", "gpt-pro": "US", "gpt-codex": "US",
    "gpt-oss": "US", "gpt-image": "US", "gpt-sol": "US", "gpt-luna": "US", "gpt-terra": "US",
    "o": "US", "o-mini": "US",
    "claude-opus": "US", "claude-sonnet": "US", "claude-haiku": "US", "claude-fable": "US",
    "gemini": "US", "gemini-flash": "US", "gemini-pro": "US", "gemini-flash-lite": "US",
    "gemma": "US", "veo": "US", "imagen": "US",
    "llama": "US", "grok": "US", "grok-build": "US", "nemotron": "US",
    "phi": "US", "sonar": "US", "muse": "US", "laguna": "US", "auto": "US",
    "qwen": "CN", "qwen3.5": "CN", "qwen3.6": "CN",
    "glm": "CN", "glm-flash": "CN", "glm-air": "CN",
    "kimi-k2": "CN", "kimi-k3": "CN", "kimi-thinking": "CN",
    "minimax": "CN", "ernie": "CN", "seed": "CN", "mimo": "CN", "ling": "CN",
    "deepseek": "CN", "deepseek-thinking": "CN", "deepseek-flash": "CN",
    "mistral-small": "FR", "mistral-medium": "FR", "mistral-large": "FR",
    "mistral": "FR", "ministral": "FR", "mistral-nemo": "FR", "devstral": "FR",
    "command-r": "CA", "command-a": "CA",
    "flux": "DE",
    "jamba": "IL",
}

_NAME_COUNTRY_KEYWORDS = [
    ("claude", "US"), ("gpt", "US"), ("openai", "US"), ("gemini", "US"), ("gemma", "US"),
    ("llama", "US"), ("grok", "US"), ("nemotron", "US"), ("phi-", "US"), ("sonar", "US"),
    ("deepseek", "CN"), ("qwen", "CN"), ("glm", "CN"), ("kimi", "CN"), ("minimax", "CN"),
    ("ernie", "CN"), ("doubao", "CN"), ("seed", "CN"), ("mimo", "CN"), ("baichuan", "CN"),
    ("mistral", "FR"), ("ministral", "FR"), ("codestral", "FR"), ("devstral", "FR"),
    ("command", "CA"), ("cohere", "CA"),
    ("flux", "DE"),
    ("jamba", "IL"), ("falcon", "AE"),
    ("exaone", "KR"), ("hyperclova", "KR"),
]

_CONTINENT = {
    "US": "North America", "CA": "North America",
    "CN": "Asia", "IL": "Asia", "AE": "Asia", "KR": "Asia", "JP": "Asia", "IN": "Asia",
    "FR": "Europe", "DE": "Europe", "GB": "Europe", "NL": "Europe", "SE": "Europe",
}


def infer_country(name, family):
    if family in _FAMILY_COUNTRY:
        return _FAMILY_COUNTRY[family]
    nm = (name or "").lower()
    for kw, cc in _NAME_COUNTRY_KEYWORDS:
        if kw in nm:
            return cc
    return None


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
            name = m.get("name") or mid
            country = infer_country(name, m.get("family"))
            models.append({
                "id": f"{prov_id}/{mid}",
                "name": name,
                "provider": provider,
                "cost": inp,
                "cost_out": cost.get("output"),
                "context": ctx or 0,
                "params": parse_params(name, mid),
                "country": country,
                "continent": _CONTINENT.get(country, "Other") if country else "Other",
                "reasoning": bool(m.get("reasoning")),
                "open_weights": bool(m.get("open_weights")),
                "release_date": m.get("release_date"),
                "last_updated": m.get("last_updated"),
            })
    return models


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _intel_to_10(idx):
    return round(min(10.0, idx / 7.0), 1)


def _speed_to_10(ts):
    import math
    v = (math.log10(max(ts, 0.1)) - 0.5) * 2.5 + 4
    return round(max(1.0, min(10.0, v)), 1)


def fetch_aa(key):
    if os.path.exists(AA_CACHE_FILE) and time.time() - os.path.getmtime(AA_CACHE_FILE) < CACHE_TTL:
        try:
            with open(AA_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    models = {}
    page = 1
    while page <= 5:
        data = _get_json(f"{AA_URL}?page={page}", key=key, aa_key=True)
        for r in data.get("data", []):
            evals = r.get("evaluations") or {}
            perf = r.get("performance") or {}
            idx = evals.get("artificial_analysis_intelligence_index")
            ts = perf.get("median_output_tokens_per_second")
            if idx is None or ts is None:
                continue
            models[(r.get("slug") or "").lower()] = {
                "intelligence_index": idx,
                "tokens_per_sec": ts,
                "name": r.get("name") or "",
            }
        pag = data.get("pagination") or {}
        if not pag.get("has_more"):
            break
        page += 1
    try:
        with open(AA_CACHE_FILE, "w") as f:
            json.dump(models, f)
    except Exception:
        pass
    return models


def _match_aa(model, aa):
    target = _norm(model["id"].rsplit("/", 1)[-1]) or _norm(model.get("name") or "")
    if not target:
        return None
    best_v, best_score = None, -1
    for slug, v in aa.items():
        sn = _norm(slug)
        nn = _norm(v.get("name") or "")
        if sn and sn == target:
            return v
        if nn and nn == target:
            return v
        if len(sn) >= 5 and (sn in target or target in sn) and len(sn) > best_score:
            best_score = len(sn)
            best_v = v
    return best_v


def apply_scores(models, aa=None):
    out = []
    for m in models:
        row = dict(m)
        live = False
        intelligence = row.get("intelligence")
        speed = row.get("speed")
        if aa:
            hit = _match_aa(m, aa)
            if hit:
                intelligence = _intel_to_10(hit["intelligence_index"])
                speed = _speed_to_10(hit["tokens_per_sec"])
                row["aa_intelligence_index"] = hit["intelligence_index"]
                row["aa_tokens_per_sec"] = hit["tokens_per_sec"]
                live = True
        if intelligence is None:
            intelligence = estimate_intelligence(m)
            speed = estimate_speed(m)
        row["intelligence"] = intelligence
        row["speed"] = speed
        row["scores_live"] = live
        if row.get("params") is None:
            row["params"] = guess_params(intelligence)
            row["params_est"] = True
        else:
            row["params_est"] = False
        out.append(row)
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


def guess_params(intelligence):
    return round(10 ** ((intelligence - 5.0) * 0.55 + 0.4), 1)


def provider_color(provider, index=0):
    h = 0
    for ch in provider:
        h = (h * 31 + ord(ch)) % 1_000_000
    return PALETTE[h % len(PALETTE)]
