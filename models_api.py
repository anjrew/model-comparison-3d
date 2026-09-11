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
AA_CACHE_VERSION = 2

EFFORT_ORDER = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
_EFFORT_ALIASES = {
    "none": "off", "non-reasoning": "off", "noreasoning": "off", "off": "off",
    "minimal": "minimal", "low": "low", "medium": "medium",
    "high": "high", "xhigh": "xhigh", "x-high": "xhigh", "max": "max",
}


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
            effort_levels, effort_style = parse_reasoning_options(m)
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
                "effort_levels": effort_levels,
                "effort_style": effort_style,
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


def normalize_effort(v):
    return _EFFORT_ALIASES.get(str(v).strip().lower())


def parse_reasoning_options(m):
    levels, styles = [], set()
    for o in (m.get("reasoning_options") or []):
        if not isinstance(o, dict):
            continue
        t = o.get("type")
        if t == "effort":
            styles.add("effort")
            for v in (o.get("values") or []):
                e = normalize_effort(v)
                if e and e not in levels:
                    levels.append(e)
        elif t in ("toggle", "budget_tokens"):
            styles.add(t)
    levels.sort(key=lambda e: EFFORT_ORDER.index(e))
    if "effort" in styles:
        style = "effort"
    elif "toggle" in styles and "budget_tokens" in styles:
        style = "toggle+budget"
    elif "toggle" in styles:
        style = "toggle"
    elif "budget_tokens" in styles:
        style = "budget_tokens"
    else:
        style = "none"
    return levels, style


_AA_EFFORT_TOKENS = [
    ("non-reasoning", "off"),
    ("xhigh", "xhigh"),
    ("minimal", "minimal"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
    ("max", "max"),
]


def parse_aa_effort(name):
    m = re.search(r"\(([^)]*)\)", name or "")
    if not m:
        return None
    inside = m.group(1).lower()
    for tok, eff in _AA_EFFORT_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", inside):
            return eff
    if "reasoning" in inside:
        return "reasoning"
    return None


def _aa_base_name(name):
    n = re.sub(r"\s*\([^)]*\)", "", name or "").strip().lower()
    return re.sub(r"\s*(non-reasoning|reasoning|thinking)\s*$", "", n).strip()


def build_aa_index(aa):
    """Precompute AA lookup structures once (groups, exact map, fuzzy candidates)."""
    groups, exact, entries = {}, {}, []
    if not aa:
        return groups, exact, entries
    for rec in aa.values():
        if not isinstance(rec, dict):
            continue
        name = rec.get("name") or rec.get("slug") or ""
        key = _norm(_aa_base_name(name))
        if key:
            groups.setdefault(key, []).append(rec)
    for key, recs in groups.items():
        recs.sort(key=lambda r: (
            r.get("effort") not in EFFORT_ORDER,
            EFFORT_ORDER.index(r["effort"]) if r.get("effort") in EFFORT_ORDER else 99,
        ))
        for rec in recs:
            sn = _norm(rec.get("slug") or "")
            nn = _norm(rec.get("name") or "")
            if sn:
                exact.setdefault(sn, key)
            if nn:
                exact.setdefault(nn, key)
            entries.append((sn, key))
    return groups, exact, entries


def _fuzzy_hit(target, entries, guarded=False):
    best_key, best_score = None, -1
    for sn, k in entries:
        if len(sn) < 5:
            continue
        if sn in target:
            score = len(sn)
        elif len(target) >= 5 and target in sn:
            if guarded:
                lo, hi = sorted((len(sn), len(target)))
                if hi - lo > max(4, 0.25 * hi):
                    continue
            score = len(target)
        else:
            continue
        if score > best_score:
            best_score = score
            best_key = k
    return best_key, best_score


def _best_match(target, groups, exact, entries, guarded=False):
    key = exact.get(target)
    if key:
        return groups[key], 1_000_000 + len(target)
    k, score = _fuzzy_hit(target, entries, guarded)
    return (groups[k], score) if k else (None, -1)


def match_aa_ladder(model, aa=None, index=None):
    if index is None:
        index = build_aa_index(aa)
    groups, exact, entries = index
    targets = ((_norm(model["id"].rsplit("/", 1)[-1]), False),
               (_norm(model.get("name") or ""), True))
    best, best_q = [], -1
    for target, guarded in targets:
        if not target:
            continue
        hit, q = _best_match(target, groups, exact, entries, guarded)
        if hit and q > best_q:
            best, best_q = hit, q
    return best


def _name_match_quality(target, sn, nn):
    best = -1
    for cand in (sn, nn):
        if not cand:
            continue
        if cand == target:
            return 1_000_000 + len(cand)
        if len(cand) < 5:
            continue
        if cand in target:
            best = max(best, len(cand))
        elif len(target) >= 5 and target in cand:
            best = max(best, len(target))
    return best


def _variant_score(target, rec):
    return _name_match_quality(target, _norm(rec.get("slug") or ""),
                               _norm(rec.get("name") or ""))


def _pick_default_variant(model, ladder):
    if not ladder:
        return None
    targets = [t for t in (_norm(model["id"].rsplit("/", 1)[-1]),
                            _norm(model.get("name") or "")) if t]
    for target in targets:
        if any(_variant_score(target, r) >= 0 for r in ladder):
            return max(ladder, key=lambda r: _variant_score(target, r))
    return ladder[0]


def fetch_aa(key):
    if os.path.exists(AA_CACHE_FILE) and time.time() - os.path.getmtime(AA_CACHE_FILE) < CACHE_TTL:
        try:
            with open(AA_CACHE_FILE) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached.get("_v") == AA_CACHE_VERSION:
                return cached["models"]
        except Exception:
            pass
    models = {}
    page = 1
    while page <= 5:
        data = _get_json(f"{AA_URL}?page={page}", key=key, aa_key=True)
        for r in data.get("data", []):
            evals = r.get("evaluations") or {}
            perf = r.get("performance") or {}
            pricing = r.get("pricing") or {}
            idx = evals.get("artificial_analysis_intelligence_index")
            if idx is None:
                continue
            ts = perf.get("median_output_tokens_per_second")
            slug = (r.get("slug") or "").lower()
            cost = r.get("artificial_analysis_intelligence_index_cost") or {}
            name = r.get("name") or ""
            models[slug] = {
                "intelligence_index": idx,
                "tokens_per_sec": ts,
                "name": name,
                "slug": slug,
                "effort": parse_aa_effort(name),
                "cost_per_task": (cost.get("cost_per_task") or {}).get("total_cost"),
                "total_cost": cost.get("total_cost"),
                "input_price": pricing.get("price_1m_input_tokens"),
                "output_price": pricing.get("price_1m_output_tokens"),
            }
        pag = data.get("pagination") or {}
        if not pag.get("has_more"):
            break
        page += 1
    try:
        with open(AA_CACHE_FILE, "w") as f:
            json.dump({"_v": AA_CACHE_VERSION, "models": models}, f)
    except Exception:
        pass
    return models


def _ladder_entry(rec):
    idx = rec.get("intelligence_index")
    ts = rec.get("tokens_per_sec")
    return {
        "effort": rec.get("effort"),
        "name": rec.get("name"),
        "slug": rec.get("slug"),
        "intelligence": _intel_to_10(idx) if idx is not None else None,
        "speed": _speed_to_10(ts) if ts is not None else None,
        "aa_intelligence_index": idx,
        "aa_tokens_per_sec": ts,
        "cost_per_task": rec.get("cost_per_task"),
        "input_price": rec.get("input_price"),
        "output_price": rec.get("output_price"),
    }


def apply_scores(models, aa=None):
    out = []
    index = build_aa_index(aa)
    for m in models:
        row = dict(m)
        preset = bool(row.get("scores_live"))
        live = False
        intelligence = row.get("intelligence")
        speed = row.get("speed")
        ladder = match_aa_ladder(m, index=index) if aa else []
        row["effort_ladder"] = [_ladder_entry(r) for r in ladder]
        if aa:
            hit = _pick_default_variant(m, ladder)
            if hit:
                intelligence = _intel_to_10(hit["intelligence_index"])
                row["aa_intelligence_index"] = hit["intelligence_index"]
                row["aa_effort"] = hit.get("effort")
                if hit.get("tokens_per_sec") is not None:
                    speed = _speed_to_10(hit["tokens_per_sec"])
                    row["aa_tokens_per_sec"] = hit["tokens_per_sec"]
                live = True
        if intelligence is None:
            intelligence = estimate_intelligence(m)
        if speed is None:
            speed = estimate_speed(m)
        row["intelligence"] = intelligence
        row["speed"] = speed
        row["scores_live"] = live or preset
        if row.get("params") is None:
            row["params"] = guess_params(intelligence)
            row["params_est"] = True
        else:
            row["params_est"] = False
        row.setdefault("effort_levels", [])
        row.setdefault("effort_style", "none")
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
