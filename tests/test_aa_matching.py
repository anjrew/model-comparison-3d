"""Regression tests for provider-agnostic model matching and effort handling.

The core invariant: the same underlying model offered by different providers
must land at (nearly) the same place in the cost/speed/intelligence space,
regardless of how each provider spells the model id.

Run:
    .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_api as api


def _aa_record(slug, name, effort, idx, tps=None):
    return {
        "slug": slug,
        "name": name,
        "effort": effort,
        "intelligence_index": idx,
        "tokens_per_sec": tps,
        "cost_per_task": None,
        "total_cost": None,
        "input_price": None,
        "output_price": None,
    }


# Small, deterministic fixture that mimics Artificial Analysis' shape.
AA = {
    r["slug"]: r
    for r in [
        _aa_record("deepseek-v4-1-flash", "DeepSeek V4.1 Flash (Reasoning, Max Effort)", "max", 39.5, 194.33),
        _aa_record("kimi-k3", "Kimi K3 (max)", "max", 43.8, 150.0),
        _aa_record("kimi-k3-low", "Kimi K3 (low)", "low", 34.5, 160.0),
        _aa_record("hyperclova-x-seed-think-32b", "HyperCLOVA X SEED Think (32B)", None, 11.4, 80.0),
        _aa_record("claude-opus-4-5", "Claude Opus 4.5 (Non-reasoning)", "off", 23.7, 44.0),
        _aa_record("claude-opus-4-5-thinking", "Claude Opus 4.5 (Reasoning)", "reasoning", 29.1, 45.0),
        _aa_record("gpt-5", "GPT-5 (high)", "high", 23.0, 70.0),
        _aa_record("gpt-5-low", "GPT-5 (low)", "low", 15.0, 90.0),
    ]
}


def _model(mid, name, provider="P", reasoning=True, cost=1.0, cost_out=4.0, context=128000):
    return {
        "id": mid,
        "name": name,
        "provider": provider,
        "cost": cost,
        "cost_out": cost_out,
        "context": context,
        "params": None,
        "reasoning": reasoning,
        "open_weights": False,
        "effort_levels": [],
        "effort_style": "none",
    }


def _scored(models):
    return {r["id"]: r for r in api.apply_scores(models, aa=AA)}


class TestSameModelDifferentProviders(unittest.TestCase):
    def test_display_name_fallback_matches_when_id_does_not(self):
        """"opencode-go/deepseek-flash" only matches via its display name."""
        models = [
            _model("opencode-go/deepseek-flash", "DeepSeek V4.1 Flash", "OpenCode Go"),
            _model("openrouter/deepseek/deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "OpenRouter"),
        ]
        got = _scored(models)
        a, b = got[models[0]["id"]], got[models[1]["id"]]
        self.assertTrue(a["scores_live"] and b["scores_live"])
        self.assertEqual(a["intelligence"], b["intelligence"])
        self.assertEqual(a["speed"], b["speed"])
        self.assertEqual(a["intelligence"], round(39.5 / 7.0, 1))

    def test_short_model_names_do_not_cross_match(self):
        """"K3" must resolve to Kimi K3, never to a longer unrelated slug."""
        models = [
            _model("kimi-for-coding/k3", "Kimi K3"),
            _model("moonshotai/kimi-k3", "Kimi K3"),
        ]
        got = _scored(models)
        for m in models:
            r = got[m["id"]]
            self.assertTrue(r["scores_live"])
            self.assertEqual(r["aa_intelligence_index"], 43.8)
            self.assertEqual(r["intelligence"], round(43.8 / 7.0, 1))

    def test_version_adjacent_matches_stay_consistent(self):
        """Even when version parsing is ambiguous, providers must agree."""
        models = [
            _model("digitalocean/anthropic-claude-opus-4", "Claude Opus 4", "DigitalOcean"),
            _model("openrouter/anthropic/claude-opus-4", "Claude Opus 4", "OpenRouter"),
        ]
        got = _scored(models)
        a, b = got[models[0]["id"]], got[models[1]["id"]]
        self.assertEqual(a["intelligence"], b["intelligence"])
        self.assertEqual(a["speed"], b["speed"])
        self.assertEqual(a["aa_intelligence_index"], b["aa_intelligence_index"])


class TestConsistencyThreshold(unittest.TestCase):
    THRESHOLD = 0.2

    def test_same_name_group_positions_within_threshold(self):
        """Many providers, many id spellings, one model -> one position."""
        spellings = [
            "deepseek/deepseek-v4.1-flash",
            "openrouter/deepseek/deepseek-v4.1-flash",
            "venice/deepseek-v4-1-flash",
            "opencode-go/deepseek-flash",
            "crossmodel/deepseek/deepseek-v4.1-flash",
        ]
        models = [_model(mid, "DeepSeek V4.1 Flash", f"Provider {i}")
                  for i, mid in enumerate(spellings)]
        scored = list(_scored(models).values())
        intel = [r["intelligence"] for r in scored]
        speed = [r["speed"] for r in scored]
        self.assertLessEqual(max(intel) - min(intel), self.THRESHOLD)
        self.assertLessEqual(max(speed) - min(speed), self.THRESHOLD)


class TestEffortLadder(unittest.TestCase):
    def test_ladder_attached_and_ordered(self):
        got = _scored([_model("x/kimi-k3", "Kimi K3")])
        r = got["x/kimi-k3"]
        efforts = [v["effort"] for v in r["effort_ladder"] if v.get("effort")]
        self.assertEqual(set(efforts), {"low", "max"})

    def test_default_variant_follows_the_requested_level(self):
        got = _scored([
            _model("x/gpt-5", "GPT-5"),
            _model("x/gpt-5-low", "GPT-5 (low)"),
        ])
        self.assertEqual(got["x/gpt-5"]["aa_effort"], "high")
        self.assertEqual(got["x/gpt-5-low"]["aa_effort"], "low")

    def test_reasoning_variant_preferred_for_thinking_models(self):
        got = _scored([_model("x/claude-opus-4-5-thinking", "Claude Opus 4.5 Thinking")])
        r = got["x/claude-opus-4-5-thinking"]
        self.assertEqual(r["aa_effort"], "reasoning")
        self.assertEqual(r["aa_intelligence_index"], 29.1)


class TestReasoningOptionsParsing(unittest.TestCase):
    def test_effort_values_normalised_and_sorted(self):
        levels, style = api.parse_reasoning_options({
            "reasoning_options": [{"type": "effort", "values": ["none", "high", "low"]}],
        })
        self.assertEqual(levels, ["off", "low", "high"])
        self.assertEqual(style, "effort")

    def test_toggle_and_budget_styles(self):
        levels, style = api.parse_reasoning_options({
            "reasoning_options": [{"type": "toggle"}, {"type": "budget_tokens"}],
        })
        self.assertEqual(levels, [])
        self.assertEqual(style, "toggle+budget")


class TestRealCatalogConsistency(unittest.TestCase):
    """If a fresh local cache exists, verify same-name providers stay aligned.

    Skips (does not fail) when offline / no cached data, so it never hits the
    network. Artificial Analysis' free tier drops some model versions, so the
    threshold tolerates variant-level differences while catching gross mismatches.
    """

    INTEL_THRESHOLD = 2.0
    SPEED_THRESHOLD = 3.0

    @classmethod
    def setUpClass(cls):
        for path in (api.CACHE_FILE, api.AA_CACHE_FILE):
            if not os.path.exists(path):
                raise unittest.SkipTest("no local cache; skipping real-data check")
            if time.time() - os.path.getmtime(path) > api.CACHE_TTL:
                raise unittest.SkipTest("local cache is stale; skipping real-data check")
        cls.catalog = api.load_catalog()
        cls.aa = api.fetch_aa(api.load_aa_key())

    def _groups(self):
        scored = api.apply_scores(self.catalog, aa=self.aa)
        groups = {}
        for r in scored:
            if r["scores_live"]:
                groups.setdefault(r["name"].strip().lower(), []).append(r)
        return {k: v for k, v in groups.items() if len(v) >= 2}

    def test_live_same_name_intelligence_within_threshold(self):
        groups = self._groups()
        self.assertTrue(groups, "expected some multi-provider live models")
        worst = []
        for name, rows in groups.items():
            vals = [r["intelligence"] for r in rows if r["intelligence"] is not None]
            if len(vals) >= 2:
                worst.append((max(vals) - min(vals), name, len(rows)))
        worst.sort(reverse=True)
        worst_spread, worst_name, _ = worst[0]
        offenders = [w for w in worst if w[0] > self.INTEL_THRESHOLD]
        self.assertFalse(
            offenders,
            f"same-name live models diverge in intelligence (>"
            f"{self.INTEL_THRESHOLD}); worst: {offenders[:5]}",
        )
        print(f"\n[live] {len(groups)} multi-provider groups; worst intel spread "
              f"{worst_spread:.2f} ({worst_name})")

    def test_live_same_name_speed_within_threshold(self):
        groups = self._groups()
        offenders = []
        for name, rows in groups.items():
            vals = [r["speed"] for r in rows if r["speed"] is not None]
            if len(vals) >= 2 and max(vals) - min(vals) > self.SPEED_THRESHOLD:
                offenders.append((round(max(vals) - min(vals), 2), name, len(rows)))
        self.assertFalse(
            offenders,
            f"same-name live models diverge in speed (>{self.SPEED_THRESHOLD}): "
            f"{sorted(offenders, reverse=True)[:5]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
