"""Tests for the effort-ladder 3D figure builder."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import app


def _row(effort, intel, speed, cost):
    return {"effort": effort, "intelligence": intel, "speed": speed, "cost": cost,
            "cost_source": "Live (AA)", "aa_cost": cost / 10, "value": intel / 10}


def _fixture():
    rows = [_row("off", 3.0, 9.0, 1.0),
            _row("medium", 5.0, 8.0, 2.0),
            _row("max", 7.0, 7.0, 4.0)]
    per_model = [("Model A", {"provider": "TestProv", "context": 100000}, rows, "medium")]
    points = [{"name": "Model A", "provider": "TestProv", "effort": r["effort"],
               "cost": r["cost"], "speed": r["speed"], "intelligence": r["intelligence"],
               "context": 100000, "params": 10, "cost_source": r["cost_source"],
               "aa_cost": r["aa_cost"], "value": r["value"]} for r in rows]
    pts = pd.DataFrame(points)
    sizes = pd.Series([10.0, 20.0, 30.0], index=pts.index)
    return per_model, pts, sizes


BASE_CFG = {
    "x_axis": "cost", "y_axis": "speed", "z_axis": "intelligence",
    "chart_type": "3D (WebGL)", "ball_size": "Context", "size_scale": "Log",
    "ball_max": 60, "log_exp": 1.0,
}


class TestEffortFigure(unittest.TestCase):
    def test_3d_lines_one_trace_per_model(self):
        per_model, pts, sizes = _fixture()
        fig = app.build_effort_figure(per_model, pts, sizes, BASE_CFG, "Effort level", True)
        self.assertEqual(len(fig.data), 1)
        tr = fig.data[0]
        self.assertEqual(tr.type, "scatter3d")
        self.assertEqual(tr.mode, "lines+markers")
        self.assertEqual(len(tr.x), 3)
        self.assertEqual(len(tr.z), 3)
        self.assertEqual(tr.name, "Model A")

    def test_points_ordered_by_effort(self):
        per_model, pts, sizes = _fixture()
        fig = app.build_effort_figure(per_model, pts, sizes, BASE_CFG, "Effort level", True)
        tr = fig.data[0]
        # off -> medium -> max: cost rises, intelligence rises, speed falls
        self.assertEqual(list(tr.x), [1.0, 2.0, 4.0])
        self.assertEqual(list(tr.y), [9.0, 8.0, 7.0])
        self.assertEqual(list(tr.z), [3.0, 5.0, 7.0])

    def test_marker_colors_follow_effort(self):
        per_model, pts, sizes = _fixture()
        fig = app.build_effort_figure(per_model, pts, sizes, BASE_CFG, "Effort level", True)
        self.assertEqual(list(fig.data[0].marker.color),
                         [app.EFFORT_COLOR_MAP[e] for e in ("off", "medium", "max")])

    def test_2d_fallback_uses_scatter(self):
        per_model, pts, sizes = _fixture()
        cfg = dict(BASE_CFG, chart_type="2D (fallback)")
        fig = app.build_effort_figure(per_model, pts, sizes, cfg, "Provider", True)
        self.assertEqual(fig.data[0].type, "scatter")

    def test_axes_follow_configuration(self):
        per_model, pts, sizes = _fixture()
        cfg = dict(BASE_CFG, x_axis="intelligence", y_axis="cost", z_axis="speed")
        fig = app.build_effort_figure(per_model, pts, sizes, cfg, "Effort level", False)
        tr = fig.data[0]
        self.assertEqual(list(tr.x), [3.0, 5.0, 7.0])
        self.assertEqual(list(tr.y), [1.0, 2.0, 4.0])
        self.assertEqual(list(tr.z), [9.0, 8.0, 7.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
