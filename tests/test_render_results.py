"""render_results_table consumes the current evaluator JSONL schema (finding 11)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "render_results_table", REPO / "scripts" / "render_results_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_renderer_consumes_current_evaluator_schema(tmp_path):
    mod = _load_module()
    lines = [
        {"checkpoint": "output/contact_damon_x/best.pth",
         "config": "configs/damon_baseline.yaml",
         "results": {"vertex": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "iou": 0.6,
                                "tp": 100, "fp": 20, "fn": 30, "tn": 500}}},
        {"checkpoint": "output/contact_climbing_x/best.pth",
         "config": "configs/climbing_baseline.yaml",
         "results": {"vertex": {"precision": 0.6, "recall": 0.5, "f1": 0.55, "iou": 0.4,
                                "tp": 50, "fp": 10, "fn": 40, "tn": 300}}},
    ]
    jsonl = tmp_path / "cross_eval.jsonl"
    jsonl.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    mod.JSONL, mod.PNG, mod.TEX = jsonl, tmp_path / "t.png", tmp_path / "t.tex"

    cells, nstr = mod.load()
    assert cells[("damon", "damon")]["f1"] == 0.75
    assert cells[("climbing", "climbing")]["precision"] == 0.6
    assert nstr["damon"] == 100 + 20 + 30 + 500

    mod.render_png(cells, nstr)          # must not KeyError on the new schema
    mod.render_tex(cells, nstr)          # (missing climbing_damon row -> "--")
    assert mod.PNG.exists() and mod.TEX.exists()
