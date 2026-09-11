"""Print one metric table over several runs from their ``eval.json`` files.

    python scripts/eval_table.py output_2/final_*

One row per run directory (its ``eval.json``; runs without one are listed as
pending), the headline metrics of every branch in one line, plus the per-group
contact F1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COLUMNS = (
    ("f1", "metric_contact/f1"), ("P", "metric_contact/precision"), ("R", "metric_contact/recall"),
    ("P@R90", "metric_contact/precision_at_r90"),
    ("mpjpe", "metric_pose/mpjpe"), ("pa", "metric_pose/pa_mpjpe"), ("pve", "metric_pose/pve"),
    ("accel", "metric_pose/accel"), ("pelvis", "metric_pose/pelvis_err"),
    ("wa100", "metric_pose/lifted_wa_mpjpe100"), ("w100", "metric_pose/lifted_w_mpjpe100"),
    ("jitter", "metric_pose/lifted_jitter"),
    ("f_mae", "metric_force/mae"), ("angle", "metric_force/angle_deg"), ("f_off", "metric_force/noncontact_mag"),
    ("rnea_f", "metric_force_consistency/force"), ("rnea_t", "metric_force_consistency/torque"),
    ("vel_r", "metric_motion/vel_pearson"), ("acc_r", "metric_motion/acc_pearson"),
    ("still", "metric_contact_consistency/speed"),
)
GROUPS = ("left_hand", "right_hand", "left_foot", "right_foot", "left_ankle", "right_ankle")


def _cell(metrics: dict, tag: str, width: int = 7) -> str:
    value = metrics.get(tag)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return " " * (width - 1) + "-"
    return f"{value:{width}.3f}" if abs(value) < 10 else f"{value:{width}.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--groups", action="store_true", help="per-group contact F1 too")
    args = parser.parse_args()
    name_width = max(len(run.name) for run in args.runs)
    header = f"{'run':<{name_width}s} " + " ".join(f"{c:>7s}" for c, _ in COLUMNS)
    if args.groups:
        header += "  " + " ".join(f"{g[:6]:>6s}" for g in GROUPS)
    print(header)
    for run in args.runs:
        path = run / "eval.json"
        if not path.is_file():
            print(f"{run.name:<{name_width}s} (pending)")
            continue
        metrics = json.loads(path.read_text())["metrics"]
        line = f"{run.name:<{name_width}s} " + " ".join(_cell(metrics, tag) for _, tag in COLUMNS)
        if args.groups:
            line += "  " + " ".join(_cell(metrics, f"metric_contact/groups/{g}_f1", 6) for g in GROUPS)
        print(line)


if __name__ == "__main__":
    main()
