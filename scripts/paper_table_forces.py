"""Paper table: climb_wall_3 and LAAS Parkour, one row per method, from the two scorers' markdown.

Reads ``scripts/score_rig.py``'s table (``--rig``) and ``scripts/score_parkour_paper.py``'s
report (``--parkour``) and prints the LaTeX ``tabular`` body of the combined table: MPJPE,
the per-limb force MAE (L foot, R foot, L hand, R hand) and their mean, the angle /
correlation / load-share agreement and the RNEA residual. Each half keeps its own protocol:
the wall rows are the rig table's pooled columns (camera-frame MPJPE, newtons in the
reconstructed mass); the Parkour rows are the paper's (Procrustes MPJPE averaged over clips,
per-channel force error in the generic 74.6 kg body, the agreement columns pooled over the
captured channel-frames). Li's Parkour ``++`` row is his own evaluator's output; its residual
is the same run's dump round-trip (``estmf_rec``). The best value of a column is bold per
half; the optimisation's residuals are its solver's own and never bold.

    .venv/bin/python scripts/paper_table_forces.py --rig output_7/logs/rig_paper_<date>.md \\
        --parkour output_7/logs/parkour_paper_<date>.md --out output_7/logs/table_forces_<date>.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Table columns: (name, decimals).
COLUMNS = (("MPJPE", 1), ("L foot", 1), ("R foot", 1), ("L hand", 1), ("R hand", 1), ("Avg", 1),
           ("Ang", 1), ("Corr", 3), ("Share", 1), ("For", 3), ("Mom", 3))
#: Columns where the best value is the largest.
HIGHER = frozenset(("Corr",))
#: The optimisation's residual is its own solver's: never bold.
OWN_RESIDUAL = ("Optimization", ("For", "Mom"))
#: Wall rows: paper label -> rig table row; every cell comes from that row.
WALL_ROWS = (("SAM 3D Body", "SAM 3D Body"), (r"\citet{li2019_forces}++", "estmf_rec"),
             ("PhysPT++", "physpt"), ("Inverse dynamics", "baseline_rnea"),
             ("Optimization", "optimisation"), (r"\ours", "joint_frames35"))
#: Rig table column of each paper column.
WALL_COLUMNS = {"MPJPE": "MPJPE", "L foot": "LF MAE N", "R foot": "RF MAE N", "L hand": "LH MAE N",
                "R hand": "RH MAE N", "Avg": "vec MAE N", "Ang": "angle", "Corr": "corr",
                "Share": "share pp", "For": "RNEA f bw", "Mom": "RNEA tau bw·m"}
#: Parkour rows: paper label -> (report row for pose / force / agreement, report row for RNEA).
LI_RERUN = ("Li et al. 2019 (rerun: our SAM-3D init, Sapiens 2D, their recogniser, 25 fps)"
            " - own evaluator")
PARKOUR_ROWS = (("SAM 3D Body", "SAM 3D Body", "SAM 3D Body"),
                (r"\citet{li2019_forces}", "Li et al. 2019 (paper)", None),
                (r"\citet{li2019_forces}++", LI_RERUN,
                 "estmf_rec (the same Li rerun through our dump round-trip)"),
                ("PhysPT++", "physpt", "physpt"),
                ("Inverse dynamics", "baseline_rnea", "baseline_rnea"),
                ("Optimization", "optimisation (scene-free)", "optimisation (scene-free)"),
                (r"\ours", "joint_frames35", "joint_frames35"))
PARKOUR_FORCE = {"L foot": "l_sole", "R foot": "r_sole", "L hand": "l_hand", "R hand": "r_hand"}
PARKOUR_RIG = {"Ang": "angle", "Corr": "corr", "Share": "share pp"}
PARKOUR_RNEA = {"For": "RNEA f bw", "Mom": "RNEA tau bw·m"}


def parse_tables(text: str) -> dict[str, dict[str, dict[str, float]]]:
    """Every markdown table of ``text`` keyed by the ``## `` heading above it (``""`` if none)."""
    tables: dict[str, dict[str, dict[str, float]]] = {}
    heading, columns = "", None
    for line in text.splitlines():
        if line.startswith("## "):
            heading, columns = line[3:].strip(), None
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if columns is None:
                columns = cells[1:]
                tables[heading] = {}
            elif not set(cells[0]) <= {"-"}:
                tables[heading][cells[0]] = {
                    c: (float(v) if v not in ("-", "nan") else float("nan"))
                    for c, v in zip(columns, cells[1:])}
    return tables


def wall_rows(rig: dict[str, dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for label, name in WALL_ROWS:
        if name not in rig:
            print(f"wall: no row `{name}` in the rig table, left as xxx")
            rows.append((label, {}))
            continue
        rows.append((label, {c: rig[name][k] for c, k in WALL_COLUMNS.items()}))
    return rows


def parkour_rows(report: dict[str, dict[str, dict[str, float]]]) -> list[tuple[str, dict[str, float]]]:
    pose = next(t for h, t in report.items() if h.startswith("Procrustes MPJPE"))
    force = next(t for h, t in report.items() if h.startswith("Mean linear force error"))
    rig = next(t for h, t in report.items() if h.startswith("Force agreement"))
    rows = []
    for label, name, rnea_name in PARKOUR_ROWS:
        cells: dict[str, float] = {}
        if name in pose:
            cells["MPJPE"] = pose[name]["Avg"]
        if name in force:
            cells |= {c: force[name][k] for c, k in PARKOUR_FORCE.items()}
            cells["Avg"] = float(np.mean([force[name][k] for k in PARKOUR_FORCE.values()]))
        if name in rig:
            cells |= {c: rig[name][k] for c, k in PARKOUR_RIG.items()}
        if rnea_name in rig:
            cells |= {c: rig[rnea_name][k] for c, k in PARKOUR_RNEA.items()}
        if not cells:
            print(f"parkour: no row `{name}` in the report, left as xxx")
        rows.append((label, cells))
    return rows


def latex_rows(rows: list[tuple[str, dict[str, float]]], dataset: str) -> list[str]:
    """One ``&``-joined line per row, the best finite value of each column in bold."""
    best: dict[str, float] = {}
    for column, _ in COLUMNS:
        values = [cells[column] for label, cells in rows
                  if np.isfinite(cells.get(column, float("nan")))
                  and not (label == OWN_RESIDUAL[0] and column in OWN_RESIDUAL[1])]
        if values:
            best[column] = max(values) if column in HIGHER else min(values)
    lines = [rf"\multirow{{{len(rows)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{dataset}}}}}}}"]
    for label, cells in rows:
        parts = []
        for column, decimals in COLUMNS:
            value = cells.get(column, float("nan"))
            if not np.isfinite(value):
                parts.append(r"\todo{xxx}" if column not in cells else "--")
                continue
            text = f"{value:.{decimals}f}"
            if column in best and abs(value - best[column]) < 0.5 * 10 ** -decimals:
                text = rf"\textbf{{{text}}}"
            parts.append(text)
        lines.append(f"& {label} & " + " & ".join(parts) + r" \\")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rig", type=Path, required=True, help="score_rig.py markdown")
    parser.add_argument("--parkour", type=Path, required=True,
                        help="score_parkour_paper.py markdown report")
    parser.add_argument("--out", type=Path, default=None, help="tex file to write")
    args = parser.parse_args()
    rig = next(iter(parse_tables(args.rig.read_text()).values()))
    report = parse_tables(args.parkour.read_text())
    lines = (latex_rows(wall_rows(rig), r"\wall") + [r"\midrule"]
             + latex_rows(parkour_rows(report), "Parkour"))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
