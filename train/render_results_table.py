"""Render the 3x2 cross-evaluation as an academic (booktabs-style) table.

Reads ``train/output/cross_eval.jsonl`` (produced by ``evaluate.py``) and writes
  train/output/cross_eval_table.png   — presentation-ready, rule-only, serif
  train/output/cross_eval_table.tex   — LaTeX booktabs source

A transfer matrix: rows are models (what each was trained on), column groups
are the held-out eval sets (DAMON test, ClimbingImages). Cells are global
precision / recall / F1 in percent; the best per column is bold.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
JSONL = REPO / "train" / "output" / "cross_eval.jsonl"
PNG = REPO / "train" / "output" / "cross_eval_table.png"
TEX = REPO / "train" / "output" / "cross_eval_table.tex"

# row order (model) and the substring that identifies each in a checkpoint path
ROWS = [
    ("damon",          "DAMON"),
    ("climbing",       "Climbing"),
    ("climbing_damon", "Climbing + DAMON"),
]
EVALS = [("damon", "DAMON"), ("climbing", "ClimbingImages")]
METRICS = [("precision", "Prec."), ("recall", "Rec."), ("f1", "F1")]


def model_key(path: str) -> str:
    if "climbing_damon" in path:
        return "climbing_damon"
    if "climbing" in path:
        return "climbing"
    return "damon"


def load():
    cells, nstr = {}, {}
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cells[(model_key(r["checkpoint"]), r["eval"])] = r
        nstr[r["eval"]] = int(r["n_used"])
    return cells, nstr


def best_mask(cells):
    """(eval, metric) -> winning model_key (max value over rows)."""
    best = {}
    for ev, _ in EVALS:
        for mk, _ in METRICS:
            vals = [(rk, cells[(rk, ev)][mk]) for rk, _ in ROWS if (rk, ev) in cells]
            if vals:
                best[(ev, mk)] = max(vals, key=lambda t: t[1])[0]
    return best


def render_png(cells, nstr):
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    best = best_mask(cells)

    fig = plt.figure(figsize=(9.2, 2.6))
    ax = fig.add_axes([0, 0, 1, 1])           # fill the figure: even tight-crop
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    mx = [0.365, 0.470, 0.575, 0.735, 0.840, 0.945]   # 6 metric column centers
    gcx = [(mx[0] + mx[2]) / 2, (mx[3] + mx[5]) / 2]   # group header centers
    groups = [(0, 2), (3, 5)]
    label_x = 0.015
    x0, x1 = 0.0, 0.99

    # generous, non-overlapping vertical rhythm (top -> bottom)
    y_top   = 0.93
    y_group = 0.83
    y_n     = 0.755
    y_cmid  = 0.695
    y_sub   = 0.625
    y_mid   = 0.57
    y_rows  = [0.43, 0.30, 0.17]
    y_bot   = 0.09

    def rule(y, a, b, lw):
        ax.plot([a, b], [y, y], color="black", lw=lw, solid_capstyle="butt")

    rule(y_top, x0, x1, 1.6)                                   # \toprule
    for (ev, ev_disp), cx, g in zip(EVALS, gcx, groups):
        ax.text(cx, y_group, ev_disp, ha="center", va="center",
                fontsize=13, fontweight="bold")
        ax.text(cx, y_n, f"(n={nstr.get(ev, '?')})", ha="center", va="center",
                fontsize=9, style="italic")
        rule(y_cmid, mx[g[0]] - 0.045, mx[g[1]] + 0.045, 1.0)  # \cmidrule
    ax.text(label_x, y_sub, "Trained on", ha="left", va="center",
            fontsize=12, fontweight="bold")
    for (ev, _), g in zip(EVALS, groups):
        for (mk, mdisp), xi in zip(METRICS, mx[g[0]:g[1] + 1]):
            ax.text(xi, y_sub, mdisp, ha="center", va="center", fontsize=11)
    rule(y_mid, x0, x1, 1.0)                                   # \midrule
    for (rk, rdisp), yr in zip(ROWS, y_rows):
        ax.text(label_x, yr, rdisp, ha="left", va="center", fontsize=12)
        for (ev, _), g in zip(EVALS, groups):
            for (mk, _), xi in zip(METRICS, mx[g[0]:g[1] + 1]):
                v = cells[(rk, ev)][mk] * 100.0
                bold = best.get((ev, mk)) == rk
                ax.text(xi, yr, f"{v:.1f}", ha="center", va="center",
                        fontsize=12, fontweight="bold" if bold else "normal")
    rule(y_bot, x0, x1, 1.6)                                   # \bottomrule

    fig.savefig(PNG, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print(f"wrote {PNG}")


def render_tex(cells, nstr):
    best = best_mask(cells)

    def fmt(rk, ev, mk):
        v = cells[(rk, ev)][mk] * 100.0
        s = f"{v:.1f}"
        return rf"\textbf{{{s}}}" if best.get((ev, mk)) == rk else s

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cross-dataset evaluation of the per-vertex contact head. Each model "
        r"(row) is trained on the named data and evaluated on held-out sets it never "
        r"trained on. Best per column in bold.}",
        r"\label{tab:contact_cross}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        rf"& \multicolumn{{3}}{{c}}{{DAMON ($n{{=}}{nstr.get('damon','?')}$)}} & "
        rf"& \multicolumn{{3}}{{c}}{{ClimbingImages ($n{{=}}{nstr.get('climbing','?')}$)}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){6-8}",
        r"Trained on & Prec. & Rec. & F1 & & Prec. & Rec. & F1 \\",
        r"\midrule",
    ]
    for rk, rdisp in ROWS:
        lines.append(
            f"{rdisp} & {fmt(rk,'damon','precision')} & {fmt(rk,'damon','recall')} & "
            f"{fmt(rk,'damon','f1')} & & {fmt(rk,'climbing','precision')} & "
            f"{fmt(rk,'climbing','recall')} & {fmt(rk,'climbing','f1')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {TEX}")


def main():
    cells, nstr = load()
    render_png(cells, nstr)
    render_tex(cells, nstr)


if __name__ == "__main__":
    main()
