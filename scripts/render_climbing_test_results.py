"""Render compact aggregate and per-joint ClimbingVideos_v1 result tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
SUMMARY_OUTPUT = REPO / "docs" / "CLIMBING_TEST_RESULTS.png"
JOINT_OUTPUT = REPO / "docs" / "CLIMBING_JOINT_RESULTS.png"

# Independently evaluated best checkpoints, threshold 0.5.
SUMMARY_RESULTS = [
    ("Always-contact baseline", 0.7826785273, 1.0, 0.8780927299),
    ("Per-frame model", 0.9119050583, 0.8672454617, 0.8890147454),
    ("Temporal all losses", 0.8685602998, 0.9271957008, 0.8969207126),
    ("Temporal mid loss", 0.9306950994, 0.8889541716, 0.9093458880),
]

# Per-joint metrics reconstructed exactly from the independently evaluated
# confusion counts. The baseline uses the full per-frame test support.
JOINT_RESULTS = {
    "Left hand": [
        (0.8040091776, 1.0, 0.8913581900),
        (0.8934010152, 0.9780714929, 0.9338208934),
        (0.8579721362, 0.9931312528, 0.9206173438),
        (0.9274611399, 0.9357729649, 0.9315985130),
    ],
    "Right hand": [
        (0.8369762106, 1.0, 0.9112542729),
        (0.9229655372, 0.9196364161, 0.9212979692),
        (0.8756187333, 0.9903818547, 0.9294712024),
        (0.9278711485, 0.9505021521, 0.9390503189),
    ],
    "Left foot": [
        (0.6946891518, 1.0, 0.8198425665),
        (0.8949936360, 0.7643115942, 0.8245065468),
        (0.8451311534, 0.8457389428, 0.8454349389),
        (0.9163306452, 0.8167115903, 0.8636579572),
    ],
    "Right foot": [
        (0.7919420437, 1.0, 0.8838924746),
        (0.9385356455, 0.7817364789, 0.8529901059),
        (0.8946318764, 0.8588066369, 0.8763532764),
        (0.9511754069, 0.8349206349, 0.8892645816),
    ],
}

ROW_NAMES = [name for name, *_ in SUMMARY_RESULTS]
TEXT = "#111111"


def _canvas(figsize: tuple[float, float]):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "text.color": TEXT,
    })
    fig = plt.figure(figsize=figsize, dpi=180, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _title(ax, title: str) -> None:
    ax.text(0.5, 0.89, title, fontsize=23, fontweight="bold",
            ha="center", va="center")


def render_summary(output: Path = SUMMARY_OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _canvas((11, 4.2))
    left, right = 0.075, 0.925
    _title(ax, "ClimbingVideos_v1 Test Results (threshold = 0.5)")

    x_metrics = (0.59, 0.73, 0.87)
    ax.plot([left, right], [0.79, 0.79], color=TEXT, lw=1.7)
    y_header = 0.715
    ax.text(left, y_header, "Model", fontsize=15, fontweight="bold", va="center")
    for x, label in zip(x_metrics, ("Precision", "Recall", "F1")):
        ax.text(x, y_header, label, fontsize=15, fontweight="bold",
                ha="center", va="center")
    ax.plot([left, right], [0.655, 0.655], color=TEXT, lw=0.9)

    best = [max(row[column] for row in SUMMARY_RESULTS[1:]) for column in range(1, 4)]
    row_ys = (0.555, 0.445, 0.335, 0.215)
    for row_index, ((name, precision, recall, f1), y) in enumerate(
        zip(SUMMARY_RESULTS, row_ys)
    ):
        if row_index == 0:
            ax.plot([left, right], [y - 0.055, y - 0.055], color=TEXT, lw=0.7)
        ax.text(left, y, name, fontsize=17, va="center")
        for metric_index, (x, value) in enumerate(
            zip(x_metrics, (precision, recall, f1))
        ):
            winner = row_index > 0 and value == best[metric_index]
            ax.text(x, y, f"{value:.3f}", fontsize=17,
                    fontweight="bold" if winner else "normal",
                    ha="center", va="center")
    ax.plot([left, right], [0.145, 0.145], color=TEXT, lw=1.7)

    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_joints(output: Path = JOINT_OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _canvas((16, 4.8))
    left, right = 0.045, 0.955
    _title(ax, "ClimbingVideos_v1 Per-Joint Test Results (threshold = 0.5)")

    model_right = 0.275
    group_edges = [model_right, 0.445, 0.615, 0.785, right]
    group_centers = [(a + b) / 2 for a, b in zip(group_edges[:-1], group_edges[1:])]
    metric_xs: list[tuple[float, float, float]] = []
    for a, b in zip(group_edges[:-1], group_edges[1:]):
        width = b - a
        metric_xs.append((a + width * 0.24, a + width * 0.50, a + width * 0.76))

    ax.plot([left, right], [0.79, 0.79], color=TEXT, lw=1.7)
    ax.text(left, 0.650, "Model", fontsize=15, fontweight="bold", va="center")
    for joint, center, xs in zip(JOINT_RESULTS, group_centers, metric_xs):
        ax.text(center, 0.725, joint, fontsize=16, fontweight="bold",
                ha="center", va="center")
        for x, metric in zip(xs, ("P", "R", "F1")):
            ax.text(x, 0.650, metric, fontsize=15, fontweight="bold",
                    ha="center", va="center")
    for a, b in zip(group_edges[:-1], group_edges[1:]):
        ax.plot([a + 0.012, b - 0.012], [0.687, 0.687], color=TEXT, lw=0.7)
    ax.plot([left, right], [0.595, 0.595], color=TEXT, lw=0.9)

    best = {
        joint: [max(row[column] for row in values[1:]) for column in range(3)]
        for joint, values in JOINT_RESULTS.items()
    }
    row_ys = (0.505, 0.405, 0.305, 0.195)
    for row_index, (name, y) in enumerate(zip(ROW_NAMES, row_ys)):
        if row_index == 0:
            ax.plot([left, right], [y - 0.050, y - 0.050], color=TEXT, lw=0.7)
        ax.text(left, y, name, fontsize=16, va="center")
        for joint, xs in zip(JOINT_RESULTS, metric_xs):
            values = JOINT_RESULTS[joint][row_index]
            for metric_index, (x, value) in enumerate(zip(xs, values)):
                winner = row_index > 0 and value == best[joint][metric_index]
                ax.text(x, y, f"{value:.3f}", fontsize=15.5,
                        fontweight="bold" if winner else "normal",
                        ha="center", va="center")
    ax.plot([left, right], [0.125, 0.125], color=TEXT, lw=1.7)

    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    render_summary()
    render_joints()
    print(SUMMARY_OUTPUT)
    print(JOINT_OUTPUT)
