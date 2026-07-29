"""Render learning-curve figures for the experiment writeup.

Usage: python3 scripts/plot_curves.py
Writes writeup/figures/{ablations,lr_sweep,batch_size}_{light,dark}.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RUNS = Path("/notebooks/runs")
OUT = Path("/notebooks/writeup/figures")

# Categorical slots in fixed order (validated: adjacent CVD dE 9.1 light / 8.4 dark).
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

THEME = {
    "light": {
        "series": SERIES_LIGHT,
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "grid": "#e4e3df",
    },
    "dark": {
        "series": SERIES_DARK,
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "grid": "#383835",
    },
}


def load_val_curve(name: str) -> tuple[list[int], list[float]]:
    """Validation loss by step, de-duplicated (a resumed run re-evaluates a step)."""
    by_step: dict[int, float] = {}
    path = RUNS / name / "log.jsonl"
    for line in path.open():
        record = json.loads(line)
        if "val_loss" in record:
            by_step[record["step"]] = record["val_loss"]
    steps = sorted(by_step)
    return steps, [by_step[s] for s in steps]


def load_config(name: str) -> dict:
    with (RUNS / name / "config.json").open() as f:
        return json.load(f)


def plot_figure(
    filename: str,
    title: str,
    subtitle: str,
    series: list[tuple[str, str]],
    mode: str,
    x_tokens: bool = False,
) -> None:
    """One multi-line chart: validation loss vs training progress, one line per run."""
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    ends = []
    for i, (run, label) in enumerate(series):
        steps, losses = load_val_curve(run)
        color = t["series"][i % len(t["series"])]
        if x_tokens:
            config = load_config(run)
            scale = config["batch_size"] * config["context_length"]
            xs = [s * scale / 1e6 for s in steps]
        else:
            xs = list(steps)
        ax.plot(xs, losses, color=color, linewidth=2.0, solid_capstyle="round", zorder=3)
        ends.append((xs[-1], losses[-1], color, label))

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.tick_params(axis="both", colors=t["text2"], labelsize=9, length=0)
    ax.tick_params(axis="y", which="minor", colors=t["text2"], labelsize=8, length=0)

    ax.set_xlabel("tokens processed (millions)" if x_tokens else "gradient step", color=t["text2"], fontsize=9.5)
    ax.set_ylabel("validation loss (log scale)", color=t["text2"], fontsize=9.5)
    ax.grid(True, which="major", color=t["grid"], linewidth=0.8, zorder=1)
    ax.grid(True, which="minor", color=t["grid"], linewidth=0.4, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["grid"])

    # Leave room on the right for the end labels.
    x_max = max(e[0] for e in ends)
    ax.set_xlim(0, x_max * 1.34)
    y_lo = min(e[1] for e in ends)
    y_hi = max(max(load_val_curve(run)[1]) for run, _ in series)
    ax.set_ylim(y_lo * 0.96, y_hi * 1.04)
    candidates = [1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ax.set_yticks([c for c in candidates if y_lo * 0.96 <= c <= y_hi * 1.04])

    # Direct labels. Runs that reach the right edge get a label column with a
    # leader drawn only in the margin; runs that stop early are labelled in
    # place, so no connector is ever drawn across the plot area.
    label_x = x_max * 1.06
    margin_x = x_max * 1.012
    right_rows = sorted([e for e in ends if e[0] >= 0.6 * x_max], key=lambda e: e[1])
    inline_rows = [e for e in ends if e[0] < 0.6 * x_max]

    min_gap = (ax.get_ylim()[1] / ax.get_ylim()[0]) ** 0.038  # multiplicative gap on a log axis
    placed: list[float] = []
    for _, y, _, _ in right_rows:
        placed.append(y if not placed else max(y, placed[-1] * min_gap))
    overflow = placed[-1] / (ax.get_ylim()[1] * 0.995) if placed else 0
    if overflow > 1:
        placed = [p / overflow for p in placed]

    for (x, y, color, label), label_y in zip(right_rows, placed):
        ax.plot([max(x, margin_x), label_x], [y, label_y], color=color, linewidth=0.8, alpha=0.6, zorder=2)
        ax.annotate(
            f"{label}  {y:.3f}",
            xy=(label_x, label_y),
            xytext=(4, 0),
            textcoords="offset points",
            color=t["text"],
            fontsize=8.5,
            va="center",
            ha="left",
            zorder=4,
        )

    for x, y, color, label in inline_rows:
        ax.plot([x], [y], marker="o", markersize=4.5, color=color, zorder=4)
        ax.annotate(
            f"{label}  {y:.3f}  (stopped)",
            xy=(x, y),
            xytext=(9, 0),
            textcoords="offset points",
            color=t["text"],
            fontsize=8.5,
            va="center",
            ha="left",
            zorder=4,
        )

    ax.set_title(title, color=t["text"], fontsize=13, fontweight="600", loc="left", pad=16)
    ax.text(
        0, 1.015, subtitle, transform=ax.transAxes, color=t["text2"], fontsize=9.5, va="bottom", ha="left"
    )
    legend = ax.legend(
        [plt.Line2D([], [], color=t["series"][i % len(t["series"])], linewidth=2.0) for i in range(len(series))],
        [label for _, label in series],
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=t["text2"],
        handlelength=1.6,
        ncol=2 if len(series) > 3 else 1,
    )
    legend.set_zorder(5)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{filename}_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


FIGURES = [
    {
        "filename": "ablations",
        "title": "Architecture ablations on TinyStories",
        "subtitle": "17M-parameter Transformer LM, 327M tokens, batch 64 x context 256, LR 1e-3 unless noted",
        "series": [
            ("tinystories-base", "base (pre-norm, RoPE, SwiGLU)"),
            ("ablation-post-norm", "post-norm"),
            ("ablation-silu", "SiLU FFN (no gating)"),
            ("ablation-no-rmsnorm", "no RMSNorm"),
            ("ablation-nope", "NoPE (no position emb.)"),
            ("ablation-no-rmsnorm-lowlr", "no RMSNorm, LR 1e-4"),
        ],
    },
    {
        "filename": "lr_sweep",
        "title": "Learning rate sweep on TinyStories",
        "subtitle": "same architecture and token budget; 1e-2 and 1e-1 are short runs, stopped once unstable",
        "series": [
            ("lr-3e-3", "LR 3e-3 (best)"),
            ("tinystories-base", "LR 1e-3 (base)"),
            ("lr-3e-4", "LR 3e-4"),
            ("lr-1e-2-divergent", "LR 1e-2 (unstable)"),
            ("lr-1e-1-divergent", "LR 1e-1 (divergent)"),
        ],
    },
    {
        "filename": "batch_size",
        "title": "Batch size variations on TinyStories",
        "subtitle": "327M-token budget held fixed; LR scaled with batch size; x-axis is tokens, not steps",
        "series": [
            ("batch-256", "batch 256, LR 2e-3"),
            ("tinystories-base", "batch 64, LR 1e-3"),
            ("batch-32", "batch 32, LR 5e-4"),
        ],
        "x_tokens": True,
    },
]


def main() -> None:
    for spec in FIGURES:
        for mode in ("light", "dark"):
            plot_figure(
                spec["filename"],
                spec["title"],
                spec["subtitle"],
                spec["series"],
                mode,
                x_tokens=spec.get("x_tokens", False),
            )
        print(f"wrote {spec['filename']}_{{light,dark}}.png")


if __name__ == "__main__":
    main()
