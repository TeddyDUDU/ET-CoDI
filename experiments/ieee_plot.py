"""Shared IEEE-style matplotlib settings for experiment figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

SAVE_DPI = 300
FIG_W_TS = 7.0
FIG_H_TS = 2.8
FIG_W_BAR = 7.0
FIG_H_BAR = 3.0
FIG_W_PANEL = 7.0
FIG_H_PANEL = 6.2
FIG_W_HALF = FIG_W_PANEL / 2
FIG_H_HALF = FIG_H_PANEL / 2


def apply_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "mathtext.fontset": "stix",
    })


def save_fig(fig, out_path: str, *, pad_inches: float = 0.04) -> None:
    fig.savefig(out_path, format="png", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
