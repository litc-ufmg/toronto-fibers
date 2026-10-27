"""Headless histogram rendering; importing this module selects Agg."""

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfcfc",
    "axes.edgecolor": "#303030",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d8d8d8",
    "grid.linestyle": "-",
    "grid.alpha": 0.45,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _format_count_axis(ax):
    """Formats y-axis ticks for histogram counts."""
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def plot_length_histograms(records, output_path, bins=30):
    """Save an EP-EP length histogram; empty records produce no plot."""
    if not records:
        return

    ep_ep_lengths = np.asarray([r["ep_ep_length"] for r in records], dtype=float)

    with (
        plt.style.context("seaborn-v0_8-whitegrid"),
        matplotlib.rc_context(_PLOT_STYLE),
    ):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.8), dpi=160)
        ax.hist(
            ep_ep_lengths,
            bins=bins,
            color="#2a9d8f",
            edgecolor="#202020",
            linewidth=0.6,
            alpha=0.88,
        )
        mean_v = float(np.mean(ep_ep_lengths))
        median_v = float(np.median(ep_ep_lengths))
        ax.axvline(
            mean_v,
            color="#8e2a2a",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean: {mean_v:.2f}",
        )
        ax.axvline(
            median_v,
            color="#5c3c92",
            linestyle=":",
            linewidth=1.6,
            label=f"Median: {median_v:.2f}",
        )
        ax.set_title("Fiber Length by Endpoint Distance")
        ax.set_xlabel("Length (physical units)")
        ax.set_ylabel("Count")
        _format_count_axis(ax)
        ax.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_orientation_histograms(records, output_path, bins=18):
    """Save acute-angle histograms as fiber percentages; skip empty records."""
    if not records:
        return

    angles_x = np.asarray([r["angle_x"] for r in records], dtype=float)
    angles_y = np.asarray([r["angle_y"] for r in records], dtype=float)
    angles_z = np.asarray([r["angle_z"] for r in records], dtype=float)

    if isinstance(bins, int):
        bins = np.linspace(0.0, 90.0, bins + 1)

    with (
        plt.style.context("seaborn-v0_8-whitegrid"),
        matplotlib.rc_context(_PLOT_STYLE),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=160, sharey=True)
        for ax, values, axis_name, color in (
            (axes[0], angles_x, "X", "#3b6fb6"),
            (axes[1], angles_y, "Y", "#c06c2f"),
            (axes[2], angles_z, "Z", "#4f8a3b"),
        ):
            weights = np.full(values.shape, 100.0 / max(len(values), 1), dtype=float)
            ax.hist(
                values,
                bins=bins,
                range=(0, 90),
                weights=weights,
                color=color,
                edgecolor="#1f1f1f",
                linewidth=0.6,
                alpha=0.86,
            )
            mean_v = float(np.mean(values))
            std_v = float(np.std(values))
            ax.axvline(mean_v, color="#7a1f4b", linestyle="--", linewidth=1.2)
            ax.text(
                0.98,
                0.96,
                f"mean={mean_v:.1f}\nstd={std_v:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.72,
                    "edgecolor": "none",
                    "pad": 2.0,
                },
            )
            ax.set_xlim(0, 90)
            ax.set_title(f"Orientation Relative to {axis_name} Axis")
            ax.set_xlabel("Angle (deg)")

        axes[0].set_ylabel("Fibers (%)")
        fig.suptitle("Fiber Orientation Distributions", y=1.03)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
