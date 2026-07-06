from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "final"
DATA = ROOT / "data" / "processed" / "example_measurements.csv"


def load_data() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_csv(DATA)

    rng = np.random.default_rng(20260706)
    rows = []
    for group, mean in [("Control", 1.0), ("Treatment", 1.42)]:
        values = rng.normal(loc=mean, scale=0.16, size=12)
        rows.extend({"group": group, "value": value} for value in values)
    df = pd.DataFrame(rows)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    sns.set_theme(style="white", context="paper")

    fig, ax = plt.subplots(figsize=(85 / 25.4, 65 / 25.4), constrained_layout=True)
    palette = ["#3B6EA8", "#B8473B"]
    sns.stripplot(
        data=df,
        x="group",
        y="value",
        hue="group",
        palette=palette,
        jitter=0.16,
        size=3.5,
        linewidth=0.4,
        edgecolor="white",
        legend=False,
        ax=ax,
    )
    sns.pointplot(
        data=df,
        x="group",
        y="value",
        color="black",
        markers="_",
        linestyles="none",
        errorbar=("ci", 95),
        capsize=0.22,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Normalised response")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)

    fig.savefig(OUT / "example_plot.pdf")
    fig.savefig(OUT / "example_plot.png", dpi=600)


if __name__ == "__main__":
    main()
