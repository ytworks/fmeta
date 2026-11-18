"""Plot the primary reweighted binding free energy of the TYK2 shared-bias trial."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "examples/binding-study/results"


def plot(results, output):
    experiment = json.loads((results / "e-tyk2.json").read_text())
    if experiment["sampling"]["independent_bias_histories"] != 1 or experiment["sampling"]["walkers_per_bias"] != 32:
        raise ValueError("this figure requires one bias shared by 32 walkers")
    cuts, values, references = [], [], set()
    for step in experiment["endpoints"]:
        primary = step["systems"]["tyk2"]["comparison_kcal_mol"]
        cuts.append(step["length_ns"])
        values.append(primary["predicted_mean"])
        references.add(primary["reference"])
    if len(references) != 1:
        raise ValueError("the TYK2 checkpoints must have one common experimental reference")
    reference = references.pop()

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.subplots_adjust(left=.13, bottom=.17, top=.84, right=.97)
    color = "#343b6b"
    ax.axhline(reference, color="#6d707a", lw=1.4, ls="--")
    ax.text(.025, reference, f"300 K reference  {reference:.2f}", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", color="#4a4d57", fontsize=10)
    ax.plot(cuts, values, marker="o", color=color, linewidth=2.4, markersize=7)
    for n, value in zip(cuts, values, strict=True):
        if value is not None:
            ax.annotate(f"{value:.2f}", (n, value), xytext=(0, -12), textcoords="offset points",
                        ha="center", va="top", color=color, fontsize=10)
    ax.set_xlim(3.5, 21.5)
    ax.margins(y=.18)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_xlabel("Time per walker (ns)")
    ax.set_ylabel("Binding free energy ΔG° (kcal/mol)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=.6, alpha=.22)
    fig.suptitle("TYK2 with ejm_46", x=.13, y=.98, ha="left", fontsize=15, fontweight="bold")
    ax.set_title("Reweighted site-to-bulk estimate  ·  one bias shared by 32 walkers",
                 loc="left", fontsize=10, color="#424242", pad=12)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output or args.results / "tyk2-shared-bias-dg.png")
