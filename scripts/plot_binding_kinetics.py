"""Compare TYK2 endpoint binding energies and sampled CV area at matched trajectory lengths."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parents[1] / "examples/binding-study/results"
SERIES = {"c": ("C · 8 × 4", "#34758a", "s"), "d": ("D · 32 × 1", "#ad653e", "^"),
          "e": ("E · 1 × 32", "#3b396d", "o")}


def read(path):
    return json.loads(path.read_text())


def plot(results, output):
    controls = read(results / "tyk2-cd-cv-coverage-checkpoints.json")["arms"]
    experiment = read(results / "e-tyk2.json")
    e_coverage = {r["cut_ns"]: r for r in read(results / "e-tyk2-cv-coverage.json")["endpoints"]}
    cuts = [p["length_ns"] for p in experiment["endpoints"]]
    reference = experiment["endpoints"][0]["systems"]["tyk2"]["comparison_kcal_mol"]["reference"]
    fig, (energy, coverage) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.subplots_adjust(left=.095, right=.98, top=.74, bottom=.21, wspace=.32)

    for arm, (label, color, marker) in SERIES.items():
        if arm == "e":
            dg = [p["systems"]["tyk2"]["method_statistics_kcal_mol"]["wt_bias_final"]["mean"]
                  for p in experiment["endpoints"]]
            area = [e_coverage[n]["zones"]["all"]["union_area_nm2"] for n in cuts]
            bulk = [e_coverage[n]["zones"]["bulk"]["physical_walkers_any"] for n in cuts]
        else:
            record = next(p["systems"]["tyk2"] for p in read(results / f"{arm}-screen.json")["endpoints"]
                          if p["length_ns"] == 20)
            times = record["wt_bias_times_ns"]
            dg = [mean(r["wt_bias_trace_kcal_mol"][times.index(n)] for r in record["replicas"]) for n in cuts]
            by_cut = {r["cut_ns"]: r for r in controls[arm]["checkpoints"]}
            area = [by_cut[n]["union_area_nm2"] for n in cuts]
            bulk = [by_cut[n]["bulk_walkers"] for n in cuts]
        energy.plot(cuts, dg, color=color, linewidth=2, zorder=2)
        for n, value, visitors in zip(cuts, dg, bulk, strict=True):
            energy.scatter(n, value, marker=marker, edgecolor=color, linewidth=1.5, s=40,
                           facecolor=color if visitors else "white", zorder=3)
        coverage.plot(cuts, area, color=color, marker=marker, linewidth=2, markersize=6, label=label)

    energy.axhline(reference, color="#696969", linewidth=1.2, linestyle="--")
    energy.set_ylabel("Binding free energy (kcal/mol)")
    coverage.set_ylabel("Area in the (s, ρ) plane (nm²)")
    for ax, title in [(energy, "WT surface integral at the endpoint"), (coverage, "Area visited by all 32 walkers")]:
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=10)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlim(4, 21)
        ax.set_xlabel("Time per walker (ns)")
        ax.grid(axis="y", linewidth=.6, alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, _ = coverage.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#696969", linestyle="--", label=f"300 K reference ({reference:.2f})"))
    fig.suptitle("TYK2 with ejm_46", x=.095, ha="left", y=.98, fontsize=15, fontweight="semibold")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085, .905), ncol=4, frameon=False, fontsize=9)
    fig.text(.095, .038, "Legend gives independent biases × walkers per bias. Open markers indicate zero bulk visits "
             "after the 2 ns discard.", fontsize=8, color="#424242")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.results, args.output or args.results / "tyk2-shared-bias-comparison.png")
