"""Plot the two TYK2 binding estimators across independent 20 ns pilot replicas."""

import argparse
import json
import math
from pathlib import Path
from statistics import stdev

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from score_binding import ARMS

from fmeta.analysis import KCAL

RESULTS = Path(__file__).resolve().parents[1] / "examples" / "binding-study" / "results"
METHODS = {"primary_reweighted": "dG", "wt_bias_late": "dG_bias"}


def pilot(directory, arm):
    path = directory / ("a-r2.json" if arm == "a" else f"{arm}-screen.json")
    report = json.loads(path.read_text())
    endpoints = [e for e in report["endpoints"] if e["purpose"] == "screening" and e["length_ns"] == 20]
    if (report["arm"] != arm or report["energy_unit"] != "kJ/mol"
            or report["length_unit"] != "ns/walker" or len(endpoints) != 1):
        raise ValueError(f"{path} must hold one 20 ns {arm.upper()} screening endpoint in the expected units")
    data = endpoints[0]["systems"]["tyk2"]
    replicas = data["replicas"]
    total = ARMS[arm][0]
    if sorted(r["replica"] for r in replicas) != list(range(total)):
        raise ValueError(f"{path} has incomplete or duplicate replica records")
    passed = [r for r in replicas if r["status"] == "available"]
    if any(r["dG"] is None for r in passed) or data.get("n_sampling_quality_pass", len(passed)) != len(passed):
        raise ValueError(f"{path} has inconsistent primary sample coverage")
    result = dict(arm=arm.upper(), total=total)
    for method, field in METHODS.items():
        points = sorted((r[field] / KCAL, r["status"] == "available") for r in replicas if r[field] is not None)
        if not all(math.isfinite(value) for value, _ in points):
            raise ValueError(f"{path} has a nonfinite {field}")
        sd = stdev(value for value, _ in points) if len(points) == total and total > 1 else None
        if "method_statistics_kcal_mol" in data:
            stats = data["method_statistics_kcal_mol"][method]
            if ((stats["n_finite"], stats["n_total"]) != (len(points), total)
                    or (sd is None) != (stats["sd"] is None)
                    or (sd is not None and not math.isclose(sd, stats["sd"], rel_tol=1.e-9, abs_tol=1.e-12))):
                raise ValueError(f"{path} disagrees with its replica coverage or sample SD for {method}")
        result[method] = dict(points=points, sd=sd)
    return result


def swarm_heights(ax, points, baseline, diameter_pt):
    """Offset only the vertical coordinate so adjacent replica estimates stay visible."""
    gap = diameter_pt * ax.figure.dpi / 72
    y_scale = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
    placed = []
    for value, _ in points:
        x = ax.transData.transform((value, baseline))[0]
        for n in range(len(points) + 1):
            level = (n + 1) // 2 if n % 2 else -(n // 2)
            y = level * gap
            if all(math.hypot(x - px, y - py) >= gap - 1.e-8 for px, py in placed):
                placed.append((x, y))
                break
    heights = [baseline + y / y_scale for _, y in placed]
    if len(heights) != len(points) or any(abs(y - baseline) > .47 for y in heights):
        raise ValueError("replica swarm exceeds its row; enlarge the figure")
    return heights


def plot(pilots, output):
    order = sorted(pilots, key=lambda p: p["arm"])
    all_values = [v for p in order for method in METHODS for v, _ in p[method]["points"]]
    if not all_values:
        raise ValueError("the pilots contain no numerical estimates")
    margin = max(1., .05 * (max(all_values) - min(all_values)))
    limits = min(all_values) - margin, max(all_values) + margin
    dark, muted, teal, purple = "#20343e", "#536773", "#00736e", "#5c4d85"
    fig = plt.figure(figsize=(12.5, 6.2), dpi=170, facecolor="white")
    fig.text(.045, .913, "TYK2 binding free energy at 20 ns per walker", fontsize=20, weight="bold", color=dark)
    fig.text(.045, .859, "Each point is one independent replica of the TYK2–ejm_46 complex.",
             fontsize=11, color=muted)
    fig.text(.045, .723, "Candidate", fontsize=10, weight="bold", color=dark)
    baselines = [3.75 - i for i in range(len(order))]
    panels = [("primary_reweighted", .125, .414, "Observed site/bulk ratio", "Reweighted trajectory populations"),
              ("wt_bias_late", .575, .864, "WT bias free-energy surface", "Mean over the final half of the run")]
    for method, left, annotation, title, subtitle in panels:
        ax = fig.add_axes([left, .228, annotation - left - .014, .453])
        ax.set(xlim=limits, ylim=(.20, 4.30), yticks=[])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.set_axisbelow(True)
        ax.grid(axis="x", linewidth=.75, color="#e2e8ea")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set(visible=True, color="#9aabb3")
        ax.tick_params(axis="x", labelsize=9, length=3, color="#9aabb3", labelcolor=muted, pad=4)
        ax.set_xlabel(r"$\Delta G^{\circ}$ (kcal/mol)", fontsize=11, color=dark, labelpad=5)
        fig.text(left, .767, title, fontsize=15, weight="bold", color=dark)
        fig.text(left, .723, subtitle, fontsize=10, color=muted)
        fig.canvas.draw()
        for p, baseline in zip(order, baselines):
            points, sd = p[method]["points"], p[method]["sd"]
            primary = method == "primary_reweighted"
            color, marker = (teal, "o") if primary else (purple, "D")
            heights = swarm_heights(ax, points, baseline, 6.1 if primary else 7.4)
            for (value, quality), y in zip(points, heights):
                ax.scatter(value, y, s=24, marker=marker, facecolor=color if quality or not primary else "white",
                           edgecolor=color, linewidth=.85, zorder=3)
            baseline_fig = fig.transFigure.inverted().transform(ax.transData.transform((0, baseline)))[1]
            if primary:
                fig.text(.045, baseline_fig + .006, p["arm"], fontsize=15, weight="bold", color=dark, va="center")
                fig.text(.045, baseline_fig - .026, f"{p['total']} replicas", fontsize=9, color=muted, va="center")
            sd_text = f"SD {sd:.2f}" if sd is not None else "SD undefined"
            fig.text(annotation, baseline_fig + .011, sd_text, fontsize=10, weight="bold", color=dark, va="center")
            fig.text(annotation, baseline_fig - .022, f"{len(points)}/{p['total']} finite",
                     fontsize=9, color=muted, va="center")
            if primary and any(not quality for _, quality in points):
                passed = sum(quality for _, quality in points)
                fig.text(annotation, baseline_fig - .053, f"{passed} pass · {len(points)-passed} sparse",
                         fontsize=8.5, color=muted, va="center")
    handles = [Line2D([], [], ls="", marker=m, color=c, markerfacecolor=f, markersize=6, label=t)
               for m, c, f, t in [("o", teal, teal, "Sampling checks passed"),
                                  ("o", teal, "white", "Sparse sample coverage"),
                                  ("D", purple, purple, "Bias-derived estimate")]]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.038, .061), frameon=False,
               ncol=3, fontsize=10, labelcolor=dark, handletextpad=.4, columnspacing=2.4)
    fig.text(.045, .036, "SD is the sample standard deviation across all planned replicas. "
             "Convergence requires sampling and stability checks.", color=muted, fontsize=9)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, help="PNG destination (default: the results directory)")
    args = parser.parse_args()
    plot([pilot(args.results, arm) for arm in ARMS], args.output or args.results / "tyk2-20ns-replicas.png")
