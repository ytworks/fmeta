"""Figures for a finished analysis (needs matplotlib).

``plot_run`` draws one replica: the free-energy profile along the funnel axis,
the binding free energy as the bias accumulates, the 2D free-energy surface and
the ligand's position over time. ``plot_summary`` puts the replicas of a
calculation side by side. The numbers behind every panel are in ``result.json``,
``fes.npz`` and ``summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..engine.hills import read_colvar
from ..engine.runner import load_run_metadata
from ..funnel.geometry import Funnel
from . import walker_directories

# Colours by role. Categorical hues are used in this fixed order and never cycled;
# the surface-magnitude ramp is one hue.
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", neutral="#898781",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
                "#e34948"],
        # low free energy (the basins) dark, high free energy fading into the surface
        ramp=["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", neutral="#898781",
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9",
                "#e66767"],
        # on a dark surface the basins are the light end
        ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
}
MAX_LINES = 8          # one categorical slot per walker; more are not drawn


def _matplotlib():
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError("plotting needs matplotlib (pip install matplotlib)") from error
    return matplotlib, Figure, FigureCanvasAgg, LinearSegmentedColormap


def _style(theme: dict) -> dict:
    return {
        "font.family": "sans-serif", "font.size": 9,
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "axes.edgecolor": theme["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": theme["grid"], "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.labelcolor": theme["secondary"], "axes.labelsize": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": theme["ink"],
        "axes.titlelocation": "left", "axes.titlepad": 9,
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": theme["ink"], "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    }


def _end_dot(ax, x, y, colour, theme) -> None:
    ax.plot([x], [y], marker="o", markersize=6.5, markerfacecolor=colour,
            markeredgecolor=theme["surface"], markeredgewidth=1.6, linestyle="none", zorder=5)


def _thin(x, y, limit: int = 3000):
    step = max(1, len(x) // limit)
    return x[::step], y[::step]


def plot_run(workdir, path=None, theme: str = "light"):
    """Draw the analysis of one replica directory (run ``analyze_run`` first).
    Returns the path of the PNG written."""
    matplotlib, Figure, Canvas, Colormap = _matplotlib()
    colours = THEMES[theme]
    workdir = Path(workdir)
    with open(workdir / "result.json") as handle:
        result = json.load(handle)
    data = np.load(workdir / "fes.npz")
    sources = walker_directories(workdir) or [workdir]
    funnel = Funnel.from_dict(load_run_metadata(sources[0])["funnel"])
    colvars = [read_colvar(d / "COLVAR") for d in sources]
    s, rho, profile, fes = data["s"], data["rho"], data["profile"], data["fes"]
    unbound = result["unbound"]
    blue = colours["series"][0]

    with matplotlib.rc_context(_style(colours)):
        figure = Figure(figsize=(11.0, 7.6), dpi=150)
        Canvas(figure)
        axes = figure.subplots(2, 2, gridspec_kw=dict(hspace=0.48, wspace=0.24, left=0.07,
                                                      right=0.97, top=0.86, bottom=0.08))

        # -- headline -------------------------------------------------------
        figure.text(0.07, 0.955, f"ΔG° = {result['dG']:.1f} kJ/mol"
                    f"  ({result['dG_kcal']:.2f} kcal/mol)", fontsize=15,
                    fontweight="bold", color=colours["ink"], ha="left", va="center")
        detail = (f"{workdir.name} · {result['simulated_ns']:.1f} ns simulated"
                  + (f" over {result['n_walkers']} walkers" if result["n_walkers"] > 1 else "")
                  + f" · {result['rebinding_events']} rebinding events")
        if result["dG_source"] == "samples":
            detail += f" · from the samples after the first {result['discard_ns']:.1f} ns"
        figure.text(0.07, 0.918, detail, fontsize=9, color=colours["secondary"], ha="left",
                    va="center")
        if result["no_binding"]:
            figure.text(0.97, 0.955, "No binding", fontsize=11, fontweight="bold",
                        color=colours["ink"], ha="right", va="center")
            figure.text(0.97, 0.918, "within 1 kT of the empty funnel", fontsize=8,
                        color=colours["secondary"], ha="right", va="center")

        # -- W(s) -----------------------------------------------------------
        ax = axes[0, 0]
        shown = (s >= funnel.lower - 0.1) & (s <= funnel.upper + 0.05)
        top = float(np.percentile(profile[shown], 99)) * 1.12 + 1.0
        ax.axvspan(*unbound, color=colours["neutral"], alpha=0.12, linewidth=0)
        ax.text(np.mean(unbound), top * 0.97, "unbound\nwindow", fontsize=8,
                color=colours["secondary"], ha="center", va="top")
        ax.plot(s[shown], profile[shown], color=blue, linewidth=1.8)
        i_min = int(np.argmin(profile))
        _end_dot(ax, s[i_min], profile[i_min], blue, colours)
        ax.annotate(f"bound, s = {s[i_min]:.2f} nm", (s[i_min], profile[i_min]),
                    xytext=(8, -2), textcoords="offset points", fontsize=8,
                    color=colours["secondary"], va="top")
        ax.set_ylim(-0.06 * top, top)
        ax.set_xlim(funnel.lower - 0.1, funnel.upper + 0.05)
        ax.set_title("Free energy along the funnel axis")
        ax.set_xlabel("projection on the axis, s (nm)")
        ax.set_ylabel("W (kJ/mol)")

        # -- dG(t) ----------------------------------------------------------
        ax = axes[0, 1]
        time, series = data["time_ns"], data["dG"]
        start = time[max(0, len(time) - max(2, int(round(len(time) * 0.5))))]
        ax.axvspan(start, time[-1], color=colours["neutral"], alpha=0.12, linewidth=0)
        ax.plot(time, series, color=blue, linewidth=1.8)
        _end_dot(ax, time[-1], series[-1], blue, colours)
        low, high = float(series.min()), float(series.max())
        pad = 0.15 * max(high - low, 1.0)
        ax.set_ylim(low - pad, high + 2.2 * pad)
        ax.text(0.5 * (start + time[-1]), high + 1.9 * pad,
                f"plateau mean {result['dG_plateau']:.1f} ± {result['dG_plateau_std']:.1f}",
                fontsize=8, color=colours["secondary"], ha="center", va="top")
        ax.set_title("Binding free energy as the bias accumulates")
        ax.set_xlabel("time per walker (ns)" if result["n_walkers"] > 1 else "time (ns)")
        ax.set_ylabel("ΔG° (kJ/mol)")

        # -- F(s, rho) ------------------------------------------------------
        ax = axes[1, 0]
        # cells the run never reached carry no bias and sit at the maximum: leave them blank
        cap = float(max(min(fes.max() - 1.0, 60.0), 0.5 * fes.max(), 1e-6))
        surface = np.ma.masked_greater(fes, cap)
        cmap = Colormap.from_list("fmeta_sequential", colours["ramp"]).with_extremes(
            bad=colours["surface"])
        mesh = ax.pcolormesh(s, rho, surface, cmap=cmap, vmin=0.0, vmax=cap, shading="auto",
                             rasterized=True)
        wall_s = np.linspace(funnel.lower, funnel.upper, 200)
        ax.plot(wall_s, funnel.wall_radius(wall_s), color=colours["secondary"], linewidth=1.4)
        ax.annotate("funnel wall", (wall_s[60], float(funnel.wall_radius(wall_s[60]))),
                    xytext=(4, 6), textcoords="offset points", fontsize=8,
                    color=colours["secondary"])
        ax.grid(False)
        ax.set_xlim(funnel.lower - 0.1, funnel.upper + 0.05)
        ax.set_ylim(0.0, min(float(rho[-1]), funnel.max_radius + 0.2))
        ax.set_title("Free-energy surface")
        ax.set_xlabel("projection on the axis, s (nm)")
        ax.set_ylabel("distance from the axis, ρ (nm)")
        bar = figure.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
        bar.outline.set_visible(False)
        bar.set_label("F (kJ/mol)", color=colours["secondary"], fontsize=8)
        bar.ax.tick_params(colors=colours["muted"], labelsize=8, length=0)

        # -- s(t) -----------------------------------------------------------
        ax = axes[1, 1]
        ax.axhspan(*unbound, color=colours["neutral"], alpha=0.12, linewidth=0)
        for index, colvar in enumerate(colvars[:MAX_LINES]):
            x, y = _thin(colvar["time"] / 1000.0, colvar["s"])
            ax.plot(x, y, color=colours["series"][index], linewidth=0.8,
                    label=f"walker {index}")
        if len(colvars) > 1:
            ax.legend(loc="upper left", ncols=min(4, len(colvars)), handlelength=1.2,
                      columnspacing=1.0, labelcolor=colours["secondary"],
                      bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
        longest = max(c["time"][-1] for c in colvars) / 1000.0
        ax.text(longest, funnel.upper + 0.62, "shaded: unbound window", fontsize=8,
                color=colours["secondary"], ha="right", va="top")
        ax.set_xlim(0.0, longest)
        ax.set_ylim(funnel.lower - 0.2, funnel.upper + (1.1 if len(colvars) > 1 else 0.7))
        title = "Ligand position along the axis"
        if len(colvars) > MAX_LINES:
            title += f" (first {MAX_LINES} of {len(colvars)} walkers)"
        ax.set_title(title)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("s (nm)")

        path = Path(path) if path is not None else workdir / "analysis.png"
        figure.savefig(path)
    return path


def plot_summary(workdir, path=None, theme: str = "light"):
    """Draw the replicas of a calculation together (run ``analyze_stage`` first).
    Returns the path of the PNG written."""
    matplotlib, Figure, Canvas, _ = _matplotlib()
    colours = THEMES[theme]
    workdir = Path(workdir)
    with open(workdir / "summary.json") as handle:
        summary = json.load(handle)
    runs = []
    for replica in summary["replicas"]:
        with open(Path(replica["workdir"]) / "result.json") as handle:
            result = json.load(handle)
        runs.append((np.array(result["dG_time_ns"]), np.array(result["dG_series"])))
    blue = colours["series"][0]

    with matplotlib.rc_context(_style(colours)):
        figure = Figure(figsize=(8.0, 5.0), dpi=150)
        Canvas(figure)
        ax = figure.subplots(gridspec_kw=dict(left=0.1, right=0.86, top=0.78, bottom=0.12))
        headline = f"ΔG° = {summary['dG']:.1f} kJ/mol  ({summary['dG_kcal']:.2f} kcal/mol)"
        figure.text(0.1, 0.93, headline, fontsize=15, fontweight="bold", ha="left", va="center")
        detail = (f"mean of {summary['n_replicas']} replica"
                  + ("s" if summary["n_replicas"] > 1 else ""))
        if summary["ci95"] is not None:
            detail += f" · 95% interval {summary['ci95'][0]:.1f} to {summary['ci95'][1]:.1f} kJ/mol"
        if summary["no_binding"]:
            detail += " · no binding (within 1 kT of the empty funnel)"
        figure.text(0.1, 0.875, detail, fontsize=9, color=colours["secondary"], ha="left", va="center")

        # the replicas are interchangeable: draw them as context, their mean as the subject
        for index, (time, series) in enumerate(runs):
            ax.plot(time, series, color=colours["neutral"], linewidth=1.0, alpha=0.75,
                    label="single replicas" if index == 0 else None)
        if len(runs) > 1:
            end = min(time[-1] for time, _ in runs)
            common = np.linspace(max(time[0] for time, _ in runs), end, 60)
            mean = np.mean([np.interp(common, time, series) for time, series in runs], axis=0)
            ax.plot(common, mean, color=blue, linewidth=1.9, label="mean of the replicas")
            _end_dot(ax, common[-1], mean[-1], blue, colours)
            ax.annotate(f"{mean[-1]:.1f}", (common[-1], mean[-1]), xytext=(9, 0),
                        textcoords="offset points", fontsize=9, color=colours["ink"], va="center")
            ax.legend(loc="upper right", labelcolor=colours["secondary"], handlelength=1.6)
        ax.set_title("Binding free energy as the bias accumulates")
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("ΔG° (kJ/mol)")

        path = Path(path) if path is not None else workdir / "summary.png"
        figure.savefig(path)
    return path
