"""Refit single-walker paper estimates at an absolute observation cadence using all recorded hills."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from fmeta import analysis
from fmeta.config import Config
from fmeta.engine.runner import load_run_metadata
from fmeta.workflow import analyze_paper_stage


def recorded_count(duration_ps, interval_ps):
    if not math.isfinite(interval_ps) or interval_ps <= 0:
        raise ValueError("the observation and recording intervals must be finite and positive")
    count = round(duration_ps / interval_ps)
    if not math.isclose(count * interval_ps, duration_ps, abs_tol=1e-6, rel_tol=0):
        raise ValueError("the requested time does not fall on the recording grid")
    return count


def copy_observations(source, destination, native_ps, period_ps, end_ps):
    every = recorded_count(period_ps, native_ps)
    total = recorded_count(end_ps, native_ps)
    if every < 1 or total % every:
        raise ValueError("the requested grid does not coincide with native COLVAR records")
    index = 0
    with source.open() as raw, destination.open("w") as selected:
        for line in raw:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                selected.write(line)
                continue
            index += 1
            time_ps = float(stripped.split(maxsplit=1)[0])
            if not math.isclose(time_ps, index * native_ps, abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"missing or irregular raw timestamp in {source} at row {index}")
            if index % every == 0:
                selected.write(line)
            if index == total:
                break
    if index != total:
        raise ValueError(f"{source} does not reach {end_ps:g} ps")


def analyze(workdir, regions, cut_ns, discard_ns, period_ps):
    workdir = workdir.resolve()
    cfg = Config.load(workdir / "config.yaml")
    if cfg.run.walkers != 1 or not 0 <= discard_ns < cut_ns:
        raise ValueError("this observation comparison requires independent single-walker runs and ordered cutoffs")
    n_samples = recorded_count(1000 * (cut_ns - discard_ns), period_ps)
    recorded_count(1000 * discard_ns, period_ps)
    n_states = math.ceil((cut_ns - discard_ns) * 1000 / analysis.SLICE_PS)
    with TemporaryDirectory() as temporary:
        view = Path(temporary)
        for name in ("config.yaml", "prepared", "equil"):
            (view / name).symlink_to(workdir / name)
        native_periods = []
        for i in range(cfg.run.replicas):
            rep = f"rep{i}"
            raw, filtered = workdir / rep, view / rep
            filtered.mkdir()
            meta = load_run_metadata(raw)
            native_ps = (meta["settings"].get("colvar_stride") or meta["bias"]["pace"]) * meta["timestep_ps"]
            native_periods.append(native_ps)
            for name in ("HILLS", "run.yaml", "device.json"):
                (filtered / name).symlink_to(raw / name)
            if (raw / "SYNC").exists():
                (filtered / "SYNC").symlink_to(raw / "SYNC")
            copy_observations(raw / "COLVAR", filtered / "COLVAR", native_ps, period_ps, cut_ns * 1000)
        old_samples, old_states = analysis.MAX_SAMPLES, analysis.MAX_SLICES
        try:
            analysis.MAX_SAMPLES = max(old_samples, n_samples)
            analysis.MAX_SLICES = max(old_states, n_states)
            result = analyze_paper_stage(view, regions, length_ns=cut_ns, discard_ns=discard_ns)
        finally:
            analysis.MAX_SAMPLES, analysis.MAX_SLICES = old_samples, old_states
    for i, row in enumerate(result["replicas"]):
        if (row.get("n_mbar_samples") != n_samples or row.get("n_mbar_states") != n_states or
                row["unknown_samples"] != 0 or row["peer_postcut_samples"] != 0):
            raise ValueError(f"rep{i} lacks the complete observation grid or its full bias history")
        row["workdir"] = str(workdir / f"rep{i}")
    result["observation_grid"] = dict(period_ps=period_ps, origin_ps=0, native_periods_ps=native_periods,
                                      samples_per_replica=n_samples, target_slice_width_ps=analysis.SLICE_PS,
                                      states_per_replica=n_states, hill_history="all native deposited hills")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--length-ns", type=float, required=True)
    parser.add_argument("--discard-ns", type=float, default=2.)
    parser.add_argument("--observation-ps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.workdir, args.regions, args.length_ns, args.discard_ns, args.observation_ps)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
