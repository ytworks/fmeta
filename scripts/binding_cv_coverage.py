"""Measure observed (s, rho) footprint in one common trajectory prefix, in nm²."""

import argparse
import json
from pathlib import Path

import numpy as np
from paper_regions import identity, walkers

from fmeta.config import Config
from fmeta.engine.hills import read_colvar
from fmeta.engine.runner import load_run_metadata

CELL_NM = np.array([0.025, 0.030])  # Fixed, zero-origin grid of the published TYK2 C/D comparison.


def coverage(root, regions, cut_ns, discard_ns=2.):
    if not 0 <= discard_ns < cut_ns or not np.isfinite(cut_ns):
        raise ValueError("the cutoff must be finite and later than the discard")
    hashes = identity(root)
    if any(regions.get(key) != value for key, value in hashes.items()):
        raise ValueError("the frozen regions belong to a different prepared system")
    run = Config.load(root / "config.yaml").run
    paths = walkers(root)
    grid = load_run_metadata(paths[0])["grid"]
    lower = np.array([grid["s_min"], 0.0])
    upper = np.array([grid["s_max"], grid["rho_max"]])
    first = np.floor(lower / CELL_NM).astype(int)
    shape = np.ceil(upper / CELL_NM).astype(int) - first
    x, y = [np.arange(count) + start for count, start in zip(shape, first, strict=True)]
    radial_width = np.minimum((y + 1) * CELL_NM[1], upper[1]) - np.maximum(y * CELL_NM[1], lower[1])
    zones = {"all": (lower[0], upper[0]), "site": regions["site"], "bulk": regions["unbound"]}
    seen = {name: np.zeros((len(paths), np.prod(shape)), dtype=bool) for name in zones}
    totals = {name: [] for name in zones}
    histories = []
    outside = 0
    for i, path in enumerate(paths):
        meta = load_run_metadata(path)
        if meta["grid"] != grid:
            raise ValueError(f"{path} uses a different native bias grid")
        data = read_colvar(path / "COLVAR")
        t = data["time"]
        start, end = 1000 * discard_ns, 1000 * cut_ns
        keep = (t > start) & (t <= end + 1.e-6)
        stride = (meta["settings"].get("colvar_stride") or meta["bias"]["pace"]) * meta["timestep_ps"]
        expected = stride * np.arange(int(np.floor(start / stride)) + 1, int(np.floor((end + 1.e-6) / stride)) + 1)
        if not expected.size or t[keep].size != expected.size or not np.allclose(t[keep], expected, rtol=0, atol=1.e-6):
            raise ValueError(f"{path} lacks the full recording window {discard_ns:g}–{cut_ns:g} ns")
        samples = np.column_stack([data[k][keep] for k in ("s", "rho")])
        if not np.isfinite(samples).all():
            raise ValueError(f"{path} has nonfinite CV samples")
        inside = np.all((samples >= lower) & (samples <= upper), axis=1)
        outside += int((~inside).sum())
        indices = np.clip(np.floor(samples[inside] / CELL_NM).astype(int) - first, 0, shape - 1)
        cells = np.ravel_multi_index(indices.T, shape)
        histories.append(path.relative_to(root).parts[0])
        for name, (lo, hi) in zones.items():
            mask = (samples[:, 0] >= lo) & (samples[:, 0] < hi) if name != "all" else np.ones(len(samples), bool)
            totals[name].append(int(mask.sum()))
            seen[name][i, cells[mask[inside]]] = True

    history_names, group_index = np.unique(histories, return_inverse=True)
    result = {}
    for name, (lo, hi) in zones.items():
        zone_lo, zone_hi = max(lo, lower[0]), min(hi, upper[0])
        axial_width = np.minimum((x + 1) * CELL_NM[0], zone_hi) - np.maximum(x * CELL_NM[0], zone_lo)
        area = np.outer(np.maximum(0, axial_width), radial_width).ravel()
        walkers_seen = seen[name]
        by_history = np.stack([walkers_seen[group_index == g].any(axis=0) for g in range(len(history_names))])
        union = walkers_seen.any(axis=0)
        visited_histories = {g for g, n in zip(histories, totals[name], strict=True) if n}
        result[name] = dict(records=sum(totals[name]), physical_walkers_any=int(np.count_nonzero(totals[name])),
                            independent_histories_any=len(visited_histories),
                            union_cells=int(union.sum()), union_area_nm2=float(area[union].sum()),
                            median_walker_cells=float(np.median(walkers_seen.sum(axis=1))),
                            median_walker_area_nm2=float(np.median(walkers_seen @ area)),
                            median_history_cells=float(np.median(by_history.sum(axis=1))),
                            median_history_area_nm2=float(np.median(by_history @ area)))
    return dict(cut_ns=cut_ns, discard_ns=discard_ns, base_seed=run.seed, prepared_hashes=hashes,
                physical_walkers=len(paths), independent_bias_histories=len(history_names),
                records_outside_native_rectangle=outside, grid_origin_nm=[0., 0.], cell_size_nm=CELL_NM.tolist(),
                native_grid_domain_nm=[lower[0], upper[0], lower[1], upper[1]], site_nm=regions["site"],
                bulk_nm=regions["unbound"], zones=result,
                walkers_both_site_and_bulk=int(np.sum((np.array(totals["site"]) > 0) & (np.array(totals["bulk"]) > 0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--length-ns", type=float, required=True)
    parser.add_argument("--discard-ns", type=float, default=2.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = coverage(args.workdir, json.loads(args.regions.read_text()), args.length_ns, args.discard_ns)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
