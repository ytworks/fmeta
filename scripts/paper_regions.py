"""Freeze binding-site and bulk intervals from an example, and check returns in its DCDs.

Use `define` only on the historical calibration run. Use `check` with that region
file on every new arm. Both commands read inputs and write one separate JSON file.
"""

import argparse
import base64
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import yaml
from openmm import unit
from scipy.spatial import cKDTree, distance

from fmeta.analysis.binding import default_unbound_window
from fmeta.engine.hills import read_colvar
from fmeta.engine.runner import load_run_metadata
from fmeta.engine.trajectory import _dcd_layout, count_dcd_frames
from fmeta.funnel.geometry import KB, Funnel, compute_cvs, funnel_frame
from fmeta.funnel.obstacles import Obstacles
from fmeta.funnel.validate import funnel_points, ligand_radius
from fmeta.system.imaging import _wrap
from fmeta.system.selection import classify

CONTACT_NM = 0.45  # The protein/ligand heavy-atom contact in system.builder.pose_contact.
CV_TOLERANCE_NM = 2e-3  # fmeta.check's periodic_images dynamic coordinate check.


def digest(path):
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def walkers(root):
    run = yaml.safe_load((root / "config.yaml").read_text())["run"]
    replicas = [root / f"rep{i}" for i in range(run["replicas"])]
    return ([rep / f"w{w}" for rep in replicas for w in range(run["walkers"])]
            if run["walkers"] > 1 else replicas)


def prepared(root):
    prep = root / "prepared"
    data = json.loads((prep / "meta.json").read_text())
    cif = app.PDBxFile(str(prep / "topology.cif"))
    selection = classify(cif.topology, data["ligand_atoms"], kept=data.get("kept_atoms", ()))
    return cif, selection, Funnel.load(prep / "funnel.yaml")


def identity(root):
    paths = {"system": "prepared/system.xml", "funnel": "prepared/funnel.yaml",
             "positions": "prepared/positions.npy", "topology": "prepared/topology.cif",
             "equil_state": "equil/state.xml"}
    return {f"{key}_sha256": digest(root / path) for key, path in paths.items()}


def initial_sample(path):
    with (path / "COLVAR").open() as handle:
        fields = handle.readline().split()[2:]
        row = next(line for line in handle if line.strip() and not line.startswith("#"))
    sample = dict(zip(fields, map(float, row.split()), strict=True))
    if sample["bias"] != 0.0:
        raise ValueError(f"first COLVAR is already biased: {path}")
    return sample


def define(root):
    cif, selection, funnel = prepared(root)
    xyz = np.load(root / "prepared/positions.npy")
    state = mm.XmlSerializer.deserialize((root / "equil/state.xml").read_text())
    snapshots = ((xyz, np.asarray(cif.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))),
                 (np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
                  np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))))
    atoms = list(cif.topology.atoms())
    protein, ligand = selection.heavy_atoms("protein"), selection.heavy_atoms("ligand")
    shell = protein[cKDTree(xyz[ligand]).query(xyz[protein])[0] < CONTACT_NM]
    origin, axis, _ = funnel_frame(funnel, xyz)
    shell_z = (xyz[shell] - origin) @ axis
    site = [float(max(funnel.lower, shell_z.min())), float(min(funnel.inflection, shell_z.max()))]
    pocket = sorted({atoms[i].residue.index for i in shell})

    initial = [initial_sample(path) for path in walkers(root)]
    initial_s = np.asarray([sample["proj"] for sample in initial])
    if len(initial_s) != 32 or any(sample["time"] != 1.0 for sample in initial):
        raise ValueError("this calibration uses 32 first zero-bias COLVAR samples at 1 ps")
    bound_below = float(initial_s.mean() + initial_s.std(ddof=1))
    if not site[0] <= initial_s.min() <= initial_s.max() <= site[1] or bound_below > site[1]:
        raise ValueError("the initial bound positions are inconsistent with the crystal shell")

    config = json.loads((root / "equil/equil.json").read_text())
    cutoff = yaml.safe_load((root / "config.yaml").read_text())["system"]["cutoff"]
    if not config["wall_ok"] or not config["funnel_checks"]["ok"]:
        raise ValueError("equilibration validation failed")
    # Start with the package's nominal bulk interval. Keep its contiguous upper suffix
    # whose axis+wall ring lies at least one interaction cutoff plus a ligand radius
    # from protein, cofactor and lipids, including neighbouring periodic copies.
    nominal = default_unbound_window(funnel)
    meta = load_run_metadata(walkers(root)[0])
    grid = np.linspace(meta["grid"]["s_min"], meta["grid"]["s_max"], meta["grid"]["n_s"])
    probes = np.unique(np.r_[nominal, grid[(grid > nominal[0]) & (grid < nominal[1])]])
    gap = np.full(len(probes), np.inf)
    centres = []
    obstacles = np.concatenate([selection.heavy_atoms(k) for k in ("protein", "other", "lipid")])
    for coords, box in snapshots:
        origin, axis, centre = funnel_frame(funnel, coords)
        centres.append(float((centre - origin) @ axis))
        surface = Obstacles(coords[obstacles])
        radius = ligand_radius(funnel, coords)
        for i, s in enumerate(probes):
            points = funnel_points(funnel, coords, [s])
            local = surface.clearance(points).min()
            periodic = surface.nearest_image_distance(points, box)
            gap[i] = min(gap[i], min(local, periodic) - radius)
    start = 1 + int(np.where(gap < cutoff)[0].max(initial=-1))
    if start >= len(probes) - 1 or min(centres) < site[0] or max(centres) > site[1]:
        raise ValueError("crystal/equil geometry does not provide usable site and bulk intervals")
    residues = list(cif.topology.residues())
    return dict(**identity(root), site=site, unbound=[float(probes[start]), float(nominal[1])],
                core_radius=funnel.buffer, bound_below=bound_below,
                pocket_residues=pocket,
                pocket_labels=[f"{residues[i].chain.id}:{residues[i].name}{residues[i].id}" for i in pocket],
                cutoff_nm=cutoff, contact_nm=CONTACT_NM, nominal_unbound=list(nominal),
                minimum_static_bulk_gap_nm=float(gap[start:].min()),
                reference_center_s_nm=centres, initial_s_mean_nm=float(initial_s.mean()),
                initial_s_sd_nm=float(initial_s.std(ddof=1)), initial_sample_count=len(initial),
                initial_sample_time_ps=initial[0]["time"], sources=["prepared/positions.npy",
                "prepared/topology.cif", "prepared/funnel.yaml", "equil/state.xml", "walkers' first COLVAR"],
                rule="Crystal heavy-atom contact shell projected onto the funnel axis, clipped to "
                     "lower wall and inflection. Bulk is the nominal bulk interval trimmed at the "
                     "lower end to one interaction cutoff in prepared and equilibrated coordinates. "
                     "Bound crossing threshold is mean + sample SD of the first zero-bias COLVAR "
                     "of baseline walkers launched from the same equilibrated pose. This measures "
                     "initial coordinate spread, not the equilibrium distribution.")


def dcd_frames(path, n_atoms, n_frames):
    header, frame_size = _dcd_layout(path)
    expected_size = 56 + 3 * (8 + 4 * n_atoms)
    if frame_size != expected_size or count_dcd_frames(path) < n_frames:
        raise ValueError(f"unexpected/incomplete periodic DCD layout: {path}")
    dtype = np.dtype([("record", "<i4"), ("xyz", "<f4", n_atoms), ("end", "<i4")])
    with path.open("rb") as handle:
        handle.seek(header)
        for _ in range(n_frames):
            raw = handle.read(frame_size)
            marker, a, cos_gamma, b, cos_beta, cos_alpha, c, end = struct.unpack_from("<i6di", raw)
            if marker != 48 or end != 48:
                raise ValueError("invalid OpenMM DCD periodic box")
            sin_gamma = np.sqrt(1 - cos_gamma**2)
            cx = c * cos_beta
            cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
            cz = np.sqrt(c*c - cx*cx - cy*cy)
            box = 0.1 * np.array([[a, 0, 0], [b * cos_gamma, b * sin_gamma, 0], [cx, cy, cz]])
            xyz = np.frombuffer(raw, dtype=dtype, count=3, offset=56)
            if (xyz["record"] != 4*n_atoms).any() or (xyz["end"] != 4*n_atoms).any():
                raise ValueError("DCD atom count/record markers disagree")
            yield np.asarray(xyz["xyz"].T, dtype=float) * 0.1, box


def lipid_checkpoint(path, colvar, selection, spec, cut_ns):
    """Where the durable full-atom checkpoint is within the analysed prefix, check the lipids."""
    lipids = selection.heavy_atoms("lipid")
    if not len(lipids):
        return None
    with zipfile.ZipFile(path / "checkpoint.zip") as checkpoint:
        checkpoint_ps = json.loads(checkpoint.read("meta.json"))["time_ps"]
        if checkpoint_ps > cut_ns * 1000 + 1e-6:
            return dict(status="outside_prefix")
        found = np.flatnonzero(np.isclose(colvar["time"], checkpoint_ps, rtol=0, atol=1e-5))
        if len(found) != 1:
            raise ValueError(f"checkpoint time absent in COLVAR: {path}")
        s = colvar["s"][found[0]]
        if not spec["unbound"][0] <= s < spec["unbound"][1]:
            return dict(status="outside_bulk", time_ns=checkpoint_ps / 1000)
        state = mm.XmlSerializer.deserialize(checkpoint.read("state.xml").decode())
    coords = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    ligand = coords[selection.heavy_atoms("ligand")]
    origin = ligand.mean(axis=0)
    gap = cKDTree(_wrap(coords[lipids] - origin, box)).query(_wrap(ligand - origin, box))[0].min()
    return dict(status="observed", time_ns=checkpoint_ps / 1000, minimum_lipid_gap_nm=float(gap),
                cutoff_clear=bool(gap >= spec["cutoff_nm"]))


def check(root, spec, cut_ns):
    if any(spec[k] != v for k, v in identity(root).items()):
        raise ValueError("frozen region and prepared system do not match")
    cif, selection, funnel = prepared(root)
    atoms = list(cif.topology.atoms())
    ligand = selection.heavy_atoms("ligand")
    receptor = np.concatenate([selection.heavy_atoms(k) for k in ("protein", "other")])
    pocket = set(spec["pocket_residues"])
    shell = np.array([i for i in receptor if atoms[i].residue.index in pocket])
    equil = mm.XmlSerializer.deserialize((root / "equil/state.xml").read_text())
    reference_xyz = {"prepared": np.load(root / "prepared/positions.npy"),
                     "equil": equil.getPositions(asNumpy=True).value_in_unit(unit.nanometer)}
    reference_wall = {name: float(funnel.restraint_energy(*compute_cvs(funnel, xyz)))
                      for name, xyz in reference_xyz.items()}
    core_cv = np.hypot(spec["core_radius"], funnel.eps)  # Logged rho includes eps squared.
    checks = []
    for path in walkers(root):
        meta = load_run_metadata(path)
        indices = meta["settings"]["trajectory_atoms"]
        if indices is None:
            indices = range(len(atoms))
        else:
            indices = sorted(indices)  # TrajectoryWriter sorts the saved atom subset.
        mapping = {atom: i for i, atom in enumerate(indices)}
        subset = [np.array([mapping[i] for i in group]) for group in (ligand, receptor, shell)]
        definition = funnel.to_dict()
        for name in ("ligand_atoms", "p0_atoms", "p1_atoms"):
            definition[name] = [mapping[i] for i in definition[name]]
        mapped_funnel = Funnel.from_dict(definition)
        lag_ps = meta["settings"]["trajectory_hills"] * meta["bias"]["pace"] * meta["timestep_ps"]
        length = int(np.floor(cut_ns * 1000 / lag_ps + 1.e-9))
        colvar = read_colvar(path / "COLVAR")
        times = np.arange(1, length + 1) * lag_ps
        site_contact_mask = np.zeros(length, dtype=bool)
        matches = np.searchsorted(colvar["time"], times)
        if length < 1 or matches.max() >= len(colvar["time"]):
            raise ValueError(f"the trajectory does not cover {cut_ns} ns: {path}")
        if not np.allclose(colvar["time"][matches], times, rtol=0, atol=1e-5):
            raise ValueError("DCD and COLVAR time grids disagree")
        s_values = colvar["s"][matches]
        rho_values = colvar["rho"][matches]
        wall_energies = funnel.restraint_energy(s_values, rho_values)
        kT = KB * meta["bias"]["temperature"]
        site_samples, site_contacts, confirmed_bounds = 0, 0, 0
        site_wall_count, site_wall_kt_count, bound_wall_kt_count = 0, 0, 0
        max_site_wall, max_bound_wall = 0., 0.
        bulk_samples, verified_bulk, core_samples, verified_core = 0, 0, 0, 0
        state = "bound"  # The shared equilibration state was verified when defining the intervals.
        visits = []
        min_bulk_gap = np.inf
        max_cv_gap = 0.0
        current = dcd_frames(path / "traj.dcd", len(indices), length)
        samples = zip(times, s_values, rho_values, wall_energies, current, strict=True)
        for index, (time_ps, s, rho, wall, (coords, box)) in enumerate(samples):
            reconstructed = np.array(compute_cvs(mapped_funnel, coords))
            max_cv_gap = max(max_cv_gap, np.max(np.abs(reconstructed - [s, rho])))
            site = spec["site"][0] <= s < spec["site"][1]
            bulk = spec["unbound"][0] <= s < spec["unbound"][1]
            if not (site or bulk):
                continue
            lig = coords[subset[0]]
            origin = lig.mean(axis=0)
            lig = _wrap(lig - origin, box)
            neighbours = coords[subset[2] if site else subset[1]]
            nearby = _wrap(neighbours - origin, box)
            # The shortest box height exceeds twice (cutoff + ligand radius); any
            # image in contact maps to these nearby coordinates under OpenMM reduction.
            if np.min(np.diag(box)) / 2 <= np.max(np.linalg.norm(lig, axis=1)) + spec["cutoff_nm"]:
                raise ValueError("box too short for the local contact test")
            dist = (distance.cdist(lig, nearby).min() if site else cKDTree(nearby).query(lig)[0].min())
            if site:
                site_samples += 1
                if dist >= spec["contact_nm"]:
                    continue
                site_contacts += 1
                site_contact_mask[index] = True
                site_wall_count += int(wall > 0)
                site_wall_kt_count += int(wall > kT)
                max_site_wall = max(max_site_wall, wall)
                if s > spec["bound_below"]:
                    continue
                confirmed_bounds += 1
                bound_wall_kt_count += int(wall > kT)
                max_bound_wall = max(max_bound_wall, wall)
                observed = "bound"
            else:
                bulk_samples += 1
                core_samples += int(rho <= core_cv)
                min_bulk_gap = min(min_bulk_gap, dist)
                if dist < spec["cutoff_nm"]:
                    continue
                verified_bulk += 1
                verified_core += int(rho <= core_cv)
                observed = "bulk"
            if state != observed:
                visits.append({"to": observed, "time_ns": float(time_ps / 1000)})
                state = observed
        if max_cv_gap > CV_TOLERANCE_NM:
            raise ValueError(f"DCD vs COLVAR CV difference {max_cv_gap} nm at {path}")
        checks.append(dict(walker=str(path.relative_to(root)), frames=length, frame_ps=lag_ps,
                           site_frames=site_samples, site_contact_frames=site_contacts,
                           site_contact_bits_b64=base64.b64encode(
                               np.packbits(site_contact_mask, bitorder="little")).decode("ascii"),
                           site_contact_wall_positive_frames=site_wall_count,
                           site_contact_wall_above_kT_frames=site_wall_kt_count,
                           maximum_site_contact_wall_kj_mol=float(max_site_wall) if site_contacts else None,
                           bound_contact_wall_above_kT_frames=bound_wall_kt_count,
                           maximum_bound_contact_wall_kj_mol=float(max_bound_wall) if confirmed_bounds else None,
                           bound_contact_frames=confirmed_bounds, bulk_frames=bulk_samples,
                           bulk_clear_frames=verified_bulk, bulk_core_frames=core_samples,
                           bulk_core_clear_frames=verified_core, max_cv_error_nm=float(max_cv_gap),
                           lipid_checkpoint=lipid_checkpoint(path, colvar, selection, spec, cut_ns),
                           minimum_bulk_receptor_gap_nm=float(min_bulk_gap) if bulk_samples else None,
                           departures=sum(v["to"] == "bulk" for v in visits),
                           returns=sum(v["to"] == "bound" for v in visits), passages=visits))
    run_config = yaml.safe_load((root / "config.yaml").read_text())["run"]
    return dict(region=spec, cut_ns=cut_ns, base_seed=run_config.get("seed", 0),
                reference_wall_kj_mol=reference_wall, walkers=checks,
                contact_mask_encoding="Base64 of little-bitorder packed bytes. Bit zero is the "
                                      "first DCD frame at frame_ps; ignore bits after frames.",
                notes="Contact-confirmed site returns are evaluated at the DCD stride. A crossing "
                      "can be missed between frames. Bulk frames require at least the configured "
                      "cutoff to protein and kept cofactors. Solute-only DCDs omit lipids/water. "
                      "Lipid clearance is tested in prepared/equilibration snapshots and any "
                      "full-atom checkpoint that lies in the bulk and within this time window.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("define", "check"):
        sub = subs.add_parser(action)
        sub.add_argument("workdir", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        if action == "check":
            sub.add_argument("--regions", type=Path, required=True)
            sub.add_argument("--length-ns", type=float, required=True)
    args = parser.parse_args()
    if args.action == "define":
        report = define(args.workdir)
    else:
        spec = json.loads(args.regions.read_text())
        report = check(args.workdir, spec, args.length_ns)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
