"""The stages of a calculation and what each leaves on disk.

    workdir/config.yaml
    workdir/prepared/   system.xml  topology.cif  funnel.yaml  funnel.pdb  report.json  meta.json
    workdir/equil/      state.xml  equil.json
    workdir/rep0/ ...   run.yaml  HILLS  COLVAR  checkpoint.zip  traj.dcd  result.json
    workdir/summary.json

Every stage reads what the previous one wrote, so a calculation can be resumed
or re-analysed from any point.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from .analysis import analyze_replicas
from .config import Config
from .engine.bias import FunnelBias
from .engine.equilibration import equilibrate
from .engine.platform import select_platform
from .engine.runner import MetadynamicsRun, RunSettings
from .funnel.geometry import Funnel
from .funnel.validate import validate_funnel
from .funnel.visualize import write_funnel_pdb
from .system import inputs
from .system.builder import (MembraneSettings, Prepared, add_barostat, prepare,
                             prepare_from_system)
from .system.selection import classify

log = logging.getLogger(__name__)


# ---- prepare ----------------------------------------------------------------

def prepare_stage(config: Config, workdir) -> Prepared:
    workdir = Path(workdir)
    out = workdir / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    config.save(workdir / "config.yaml")
    source = config.input
    platform, _ = select_platform(config.run.platform, config.run.precision)

    if source.protein and source.ligand:
        prepared = prepare(source.protein, source.ligand, config.system, config.membrane,
                           config.funnel, config.funnel_overrides, platform=platform)
    else:
        kwargs = dict(cutoff=config.system.cutoff, hydrogen_mass=config.system.hydrogen_mass)
        if source.amber:
            loaded = inputs.load_amber(source.amber["prmtop"], source.amber["coordinates"], **kwargs)
        elif source.gromacs:
            loaded = inputs.load_gromacs(source.gromacs["top"], source.gromacs["gro"],
                                         source.gromacs.get("include_dir"), **kwargs)
        elif source.charmm:
            kwargs["cutoff"] = max(kwargs["cutoff"], 1.2)
            loaded = inputs.load_charmm(source.charmm["psf"], source.charmm["coordinates"],
                                        source.charmm["parameters"], source.charmm.get("box"),
                                        **kwargs)
        else:
            raise ValueError("config.input needs protein + ligand, or one of amber/gromacs/charmm")
        prepared = prepare_from_system(*loaded, ligand=source.ligand_selection,
                                       settings=config.system, funnel_settings=config.funnel,
                                       funnel_overrides=config.funnel_overrides)
    save_prepared(prepared, out)
    log.info("prepared system: %d atoms\n%s", prepared.topology.getNumAtoms(), prepared.report)
    return prepared


def save_prepared(prepared: Prepared, out: Path) -> None:
    with open(out / "system.xml", "w") as handle:
        handle.write(mm.XmlSerializer.serialize(prepared.system))
    with open(out / "topology.cif", "w") as handle:
        app.PDBxFile.writeFile(prepared.topology, prepared.positions * unit.nanometer, handle,
                               keepIds=True)
    np.save(out / "positions.npy", prepared.positions)     # full precision, unlike the CIF
    prepared.funnel.save(out / "funnel.yaml")
    write_funnel_pdb(out / "funnel.pdb", prepared.funnel, prepared.positions)
    with open(out / "report.json", "w") as handle:
        json.dump(prepared.report.to_dict(), handle, indent=2, default=float)
    meta = dict(restraint_group=prepared.restraint_group, membrane=prepared.membrane,
                ligand_atoms=[int(i) for i in prepared.selection.ligand],
                n_atoms=prepared.topology.getNumAtoms())
    with open(out / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)


def load_prepared(workdir, config: Optional[Config] = None) -> Prepared:
    workdir = Path(workdir)
    out = workdir / "prepared"
    config = config or Config.load(workdir / "config.yaml")
    with open(out / "meta.json") as handle:
        meta = json.load(handle)
    with open(out / "system.xml") as handle:
        system = mm.XmlSerializer.deserialize(handle.read())
    topology = app.PDBxFile(str(out / "topology.cif")).topology
    positions = np.load(out / "positions.npy")
    funnel = Funnel.load(out / "funnel.yaml")
    selection = classify(topology, meta["ligand_atoms"])
    from .funnel.validate import FunnelReport
    return Prepared(topology, system, positions, funnel, selection, FunnelReport(),
                    meta["restraint_group"], meta["membrane"], config.system)


# ---- equilibrate ------------------------------------------------------------

def equilibrate_stage(config: Config, workdir, device: Optional[int] = None) -> dict:
    workdir = Path(workdir)
    out = workdir / "equil"
    out.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared(workdir, config)
    platform, properties = select_platform(config.run.platform, config.run.precision, device)
    result = equilibrate(prepared, platform, properties, config.equilibration)

    positions = np.array(result.state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = np.array(result.state.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    report = validate_funnel(prepared.funnel, positions, prepared.selection, box=box,
                             cutoff=config.system.cutoff, tumbling=not prepared.membrane)
    wall_ok = result.wall_ok(config.system.temperature)
    summary = dict(wall_energy_mean=result.wall_energy_mean, wall_energy_max=result.wall_energy_max,
                   wall_ok=wall_ok, box_lengths=result.box_lengths.tolist(),
                   funnel_checks=report.to_dict())
    with open(out / "state.xml", "w") as handle:
        handle.write(mm.XmlSerializer.serialize(result.state))
    with open(out / "equil.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    with open(out / "equilibrated.cif", "w") as handle:
        app.PDBxFile.writeFile(prepared.topology, result.state.getPositions(), handle, keepIds=True)
    if not wall_ok:
        log.warning("the funnel walls act on the bound ligand (mean %.2f kJ/mol); widen the "
                    "funnel (funnel_overrides: {width: ...}) before production",
                    result.wall_energy_mean)
    log.info("after equilibration:\n%s", report)
    report.raise_for_errors()
    return summary


# ---- production -------------------------------------------------------------

def run_stage(config: Config, workdir, replica: int = 0, device: Optional[int] = None,
              length_ns: Optional[float] = None) -> Path:
    """Run (or continue) one replica. Returns its directory."""
    workdir = Path(workdir)
    prepared = load_prepared(workdir, config)
    with open(workdir / "equil" / "state.xml") as handle:
        state = mm.XmlSerializer.deserialize(handle.read())
    run_cfg, temperature = config.run, config.system.temperature
    dt = run_cfg.timestep_fs / 1000.0

    system = prepared.system
    add_barostat(system, temperature, config.system.pressure, prepared.membrane)
    bias = FunnelBias(system, prepared.funnel, config.metad, periodic=True)
    integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin,
                                             run_cfg.friction / unit.picosecond,
                                             dt * unit.picosecond)
    seed = run_cfg.seed + replica if run_cfg.seed else 0
    if seed:
        integrator.setRandomNumberSeed(seed)
    platform, properties = select_platform(run_cfg.platform, run_cfg.precision, device)
    simulation = app.Simulation(prepared.topology, system, integrator, platform, properties)
    simulation.context.setState(state)
    # fresh velocities decorrelate the replicas (a restart overwrites them again)
    if seed:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    else:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    hill_ps = config.metad.pace * dt
    selection = prepared.selection
    solute = np.sort(np.concatenate([selection.protein, selection.ligand, selection.other]))
    settings = RunSettings(
        checkpoint_hills=max(1, round(run_cfg.checkpoint_ps / hill_ps)),
        trajectory_hills=round(run_cfg.trajectory_ps / hill_ps) if run_cfg.trajectory_ps else 0,
        trajectory_atoms=[int(i) for i in solute] if run_cfg.trajectory_solute_only else None)
    directory = workdir / f"rep{replica}"
    run = MetadynamicsRun(simulation, bias, directory, prepared.restraint_group, settings)
    length = run_cfg.length_ns if length_ns is None else length_ns
    run.run(round(length * 1000.0 / hill_ps))
    run.close()
    return directory


def run_replicas(config: Config, workdir, gpus: Optional[Sequence[int]] = None,
                 length_ns: Optional[float] = None) -> list[Path]:
    """Run all replicas: in this process one after another, or, with ``gpus``,
    as one subprocess per replica with each device working through its share."""
    workdir = Path(workdir)
    replicas = list(range(config.run.replicas))
    gpus = gpus if gpus is not None else config.run.gpus
    if not gpus:
        return [run_stage(config, workdir, i, None, length_ns) for i in replicas]

    failures = []

    def work(device, mine):
        for replica in mine:
            command = [sys.executable, "-m", "fmeta.cli", "run", str(workdir),
                       "--replica", str(replica), "--device", str(device)]
            if length_ns is not None:
                command += ["--length-ns", str(length_ns)]
            with open(workdir / f"rep{replica}.log", "a") as logfile:
                if subprocess.call(command, stdout=logfile, stderr=subprocess.STDOUT) != 0:
                    failures.append(replica)

    threads = [threading.Thread(target=work, args=(device, replicas[k::len(gpus)]))
               for k, device in enumerate(gpus)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"replica(s) {sorted(failures)} failed; see rep<N>.log in {workdir}")
    return [workdir / f"rep{i}" for i in replicas]


# ---- analysis ---------------------------------------------------------------

def analyze_stage(workdir, **kwargs) -> dict:
    workdir = Path(workdir)
    directories = sorted((d for d in workdir.glob("rep*") if (d / "HILLS").exists()),
                         key=lambda d: int(d.name[3:]))
    if not directories:
        raise FileNotFoundError(f"no replica directories with a HILLS file under {workdir}")
    return analyze_replicas(directories, write_to=workdir / "summary.json", **kwargs)


def format_summary(summary: dict) -> str:
    lines = [f"dG0 = {summary['dG']:.2f} kJ/mol  ({summary['dG_kcal']:.2f} kcal/mol)"
             f"  from {summary['n_replicas']} replica(s)"]
    if summary["ci95"] is not None:
        low, high = summary["ci95"]
        lines.append(f"95% CI: [{low:.2f}, {high:.2f}] kJ/mol   (SEM {summary['sem']:.2f})")
    for replica in summary["replicas"]:
        lines.append(f"  {Path(replica['workdir']).name}: {replica['dG_plateau']:8.2f} "
                     f"+/- {replica['dG_plateau_std']:.2f} kJ/mol over {replica['simulated_ns']:.1f} ns, "
                     f"{replica['rebinding_events']} rebinding event(s)")
    if not summary["converged"]:
        lines.append("NOT CONVERGED:")
        for directory, problems in summary["problems"].items():
            lines += [f"  {Path(directory).name}: {problem}" for problem in problems]
    return "\n".join(lines)


def run_all(config: Config, workdir) -> dict:
    prepare_stage(config, workdir)
    equilibrate_stage(config, workdir)
    run_replicas(config, workdir)
    return analyze_stage(workdir)
