"""Production loop for funnel metadynamics, with exact restart.

Files written to the work directory:

    run.yaml         funnel, bias settings and grid (what the analysis needs)
    HILLS, COLVAR    PLUMED-style text logs
    checkpoint.zip   binary checkpoint + portable State XML + bias grid + counters,
                     replaced atomically so a crash never leaves a mixed state
    traj.dcd         optional trajectory of a subset of atoms
    run.log          progress and speed
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import yaml

from .bias import FunnelBias
from .hills import ColvarWriter, HillsFollower, HillsWriter, read_hills, sum_hills, truncate_data_file
from .trajectory import TrajectoryWriter

log = logging.getLogger(__name__)
_KJ = unit.kilojoule_per_mole


@dataclass
class RunSettings:
    colvar_stride: Optional[int] = None     # MD steps between COLVAR rows (default: the pace)
    checkpoint_hills: int = 250             # hills between checkpoints
    trajectory_hills: int = 0               # hills between trajectory frames (0: no trajectory)
    trajectory_atoms: Optional[Sequence[int]] = None   # default: all atoms
    # Multiple walkers: run directories of the other walkers that share this bias.
    # Their hills are read from their HILLS files every ``sync_hills`` of our own.
    walker_dirs: Optional[Sequence] = None
    sync_hills: int = 1


class MetadynamicsRun:
    """Drives ``simulation`` with ``bias`` and keeps the on-disk record consistent."""

    def __init__(self, simulation: app.Simulation, bias: FunnelBias, workdir,
                 restraint_group: Optional[int] = None, settings: Optional[RunSettings] = None,
                 restart: bool = True):
        self.simulation = simulation
        self.bias = bias
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.restraint_group = restraint_group
        self.settings = settings or RunSettings()
        self.dt = simulation.integrator.getStepSize().value_in_unit(unit.picosecond)
        bias.check_platform(simulation.context)
        bias.exclude_probe(simulation.integrator)
        self.pace = bias.settings.pace
        self.stride = self.settings.colvar_stride or self.pace
        if self.pace % self.stride:
            raise ValueError("colvar_stride must divide the hill pace")
        self.n_hills = 0
        self._traj: Optional[TrajectoryWriter] = None
        self._others = [HillsFollower(Path(d) / "HILLS")
                        for d in (self.settings.walker_dirs or [])
                        if Path(d).resolve() != self.workdir.resolve()]

        resume = restart and (self.workdir / "checkpoint.zip").exists()
        n_frames = self._restore() if resume else None
        if not resume:
            # A fresh run counts steps from zero, whatever equilibration came before.
            simulation.context.setStepCount(0)
            simulation.context.setTime(0.0)
            self._write_metadata()
        # With shared bias, start from everything the other walkers have deposited so far
        # (after a restart _restore() has put our own hills on the grid already).
        if self._others and self.sync_walkers():
            self.bias.push(simulation.context)
        self._hills = HillsWriter(self.workdir / "HILLS", bias.settings.bias_factor, append=resume)
        self._colvar = ColvarWriter(self.workdir / "COLVAR", append=resume)
        if self.settings.trajectory_hills > 0:
            self._traj = TrajectoryWriter(
                self.workdir / "traj.dcd", simulation.topology, self.dt,
                self.settings.trajectory_hills * self.pace, self.settings.trajectory_atoms,
                resume_frames=n_frames)

    # ---- bookkeeping ----------------------------------------------------
    def _write_metadata(self) -> None:
        data = dict(funnel=self.bias.funnel.to_dict(), bias=self.bias.settings.to_dict(),
                    grid=self.bias.grid.spec(), timestep_ps=self.dt,
                    settings=asdict(self.settings))
        if data["settings"]["walker_dirs"] is not None:
            data["settings"]["walker_dirs"] = [str(d) for d in data["settings"]["walker_dirs"]]
        if data["settings"]["trajectory_atoms"] is not None:
            data["settings"]["trajectory_atoms"] = [int(i) for i in data["settings"]["trajectory_atoms"]]
        with open(self.workdir / "run.yaml", "w") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=None)

    def _restraint_energy(self) -> float:
        if self.restraint_group is None:
            return 0.0
        state = self.simulation.context.getState(getEnergy=True, groups={self.restraint_group})
        return state.getPotentialEnergy().value_in_unit(_KJ)

    def checkpoint(self) -> None:
        context = self.simulation.context
        for writer in (self._hills, self._colvar, self._traj):
            if writer is not None:
                writer.flush()
        state = context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                 getIntegratorParameters=True)
        meta = dict(n_hills=self.n_hills, step=int(context.getStepCount()),
                    time_ps=context.getStepCount() * self.dt,
                    n_frames=None if self._traj is None else self._traj.n_frames)
        grid_bytes = io.BytesIO()
        np.save(grid_bytes, self.bias.grid.values)
        path = self.workdir / "checkpoint.zip"
        tmp = self.workdir / "checkpoint.zip.tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("checkpoint.chk", context.createCheckpoint())
            archive.writestr("state.xml", mm.XmlSerializer.serialize(state))
            archive.writestr("bias.npy", grid_bytes.getvalue())
            archive.writestr("meta.json", json.dumps(meta))
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def sync_walkers(self) -> int:
        """Add the hills the other walkers wrote since the last call. Returns how many."""
        added = 0
        for other in self._others:
            hills = other.read_new()
            if hills is not None:
                self.bias.grid.add_hills(hills.s, hills.rho, hills.height)
                added += len(hills)
        return added

    def _restore(self) -> Optional[int]:
        context = self.simulation.context
        meta = rollback_to_checkpoint(self.workdir)
        with zipfile.ZipFile(self.workdir / "checkpoint.zip") as archive:
            values = np.load(io.BytesIO(archive.read("bias.npy")))
            binary = archive.read("checkpoint.chk")
            xml = archive.read("state.xml").decode()
        try:
            context.loadCheckpoint(binary)
        except Exception as error:  # different platform or hardware: use the portable state
            log.warning("binary checkpoint unusable (%s); restoring from State XML", error)
            context.setState(mm.XmlSerializer.deserialize(xml))
        self.n_hills = int(meta["n_hills"])

        replayed = sum_hills(read_hills(self.workdir / "HILLS"), self.bias.grid)
        if self._others:
            # A shared bias is whatever all the HILLS files hold right now: the
            # other walkers have moved on (or rolled back) since our checkpoint.
            self.bias.set_values(replayed.values, context)
        else:
            self.bias.set_values(values, context)
            scale = max(1.0, float(np.abs(values).max()))
            if np.abs(replayed.values - values).max() > 1e-6 * scale:
                log.warning("bias grid in the checkpoint differs from the replayed HILLS file")
        log.info("restarted from %d hills (%.3f ns)", self.n_hills, meta["time_ps"] / 1000)
        return meta.get("n_frames")

    # ---- main loop ------------------------------------------------------
    def run(self, n_hills: int) -> None:
        """Run until ``n_hills`` hills have been deposited in total."""
        context = self.simulation.context
        settings = self.bias.settings
        start_hills, start_time = self.n_hills, time.time()
        try:
            while self.n_hills < n_hills:
                for _ in range(self.pace // self.stride):
                    self.simulation.step(self.stride)
                    t = context.getStepCount() * self.dt
                    s, rho = self.bias.collective_variables(context)
                    bias, wall = self.bias.energy(context), self._restraint_energy()
                    if not np.isfinite([s, rho, bias, wall]).all():
                        # Nothing of this is written: a restart continues from the last checkpoint.
                        raise RuntimeError(
                            f"the simulation broke down before t = {t:.3f} ps: values that are not finite "
                            f"(s = {s} nm, rho = {rho} nm, bias = {bias} kJ/mol, funnel walls = {wall} kJ/mol)")
                    self._colvar.write(t, s, rho, bias, wall)
                height = self.bias.hill_height(bias)
                self.bias.grid.add_hill(s, rho, height)
                self._hills.write(t, s, rho, settings.sigma_s, settings.sigma_rho, height)
                self.n_hills += 1
                if self._others:
                    self._hills.release()               # let the other walkers see this hill
                    if self.n_hills % self.settings.sync_hills == 0:
                        self.sync_walkers()
                self.bias.push(context)
                if self._traj is not None and self.n_hills % self.settings.trajectory_hills == 0:
                    self._traj.write(context.getState(getPositions=True))
                if self.n_hills % self.settings.checkpoint_hills == 0:
                    self.checkpoint()
                    self._log_progress(start_hills, start_time, n_hills)
        finally:
            for writer in (self._hills, self._colvar, self._traj):
                if writer is not None:
                    writer.flush()
        if self.n_hills % self.settings.checkpoint_hills:   # not already written in the loop
            self.checkpoint()
            self._log_progress(start_hills, start_time, n_hills)

    def _log_progress(self, start_hills: int, start_time: float, target: int) -> None:
        elapsed = max(time.time() - start_time, 1e-9)
        simulated_ns = (self.n_hills - start_hills) * self.pace * self.dt / 1000.0
        line = (f"hills {self.n_hills}/{target}  time {self.n_hills * self.pace * self.dt / 1000:.3f} ns"
                f"  speed {simulated_ns / elapsed * 86400:.1f} ns/day")
        log.info(line)
        with open(self.workdir / "run.log", "a") as handle:
            handle.write(line + "\n")

    def close(self) -> None:
        for writer in (self._hills, self._colvar, self._traj):
            if writer is not None:
                writer.close()


def rollback_to_checkpoint(workdir) -> dict:
    """Cut HILLS and COLVAR back to what the last checkpoint knows about and
    return the checkpoint's counters. Safe to call more than once."""
    workdir = Path(workdir)
    with zipfile.ZipFile(workdir / "checkpoint.zip") as archive:
        meta = json.loads(archive.read("meta.json"))
    kept = truncate_data_file(workdir / "HILLS", keep_rows=int(meta["n_hills"]))
    if kept != int(meta["n_hills"]):
        raise RuntimeError(f"{workdir}/HILLS has {kept} rows but the checkpoint expects "
                           f"{meta['n_hills']}")
    truncate_data_file(workdir / "COLVAR", max_time=meta["time_ps"])
    return meta


def discard_run_without_checkpoint(workdir) -> None:
    """Remove the record of a run that never wrote a checkpoint. Such a run starts again from
    nothing, and with walkers no other walker may read the hills it leaves behind."""
    for name in ("HILLS", "COLVAR", "traj.dcd"):
        (Path(workdir) / name).unlink(missing_ok=True)


def load_run_metadata(workdir) -> dict:
    with open(Path(workdir) / "run.yaml") as handle:
        return yaml.safe_load(handle)
