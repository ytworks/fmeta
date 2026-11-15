"""Staged equilibration before production.

minimise -> heat at constant volume with the solute held -> constant pressure
while the positional restraints are released in steps -> unrestrained run with
the funnel walls on (and no bias). Membrane systems use a tensionless
semi-isotropic barostat and also hold the lipids along z during the release.

The last stage doubles as a check: if the funnel walls do work on the bound
ligand there, the funnel is too tight for this pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from ..funnel.geometry import KB
from ..system.builder import Prepared, add_barostat

log = logging.getLogger(__name__)
_KJ = unit.kilojoule_per_mole


@dataclass
class EquilibrationSettings:
    minimize_iterations: int = 5000
    restraint_k: float = 1000.0             # kJ/mol/nm^2 on solute heavy atoms
    heating_ps: float = 100.0               # NVT, 10 K -> target
    release_ps: float = 400.0               # NPT, split evenly over the release stages
    release_factors: Sequence[float] = (1.0, 0.5, 0.2, 0.05)
    free_ps: Optional[float] = None         # unrestrained NPT; default 1 ns (10 ns with a membrane)
    timestep_fs: float = 2.0
    friction: float = 1.0                   # 1/ps
    seed: int = 0                           # 0: random


@dataclass
class EquilibrationResult:
    state: mm.State
    wall_energy_mean: float                 # kJ/mol, funnel wall energy over the free stage
    wall_energy_max: float
    box_lengths: np.ndarray                 # nm, at the end

    def wall_ok(self, temperature: float) -> bool:
        """The funnel leaves the bound state alone if it does less than 0.1 kT of work."""
        return self.wall_energy_mean < 0.1 * KB * temperature


def _clone(system: mm.System) -> mm.System:
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def _add_position_restraints(system: mm.System, prepared: Prepared, k: float) -> None:
    positions = prepared.positions
    selection = prepared.selection
    solute = mm.CustomExternalForce("0.5*fm_posres_scale*k*periodicdistance(x,y,z,x0,y0,z0)^2")
    solute.addGlobalParameter("fm_posres_scale", 1.0)
    solute.addPerParticleParameter("k")
    for name in ("x0", "y0", "z0"):
        solute.addPerParticleParameter(name)
    held = np.concatenate([selection.heavy_atoms("protein"), selection.heavy_atoms("ligand"),
                           selection.heavy_atoms("other")])
    for index in held:
        solute.addParticle(int(index), [k, *positions[index]])
    system.addForce(solute)
    if prepared.membrane:
        # hold lipid heavy atoms in z only, so the bilayer can pack laterally
        lipid = mm.CustomExternalForce("0.5*fm_posres_scale*k*periodicdistance(x,y,z,x,y,z0)^2")
        lipid.addGlobalParameter("fm_posres_scale", 1.0)
        lipid.addPerParticleParameter("k")
        lipid.addPerParticleParameter("z0")
        atoms = list(prepared.topology.atoms())
        for index in selection.heavy_atoms("lipid"):
            if atoms[index].element.symbol == "P":
                lipid.addParticle(int(index), [k, positions[index, 2]])
        system.addForce(lipid)


def equilibrate(prepared: Prepared, platform: mm.Platform, properties: Optional[dict] = None,
                settings: Optional[EquilibrationSettings] = None) -> EquilibrationResult:
    settings = settings or EquilibrationSettings()
    properties = properties or {}
    temperature = prepared.settings.temperature
    dt = settings.timestep_fs / 1000.0
    free_ps = settings.free_ps if settings.free_ps is not None else (
        10000.0 if prepared.membrane else 1000.0)

    def integrator(start_temperature: float) -> mm.LangevinMiddleIntegrator:
        result = mm.LangevinMiddleIntegrator(start_temperature * unit.kelvin,
                                             settings.friction / unit.picosecond,
                                             dt * unit.picosecond)
        if settings.seed:
            result.setRandomNumberSeed(settings.seed)
        return result

    # -- constant volume: minimise and heat -----------------------------------
    nvt = _clone(prepared.system)
    _add_position_restraints(nvt, prepared, settings.restraint_k)
    simulation = app.Simulation(prepared.topology, nvt, integrator(10.0), platform, properties)
    simulation.context.setPositions(prepared.positions * unit.nanometer)
    simulation.minimizeEnergy(maxIterations=settings.minimize_iterations)
    simulation.context.setVelocitiesToTemperature(10.0 * unit.kelvin)
    n_heat = max(1, int(round(settings.heating_ps / dt)))
    chunks = min(50, n_heat)
    for chunk in range(chunks):
        simulation.integrator.setTemperature(
            (10.0 + (temperature - 10.0) * (chunk + 1) / chunks) * unit.kelvin)
        simulation.step(max(1, n_heat // chunks))
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    log.info("heated to %.0f K", temperature)

    # -- constant pressure: release the restraints, then run free ---------------
    npt = _clone(prepared.system)
    _add_position_restraints(npt, prepared, settings.restraint_k)
    add_barostat(npt, temperature, prepared.settings.pressure, prepared.membrane)
    simulation = app.Simulation(prepared.topology, npt, integrator(temperature), platform, properties)
    simulation.context.setState(state)
    n_release = max(1, int(round(settings.release_ps / dt / max(1, len(settings.release_factors)))))
    for factor in settings.release_factors:
        simulation.context.setParameter("fm_posres_scale", factor)
        simulation.step(n_release)
        log.info("restraints at %.0f%%", 100 * factor)
    simulation.context.setParameter("fm_posres_scale", 0.0)

    n_free = max(1, int(round(free_ps / dt)))
    stride = max(1, int(round(1.0 / dt)))           # sample the walls every ps
    wall = []
    for _ in range(max(1, n_free // stride)):
        simulation.step(min(stride, n_free))
        energy = simulation.context.getState(getEnergy=True, groups={prepared.restraint_group})
        wall.append(energy.getPotentialEnergy().value_in_unit(_KJ))
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    box = np.diag(np.array(state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)))
    return EquilibrationResult(state, float(np.mean(wall)), float(np.max(wall)), box)
