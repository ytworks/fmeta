"""An exactly solvable funnel system for end-to-end validation.

One "ligand" particle moves in a Gaussian well centred on the funnel axis; two
massless particles are the p0 and p1 anchors. Because the potential is known,
the binding constant that funnel metadynamics should recover can be computed
by direct quadrature, independently of any simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from ..funnel.forces import add_funnel_restraint
from ..funnel.geometry import KB, STANDARD_VOLUME, Funnel


@dataclass
class ToySystem:
    system: mm.System
    topology: app.Topology
    positions: np.ndarray
    funnel: Funnel
    restraint_group: int
    well_depth: float
    well_sigma: float
    site_s: float

    def well_energy(self, s, rho):
        """Binding-site potential in funnel coordinates (kJ/mol)."""
        r_sq = (np.asarray(s) - self.site_s) ** 2 + np.asarray(rho) ** 2
        return -self.well_depth * np.exp(-r_sq / (2.0 * self.well_sigma**2))

    def exact_free_energy_profile(self, s, temperature: float, rho_max: float = 1.5,
                                  n_rho: int = 1501):
        """W(s) = -kT ln int 2 pi rho exp(-U/kT) drho, with U = well + funnel walls."""
        kT = KB * temperature
        rho = np.linspace(0.0, rho_max, n_rho)
        s_grid, rho_grid = np.meshgrid(np.atleast_1d(s), rho, indexing="ij")
        energy = self.well_energy(s_grid, rho_grid) + self.funnel.restraint_energy(s_grid, rho_grid)
        weight = np.trapezoid(2.0 * np.pi * rho_grid * np.exp(-energy / kT), rho, axis=1)
        with np.errstate(divide="ignore"):   # far behind the lower wall the weight underflows
            return -kT * np.log(weight)

    def exact_binding_free_energy(self, temperature: float, site_max: float,
                                  s_min: float = -0.5, n_s: int = 3501) -> float:
        """Standard binding free energy with the site defined as ``s < site_max``."""
        kT = KB * temperature
        s = np.linspace(s_min, site_max, n_s)
        k_bind = np.trapezoid(np.exp(-self.exact_free_energy_profile(s, temperature) / kT), s)
        return float(-kT * np.log(k_bind / STANDARD_VOLUME))


def make_toy(well_depth: float = 20.0, well_sigma: float = 0.15, site_s: float = 1.0,
             box: float = 12.0, mass: float = 100.0, **funnel_kwargs) -> ToySystem:
    """Build the toy system. Particle 0 is the ligand, 1 is p0, 2 is p1."""
    system = mm.System()
    system.addParticle(mass)
    system.addParticle(0.0)
    system.addParticle(0.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))

    centre = np.full(3, box / 2.0)
    positions = np.array([[0.0, 0.0, site_s], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) + centre
    site = positions[0]

    well = mm.CustomExternalForce(
        "-depth*exp(-periodicdistance(x,y,z,x0,y0,z0)^2/(2*sigma^2))")
    well.addGlobalParameter("depth", well_depth)
    well.addGlobalParameter("sigma", well_sigma)
    for name in ("x0", "y0", "z0"):
        well.addPerParticleParameter(name)
    well.addParticle(0, list(site))
    system.addForce(well)

    funnel_kwargs.setdefault("upper", 3.5)
    funnel = Funnel(ligand_atoms=[0], p0_atoms=[1], p1_atoms=[2], **funnel_kwargs)
    restraint = add_funnel_restraint(system, funnel, periodic=True)

    topology = app.Topology()
    chain = topology.addChain()
    for name in ("LIG", "P0", "P1"):
        residue = topology.addResidue(name, chain)
        topology.addAtom(name, app.Element.getBySymbol("C"), residue)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    return ToySystem(system, topology, positions, funnel, restraint.getForceGroup(),
                     well_depth, well_sigma, site_s)


def make_simulation(toy: ToySystem, temperature: float = 300.0, friction: float = 1.0,
                    timestep: float = 0.02, seed: int = 1, platform: str = "Reference"):
    """A Langevin simulation of the toy. Call after the bias force was added."""
    integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin, friction / unit.picosecond,
                                             timestep * unit.picosecond)
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(toy.topology, toy.system, integrator,
                                mm.Platform.getPlatformByName(platform))
    simulation.context.setPositions(toy.positions)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)
    return simulation
