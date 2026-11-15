"""OpenMM forces for the funnel: the two collective variables and the walls.

Everything is a ``CustomCentroidBondForce`` over the groups (ligand, p0, p1),
written with centroid distances only:

    s   = (dL0^2 + d01^2 - dL1^2) / (2*d01)          (law of cosines)
    rho = sqrt(max(dL0^2 - s^2, 0) + eps^2)

This is the same pair of CVs as ``d*cos(angle)`` / ``d*sin(angle)`` but has a
well-defined, bounded gradient on the funnel axis (rho -> 0).
"""

from __future__ import annotations

import openmm as mm

from .geometry import Funnel

_DISTANCES = "dL0=distance(g1,g2); dL1=distance(g1,g3); d01=distance(g2,g3)"
_S = "(dL0^2 + d01^2 - dL1^2)/(2*d01)"


def projection_expression() -> str:
    return f"{_S}; {_DISTANCES}"


def extent_expression(eps: float) -> str:
    return f"sqrt(max(dL0^2 - s^2, 0) + {eps!r}^2); s={_S}; {_DISTANCES}"


def restraint_expression(funnel: Funnel) -> str:
    f = funnel
    return (
        f"0.5*{f.k_wall!r}*max(rho - R, 0)^2"
        f" + 0.5*{f.k_lower!r}*max({f.lower!r} - s, 0)^2"
        f" + 0.5*{f.k_upper!r}*max(s - {f.upper!r}, 0)^2;"
        f" R = {f.width!r}/(1 + exp({f.steepness!r}*(s - {f.inflection!r}))) + {f.buffer!r};"
        f" rho = sqrt(max(dL0^2 - s^2, 0) + {f.eps!r}^2);"
        f" s = {_S}; {_DISTANCES}"
    )


def _centroid_force(expression: str, funnel: Funnel, periodic: bool) -> mm.CustomCentroidBondForce:
    force = mm.CustomCentroidBondForce(3, expression)
    g1 = force.addGroup(funnel.ligand_atoms, funnel.ligand_weights)
    g2 = force.addGroup(funnel.p0_atoms, funnel.p0_weights)
    g3 = force.addGroup(funnel.p1_atoms, funnel.p1_weights)
    force.addBond([g1, g2, g3], [])
    force.setUsesPeriodicBoundaryConditions(periodic)
    return force


def projection_force(funnel: Funnel, periodic: bool = True) -> mm.CustomCentroidBondForce:
    """Force whose energy is the projection CV ``s`` (nm)."""
    return _centroid_force(projection_expression(), funnel, periodic)


def extent_force(funnel: Funnel, periodic: bool = True) -> mm.CustomCentroidBondForce:
    """Force whose energy is the extent CV ``rho`` (nm)."""
    return _centroid_force(extent_expression(funnel.eps), funnel, periodic)


def restraint_force(funnel: Funnel, periodic: bool = True) -> mm.CustomCentroidBondForce:
    """Side wall plus lower and upper walls, as a single force."""
    return _centroid_force(restraint_expression(funnel), funnel, periodic)


def free_force_group(system: mm.System) -> int:
    """Highest force group not used by any force in the system."""
    used = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
    free = set(range(32)) - used
    if not free:
        raise RuntimeError("all 32 force groups are in use")
    return max(free)


def add_funnel_restraint(system: mm.System, funnel: Funnel,
                         periodic: bool | None = None) -> mm.CustomCentroidBondForce:
    """Add the funnel walls to ``system`` in their own force group.

    ``periodic`` defaults to whether the system already uses periodic boundaries.
    """
    if periodic is None:
        periodic = system.usesPeriodicBoundaryConditions()
    force = restraint_force(funnel, periodic=periodic)
    force.setForceGroup(free_force_group(system))
    force.setName("FunnelRestraint")
    system.addForce(force)
    return force
