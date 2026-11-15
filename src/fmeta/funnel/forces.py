"""OpenMM forces for the funnel: the two collective variables and the walls.

Everything is a ``CustomCentroidBondForce`` over the groups (ligand, p0, p1):

    d     = |L - p0|,   theta = angle(L, p0, p1)
    s     = d*cos(theta)
    rho   = sqrt((d*sin(theta))^2 + eps^2)

``eps`` gives ``rho`` a well-defined gradient on the funnel axis (theta -> 0, where
``d*sin(theta)`` alone has a cusp). The angle form is the one that keeps its accuracy in
single precision: OpenMM evaluates a nearly straight angle from the cross product of the two
vectors, so ``rho`` near the axis does not come from a difference of large numbers.

The forces never apply periodic boundary conditions. They do not need them: a force over the
three groups makes OpenMM treat every molecule the groups touch as one molecule, so the
ligand and the protein are always moved to another periodic cell together and the vectors
between the centroids are the true ones. And they must not use them: OpenMM wraps a
displacement one box vector after the other, which is the nearest image only while each
component stays below half of the box width along its axis. In a rhombic dodecahedron the
width along z is 0.71 of the image distance, so a ligand near the upper wall of a funnel that
points along z was wrapped to another image: the CVs jumped and the wall threw the ligand
(FM-D2 in ``docs/design/engine.md``).
"""

from __future__ import annotations

import openmm as mm

from .geometry import Funnel


def cv_force(energy: str, funnel: Funnel) -> mm.CustomCentroidBondForce:
    """Force whose energy is ``energy``, an expression of the two CVs ``s`` and ``rho`` (nm)."""
    force = mm.CustomCentroidBondForce(
        3, f"{energy}; rho = sqrt((d*sin(theta))^2 + {funnel.eps!r}^2); s = d*cos(theta);"
           " d=distance(g1,g2); theta=angle(g1,g2,g3)")
    g1 = force.addGroup(funnel.ligand_atoms, funnel.ligand_weights)
    g2 = force.addGroup(funnel.p0_atoms, funnel.p0_weights)
    g3 = force.addGroup(funnel.p1_atoms, funnel.p1_weights)
    force.addBond([g1, g2, g3], [])
    force.setUsesPeriodicBoundaryConditions(False)
    return force


def projection_force(funnel: Funnel) -> mm.CustomCentroidBondForce:
    """Force whose energy is the projection CV ``s`` (nm)."""
    return cv_force("s", funnel)


def extent_force(funnel: Funnel) -> mm.CustomCentroidBondForce:
    """Force whose energy is the extent CV ``rho`` (nm)."""
    return cv_force("rho", funnel)


def restraint_force(funnel: Funnel) -> mm.CustomCentroidBondForce:
    """Side wall plus lower and upper walls, as a single force."""
    f = funnel
    return cv_force(
        f"0.5*{f.k_wall!r}*max(rho - R, 0)^2"
        f" + 0.5*{f.k_lower!r}*max({f.lower!r} - s, 0)^2"
        f" + 0.5*{f.k_upper!r}*max(s - {f.upper!r}, 0)^2;"
        f" R = {f.width!r}/(1 + exp({f.steepness!r}*(s - {f.inflection!r}))) + {f.buffer!r}",
        funnel)


def free_force_group(system: mm.System) -> int:
    """Highest force group not used by any force in the system."""
    used = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
    free = set(range(32)) - used
    if not free:
        raise RuntimeError("all 32 force groups are in use")
    return max(free)


def add_funnel_restraint(system: mm.System, funnel: Funnel) -> mm.CustomCentroidBondForce:
    """Add the funnel walls to ``system`` in their own force group."""
    force = restraint_force(funnel)
    force.setForceGroup(free_force_group(system))
    force.setName("FunnelRestraint")
    system.addForce(force)
    return force
