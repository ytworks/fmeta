"""The metadynamics bias force.

The bias is a cubic spline of the grid, evaluated at the funnel CVs on the run platform and
refreshed after every hill. It is the same function as the ``Continuous2DFunction`` table of
``openmm.app.Metadynamics`` (the natural bicubic spline through the grid values), reached in
a cheaper way:

* the force expression evaluates the spline itself, from a table of B-spline coefficients
  (``Discrete2DFunction``): the spline is fitted here, in a fraction of a millisecond, and a
  table update sends one number per grid point;
* the lookup is clamped to the grid, so the bias keeps its edge value when a CV overshoots
  the grid (an OpenMM table evaluates to 0 outside its range);
* hills can be mirrored at rho = 0, which keeps the bias flat on the funnel axis;
* every hill is returned to the caller so it can be logged and replayed.

The expression and the table are the same on every platform. The force that carries them is
not, because the platforms differ in what is expensive (measurements: FM-D4 in
``docs/design/engine.md``):

* GPU platforms: one ``CustomCentroidBondForce``, so that no ``CustomCVForce`` is part of a
  step (it costs about a fifth of the speed there). The CV values are read through a second
  force, a ``CustomCVForce`` with zero energy, whose force group a run leaves out of the
  forces the integrator uses (``exclude_probe``). The readout is not an energy parameter
  derivative of the bias force because the GPU platforms return zero energy and zero forces
  for a ``CustomCentroidBondForce`` that has both a tabulated function and a parameter
  derivative (OpenMM 8.6.1).
* Reference and CPU: one ``CustomCVForce`` over the two CVs, which is the readout as well.
  There a ``CustomCentroidBondForce`` builds its whole expression again at every table
  update (about 0.8 s per hill, whatever the size of the table), while a ``CustomCVForce``
  replaces the table in place.

``FunnelBias`` is told the platform when it is built and refuses a context of the other
kind: the wrong form gives the same energies and forces, only far more slowly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import openmm as mm
import openmm.unit as unit
from scipy.linalg import solve_banded

from ..funnel import forces
from ..funnel.geometry import Funnel
from .grid import BiasGrid, BiasSettings
from .platform import GPU_PLATFORMS

_KJ = unit.kilojoule_per_mole


@dataclass
class Hill:
    s: float
    rho: float
    height: float
    bias: float      # bias at (s, rho) just before this hill was added


def _spline_axis(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    c = np.zeros((n + 3,) + values.shape[1:])
    c[1], c[n] = values[0], values[-1]          # zero curvature at the ends: c = f there
    rhs = 6.0 * values[1:-1]
    rhs[0] -= values[0]
    rhs[-1] -= values[-1]
    band = np.ones((3, n - 2))
    band[1] = 4.0
    c[2:n] = solve_banded((1, 1), band, rhs)    # c[k-1] + 4 c[k] + c[k+1] = 6 f[k]
    c[0] = 2.0 * c[1] - c[2]
    c[n + 1] = 2.0 * c[n] - c[n - 1]
    return c


def spline_coefficients(values: np.ndarray) -> np.ndarray:
    """Cubic B-spline coefficients of the natural bicubic spline through ``values``.

    An axis with n nodes gets n + 3 entries: the coefficient of node k is at k + 1, with one
    more on either side (they set the curvature at the ends to zero) and a last entry that
    only keeps a lookup at the final node inside the array.
    """
    return _spline_axis(_spline_axis(values).T).T


def _lookup(grid: BiasGrid) -> str:
    """Expression of ``s`` and ``rho`` for the spline of ``table``, clamped to the grid."""
    weights = ("{w}0 = (1-{t})^3/6; {w}1 = (3*{t}^3 - 6*{t}^2 + 4)/6;"
               " {w}2 = (-3*{t}^3 + 3*{t}^2 + 3*{t} + 1)/6; {w}3 = {t}^3/6")
    rows = " + ".join(
        f"wy{b}*(" + " + ".join(f"wx{a}*table(i+{a}, j+{b})" for a in range(4)) + ")"
        for b in range(4))
    ds = (grid.s_max - grid.s_min) / (grid.n_s - 1)
    drho = grid.rho_max / (grid.n_rho - 1)
    return (f"{rows}; {weights.format(w='wx', t='tx')}; {weights.format(w='wy', t='ty')};"
            " tx = x - i; i = floor(x); ty = y - j; j = floor(y);"
            f" x = min(max((s - ({grid.s_min!r}))/{ds!r}, 0), {grid.n_s - 1});"
            f" y = min(rho/{drho!r}, {grid.n_rho - 1})")


class FunnelBias:
    """The bias of one run, added to ``system``.

    ``platform`` is the name of the OpenMM platform the system will run on; it selects the
    form of the force (see the module text).
    """

    def __init__(self, system: mm.System, funnel: Funnel, settings: BiasSettings,
                 periodic: Optional[bool] = None, grid: Optional[BiasGrid] = None,
                 platform: str = "Reference"):
        self.funnel = funnel
        self.settings = settings
        self.grid = grid if grid is not None else BiasGrid.from_funnel(funnel, settings)
        if periodic is None:
            periodic = system.usesPeriodicBoundaryConditions()
        self._on_gpu = platform in GPU_PLATFORMS

        g = self.grid
        self._table = mm.Discrete2DFunction(g.n_s + 3, g.n_rho + 3,
                                            spline_coefficients(g.values).ravel())
        if self._on_gpu:
            self.force = forces.cv_force(_lookup(g), funnel, periodic)
            self._probe = mm.CustomCVForce("0")
            self._probe.setName("FunnelCVs")
        else:
            self.force = self._probe = mm.CustomCVForce(_lookup(g))
        self._probe.addCollectiveVariable("s", forces.projection_force(funnel, periodic))
        self._probe.addCollectiveVariable("rho", forces.extent_force(funnel, periodic))
        self.force.addTabulatedFunction("table", self._table)
        self.force.setName("FunnelBias")
        for force in [self.force] + [self._probe] * self._on_gpu:
            force.setForceGroup(forces.free_force_group(system))
            system.addForce(force)

    @property
    def force_group(self) -> int:
        return self.force.getForceGroup()

    def check_platform(self, context: mm.Context) -> None:
        """Refuse a context that runs on the other kind of platform than this bias was built for."""
        name = context.getPlatform().getName()
        if (name in GPU_PLATFORMS) != self._on_gpu:
            raise ValueError(f"the bias was built for a {'GPU' if self._on_gpu else 'CPU'} platform "
                             f"but the context runs on {name}: pass platform={name!r} to FunnelBias")

    def exclude_probe(self, integrator: mm.Integrator) -> None:
        """Keep the CV readout out of the forces ``integrator`` evaluates at every step."""
        if self._probe is self.force:
            return
        mask = integrator.getIntegrationForceGroups()
        groups = {group for group in range(32) if mask >> group & 1}
        integrator.setIntegrationForceGroups(groups - {self._probe.getForceGroup()})

    def collective_variables(self, context: mm.Context) -> tuple[float, float]:
        s, rho = self._probe.getCollectiveVariableValues(context)
        return float(s), float(rho)

    def energy(self, context: mm.Context) -> float:
        state = context.getState(getEnergy=True, groups={self.force_group})
        return state.getPotentialEnergy().value_in_unit(_KJ)

    def hill_height(self, bias: float) -> float:
        """Well-tempered height for a hill deposited where the bias is ``bias``."""
        delta = self.settings.delta_kT
        if delta is None:
            return self.settings.height
        return self.settings.height * math.exp(-bias / delta)

    def push(self, context: Optional[mm.Context]) -> None:
        """Rebuild the table from the grid and, with a context, send it there."""
        coefficients = spline_coefficients(self.grid.values)
        self._table.setFunctionParameters(coefficients.shape[1], coefficients.shape[0],
                                          coefficients.ravel())
        if context is not None:
            self.check_platform(context)
            self.force.updateParametersInContext(context)

    def deposit(self, context: mm.Context) -> Hill:
        """Add one hill at the current CV values and update the context."""
        s, rho = self.collective_variables(context)
        bias = self.energy(context)
        hill = Hill(s, rho, self.hill_height(bias), bias)
        self.grid.add_hill(s, rho, hill.height)
        self.push(context)
        return hill

    def set_values(self, values: np.ndarray, context: Optional[mm.Context] = None) -> None:
        """Replace the grid contents (restart). Call before creating the Context,
        or pass the context to update it in place."""
        if values.shape != self.grid.values.shape:
            raise ValueError(f"bias grid shape {values.shape} != {self.grid.values.shape}")
        self.grid.values[...] = values
        self.push(context)
