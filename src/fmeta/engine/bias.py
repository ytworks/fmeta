"""The metadynamics bias force.

Same mechanism as ``openmm.app.Metadynamics`` (a ``CustomCVForce`` reading a
tabulated function that is refreshed after every hill, so it runs on the GPU),
specialised to the two funnel CVs and with three differences:

* the table lookup is clamped to the grid, so the bias stays continuous when a
  CV overshoots the grid (an OpenMM table evaluates to 0 outside its range);
* hills can be mirrored at rho = 0, which keeps the bias flat on the funnel axis;
* every hill is returned to the caller so it can be logged and replayed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import openmm as mm
import openmm.unit as unit

from ..funnel import forces
from ..funnel.geometry import Funnel
from .grid import BiasGrid, BiasSettings

_KJ = unit.kilojoule_per_mole


@dataclass
class Hill:
    s: float
    rho: float
    height: float
    bias: float      # bias at (s, rho) just before this hill was added


class FunnelBias:
    def __init__(self, system: mm.System, funnel: Funnel, settings: BiasSettings,
                 periodic: Optional[bool] = None, grid: Optional[BiasGrid] = None):
        self.funnel = funnel
        self.settings = settings
        self.grid = grid if grid is not None else BiasGrid.from_funnel(funnel, settings)
        if periodic is None:
            periodic = system.usesPeriodicBoundaryConditions()

        g = self.grid
        # Clamp a hair inside the table so single-precision rounding on the GPU
        # can never push the lookup outside it.
        inset = 1.0e-4
        expression = (f"table(min(max(s, {g.s_min + inset!r}), {g.s_max - inset!r}),"
                      f" min(rho, {g.rho_max - inset!r}))")
        self._table = mm.Continuous2DFunction(g.n_s, g.n_rho, g.values.ravel(),
                                              g.s_min, g.s_max, 0.0, g.rho_max)
        self.force = mm.CustomCVForce(expression)
        self.force.addCollectiveVariable("s", forces.projection_force(funnel, periodic))
        self.force.addCollectiveVariable("rho", forces.extent_force(funnel, periodic))
        self.force.addTabulatedFunction("table", self._table)
        self.force.setForceGroup(forces.free_force_group(system))
        self.force.setName("FunnelBias")
        system.addForce(self.force)

    @property
    def force_group(self) -> int:
        return self.force.getForceGroup()

    def collective_variables(self, context: mm.Context) -> tuple[float, float]:
        s, rho = self.force.getCollectiveVariableValues(context)
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

    def push(self, context: mm.Context) -> None:
        """Send the current grid to the context."""
        g = self.grid
        self._table.setFunctionParameters(g.n_s, g.n_rho, g.values.ravel(),
                                          g.s_min, g.s_max, 0.0, g.rho_max)
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
        g = self.grid
        self._table.setFunctionParameters(g.n_s, g.n_rho, g.values.ravel(),
                                          g.s_min, g.s_max, 0.0, g.rho_max)
        if context is not None:
            self.force.updateParametersInContext(context)
