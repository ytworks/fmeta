"""Standard binding free energy from the 1D profile along the funnel axis.

Following Limongelli, Bonomi and Parrinello (PNAS 2013), the binding constant is
the Boltzmann-weighted volume of the bound region relative to bulk:

    K_b = int_site ds exp(-W(s)/kT)  /  < exp(-W(s_u)/kT) / A_eff(s_u) >_unbound
    dG0 = -kT ln(K_b / V0),   V0 = 1.6605 nm^3  (1 mol/L)

In the unbound window the ligand feels only the funnel wall, so
exp(-W(s_u)/kT) / A_eff(s_u) is the bulk Boltzmann density per unit volume.
A_eff is the cross-section actually available to the ligand centroid,
including the thermally populated skin of the harmonic wall, and may vary
with s (the sigmoid wall never becomes an exact cylinder).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..funnel.geometry import STANDARD_VOLUME, Funnel

SIMPLE_WELL_LENGTH = 0.1  # nm; the bound-state length implied by the "area / 1660 A^3" shortcut


@dataclass
class BindingEstimate:
    dG: float                 # kJ/mol, standard state 1 mol/L
    k_bind: float             # nm^3
    dG_simple: float          # kJ/mol, well-depth shortcut (see ``binding_free_energy``)
    well_depth: float         # kJ/mol, mean unbound W minus the minimum of W
    site_max: float           # nm
    unbound: tuple[float, float]
    unbound_roughness: float  # kJ/mol, std of the bulk-normalised W over the unbound window

    def to_dict(self) -> dict:
        return asdict(self)


def default_unbound_window(funnel: Funnel) -> tuple[float, float]:
    """The last 0.5 nm below the upper wall, kept 0.1 nm clear of the wall itself."""
    return (funnel.upper - 0.6, funnel.upper - 0.1)


def binding_free_energy(s: np.ndarray, profile: np.ndarray, funnel: Funnel, kT: float,
                        site_max: Optional[float] = None,
                        unbound: Optional[tuple[float, float]] = None) -> BindingEstimate:
    """Estimate dG0 from W(s).

    ``site_max`` is the upper end of the bound region (default: the start of
    the unbound window). ``dG_simple`` is the common shortcut
    ``-(well depth + kT ln(pi R^2 * 1 A / V0))`` which replaces the integral
    over the bound basin by its depth; it is reported for comparison only.
    """
    s = np.asarray(s, dtype=float)
    profile = np.asarray(profile, dtype=float)
    unbound = tuple(unbound) if unbound is not None else default_unbound_window(funnel)
    site_max = float(site_max) if site_max is not None else unbound[0]
    in_site = s <= site_max
    in_bulk = (s >= unbound[0]) & (s <= unbound[1])
    if in_site.sum() < 2 or in_bulk.sum() < 2:
        raise ValueError("the bound region and the unbound window must each cover grid points")

    reference = profile.min()
    boltzmann = np.exp(-(profile - reference) / kT)
    site_integral = np.trapezoid(boltzmann[in_site], s[in_site])
    area = funnel.effective_area(s[in_bulk], kT)
    bulk_density = np.mean(boltzmann[in_bulk] / area)
    k_bind = site_integral / bulk_density
    dG = -kT * np.log(k_bind / STANDARD_VOLUME)

    # W with the area term removed is flat in true bulk; its spread is a quality measure.
    bulk_level = profile[in_bulk] + kT * np.log(area)
    well_depth = float(profile[in_bulk].mean() - reference)
    geometric_area = float(np.mean(np.pi * funnel.wall_radius(s[in_bulk]) ** 2))
    dG_simple = -(well_depth + kT * np.log(geometric_area * SIMPLE_WELL_LENGTH / STANDARD_VOLUME))

    return BindingEstimate(float(dG), float(k_bind), float(dG_simple), well_depth, site_max,
                           (float(unbound[0]), float(unbound[1])), float(np.std(bulk_level)))


def empty_funnel_free_energy(s: np.ndarray, funnel: Funnel, kT: float, site_max: float) -> float:
    """The dG0 this analysis would report if nothing bound at all.

    Without any interaction the profile is W(s) = -kT ln A_eff(s), and the
    "binding constant" reduces to the volume of funnel between the lower wall
    and ``site_max``. A measured dG0 close to this value says the bound region
    is no more populated than empty solvent would be.
    """
    s = np.asarray(s, dtype=float)
    in_site = (s >= funnel.lower) & (s <= site_max)
    volume = np.trapezoid(funnel.effective_area(s[in_site], kT), s[in_site])
    return float(-kT * np.log(volume / STANDARD_VOLUME))
