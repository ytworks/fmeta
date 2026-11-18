"""Compare the primary TYK2 analysis formula with independent, known binding integrals."""

import json
import os

import numpy as np
from scipy.constants import Avogadro, R

from fmeta.analysis.fes import sliced_log_weights
from fmeta.analysis.paper import sample_binding
from fmeta.testing.toy import make_toy


def report():
    kt = R * 300 / 1000
    standard_volume = 1.e24 / Avogadro  # nm³ per molecule at 1 mol/l
    radius = .15
    site, bulk = (.614, 1.316), (2.701, 3.299)
    area = np.pi * radius**2
    cylinder = make_toy(width=0., buffer=radius, k_wall=1.e25, eps=0.).funnel
    cases = {}

    def compare(name, measured, exact, tolerance):
        difference = (measured - exact) / 4.184
        if not np.isfinite(difference) or abs(difference) > tolerance:
            raise ValueError(f"{name} differs from the independent reference by {difference} kcal/mol")
        cases[name] = dict(exact=exact / 4.184, measured=measured / 4.184,
                           difference=difference, tolerance=tolerance)

    # Constant site and bulk energies in a hard cylinder give an analytic 1 M binding constant.
    for depth in (0, 6):
        weights = np.array([np.log(np.ptp(site)) + depth, np.log(np.ptp(bulk))])
        estimate = sample_binding(np.array([1., 3.]), np.array([.05, .05]), weights,
                                  cylinder, kt, site, bulk, radius)
        exact = -kt * np.log(area * np.ptp(site) * np.exp(depth) / standard_volume)
        compare(f"cylinder_well_{depth}_kT", estimate["dG"], exact, 1.e-6)

    # Unbiased populations of site, bulk and intermediate are 6:1:2. Neither source
    # ensemble is unbiased. Multiplying those populations by the two Boltzmann factors
    # gives integer sample counts without random error, 2:2:2 and 6:3:1 respectively.
    boltzmann = np.array([[1/3, 2, 1], [1, 3, .5]])
    source_counts = np.array([[2, 2, 2], [6, 3, 1]])
    labels = np.concatenate([np.repeat(np.arange(3), counts) for counts in source_counts])
    weights = sliced_log_weights(-np.log(boltzmann[:, labels]), source_counts.sum(axis=1), fast=True)
    measured = sample_binding(np.array([1., 3., 2.])[labels], np.full(len(labels), .05), weights,
                              cylinder, kt, site, bulk, radius)
    exact = -kt * np.log(6 * area * np.ptp(bulk) / standard_volume)
    compare("two_biased_ensembles_with_known_unbiased_populations", measured["dG"], exact, 1.e-6)

    # Independent cylindrical quadrature of a smooth three-dimensional well.
    toy = make_toy(well_depth=20., eps=0.)
    site, bulk = (.6, 1.5), (2.8, 3.3)
    s, rho = np.meshgrid(np.arange(.501, 3.499, .002), np.arange(.001, 1.20, .002))
    s, rho = s.ravel(), rho.ravel()
    weights = np.log(rho) - (toy.well_energy(s, rho) + toy.funnel.restraint_energy(s, rho)) / kt
    measured = sample_binding(s, rho, weights, toy.funnel, kt, site, bulk, radius)
    exact = toy.exact_binding_free_energy(300, site_max=site[1], s_min=site[0])
    compare("three_dimensional_gaussian_well", measured["dG"], exact, .001 / 4.184)
    return dict(analysis_commit=os.environ.get("FMETA_CODE_COMMIT"), energy_unit="kcal/mol",
                temperature_K=300, standard_volume_nm3=standard_volume, cases=cases)


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
