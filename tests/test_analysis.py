"""Free-energy analysis: estimator against quadrature, and the toy end to end."""

import dataclasses

import numpy as np
import pytest

from fmeta.analysis import analyze_replicas, analyze_run
from fmeta.analysis.binding import binding_free_energy, empty_funnel_free_energy
from fmeta.analysis.convergence import count_transitions, plateau_statistics, summarize_replicas
from fmeta.analysis.fes import projection_profile, reweighted_profile, sliced_log_weights
from fmeta.engine import BiasGrid, BiasSettings, FunnelBias, MetadynamicsRun, RunSettings
from fmeta.funnel.geometry import KB, STANDARD_VOLUME
from fmeta.testing.toy import make_simulation, make_toy

T = 300.0
KT = KB * T


def test_estimator_is_exact_on_the_exact_profile():
    """Given the true W(s), the estimator returns the true binding free energy.
    The reference is a direct quadrature of the binding constant, not W."""
    for depth, sigma in ((20.0, 0.15), (30.0, 0.12), (12.0, 0.2)):
        toy = make_toy(well_depth=depth, well_sigma=sigma)
        s = np.linspace(-0.5, 3.45, 3951)
        estimate = binding_free_energy(s, toy.exact_free_energy_profile(s, T), toy.funnel, KT)
        exact = toy.exact_binding_free_energy(T, estimate.site_max)
        assert estimate.dG == pytest.approx(exact, abs=0.01)
        assert estimate.unbound_roughness < 0.05


def test_funnel_does_not_change_the_binding_constant():
    """The toy well sits inside the cone, so K_b with the funnel equals the plain
    3D integral of exp(-U/kT) over the same region of space."""
    toy = make_toy(well_depth=30.0, well_sigma=0.12)
    site_max = 1.6
    with_funnel = toy.exact_binding_free_energy(T, site_max)
    r = np.linspace(0.0, 0.6, 200001)   # sphere around the site, inside s < site_max and the cone
    k_free = np.trapezoid(4 * np.pi * r**2 * np.exp(toy.well_depth * np.exp(-r**2 / (2 * toy.well_sigma**2)) / KT), r)
    assert with_funnel == pytest.approx(-KT * np.log(k_free / STANDARD_VOLUME), abs=0.05)


def test_soft_wall_area_and_simple_estimate_are_measurably_different():
    """Negative controls: dropping the soft-wall skin from the area, or using the
    well-depth shortcut, shifts dG by a predictable amount."""
    toy = make_toy()
    s = np.linspace(-0.5, 3.45, 3951)
    profile = toy.exact_free_energy_profile(s, T)
    estimate = binding_free_energy(s, profile, toy.funnel, KT)

    hard = dataclasses.replace(toy.funnel, k_wall=1e12)   # area = pi R^2 exactly
    hard_estimate = binding_free_energy(s, profile, hard, KT)
    window = (s >= estimate.unbound[0]) & (s <= estimate.unbound[1])
    ratio = toy.funnel.effective_area(s[window], KT) / hard.effective_area(s[window], KT)
    assert ratio.min() > 1.1
    # dG = -kT ln(site integral) + kT ln(bulk density); only the bulk density changes
    boltzmann = np.exp(-profile[window] / KT)
    density_ratio = (np.mean(boltzmann / hard.effective_area(s[window], KT))
                     / np.mean(boltzmann / toy.funnel.effective_area(s[window], KT)))
    assert hard_estimate.dG - estimate.dG == pytest.approx(KT * np.log(density_ratio), abs=1e-9)
    assert 0.3 < hard_estimate.dG - estimate.dG < 0.6          # ~ kT ln(1.18)
    assert abs(estimate.dG_simple - estimate.dG) > 1.0         # the shortcut is not the same quantity


def test_projection_profile_integrates_out_rho():
    """For a separable bias V = a(s) + b(rho), W(s) is -scale*a(s) up to a constant."""
    grid = BiasGrid(0.0, 3.0, 61, 1.0, 41, 0.05, 0.05)
    a = 4.0 * np.exp(-((grid.s - 1.0) / 0.3) ** 2)
    b = 2.0 * np.cos(3 * grid.rho)
    grid.values[...] = a[None, :] + b[:, None]
    profile = projection_profile(grid, 1.25, KT)
    expected = -1.25 * a
    assert np.allclose(profile, expected - expected.min(), atol=1e-9)


def test_reweighted_profile_recovers_the_unbiased_profile():
    """Samples drawn from exp(-(F + V)/kT) reweight back to F with weights exp(V/kT)."""
    rng = np.random.default_rng(0)
    s_grid = np.linspace(0.0, 2.0, 2001)
    free_energy = 6.0 * (s_grid - 1.0) ** 2
    bias = 0.8 * (free_energy.max() - free_energy)     # cancels 80% of F
    biased = np.exp(-(free_energy + bias) / KT)
    cdf = np.cumsum(biased) / biased.sum()
    s = np.interp(rng.random(400000), cdf, s_grid)
    edges = np.linspace(0.3, 1.7, 15)
    profile = reweighted_profile(s, np.interp(s, s_grid, bias) / KT, KT, edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    expected = 6.0 * (centres - 1.0) ** 2
    assert np.allclose(profile - profile.min(), expected - expected.min(), atol=0.25)


def test_sliced_weights_recover_the_unbiased_populations():
    """Three states with populations 1/2, 1/4, 1/4. Four samples without a bias and four under a
    bias that turns the populations into 1/4, 1/4, 1/2, each set in exactly those proportions:
    the weights then give the unbiased populations exactly, whatever constant a bias carries."""
    states = np.array([0, 0, 1, 2, 0, 1, 2, 2])
    second = np.log(2.0) * np.array([1.0, 0.0, -1.0])
    for constant in (0.0, 30.0):
        reduced = np.array([np.zeros(8), second[states] + constant])
        weights = np.exp(sliced_log_weights(reduced, np.array([4, 4])))
        populations = np.bincount(states, weights=weights) / weights.sum()
        assert populations == pytest.approx([0.5, 0.25, 0.25], abs=1e-7)


def test_transition_counting_uses_hysteresis():
    s = np.array([1.0, 1.4, 1.6, 1.4, 1.6, 3.2, 2.0, 3.3, 1.5, 1.1, 2.5, 1.0, 3.1])
    transitions = count_transitions(s, bound_below=1.3, unbound_above=3.0)
    assert (transitions.unbinding, transitions.rebinding) == (2, 1)


def test_replica_summary_and_plateau():
    summary = summarize_replicas([-20.0, -21.0, -19.0, -20.5, -19.5])
    assert summary.mean == pytest.approx(-20.0)
    assert summary.sem == pytest.approx(np.std([-20, -21, -19, -20.5, -19.5], ddof=1) / np.sqrt(5))
    assert summary.ci95[0] < -20.0 < summary.ci95[1]
    assert summarize_replicas([-3.0]).ci95 is None
    stats = plateau_statistics(np.arange(10.0), np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5.0]), 0.5)
    assert stats["mean"] == pytest.approx(3.0) and stats["drift"] == pytest.approx(4.0)
    # the error: standard error of the mean of 5 blocks of the kept half (here 2 points each)
    series = np.r_[np.full(10, 50.0), 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    assert plateau_statistics(np.arange(20.0), series, 0.5)["error"] == pytest.approx(np.sqrt(2.5 / 5))


def run_toy(workdir, seed, n_hills, reflect=True):
    toy = make_toy()
    settings = BiasSettings(temperature=T, bias_factor=10.0, height=1.2, sigma_s=0.05,
                            sigma_rho=0.05, pace=100, points_per_sigma=2.0, reflect=reflect)
    bias = FunnelBias(toy.system, toy.funnel, settings)
    simulation = make_simulation(toy, temperature=T, seed=seed)
    run = MetadynamicsRun(simulation, bias, workdir, toy.restraint_group,
                          RunSettings(checkpoint_hills=5000))
    run.run(n_hills)
    run.close()
    return toy


def test_toy_binding_free_energy_end_to_end(tmp_path):
    """Funnel metadynamics on the solvable toy recovers the exact dG0."""
    toy = run_toy(tmp_path, seed=12, n_hills=10000)
    result = analyze_run(tmp_path)
    exact = toy.exact_binding_free_energy(T, result["site_max"])
    assert result["simulated_ns"] == pytest.approx(20.0)
    assert result["rebinding_events"] >= 20
    # One 20 ns replica scatters by roughly 0.5-1 kJ/mol around the exact value
    # (the well is 14 kJ/mol deep); the replica test below checks the mean tightly.
    assert result["dG_plateau"] == pytest.approx(exact, abs=2.0)
    assert not result["no_binding"]
    assert not [note for note in result["diagnostics"] if "rebinding" in note or "flat" in note]
    # the headline is that mean of dG(t) from the bias; the estimate from the samples after the
    # first 2 ns is its cross-check
    assert result["dG_reweighted"] == pytest.approx(exact, abs=1.0)
    assert (result["dG"], result["discard_ns"]) == (result["dG_plateau"], 2.0)
    assert (tmp_path / "result.json").exists() and (tmp_path / "fes.npz").exists()


@pytest.mark.slow
@pytest.mark.parametrize("reflect", [True, False])
def test_toy_binding_free_energy_replicas(tmp_path, reflect):
    """Four independent replicas: each within 1.5 kJ/mol, their mean within 0.75."""
    dirs = []
    for seed in (11, 12, 13, 14):
        toy = run_toy(tmp_path / f"r{seed}", seed=seed, n_hills=20000, reflect=reflect)
        dirs.append(tmp_path / f"r{seed}")
    summary = analyze_replicas(dirs, write_to=tmp_path / "summary.json")
    exact = toy.exact_binding_free_energy(T, 2.9)
    for replica in summary["replicas"]:
        assert replica["dG"] == pytest.approx(exact, abs=1.5)
    assert summary["dG"] == pytest.approx(exact, abs=0.75)
    assert summary["ci95"][0] - 1.0 < exact < summary["ci95"][1] + 1.0


def test_empty_funnel_reference():
    """With the well switched off, the exact profile gives exactly the empty-funnel
    value; the toy's standard well beats it by 1.7 kT, a deep well by much more."""
    s = np.linspace(-0.5, 3.45, 3951)
    excess = {}
    for depth in (0.0, 20.0, 30.0):
        toy = make_toy(well_depth=depth)
        estimate = binding_free_energy(s, toy.exact_free_energy_profile(s, T), toy.funnel, KT)
        empty = empty_funnel_free_energy(s, toy.funnel, KT, estimate.site_max)
        excess[depth] = (empty - estimate.dG) / KT
    assert excess[0.0] == pytest.approx(0.0, abs=0.02)
    assert excess[20.0] == pytest.approx(1.7, abs=0.1)
    assert excess[30.0] > 4.0


def test_run_without_binding_is_flagged(tmp_path):
    """Funnel metadynamics on the toy with no well at all: the analysis reports a
    number near the empty-funnel value and says that it is not a binding free energy."""
    toy = make_toy(well_depth=0.0)
    settings = BiasSettings(temperature=T, bias_factor=10.0, height=1.2, sigma_s=0.05,
                            sigma_rho=0.05, pace=100, points_per_sigma=2.0)
    bias = FunnelBias(toy.system, toy.funnel, settings)
    run = MetadynamicsRun(make_simulation(toy, temperature=T, seed=4), bias, tmp_path,
                          toy.restraint_group, RunSettings(checkpoint_hills=5000))
    run.run(10000)
    run.close()
    result = analyze_run(tmp_path, write=False)
    # 20 ns: dG0 scatters by 0.35 kJ/mol between eight seeds around the value of the empty
    # funnel, so both bounds below are five or more standard deviations wide
    assert result["dG"] == pytest.approx(result["dG_empty_funnel"], abs=2.0)
    assert result["binding_excess"] < 1.0 * KT
    assert result["no_binding"] and any("empty funnel" in note for note in result["diagnostics"])


def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from fmeta.analysis.plots import plot_run, plot_summary
    dirs = []
    for replica, seed in enumerate((1, 2)):
        run_toy(tmp_path / f"rep{replica}", seed=seed, n_hills=600)
        dirs.append(tmp_path / f"rep{replica}")
    analyze_replicas(dirs, write_to=tmp_path / "summary.json")
    written = [plot_run(dirs[0]), plot_run(dirs[0], tmp_path / "dark.png", theme="dark"),
               plot_summary(tmp_path)]
    assert [p.name for p in written] == ["analysis.png", "dark.png", "summary.png"]
    for path in written:
        data = path.read_bytes()
        assert data[:4] == bytes([0x89]) + b"PNG" and len(data) > 30000


def test_dg_series_of_a_run_with_any_number_of_hills(tmp_path):
    """A run analysed before its end has any number of hills, and dG(t) keeps its 20 points:
    with 650 hills it had 7, so the plateau was the mean of 3 values."""
    run_toy(tmp_path, seed=5, n_hills=650)
    result = analyze_run(tmp_path, write=False)
    assert 20 <= len(result["dG_series"]) <= 21
