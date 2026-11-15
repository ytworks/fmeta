"""Multiple walkers sharing one bias through their HILLS files."""

import numpy as np
import pytest

from fmeta.analysis import analyze_run
from fmeta.engine import BiasSettings, FunnelBias, MetadynamicsRun, RunSettings
from fmeta.engine.hills import HillsFollower, merge_hills, read_hills, sum_hills
from fmeta.testing.toy import make_simulation, make_toy

SETTINGS = dict(temperature=300.0, bias_factor=10.0, height=1.2, sigma_s=0.05, sigma_rho=0.05,
                pace=100, points_per_sigma=2.0)


def make_walker(base, index, n_walkers, seed, **run_kwargs):
    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS), periodic=True)
    simulation = make_simulation(toy, seed=seed)
    dirs = [base / f"w{k}" for k in range(n_walkers)]
    settings = RunSettings(walker_dirs=dirs, checkpoint_hills=20, **run_kwargs)
    return toy, MetadynamicsRun(simulation, bias, dirs[index], toy.restraint_group, settings)


def total_bias(base, grid, n_walkers):
    hills = merge_hills([read_hills(base / f"w{k}" / "HILLS") for k in range(n_walkers)])
    return sum_hills(hills, grid).values, hills


def test_follower_reads_only_complete_new_lines(tmp_path):
    path = tmp_path / "HILLS"
    follower = HillsFollower(path)
    assert follower.read_new() is None                         # no file yet
    path.write_text("#! FIELDS time proj ext sigma_proj sigma_ext height biasf\n"
                    "1.0 1.0 0.1 0.05 0.05 2.0 10\n2.0 1.1 0.1 0.05 0.05 2.0 10\n3.0 1.2 0.")
    first = follower.read_new()
    assert len(first) == 2 and first.height[0] == pytest.approx(2.0 * 9 / 10)
    assert follower.read_new() is None                         # the partial line is not consumed
    with open(path, "a") as handle:
        handle.write("1 0.05 0.05 2.0 10\n")
    second = follower.read_new()
    assert len(second) == 1 and second.s[0] == pytest.approx(1.2) and follower.rows == 3


def test_walkers_share_the_bias(tmp_path):
    """Two walkers taking turns: each ends up with the sum of both HILLS files."""
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    for target in (10, 20, 30, 40):
        a.run(target)
        b.run(target)
    for walker in (a, b):                 # pick up whatever the other wrote last
        walker.sync_walkers()
    expected, hills = total_bias(tmp_path, a.bias.grid, 2)
    assert len(hills) == 80 and np.all(np.diff(hills.time) >= 0)
    assert expected.max() > 5.0
    assert np.allclose(a.bias.grid.values, expected, rtol=1e-7, atol=1e-7)
    assert np.allclose(b.bias.grid.values, expected, rtol=1e-7, atol=1e-7)
    # the bias a walker feels includes the other's hills: where only B deposited, A's table is non-zero
    own = sum_hills(read_hills(tmp_path / "w0" / "HILLS"), a.bias.grid).values
    assert np.abs(a.bias.grid.values - own).max() > 1.0
    a.close()
    b.close()

    # the analysis treats the two walkers as one bias, and the figure draws both
    result = analyze_run(tmp_path, write=True)
    assert result["n_walkers"] == 2 and result["n_hills"] == 80
    assert result["simulated_ns"] == pytest.approx(2 * 40 * 100 * 0.02 / 1000)
    pytest.importorskip("matplotlib")
    from fmeta.analysis.plots import plot_run
    assert plot_run(tmp_path).stat().st_size > 30000


def test_walker_restart_rebuilds_from_all_files(tmp_path):
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    a.run(20)             # checkpoint written at 20 hills
    b.run(30)
    a.close()
    b.close()
    # work A did after its checkpoint and lost in a crash: rows on disk the checkpoint does not know
    hills_path = tmp_path / "w0" / "HILLS"
    last = hills_path.read_text().splitlines()[-1].split()
    with open(hills_path, "a") as handle:
        for k in range(1, 4):
            handle.write(" ".join([str(float(last[0]) + 2.0 * k), "2.0", "0.1"] + last[3:]) + "\n")
    assert len(read_hills(hills_path)) == 23

    _, again = make_walker(tmp_path, 0, 2, seed=1)
    assert again.n_hills == 20 and len(read_hills(hills_path)) == 20      # rolled back
    expected, hills = total_bias(tmp_path, again.bias.grid, 2)
    assert len(hills) == 50
    assert np.allclose(again.bias.grid.values, expected, rtol=1e-7, atol=1e-7)   # ours + all of B's
    again.run(40)
    again.close()
    assert len(read_hills(hills_path)) == 40


@pytest.mark.slow
def test_two_walkers_recover_the_exact_free_energy(tmp_path):
    """Two walkers of 10 ns each give the same answer as one long run."""
    toy, a = make_walker(tmp_path, 0, 2, seed=21)
    _, b = make_walker(tmp_path, 1, 2, seed=22)
    for target in range(50, 5001, 50):
        a.run(target)
        b.run(target)
    a.close()
    b.close()
    result = analyze_run(tmp_path)
    exact = toy.exact_binding_free_energy(300.0, result["site_max"])
    assert result["n_walkers"] == 2 and result["simulated_ns"] == pytest.approx(20.0)
    assert result["dG_plateau"] == pytest.approx(exact, abs=2.0)
    assert result["dG_reweighted"] == pytest.approx(exact, abs=1.5)
    assert result["rebinding_events"] >= 20
