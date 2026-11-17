"""Multiple walkers sharing one bias through their HILLS files."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from fmeta.analysis import analyze_run
from fmeta.analysis.walkers import SharedBias, hills_on_grid
from fmeta.config import Config, RunConfig
from fmeta.engine import BiasSettings, FunnelBias, MetadynamicsRun, RunSettings
from fmeta.engine.hills import HillsFollower, merge_hills, read_colvar, read_hills, sum_hills
from fmeta.testing.toy import make_simulation, make_toy
from fmeta.workflow import _rollback_walkers, run_stage

SETTINGS = dict(temperature=300.0, bias_factor=10.0, height=1.2, sigma_s=0.05, sigma_rho=0.05,
                pace=100, points_per_sigma=2.0)
CONDITIONS = dict(bias=BiasSettings(**SETTINGS).to_dict(), timestep_ps=0.02)     # as the workflow passes them


def make_walker(base, index, n_walkers, seed, **run_kwargs):
    toy = make_toy()
    bias = FunnelBias(toy.system, toy.funnel, BiasSettings(**SETTINGS))
    simulation = make_simulation(toy, seed=seed)
    dirs = [base / f"w{k}" for k in range(n_walkers)]
    settings = RunSettings(walker_dirs=dirs, checkpoint_hills=20, **run_kwargs)
    return toy, MetadynamicsRun(simulation, bias, dirs[index], toy.restraint_group, settings)


def total_bias(base, grid, n_walkers):
    hills = merge_hills([read_hills(base / f"w{k}" / "HILLS") for k in range(n_walkers)])
    return sum_hills(hills, grid).values, hills


def felt_bias(grid, s, rho):
    """The bias as the force evaluates it: the natural bicubic spline through the grid values."""
    along_s = CubicSpline(grid.s, grid.values, axis=1, bc_type="natural")(np.clip(s, grid.s_min, grid.s_max))
    return float(CubicSpline(grid.rho, along_s, bc_type="natural")(min(rho, grid.rho_max)))


def rebuilt_and_written(base, grid):
    """For each of two walkers: the bias of every sample rebuilt from the HILLS files and the
    record of the walker (NaN where the record says it cannot be rebuilt), and the bias column
    of its COLVAR."""
    sources = [base / "w0", base / "w1"]
    parts = [read_hills(directory / "HILLS") for directory in sources]
    pairs = []
    for directory, own, table in zip(sources, parts, hills_on_grid(sources, parts)):
        colvar = read_colvar(directory / "COLVAR")
        shared = SharedBias(grid, parts)
        counts = table[np.searchsorted(own.time, colvar["time"], side="left")]     # own hills before the sample
        rebuilt = [felt_bias(shared.at(row), s, rho) if row.min() >= 0 else np.nan
                   for row, s, rho in zip(counts, colvar["s"], colvar["rho"])]
        pairs.append((np.array(rebuilt), colvar["bias"]))
    return pairs


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


@pytest.mark.parametrize("first_has_checkpoint", [False, True])
def test_walkers_stopped_before_a_checkpoint_leave_no_hills_behind(tmp_path, first_has_checkpoint):
    """A walker stopped before its first checkpoint starts again from nothing: its hills are
    gone before another walker starts and reads them."""
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    a.run(20 if first_has_checkpoint else 3)
    b.run(3)
    a.close()
    b.close()
    for index in ([1] if first_has_checkpoint else [0, 1]):    # stopped before a checkpoint was written
        (tmp_path / f"w{index}" / "checkpoint.zip").unlink()

    if first_has_checkpoint:        # a replica that was run with another bias is refused before a row is lost
        with pytest.raises(RuntimeError, match="bias.bias_factor: 10.0 there, 2.0 now"):
            _rollback_walkers([tmp_path / "w0", tmp_path / "w1"],
                              dict(CONDITIONS, bias=dict(CONDITIONS["bias"], bias_factor=2.0)))
        assert len(read_hills(tmp_path / "w1" / "HILLS")) == 3
    _rollback_walkers([tmp_path / "w0", tmp_path / "w1"], CONDITIONS)
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    assert (a.n_hills, b.n_hills) == (20 if first_has_checkpoint else 0, 0)
    for target in (25, 30):
        a.run(target)
        b.run(target)
    for walker in (a, b):
        walker.sync_walkers()
    expected, hills = total_bias(tmp_path, a.bias.grid, 2)
    assert len(hills) == 60
    for walker in (a, b):
        assert np.allclose(walker.bias.grid.values, expected, rtol=1e-7, atol=1e-7)
        walker.close()


def test_the_bias_of_a_sample_is_rebuilt_from_what_its_walker_had_read(tmp_path):
    """Walkers out of step: A deposits 30 hills before B starts, and B is at 10 when A ends. The
    bias of a sample is the sum of its walker's own hills and of the rows of the other file that
    the record of the walker names. Without the record there is no estimate from the samples."""
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    a.run(30)
    b.run(10)
    a.run(40)
    b.run(40)
    grid = a.bias.grid.empty_copy()
    a.close()
    b.close()
    for rebuilt, written in rebuilt_and_written(tmp_path, grid):
        # measured: 3e-7 kJ/mol. With all hills up to the time of the sample: 1.1 and 3.3 kJ/mol.
        assert written.max() > 5.0 and np.abs(rebuilt - written).max() < 1e-5

    (tmp_path / "w1" / "SYNC").unlink()            # a run from before the record existed
    result = analyze_run(tmp_path, write=False)
    assert result["dG_reweighted"] is None and result["dG_second_half"] is None
    assert any("kept no record" in note for note in result["diagnostics"])


def test_samples_under_hills_that_a_restart_removed_are_left_out(tmp_path):
    """B loses 5 hills in a crash after A had read them: the bias of A's samples in between cannot
    be rebuilt. Those samples are left out and counted; every other sample is rebuilt as before."""
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    b.run(20)
    checkpoint = (tmp_path / "w1" / "checkpoint.zip").read_bytes()     # the last one before the crash
    b.run(25)
    a.run(20)             # reads the 25 hills of B after its first one
    a.close()
    b.close()
    (tmp_path / "w1" / "checkpoint.zip").write_bytes(checkpoint)
    _rollback_walkers([tmp_path / "w0", tmp_path / "w1"], CONDITIONS)
    _, a = make_walker(tmp_path, 0, 2, seed=1)
    _, b = make_walker(tmp_path, 1, 2, seed=2)
    a.run(40)
    b.run(40)
    grid = a.bias.grid.empty_copy()
    a.close()
    b.close()

    (of_a, written_a), (of_b, written_b) = rebuilt_and_written(tmp_path, grid)
    lost = np.isnan(of_a)
    assert lost.tolist() == [False] + [True] * 19 + [False] * 20       # A's samples after its hills 1 to 19
    assert np.abs(of_a - written_a)[~lost].max() < 1e-5 and np.abs(of_b - written_b).max() < 1e-5
    result = analyze_run(tmp_path, write=False)
    assert any(note.startswith("19 sample(s) are left out") for note in result["diagnostics"])


def test_a_run_that_leaves_out_a_walker_with_hills_is_refused(tmp_path):
    """The analysis adds every HILLS file of a replica, and a walker reads only the ones it is
    run with: with a walker left out, the bias of the analysis is not the bias of the run."""
    base = tmp_path / "rep0"
    for k in range(3):
        walker = make_walker(base, k, 3, seed=k + 1)[1]
        walker.run(20)                  # a checkpoint each
        walker.close()
    files = [base / f"w{k}" / name for k in range(3) for name in ("HILLS", "COLVAR", "SYNC")]
    before = [path.read_bytes() for path in files]
    with pytest.raises(RuntimeError, match="w2"):
        _rollback_walkers([base / "w0", base / "w1"], CONDITIONS)
    for n_walkers, index in ((2, 0), (1, None)):
        with pytest.raises(RuntimeError, match="w2"):
            run_stage(Config(run=RunConfig(walkers=n_walkers)), tmp_path, walker=index)
    assert before == [path.read_bytes() for path in files]

    (tmp_path / "rep1").mkdir()                     # a run without walkers is not continued with walkers
    (tmp_path / "rep1" / "HILLS").write_text("")
    with pytest.raises(RuntimeError, match="without walkers"):
        run_stage(Config(run=RunConfig(walkers=2)), tmp_path, replica=1, walker=0)

    # one walker more: it reads the hills that are there, and the others read its hills
    _rollback_walkers([base / f"w{k}" for k in range(4)], CONDITIONS)
    walkers = [make_walker(base, k, 4, seed=k + 1)[1] for k in range(4)]
    for walker in walkers:
        walker.run(25)
    for walker in walkers:
        walker.sync_walkers()
    expected, hills = total_bias(base, walkers[0].bias.grid, 4)
    assert len(hills) == 100
    for walker in walkers:
        assert np.allclose(walker.bias.grid.values, expected, rtol=1e-7, atol=1e-7)
        walker.close()


@pytest.mark.slow
def test_walkers_far_out_of_step_recover_the_exact_free_energy(tmp_path):
    """One walker ends its 10 ns before the other starts: half of the samples were taken under a
    bias far from the sum of the hills up to their time."""
    errors = []
    for pair in range(4):
        base = tmp_path / str(pair)
        base.mkdir()
        toy, a = make_walker(base, 0, 2, seed=100 + 2 * pair)
        _, b = make_walker(base, 1, 2, seed=101 + 2 * pair)
        a.run(5000)
        b.run(5000)
        a.close()
        b.close()
        result = analyze_run(base, write=False)
        errors.append(result["dG_reweighted"] - toy.exact_binding_free_energy(300.0, result["site_max"]))
    # 12 such pairs: mean -0.17, standard deviation 0.67 kJ/mol. Weighted with all hills up to the
    # time of a sample: standard deviation 1.63, and -2.28, 0.14, 1.74, -2.10 for these four.
    assert np.abs(errors).max() < 2.0 and abs(np.mean(errors)) < 1.0


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
