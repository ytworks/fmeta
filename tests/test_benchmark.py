"""Benchmark sets (FM-D17): the statistics of a report, the set definition, the order of the
jobs and the restart, the report and its check."""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from fixtures import BENZENE_XML
from scipy import stats

from fmeta.benchmark import REFERENCE_COLUMNS, agreement, load_set

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "benchmarks"
NAMES = ("a", "b D", "c")                                 # "b D": a name that is not a directory name
ROWS = ("a,-8.0,,,10.1000/exp,-8.5,0.1,relative,10.1000/fep,,,",
        "b D,-9.0,,,10.1000/exp,-9.2,0.1,relative,10.1000/fep,,,",
        "c,-10.0,,,10.1000/exp,-9.6,0.1,relative,10.1000/fep,,,")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_set(directory, inputs, edit=None, rows=ROWS) -> Path:
    """A set of three benzene poses on the small test protein, its files beside ``set.yaml``.
    ``edit`` changes the definition before it is written."""
    directory.mkdir()
    protein, benzene = inputs
    shutil.copy(protein, directory / "protein.pdb")
    record = Path(benzene).read_text().split("\n", 1)[1]            # without the line of the name
    (directory / "ligands.sdf").write_text("".join(f"{name}\n{record}" for name in NAMES))
    data = dict(
        name="toy",
        files={key: dict(file=name, sha256=sha256(directory / name))
               for key, name in (("protein", "protein.pdb"), ("ligands", "ligands.sdf"))},
        ligands=[dict(name=name, charge=0) for name in NAMES],
        pilot=["a", "b D"],
        config=dict(system=dict(ligand_method="xml", ligand_xml=[BENZENE_XML], ligand_resname="BNZ"),
                    run=dict(length_ns=0.004, replicas=2)))
    if edit is not None:
        edit(data)
    (directory / "set.yaml").write_text(yaml.safe_dump(data))
    (directory / "reference.csv").write_text("\n".join([",".join(REFERENCE_COLUMNS), *rows]) + "\n")
    return directory


# ---- the set definition -----------------------------------------------------


@pytest.mark.parametrize("directory", sorted(path.parent for path in EXAMPLES.glob("*/set.yaml")),
                         ids=lambda path: path.name)
def test_the_sets_of_the_repository_are_valid(directory):
    benchmark = load_set(directory)
    assert benchmark.name == directory.name
    assert all(source.url.startswith("https://") for source in benchmark.files.values())      # nothing local
    assert {item.fep_kind for item in benchmark.reference.values()} == {"relative"}


def test_a_set_keeps_the_order_of_its_ligands_and_gives_each_a_directory(tmp_path, inputs):
    benchmark = load_set(write_set(tmp_path / "set", inputs, lambda data: data.update(pilot=["c", "a"])))
    assert [(ligand.name, ligand.directory) for ligand in benchmark.ligands] == [
        ("a", "a"), ("b D", "b_D"), ("c", "c")]
    assert benchmark.pilot == ("a", "c")                              # in the order of the set
    assert benchmark.reference["b D"].exp_dg == pytest.approx(-9.0 * 4.184)
    assert benchmark.reference["b D"].fep_dg == pytest.approx(-9.2 * 4.184)


def _url_without_commit(data):
    data["source"] = dict(commit="0" * 40, raw="https://example.org/main/")


MISTAKES = {
    "a hash that is not a sha256": (lambda data: data["files"]["protein"].update(sha256="abc"), ROWS,
                                    "sha256 must be"),
    "a URL that does not name the commit": (_url_without_commit, ROWS, "contains source.commit"),
    "a pilot ligand that is not in the set": (lambda data: data.update(pilot=["a", "z"]), ROWS, "pilot must name"),
    "two ligands with one directory": (lambda data: data["ligands"].append(dict(name="b_D", charge=0)), ROWS,
                                       "used twice"),
    "a reference row for another ligand": (None, ROWS + (ROWS[0].replace("a,", "z,", 1),), "not a ligand"),
    "a ligand without a reference row": (None, ROWS[:2], "no row for c"),
    "relative and absolute FEP values": (None, ROWS[:2] + (ROWS[2].replace("relative", "absolute"),),
                                         "cannot be compared"),
    "an input in the configuration": (lambda data: data["config"].update(input=dict(protein="p.pdb")), ROWS,
                                      "config.input"),
}


@pytest.mark.parametrize("mistake", MISTAKES)
def test_a_set_definition_with_a_mistake_is_refused(tmp_path, inputs, mistake):
    edit, rows, message = MISTAKES[mistake]
    with pytest.raises(ValueError, match=message):
        load_set(write_set(tmp_path / "set", inputs, edit, rows))


def test_a_measured_ki_becomes_a_free_energy_at_the_temperature_of_the_measurement(tmp_path, inputs):
    """1 nM is -12.28 kcal/mol at 298.15 K; the same Ki at 310 K is 310 / 298.15 times that."""
    rows = ("a,,1.0,298.15,10.1000/exp,,,,,,,", "b D,,1.0,310.0,10.1000/exp,,,,,,,", ROWS[2])
    reference = load_set(write_set(tmp_path / "set", inputs, rows=rows)).reference
    assert reference["a"].exp_dg == pytest.approx(-12.278 * 4.184, abs=0.01)
    assert reference["b D"].exp_dg == pytest.approx(reference["a"].exp_dg * 310.0 / 298.15)
    assert reference["b D"].exp_temperature == 310.0
    with pytest.raises(ValueError, match="needs exp_temperature_k"):
        load_set(write_set(tmp_path / "other", inputs, rows=("a,,1.0,,10.1000/exp,,,,,,,",) + ROWS[1:]))


# ---- statistics -------------------------------------------------------------


def test_a_constant_offset_is_the_whole_error_and_vanishes_when_the_offset_is_removed():
    reference = np.array([-40.0, -35.0, -30.0, -25.0])
    result = agreement(reference + 1.0, reference)
    assert result["n"] == 4
    for name in ("mue", "rmse", "mean_signed_error"):
        assert result[name] == dict(value=1.0, ci95=[1.0, 1.0])       # every resample has it too
    for name in ("mue_offset_removed", "rmse_offset_removed"):
        assert result[name] == dict(value=0.0, ci95=[0.0, 0.0])
    assert result["pearson_r"]["value"] == pytest.approx(1.0, abs=1e-12)
    assert result["kendall_tau"]["value"] == 1.0


def test_errors_of_both_signs_give_the_known_measures():
    reference = np.array([-40.0, -35.0, -30.0, -25.0])
    result = agreement(reference + np.array([1.0, -1.0, 2.0, -2.0]), reference)
    assert result["mue"]["value"] == pytest.approx(1.5)
    assert result["rmse"]["value"] == pytest.approx(np.sqrt(2.5))
    assert result["mean_signed_error"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert result["mue_offset_removed"]["value"] == pytest.approx(1.5)
    assert result["rmse_offset_removed"]["value"] == pytest.approx(np.sqrt(2.5))


def test_the_correlations_are_those_of_scipy_also_with_ties():
    reference = np.array([-45.0, -42.0, -42.0, -40.0, -38.0, -38.0, -38.0, -33.0, -30.0, -30.0])
    predicted = np.array([-44.0, -40.0, -43.0, -40.0, -35.0, -39.0, -36.0, -36.0, -31.0, -28.0])
    result = agreement(predicted, reference, resamples=0)
    assert result["pearson_r"]["value"] == pytest.approx(stats.pearsonr(predicted, reference)[0], abs=1e-12)
    assert result["kendall_tau"]["value"] == pytest.approx(stats.kendalltau(predicted, reference)[0], abs=1e-12)
    assert result["kendall_tau"]["ci95"] is None                      # no resamples, no interval


def test_the_bootstrap_interval_is_that_of_the_mean_of_the_errors():
    """40 errors of mean 0.5 and standard deviation 1: the 95% interval of their mean is the
    sample mean -+ 1.96 s / sqrt(40), to the precision of 10000 resamples."""
    rng = np.random.default_rng(7)
    reference = rng.uniform(-50.0, -25.0, 40)
    error = rng.normal(0.5, 1.0, 40)
    result = agreement(reference + error, reference)["mean_signed_error"]
    half = 1.96 * error.std() / np.sqrt(40)
    assert result["value"] == pytest.approx(error.mean())
    assert result["ci95"] == pytest.approx([error.mean() - half, error.mean() + half], abs=0.03)
    assert result == agreement(reference + error, reference)["mean_signed_error"]     # fixed seed


def test_measures_that_do_not_exist_are_none():
    two = agreement([-30.0, -40.0], [-31.0, -38.0])
    assert two["mue"] == dict(value=1.5, ci95=None)                   # no interval below three ligands
    assert two["pearson_r"] == two["kendall_tau"] == dict(value=None, ci95=None)
    flat = agreement([-30.0, -30.0, -30.0], [-31.0, -38.0, -35.0])
    assert flat["pearson_r"]["value"] is None and flat["kendall_tau"]["value"] is None
    assert flat["mue"]["ci95"] is not None
    assert agreement([], [])["mue"] == dict(value=None, ci95=None)
