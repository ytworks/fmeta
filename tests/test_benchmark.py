"""Benchmark sets (FM-D17): the statistics of a report, the set definition, the order of the
jobs and the restart, the report and its check."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fixtures import BENZENE_XML
from scipy import stats
from scipy.spatial.transform import Rotation

from fmeta.benchmark import (
    REFERENCE_COLUMNS,
    agreement,
    code_version,
    fetch_inputs,
    format_status,
    load_set,
    prepare_set,
    prepared_ligands,
    report_set,
    run_set,
    sdf_charge,
    sdf_coordinates,
    sdf_records,
    set_statistics,
    status_rows,
)
from fmeta.cli import main
from fmeta.config import Config
from fmeta.engine import BiasSettings, FunnelBias, MetadynamicsRun, RunSettings
from fmeta.scheduler import read_schedule, verify_schedule
from fmeta.testing.toy import make_simulation, make_toy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_bench_results  # noqa: E402

EXAMPLES = ROOT / "examples" / "benchmarks"
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


# ---- the inputs and the work directories ------------------------------------


def test_a_file_is_fetched_once_and_content_with_another_hash_is_refused(tmp_path, inputs):
    def remote(data):
        for entry in data["files"].values():
            entry["url"] = f"https://example.org/{entry.pop('file')}"

    benchmark = load_set(write_set(tmp_path / "set", inputs, remote))
    fetched = []

    def download(url):
        fetched.append(url)
        return (benchmark.directory / url.rsplit("/", 1)[1]).read_bytes()

    paths = fetch_inputs(benchmark, tmp_path / "work", download)
    assert {key: sha256(path) for key, path in paths.items()} == {
        key: source.sha256 for key, source in benchmark.files.items()}
    fetch_inputs(benchmark, tmp_path / "work", download)
    assert len(fetched) == 2                                          # what is verified is not fetched again
    with pytest.raises(ValueError, match="not the file the set was written for"):
        fetch_inputs(benchmark, tmp_path / "other", lambda url: b"something else")
    assert list((tmp_path / "other" / "inputs").iterdir()) == []      # and nothing of it was written


def test_a_charge_that_is_not_that_of_the_record_is_refused(tmp_path, inputs):
    benchmark = load_set(write_set(tmp_path / "set", inputs, lambda data: data["ligands"][0].update(charge=1)))
    with pytest.raises(ValueError, match=r"formal charge \+0 in the ligand file, \+1 in set.yaml"):
        prepare_set(benchmark, tmp_path / "work")
    assert not (tmp_path / "work" / "a").exists()
    record = sdf_records(benchmark.files["ligands"].file)["a"][0]
    assert sdf_charge(record + ["M  CHG  2   1  -1   2  -1"], "a") == -2          # the property line
    anion = [line[:36] + "  5" + line[39:] if index == 4 else line for index, line in enumerate(record)]
    assert sdf_charge(anion, "a") == -1                                          # the field of an atom line


def ca_atoms(path) -> dict:
    """Residue number -> the coordinates of its CA atom (Å)."""
    return {int(line[22:26]): np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            for line in Path(path).read_text().splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"}


def write_reference(path, protein, rotation, translation, shift=0.0) -> None:
    """Another structure of the protein in another frame: the CA atom of every second residue
    at ``rotation @ x + translation``. ``shift`` (Å) displaces half of them."""
    lines = []
    for count, (number, xyz) in enumerate(sorted(ca_atoms(protein).items())[::2]):
        x, y, z = rotation @ xyz + translation + (shift if count % 2 else 0.0)
        lines.append(f"ATOM  {count + 1:5d}  CA  GLY A{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def superposed_set(directory, inputs, rotation, translation, shift=0.0):
    def edit(data):
        write_reference(directory / "reference.pdb", inputs[0], rotation, translation, shift)
        data["superpose_on"] = dict(file="reference.pdb", sha256=sha256(directory / "reference.pdb"), max_rmsd=0.1)

    return load_set(write_set(directory, inputs, edit))


def test_the_protein_is_put_on_the_reference_structure_and_the_ligands_move_with_it(tmp_path, inputs):
    """A known rotation and translation, recovered from the CA atoms that the two structures
    share by residue number."""
    rotation = Rotation.from_rotvec(np.deg2rad(100.0) * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)).as_matrix()
    translation = np.array([12.0, -7.0, 30.0])
    benchmark = superposed_set(tmp_path / "set", inputs, rotation, translation)
    work = tmp_path / "work"
    prepare_set(benchmark, work)
    target = ca_atoms(benchmark.directory / "reference.pdb")
    moved = ca_atoms(work / "inputs" / "protein_superposed.pdb")
    assert len(moved) == 35 and len(target) == 18
    assert max(np.abs(moved[number] - xyz).max() for number, xyz in target.items()) < 0.005     # Å
    before = sdf_coordinates(sdf_records(benchmark.files["ligands"].file)["b D"][0], "b D")
    after = sdf_coordinates(sdf_records(work / "b_D" / "ligand.sdf")["b D"][0], "b D")
    assert np.abs(after - (before @ rotation.T + translation)).max() < 0.005
    assert Path(Config.load(work / "b_D" / "config.yaml").input.protein) == work / "inputs" / "protein_superposed.pdb"
    assert json.loads((work / "inputs" / "superposition.json").read_text())["matched_ca"] == 18


def test_a_reference_structure_of_another_shape_is_refused(tmp_path, inputs):
    benchmark = superposed_set(tmp_path / "set", inputs, np.eye(3), np.zeros(3), shift=5.0)
    with pytest.raises(ValueError, match="above max_rmsd = 0.1 nm"):
        prepare_set(benchmark, tmp_path / "work")
    assert not (tmp_path / "work" / "a").exists()


def test_prepare_writes_a_work_directory_per_ligand_and_one_configuration_only(tmp_path, inputs):
    benchmark = load_set(write_set(tmp_path / "set", inputs))
    work = tmp_path / "work"
    prepare_set(benchmark, work)                                      # the pilot ligands
    assert sorted(path.name for path in work.iterdir() if path.is_dir()) == ["a", "b_D", "inputs"]
    config = Config.load(work / "b_D" / "config.yaml")
    assert Path(config.input.protein) == work / "inputs" / "protein.pdb"
    assert Path(config.input.ligand) == work / "b_D" / "ligand.sdf"
    assert list(sdf_records(config.input.ligand)) == ["b D"]          # its own record, nothing else
    assert (config.run.length_ns, config.run.replicas) == (0.004, 2)

    prepare_set(benchmark, work, "all", length_ns=0.008)              # more ligands and a longer plan: allowed
    assert [(item.ligand.name, item.config.run.length_ns) for item in prepared_ligands(benchmark, work)] == [
        ("a", 0.008), ("b D", 0.008), ("c", 0.008)]

    other = load_set(write_set(tmp_path / "other", inputs, lambda data: data["config"].update(metad=dict(height=2.0))))
    for call in (lambda: prepare_set(other, work), lambda: prepared_ligands(other, work)):
        with pytest.raises(ValueError, match="another configuration or set definition"):
            call()
    text = (work / "a" / "config.yaml").read_text()
    (work / "a" / "config.yaml").write_text(text.replace("temperature: 300.0", "temperature: 310.0"))
    assert "temperature: 310.0" in (work / "a" / "config.yaml").read_text()
    with pytest.raises(ValueError, match="config.yaml was changed"):
        prepared_ligands(benchmark, work)


# ---- the order of the jobs and the restart ----------------------------------

class Ended:
    """A child process that has ended."""

    def __init__(self, code):
        self.pid, self.code = None, code

    def wait(self):
        return self.code

    def terminate(self):
        pass

    kill = terminate


class Stages:
    """Stands in for the child processes of ``run_set``: a stage that is started ends at once
    and leaves in the work directory of its ligand what marks it as finished (a replica: a log
    that ends at ``hills`` hills). The stages named in ``fail`` end with an error and leave
    nothing."""

    def __init__(self, hills=4, fail=()):
        self.started, self.hills, self.fail = [], hills, fail

    def __call__(self, task, device):
        self.started.append(task.name)
        if task.name in self.fail:
            return Ended(1)
        directory, stage = task.log.parent, task.name.split("/")[1]
        if stage == "prepare":
            (directory / "prepared").mkdir()
            (directory / "prepared" / "meta.json").write_text("{}")
        elif stage == "equilibrate":
            (directory / "equil").mkdir()
            (directory / "equil" / "equil.json").write_text(json.dumps(dict(funnel_checks=dict(ok=True))))
        else:
            (directory / stage).mkdir(exist_ok=True)
            (directory / stage / "run.log").write_text(
                f"hills {self.hills}/{self.hills}  time {self.hills / 1000:.3f} ns  speed 100.0 ns/day\n")
        return Ended(0)


def stages(stage, ligands=("a", "b_D", "c")) -> list:
    return [f"{ligand}/{stage}" for ligand in ligands]


def test_every_ligand_has_a_first_replica_before_any_has_a_second_and_finished_stages_are_not_run_again(
        tmp_path, inputs):
    benchmark = load_set(write_set(tmp_path / "set", inputs))
    work = tmp_path / "work"
    prepare_set(benchmark, work, "all")                               # 2 replicas of 4 hills per ligand

    children = Stages()
    run_set(benchmark, work, until="prepare", launcher=children)
    assert children.started == stages("prepare")
    status = run_set(benchmark, work, launcher=children)              # one device: the order of the list
    assert children.started[3:] == stages("equilibrate") + stages("rep0") + stages("rep1")
    assert set(status.values()) == {"done"}
    assert verify_schedule(read_schedule(work / "schedule.jsonl")[-1]) == []

    assert run_set(benchmark, work, launcher=children) == {}          # the work directory is the truth
    assert len(children.started) == 12 and len(read_schedule(work / "schedule.jsonl")) == 2
    assert len((work / "runs.jsonl").read_text().splitlines()) == 2   # the code and the image of each call
    rows = status_rows(benchmark, work)
    assert {row["state"] for row in rows} == {"done"}
    assert format_status(rows).splitlines()[-1] == "3 ligand(s): 6 of 6 run(s) done, 0.024 of 0.024 ns"

    prepare_set(benchmark, work, length_ns=0.008)                     # a longer plan: only the replicas are left
    longer = Stages(hills=8)
    run_set(benchmark, work, launcher=longer)
    assert longer.started == stages("rep0") + stages("rep1")

    shutil.rmtree(work / "a" / "prepared")
    with pytest.raises(ValueError, match="without the stage before it"):
        run_set(benchmark, work, launcher=longer)


def test_a_stage_that_fails_holds_back_its_own_ligand_only_and_the_next_call_takes_it_up(tmp_path, inputs):
    benchmark = load_set(write_set(tmp_path / "set", inputs))
    work = tmp_path / "work"
    prepare_set(benchmark, work, "all")

    children = Stages(fail=["b_D/prepare"])
    with pytest.raises(RuntimeError, match=r"^b_D/prepare failed: no output \(see .*/b_D/prepare\.log\)$"):
        run_set(benchmark, work, launcher=children)
    others = ("a", "c")
    assert children.started == (stages("prepare") + stages("equilibrate", others) + stages("rep0", others)
                                + stages("rep1", others))
    states = {(row["ligand"], row["stage"]): row["state"] for row in status_rows(benchmark, work)}
    assert [states["b D", stage] for stage in ("prepare", "equilibrate", "rep0", "rep1")] == [
        "failed", "skipped", "skipped", "skipped"]
    assert {state for (ligand, _), state in states.items() if ligand != "b D"} == {"done"}

    again = Stages()
    run_set(benchmark, work, launcher=again)
    assert again.started == [f"b_D/{stage}" for stage in ("prepare", "equilibrate", "rep0", "rep1")]


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


# ---- the report and its check -----------------------------------------------

VERSION = ("0123456789abcdef0123456789abcdef01234567", False)        # (commit, dirty) of the code of a report


def run_toy(directory, seed) -> None:
    """A short run of the solvable toy system: a replica that the analysis can read."""
    toy = make_toy()
    settings = BiasSettings(temperature=300.0, bias_factor=10.0, height=1.2, sigma_s=0.05, sigma_rho=0.05,
                            pace=100, points_per_sigma=2.0)
    bias = FunnelBias(toy.system, toy.funnel, settings)             # its force joins the system before the context
    run = MetadynamicsRun(make_simulation(toy, temperature=300.0, seed=seed), bias, directory, toy.restraint_group,
                          RunSettings(checkpoint_hills=5000))
    run.run(600)
    run.close()


@pytest.fixture(scope="module")
def reported(tmp_path_factory, inputs):
    """The report of a set of which ``a`` has two replicas, ``b D`` one and a second without
    a hill, and ``c`` none: ``(directory of the reports, path of the JSON file)``."""
    root = tmp_path_factory.mktemp("reported")
    benchmark = load_set(write_set(root / "set", inputs))
    work = root / "work"
    prepare_set(benchmark, work, "all")
    for replica, seed in (("a/rep0", 1), ("a/rep1", 2), ("b_D/rep0", 3)):
        run_toy(work / replica, seed)
    (work / "b_D" / "rep1").mkdir()
    (work / "b_D" / "rep1" / "HILLS").write_text("")
    call = dict(commit=VERSION[0], dirty=False, image="sha256:0a", lock_sha256="0b")
    (work / "runs.jsonl").write_text("".join(json.dumps(dict(time=f"2026-11-1{day}T00:00:00+00:00", **call)) + "\n"
                                             for day in (4, 5)))
    path = report_set(benchmark, work, root / "reports", VERSION,
                      environment=dict(image="sha256:0a", lock_sha256="0b"), date="2026-11-16")
    return root / "reports", path


def test_a_report_keeps_every_ligand_gives_both_units_and_compares_the_values_it_lists(reported):
    reports, path = reported
    assert path == reports / "2026-11-16-012345678-toy.json"
    report = json.loads(path.read_text())
    a, b, c = report["ligands"]
    assert [ligand["name"] for ligand in (a, b, c)] == list(NAMES)   # no ligand is dropped
    assert [ligand["n_replicas"] for ligand in (a, b, c)] == [2, 1, 0]
    assert a["dg"]["kj_mol"] == pytest.approx(np.mean([replica["dg"]["kj_mol"] for replica in a["replicas"]]), abs=1e-4)
    assert a["dg"]["kcal_mol"] == pytest.approx(a["dg"]["kj_mol"] / 4.184, abs=1e-4)
    assert a["ci95"]["kj_mol"][0] < a["dg"]["kj_mol"] < a["ci95"]["kj_mol"][1]
    assert b["ci95"] is None and list(b["not_analysed"]) == ["rep1"]  # one replica: a value without an interval
    assert c["dg"] is None and c["error"] == "no replica has hills yet"
    assert a["experiment"]["dg"] == dict(kj_mol=-33.472, kcal_mol=-8.0)
    assert [call["time"] for call in report["runs"]] == ["2026-11-14T00:00:00+00:00"]    # one code, its first call

    statistics = report["statistics"]
    fmeta, fep = (statistics[key] for key in ("fmeta_vs_experiment", "fep_vs_experiment"))
    errors = [ligand["dg"]["kj_mol"] - ligand["experiment"]["dg"]["kj_mol"] for ligand in (a, b)]
    assert fmeta["ligands"] == fep["ligands"] == ["a", "b D"] and statistics["without_value"] == ["c"]
    assert fmeta["mean_signed_error"]["value"]["kj_mol"] == pytest.approx(np.mean(errors), abs=1e-4)
    assert fep["mue"]["value"] == dict(kj_mol=1.4644, kcal_mol=0.35)   # FEP - experiment: -0.5 and -0.2 kcal/mol

    text = path.with_suffix(".md").read_text()
    assert "## Ligands (kcal/mol)" in text and "## Ligands (kJ/mol)" in text
    assert f"| a | 2 (2 / 2) | {a['dg']['kcal_mol']:.2f} |" in text and "| c | 0 (0 / 2) |  |" in text
    index = json.loads((reports / "index.json").read_text())
    assert index == dict(entries=[dict(file=path.name, date="2026-11-16", commit=VERSION[0], dirty=False, set="toy")])


def listed(name, dg, experiment, fep) -> dict:
    """A ligand as a report lists it, with what the statistics read; values in kJ/mol."""
    def energy(value):
        return None if value is None else dict(kj_mol=value, kcal_mol=value / 4.184)
    return dict(name=name, dg=energy(dg), experiment=dict(dg=energy(experiment)), fep=dict(dg=energy(fep)), afe=None)


def test_the_statistics_compare_every_method_on_the_ligands_that_have_a_value_of_fmeta():
    statistics = set_statistics([listed("a", -30.0, -31.0, -31.5), listed("b", -36.0, -35.0, -35.5),
                                 listed("c", -20.0, -40.0, -40.5), listed("d", None, -38.0, -38.5)])
    fmeta, fep = statistics["fmeta_vs_experiment"], statistics["fep_vs_experiment"]
    assert fmeta["ligands"] == fep["ligands"] == ["a", "b", "c"] and statistics["without_value"] == ["d"]
    assert fmeta["mue"]["value"]["kj_mol"] == pytest.approx(22.0 / 3.0, abs=1e-4)
    assert fmeta["mean_signed_error"]["value"]["kj_mol"] == pytest.approx(20.0 / 3.0, abs=1e-4)
    assert fep["mue"]["value"] == dict(kj_mol=0.5, kcal_mol=0.1195)             # a constant offset only
    assert fep["mue_offset_removed"]["value"]["kj_mol"] == 0.0
    assert statistics["afe_vs_experiment"] is None


def test_the_check_accepts_a_report_and_catches_one_that_was_changed(reported, tmp_path):
    reports, path = reported
    assert verify_bench_results.verify_reports(reports) == ([], 1)
    assert verify_bench_results.verify_reports(tmp_path) == ([], -1)  # no report yet is fine

    def changed(change) -> str:
        directory = tmp_path / f"reports{len(list(tmp_path.iterdir()))}"
        shutil.copytree(reports, directory)
        report = json.loads((directory / path.name).read_text())
        change(report, directory)
        (directory / path.name).write_text(json.dumps(report))
        errors, _ = verify_bench_results.verify_reports(directory)
        return "\n".join(errors)

    assert "holds no verdict" in changed(lambda report, directory: report["statistics"].update(passed=True))
    assert "no environment" in changed(lambda report, directory: report["environment"].update(image=None))
    assert "commit differs" in changed(lambda report, directory: report.update(commit="f" * 40))
    assert "not beside it" in changed(lambda report, directory: (directory / path.name).with_suffix(".md").unlink())
    assert "not listed" in changed(lambda report, directory: shutil.copy(directory / path.name, directory / "x.json"))


def test_the_commit_of_a_report_is_given_where_git_cannot_tell(tmp_path, monkeypatch):
    monkeypatch.delenv("FMETA_CODE_COMMIT", raising=False)
    assert code_version(VERSION[0], root=tmp_path) == (VERSION[0], None)      # dirty: not known
    with pytest.raises(ValueError, match="give it with --commit"):
        code_version(root=tmp_path)
    with pytest.raises(ValueError, match="full hash"):
        code_version("0123456", root=tmp_path)

    # scripts/in_docker.sh passes what git said on the host: a worktree has no git in the container
    monkeypatch.setenv("FMETA_CODE_COMMIT", VERSION[0])
    monkeypatch.setenv("FMETA_CODE_DIRTY", "1")
    assert code_version(root=tmp_path) == (VERSION[0], True)
    assert code_version(VERSION[0], root=tmp_path) == (VERSION[0], True)
    with pytest.raises(ValueError, match="is not the commit of this code"):
        code_version("f" * 40, root=tmp_path)


# ---- the commands -----------------------------------------------------------

def test_the_commands_prepare_build_show_and_report_a_set(tmp_path, inputs, capsys, monkeypatch):
    """The prepare stage is a real child process of the scheduler here: the command it is given
    builds the system of the ligand."""
    common = [str(write_set(tmp_path / "set", inputs)), "-o", str(tmp_path / "work")]
    main(["benchmark", "prepare", *common, "--ligands", "b D", "--replicas", "1", "--length-ns", "0.002"])
    assert "plan: 1 replica(s) of 0.002 ns per ligand" in capsys.readouterr().out

    main(["benchmark", "run", *common, "--until", "prepare"])
    meta = json.loads((tmp_path / "work" / "b_D" / "prepared" / "meta.json").read_text())
    assert len(meta["ligand_atoms"]) == 12                            # benzene, in the system that was built
    capsys.readouterr()
    main(["benchmark", "status", *common])
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["b", "D", "prepare", "done"] in rows and ["b", "D", "equilibrate", "not", "started"] in rows

    monkeypatch.setattr("fmeta.benchmark.code_version", lambda commit: VERSION)    # git or --commit: not the subject
    main(["benchmark", "report", *common, "--reports", str(tmp_path / "reports")])
    (path,) = (tmp_path / "reports").glob("*-012345678-toy.json")
    (ligand,) = json.loads(path.read_text())["ligands"]
    assert (ligand["name"], ligand["error"]) == ("b D", "no replica has hills yet")
