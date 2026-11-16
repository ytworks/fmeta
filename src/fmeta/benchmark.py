"""Benchmark sets (FM-D17): the ligands of one target through the ordinary stages, and their
binding free energies beside experiment and FEP.

A set is a directory with ``set.yaml`` (where the structures come from, pinned by commit and
sha256; the ligands; the differences from the default configuration), ``reference.csv`` (the
experimental and the FEP values per ligand) and a ``README.md``. ``fmeta benchmark prepare``
fetches and verifies the inputs and writes one ordinary fmeta work directory per ligand;
``run`` gives every stage of every ligand to one call of the scheduler (FM-D14); ``status``
shows where it stands; ``report`` writes the comparison. A report carries no verdict.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import numpy as np
import yaml

from .analysis import KCAL
from .config import Config
from .engine.platform import resolve_devices
from .funnel.geometry import KB
from .scheduler import Job, Task, directory_lock, is_locked, read_schedule, run_jobs
from .workflow import replica_jobs, status_table, unfinished_jobs

SET_KEYS = ("name", "title", "source", "files", "superpose_on", "ligands", "pilot", "config")
SOURCE_KEYS = ("repository", "commit", "raw", "licence")
FILE_KEYS = ("path", "url", "file", "sha256")
SUPERPOSE_KEYS = FILE_KEYS + ("reference_chain", "protein_chain", "max_rmsd")
REFERENCE_COLUMNS = ("ligand", "exp_dg_kcal", "exp_ki_nm", "exp_temperature_k", "exp_doi",
                     "fep_dg_kcal", "fep_se_kcal", "fep_kind", "fep_source",
                     "afe_dg_kcal", "afe_se_kcal", "afe_source")
FEP_KINDS = ("relative", "absolute")
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261116
ENERGY_MEASURES = ("mue", "rmse", "mean_signed_error", "mue_offset_removed", "rmse_offset_removed")
MEASURES = ENERGY_MEASURES + ("pearson_r", "kendall_tau")

_NAME = re.compile(r"[A-Za-z0-9._-]+")
_KEY = re.compile(r"[A-Za-z0-9_-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


# ---- the set definition -----------------------------------------------------

@dataclass(frozen=True)
class Source:
    """One input file: where it comes from (an https URL or a local file) and its sha256."""
    key: str
    sha256: str
    url: Optional[str] = None
    file: Optional[Path] = None
    suffix: str = ""


@dataclass(frozen=True)
class Superposition:
    """The structure the protein is superposed on before the work directories are written."""
    source: Source
    reference_chain: Optional[str] = None
    protein_chain: Optional[str] = None
    max_rmsd: float = 0.3                    # nm, over the matched CA atoms


@dataclass(frozen=True)
class Ligand:
    name: str                                # the name of its record in the SDF file
    charge: int
    directory: str                           # its work directory under the benchmark work directory


@dataclass(frozen=True)
class Reference:
    """The reference values of one ligand, in kJ/mol at the standard state of 1 mol/L."""
    exp_dg: float
    exp_temperature: Optional[float]
    exp_doi: str
    fep_dg: Optional[float]
    fep_se: Optional[float]
    fep_kind: Optional[str]
    fep_source: str
    afe_dg: Optional[float]
    afe_se: Optional[float]
    afe_source: str


@dataclass(frozen=True)
class BenchmarkSet:
    directory: Path
    name: str
    title: str
    source: dict
    files: dict                              # key -> Source; "protein" and "ligands" are used
    superpose_on: Optional[Superposition]
    ligands: tuple
    pilot: tuple                             # names of the ligands that run first
    config: dict                             # the differences from the default configuration
    reference: dict                          # ligand name -> Reference


def _mapping(kind: str, data, allowed, required=()) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{kind} must be a mapping")
    unknown = sorted(str(key) for key in set(data) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) in {kind}: {', '.join(unknown)}")
    missing = [key for key in required if data.get(key) in (None, "", [], {})]
    if missing:
        raise ValueError(f"{kind} needs {', '.join(missing)}")
    return data


def _raw_base(source: dict) -> Optional[str]:
    """``source.raw`` with a closing slash: the URL the ``path`` entries are relative to. It
    has to name the commit, so that what is fetched cannot change under the same name."""
    commit, raw = source.get("commit"), source.get("raw")
    if commit is not None and not _COMMIT.fullmatch(str(commit)):
        raise ValueError("source.commit must be a full commit hash (40 hexadecimal digits)")
    if raw is None:
        return None
    if commit is None or not str(raw).startswith("https://") or str(commit) not in str(raw):
        raise ValueError("source.raw must be an https:// URL that contains source.commit")
    return str(raw) if str(raw).endswith("/") else f"{raw}/"


def _source(key: str, entry: dict, raw: Optional[str], directory: Path) -> Source:
    kind = f"files.{key}" if key != "superpose_on" else key
    places = [place for place in ("path", "url", "file") if entry.get(place)]
    if len(places) != 1:
        raise ValueError(f"{kind} needs exactly one of path, url and file")
    sha256 = str(entry.get("sha256", ""))
    if not _SHA256.fullmatch(sha256):
        raise ValueError(f"{kind}.sha256 must be 64 lowercase hexadecimal digits")
    place, value = places[0], str(entry[places[0]])
    suffix = Path(value).suffix
    if place == "file":
        return Source(key, sha256, file=(directory / value).resolve(), suffix=suffix)
    if place == "path":
        if raw is None:
            raise ValueError(f"{kind}.path needs source.raw and source.commit")
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError(f"{kind}.path must be relative to source.raw")
        value = raw + value
    if not value.startswith("https://"):
        raise ValueError(f"{kind}.url must start with https://")
    return Source(key, sha256, url=value, suffix=suffix)


def _ligands(entries, pilot) -> tuple:
    if not isinstance(entries, list) or not entries:
        raise ValueError("ligands must be a list of {name, charge}")
    ligands = []
    for entry in entries:
        _mapping("a ligand", entry, ("name", "charge"))
        name, charge = str(entry.get("name", "")).strip(), entry.get("charge")
        if not name or isinstance(charge, bool) or not isinstance(charge, int):
            raise ValueError(f"ligand {name!r} needs a name and its formal charge (an integer)")
        if "," in name:
            raise ValueError(f"ligand name {name!r} contains a comma")
        directory = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        if directory.startswith(".") or directory == "inputs":
            raise ValueError(f"ligand name {name!r} cannot name a work directory")
        ligands.append(Ligand(name, charge, directory))
    for attribute in ("name", "directory"):
        values = [getattr(ligand, attribute) for ligand in ligands]
        twice = sorted({value for value in values if values.count(value) > 1})
        if twice:
            raise ValueError(f"ligand {attribute}(s) used twice: {', '.join(twice)}")
    names = [ligand.name for ligand in ligands]
    pilot = names if pilot is None else [str(name) for name in pilot]
    unknown = [name for name in pilot if name not in names]
    if unknown or not pilot:
        raise ValueError(f"pilot must name ligands of the set (not in it: {', '.join(unknown) or 'none given'})")
    return tuple(ligands), tuple(name for name in names if name in pilot)


def _number(text: str, what: str) -> Optional[float]:
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"{what} is not a number: {text!r}")
    return value


def read_reference(path, names) -> dict:
    """``reference.csv`` of a set: one row for every ligand in ``names``, none for others.

    The experimental value is ``exp_dg_kcal`` as its source gives it, or ``exp_ki_nm`` with
    ``exp_temperature_k``, the temperature of the measurement, from which dG = RT ln(Ki / 1 M).
    A FEP value says in ``fep_kind`` whether it is ``relative`` (differences between the
    ligands, shifted by a constant) or ``absolute``; one file holds one kind."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: a benchmark set needs the reference values")
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REFERENCE_COLUMNS:
            raise ValueError(f"{path}: the columns must be {','.join(REFERENCE_COLUMNS)}")
        rows = list(reader)
    reference: dict = {}
    for line, row in enumerate(rows, start=2):
        where = f"{path} line {line}"
        if None in row or None in row.values():
            raise ValueError(f"{where}: {len(REFERENCE_COLUMNS)} fields expected")
        text = {column: value.strip() for column, value in row.items()}

        def number(column, where=where, text=text):
            return _number(text[column], f"{where}: {column}")

        name = text["ligand"]
        if name not in names:
            raise ValueError(f"{where}: {name!r} is not a ligand of set.yaml")
        if name in reference:
            raise ValueError(f"{where}: a second row for {name!r}")
        dg, ki, temperature = number("exp_dg_kcal"), number("exp_ki_nm"), number("exp_temperature_k")
        if (dg is None) == (ki is None):
            raise ValueError(f"{where}: give exactly one of exp_dg_kcal and exp_ki_nm")
        if ki is not None:
            if temperature is None or ki <= 0 or temperature <= 0:
                raise ValueError(f"{where}: exp_ki_nm must be positive and needs exp_temperature_k, "
                                 "the temperature of the measurement")
            dg = KB * temperature * math.log(ki * 1e-9) / KCAL
        fep, fep_se, kind = number("fep_dg_kcal"), number("fep_se_kcal"), text["fep_kind"] or None
        if (fep is None and (kind or fep_se is not None)) or (fep is not None and kind not in FEP_KINDS):
            raise ValueError(f"{where}: a FEP value needs fep_kind ({' or '.join(FEP_KINDS)}), "
                             "and only a FEP value has a kind and a standard error")
        afe, afe_se = number("afe_dg_kcal"), number("afe_se_kcal")
        if afe is None and afe_se is not None:
            raise ValueError(f"{where}: afe_se_kcal without afe_dg_kcal")
        reference[name] = Reference(
            dg * KCAL, temperature, text["exp_doi"],
            None if fep is None else fep * KCAL, None if fep_se is None else fep_se * KCAL, kind,
            text["fep_source"], None if afe is None else afe * KCAL,
            None if afe_se is None else afe_se * KCAL, text["afe_source"])
    missing = [name for name in names if name not in reference]
    if missing:
        raise ValueError(f"{path} has no row for {', '.join(missing)}")
    if len({item.fep_kind for item in reference.values() if item.fep_kind}) > 1:
        raise ValueError(f"{path}: relative and absolute FEP values in one set cannot be compared")
    return reference


def load_set(directory) -> BenchmarkSet:
    """Read and check the set definition in ``directory`` (``set.yaml``, ``reference.csv``)."""
    directory = Path(directory).resolve()
    path = directory / "set.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: a benchmark set is a directory with set.yaml")
    with open(path) as handle:
        data = _mapping(str(path), yaml.safe_load(handle) or {}, SET_KEYS, ("name", "files", "ligands"))
    name = str(data["name"])
    if not _NAME.fullmatch(name):
        raise ValueError(f"name {name!r}: letters, digits, '.', '_' and '-' only (it names the report files)")
    source = dict(_mapping("source", data.get("source") or {}, SOURCE_KEYS))
    raw = _raw_base(source)
    files = {}
    for key, entry in _mapping("files", data["files"], data["files"], ("protein", "ligands")).items():
        if not _KEY.fullmatch(str(key)) or key == "superpose_on":
            raise ValueError(f"files.{key}: not a usable name for a file")
        files[key] = _source(key, _mapping(f"files.{key}", entry, FILE_KEYS), raw, directory)
    superpose_on = None
    if data.get("superpose_on") is not None:
        entry = _mapping("superpose_on", data["superpose_on"], SUPERPOSE_KEYS)
        max_rmsd = entry.get("max_rmsd", 0.3)
        if isinstance(max_rmsd, bool) or not isinstance(max_rmsd, (int, float)) or max_rmsd <= 0:
            raise ValueError("superpose_on.max_rmsd must be a positive length in nm")
        chains = [entry.get(key) for key in ("reference_chain", "protein_chain")]
        if any(chain is not None and len(str(chain)) != 1 for chain in chains):
            raise ValueError("superpose_on: a chain is one character")
        superpose_on = Superposition(_source("superpose_on", entry, raw, directory),
                                     *(None if chain is None else str(chain) for chain in chains),
                                     max_rmsd=float(max_rmsd))
    ligands, pilot = _ligands(data["ligands"], data.get("pilot"))
    config = _mapping("config", data.get("config") or {}, data.get("config") or {})
    if "input" in config:
        raise ValueError("config.input is written by 'fmeta benchmark prepare': leave it out of set.yaml")
    benchmark = BenchmarkSet(directory, name, str(data.get("title") or name), source, files, superpose_on,
                             ligands, pilot, config,
                             read_reference(directory / "reference.csv", [ligand.name for ligand in ligands]))
    set_config(benchmark)                    # unknown sections and options are errors here, not at run time
    return benchmark


def set_config(benchmark: BenchmarkSet, length_ns: Optional[float] = None,
               replicas: Optional[int] = None) -> Config:
    """The configuration of the set, without inputs. Relative paths in it are relative to the
    set directory. ``length_ns`` and ``replicas`` stand in for ``run.length_ns`` and
    ``run.replicas``."""
    data = copy.deepcopy(benchmark.config)
    plan = dict(length_ns=length_ns, replicas=replicas)
    if any(value is not None for value in plan.values()):
        data["run"] = dict(data.get("run") or {}, **{key: value for key, value in plan.items() if value is not None})
    return Config.from_dict(data, base_dir=benchmark.directory)


# ---- the inputs: fetched, verified, read as text ----------------------------

def download(url: str) -> bytes:
    """The content at ``url``. What is downloaded is data: it is hashed, read as text and
    written to files, never imported or executed."""
    request = urllib.request.Request(url, headers={"User-Agent": "fmeta-benchmark"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except OSError as error:
        raise RuntimeError(f"could not download {url}: {error}") from error


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _replace(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that the file is never seen half written."""
    part = path.with_name(path.name + ".part")
    part.write_text(text)
    os.replace(part, path)


def fetch_inputs(benchmark: BenchmarkSet, workdir, download=download) -> dict:
    """The files of the set in ``workdir/inputs``, each with the sha256 the set names: key ->
    path. A file that is there with that hash is not fetched again; content with another
    hash is refused and not written."""
    directory = Path(workdir) / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    sources = list(benchmark.files.values())
    if benchmark.superpose_on is not None:
        sources.append(benchmark.superpose_on.source)
    paths = {}
    for source in sources:
        target = directory / f"{source.key}{source.suffix}"
        if not (target.exists() and _sha256(target) == source.sha256):
            content = source.file.read_bytes() if source.file is not None else download(source.url)
            found = hashlib.sha256(content).hexdigest()
            if found != source.sha256:
                raise ValueError(f"{source.url or source.file} has sha256 {found}, the set names "
                                 f"{source.sha256}: it is not the file the set was written for")
            part = target.with_name(target.name + ".part")
            part.write_bytes(content)
            os.replace(part, target)
        paths[source.key] = target
    return paths


_SDF_CHARGES = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}       # the charge field of a V2000 atom line


def sdf_records(path) -> dict:
    """The records of an SDF file: the name in the first line -> the records of that name,
    each a list of lines without the closing ``$$$$``."""
    records: dict = {}
    lines: list = []
    for line in Path(path).read_text().splitlines() + ["$$$$"]:
        if line.strip() != "$$$$":
            lines.append(line)
            continue
        if any(text.strip() for text in lines):
            records.setdefault(lines[0].strip(), []).append(lines)
        lines = []
    return records


def _atom_lines(record: list, name: str) -> range:
    """Where the atom block of a V2000 record is."""
    try:
        if "V2000" not in record[3]:
            raise ValueError
        atoms = range(4, 4 + int(record[3][0:3]))
        if not atoms or len(record) < atoms.stop:
            raise ValueError
    except (ValueError, IndexError):
        raise ValueError(f"ligand {name!r}: its record is not a V2000 molfile") from None
    return atoms


def sdf_charge(record: list, name: str) -> int:
    """The formal charge of a V2000 record: its ``M  CHG`` lines or, without any, the charge
    fields of the atom lines."""
    atoms = _atom_lines(record, name)
    try:
        properties = [line.split() for line in record[atoms.stop:] if line.startswith("M  CHG")]
        if properties:
            return sum(int(value) for tokens in properties for value in tokens[4:3 + 2 * int(tokens[2]):2])
        return sum(_SDF_CHARGES.get(int(record[index][36:39].strip() or 0), 0) for index in atoms)
    except (ValueError, IndexError):
        raise ValueError(f"ligand {name!r}: the charges of its record cannot be read") from None


def sdf_coordinates(record: list, name: str) -> np.ndarray:
    """The coordinates of a V2000 record in Å."""
    try:
        return np.array([[float(record[index][start:start + 10]) for start in (0, 10, 20)]
                         for index in _atom_lines(record, name)])
    except ValueError:
        raise ValueError(f"ligand {name!r}: the coordinates of its record cannot be read") from None


def _moved_record(record: list, name: str, rotation: np.ndarray, translation: np.ndarray) -> list:
    """The record with its atoms at ``rotation @ x + translation``."""
    atoms = _atom_lines(record, name)
    moved = sdf_coordinates(record, name) @ rotation.T + translation
    lines = list(record)
    for index, (x, y, z) in zip(atoms, moved):
        lines[index] = f"{x:10.4f}{y:10.4f}{z:10.4f}{record[index][30:]}"
    return lines


def pdb_ca(path, chain: Optional[str] = None, option: str = "chain") -> dict:
    """The CA atoms of the ``ATOM`` records of the first model of a PDB file: (residue number,
    insertion code) -> coordinates in Å. Of alternate locations the first one counts.
    ``option`` is the key of the set definition that selects the chain."""
    found: dict = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or chain not in (None, line[21]):
            continue
        key = (int(line[22:26]), line[26].strip())
        if key not in found:
            found[key] = (line[21], [float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif found[key][0] != line[21]:
            raise ValueError(f"{path}: residue {key[0]}{key[1]} is in chains {found[key][0]!r} and "
                             f"{line[21]!r}; name the chain with superpose_on.{option}")
    return {key: xyz for key, (_, xyz) in found.items()}


def superposition(mobile, target) -> tuple:
    """The rotation and the translation that put the points ``mobile`` on ``target`` with the
    least RMSD (Kabsch), and that RMSD: a point x goes to ``rotation @ x + translation``."""
    mobile, target = np.asarray(mobile, dtype=float), np.asarray(target, dtype=float)
    centre_mobile, centre_target = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - centre_mobile).T @ (target - centre_target))
    proper = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0             # a rotation, never a mirror image
    rotation = vt.T @ np.diag([1.0, 1.0, proper]) @ u.T
    translation = centre_target - rotation @ centre_mobile
    deviation = mobile @ rotation.T + translation - target
    return rotation, translation, float(np.sqrt((deviation ** 2).sum(axis=1).mean()))


def _moved_pdb(text: str, rotation: np.ndarray, translation: np.ndarray) -> str:
    """A PDB file with its atoms at ``rotation @ x + translation``. Records that describe the
    old frame (the unit cell, its transformations, anisotropic temperature factors) are left out."""
    lines = []
    for line in text.splitlines():
        if line.startswith(("ANISOU", "CRYST1", "SCALE", "ORIGX", "MTRIX")):
            continue
        if line.startswith(("ATOM", "HETATM")):
            x, y, z = rotation @ [float(line[30:38]), float(line[38:46]), float(line[46:54])] + translation
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def superpose_protein(benchmark: BenchmarkSet, paths: dict) -> tuple:
    """The protein file the work directories use, and the motion that the ligands have to
    follow. With ``superpose_on`` the protein is put on that structure, CA atoms of the same
    residue number and insertion code on each other, and written beside the fetched file
    with a record of the fit; without it the fetched file is used as it is."""
    on = benchmark.superpose_on
    if on is None:
        return paths["protein"], np.eye(3), np.zeros(3)
    mobile = pdb_ca(paths["protein"], on.protein_chain, "protein_chain")
    target = pdb_ca(paths["superpose_on"], on.reference_chain, "reference_chain")
    common = sorted(set(mobile) & set(target))
    if len(common) < 3:
        raise ValueError(f"superpose_on: {len(common)} CA atom(s) with the same residue number in the two "
                         "structures; three are needed")
    rotation, translation, rmsd = superposition([mobile[key] for key in common], [target[key] for key in common])
    rmsd /= 10.0                                                        # nm
    if rmsd > on.max_rmsd:
        raise ValueError(f"superpose_on: {len(common)} CA atoms fit with an RMSD of {rmsd:.3f} nm, above "
                         f"max_rmsd = {on.max_rmsd} nm: the two structures are not the same protein state, "
                         "or their residue numbers differ")
    directory = paths["protein"].parent
    path = directory / "protein_superposed.pdb"
    _replace(path, _moved_pdb(paths["protein"].read_text(), rotation, translation))
    record = dict(reference_sha256=on.source.sha256, matched_ca=len(common), rmsd_nm=round(rmsd, 4),
                  rotation=rotation.tolist(), translation_angstrom=translation.tolist())
    _replace(directory / "superposition.json", json.dumps(record, indent=1) + "\n")
    return path, rotation, translation


# ---- the work directory: one ordinary fmeta work directory per ligand -------

MARKER = "benchmark.json"                    # written last: a ligand directory with it is complete


class Prepared(NamedTuple):
    ligand: Ligand
    directory: Path
    config: Config


def configuration_hash(benchmark: BenchmarkSet, config: Config) -> str:
    """Everything that decides the free energy of a ligand, as one hash: the configuration,
    files it names by their content, the files of the set and the superposition by their
    sha256. Left out: the inputs (they are these files), the devices, and the sampling plan
    (``run.length_ns``, ``run.replicas``), which may grow while the results stay comparable."""
    data = config.to_dict()
    del data["input"]
    for option in ("mode", "gpus", "length_ns", "replicas"):
        del data["run"][option]
    data["system"]["ligand_xml"] = [_sha256(path) for path in data["system"]["ligand_xml"]]
    if data["membrane"] is not None and data["membrane"].get("patch"):
        data["membrane"]["patch"] = _sha256(data["membrane"]["patch"])
    on = benchmark.superpose_on
    payload = dict(set=benchmark.name, config=data,
                   files={key: benchmark.files[key].sha256 for key in ("protein", "ligands")},
                   superpose_on=None if on is None else [on.source.sha256, on.reference_chain,
                                                         on.protein_chain, on.max_rmsd])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _plan(config: Config) -> tuple:
    return float(config.run.length_ns), int(config.run.replicas)


def _stored(benchmark: BenchmarkSet, workdir: Path) -> list:
    """The ligands of the set that have a work directory, in the order of the set. A work
    directory of another configuration or set definition is an error: one work directory,
    one configuration."""
    expected = configuration_hash(benchmark, set_config(benchmark))
    stored = []
    for ligand in benchmark.ligands:
        directory = workdir / ligand.directory
        if not (directory / MARKER).exists():
            continue
        marker = json.loads((directory / MARKER).read_text())
        if marker.get("configuration_sha256") != expected or marker.get("ligand") != ligand.name:
            raise ValueError(f"{directory} was prepared with another configuration or set definition. Results "
                             "of two configurations do not go into one report: use a new work directory")
        config = Config.load(directory / "config.yaml")
        if configuration_hash(benchmark, config) != expected:
            raise ValueError(f"{directory}/config.yaml was changed after 'fmeta benchmark prepare'. The ligands "
                             "of a set run with one configuration: change set.yaml and use a new work directory")
        stored.append(Prepared(ligand, directory, config))
    return stored


def prepared_ligands(benchmark: BenchmarkSet, workdir) -> list:
    """The prepared ligands of ``workdir`` as ``Prepared(ligand, directory, config)``, checked:
    one configuration (that of the set as it is now) and one sampling plan for all."""
    workdir = Path(workdir).resolve()
    stored = _stored(benchmark, workdir) if workdir.is_dir() else []
    if not stored:
        raise ValueError(f"no ligand of the set {benchmark.name} is prepared in {workdir}: "
                         "run 'fmeta benchmark prepare' first")
    if len({_plan(item.config) for item in stored}) > 1:
        raise ValueError(f"the ligands in {workdir} have different lengths or numbers of replicas: set one plan "
                         "with 'fmeta benchmark prepare ... --length-ns X --replicas N'")
    return stored


def _selection(benchmark: BenchmarkSet, ligands) -> list:
    names = [ligand.name for ligand in benchmark.ligands]
    if ligands == "pilot":
        chosen = list(benchmark.pilot)
    elif ligands == "all":
        chosen = names
    else:
        chosen = [name.strip() for name in ligands.split(",")] if isinstance(ligands, str) else list(ligands)
    unknown = [name for name in chosen if name not in names]
    if unknown or not chosen:
        raise ValueError(f"not in the set {benchmark.name}: {', '.join(unknown) or 'no ligand given'} "
                         f"(its ligands: {', '.join(names)}; or 'pilot', 'all')")
    return [ligand for ligand in benchmark.ligands if ligand.name in chosen]


def _write_config(benchmark: BenchmarkSet, directory: Path, protein: Path, plan: tuple) -> None:
    config = set_config(benchmark, *plan)
    config.input.protein = os.path.relpath(protein, directory)       # relative: the work directory can move
    config.input.ligand = "ligand.sdf"
    part = directory / "config.yaml.part"
    config.save(part)
    os.replace(part, directory / "config.yaml")


def prepare_set(benchmark: BenchmarkSet, workdir, ligands="pilot", length_ns: Optional[float] = None,
                replicas: Optional[int] = None, download=download) -> list:
    """Fetch and verify the files of the set and write one fmeta work directory per ligand
    (``config.yaml``, ``ligand.sdf``) under ``workdir``. Returns one line per ligand.

    ``ligands`` is ``pilot``, ``all`` or names; ligands that are already there stay as they
    are, so a later call adds ligands. ``length_ns`` and ``replicas`` set the sampling plan
    of every ligand in the work directory, also of those prepared before: without them a new
    ligand gets the plan of the ones that are there, and the first ones that of the set.
    Nothing is simulated here: building the systems is the first stage of ``run_set``."""
    workdir = Path(workdir).resolve()
    selected = _selection(benchmark, ligands)
    if (length_ns is not None and not length_ns > 0) or (replicas is not None and replicas < 1):
        raise ValueError("--length-ns must be positive and --replicas at least 1")
    with directory_lock(workdir):
        stored = {item.ligand.name: item for item in _stored(benchmark, workdir)}
        plans = {_plan(item.config) for item in stored.values()}
        if len(plans) > 1 and (length_ns is None or replicas is None):
            raise ValueError(f"the ligands in {workdir} have different lengths or numbers of replicas: "
                             "give --length-ns and --replicas")
        base = set_config(benchmark)
        current = plans.pop() if len(plans) == 1 else _plan(base)
        plan = (current[0] if length_ns is None else float(length_ns),
                current[1] if replicas is None else int(replicas))
        paths = fetch_inputs(benchmark, workdir, download)
        records = sdf_records(paths["ligands"])
        chosen = {}
        for ligand in selected:
            found = records.get(ligand.name, [])
            if len(found) != 1:
                raise ValueError(f"{len(found)} records named {ligand.name!r} in the ligand file of the set "
                                 f"(it has: {', '.join(sorted(records))})")
            charge = sdf_charge(found[0], ligand.name)
            if charge != ligand.charge:
                raise ValueError(f"ligand {ligand.name!r}: formal charge {charge:+d} in the ligand file, "
                                 f"{ligand.charge:+d} in set.yaml")
            sdf_coordinates(found[0], ligand.name)
            chosen[ligand.name] = found[0]
        protein, rotation, translation = superpose_protein(benchmark, paths)
        expected = configuration_hash(benchmark, base)
        lines = []
        for ligand in benchmark.ligands:
            directory = workdir / ligand.directory
            if ligand.name in stored:
                if _plan(stored[ligand.name].config) != plan:
                    beyond = sorted(path.name for path in directory.glob("rep*") if path.name[3:].isdigit()
                                    and int(path.name[3:]) >= plan[1] and any(path.rglob("HILLS")))
                    if beyond:
                        raise ValueError(f"{directory} has {', '.join(beyond)}: the number of replicas cannot "
                                         "go below those that have run")
                    _write_config(benchmark, directory, protein, plan)
                    lines.append(f"{ligand.name}: {directory} (plan changed)")
                elif ligand.name in chosen:
                    lines.append(f"{ligand.name}: {directory} (was there)")
                continue
            if ligand.name not in chosen:
                continue
            directory.mkdir(exist_ok=True)
            record = chosen[ligand.name]
            if benchmark.superpose_on is not None:
                record = _moved_record(record, ligand.name, rotation, translation)
            _replace(directory / "ligand.sdf", "\n".join(record) + "\n$$$$\n")
            _write_config(benchmark, directory, protein, plan)
            marker = dict(set=benchmark.name, ligand=ligand.name, charge=ligand.charge,
                          configuration_sha256=expected)
            _replace(directory / MARKER, json.dumps(marker, indent=1) + "\n")
            lines.append(f"{ligand.name}: {directory}")
        lines.append(f"plan: {plan[1]} replica(s) of {plan[0]:g} ns per ligand")
    return lines


# ---- the stages of every ligand: jobs of one call of the scheduler ----------

STAGES = ("prepare", "equilibrate", "run")


def _read_json(path: Path):
    """The content of a JSON file; None when it is not there or not complete."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _built(directory: Path) -> bool:
    return isinstance(_read_json(directory / "prepared" / "meta.json"), dict)       # the last file of the stage


def _equilibration(directory: Path) -> Optional[bool]:
    """None when the equilibration did not come to its end; else whether the structure it
    ended with passed the checks of the funnel. Only then is the stage finished."""
    summary = _read_json(directory / "equil" / "equil.json")
    try:
        return bool(summary["funnel_checks"]["ok"])
    except (TypeError, KeyError):
        return None


def _replica_rows(item: Prepared) -> dict:
    """Replica number -> its rows of :func:`fmeta.workflow.status_table` (one per walker),
    measured against the plan as it is now: a replica that reached a shorter target is
    ``stopped``, not ``done``."""
    hill_ps = item.config.metad.pace * item.config.run.timestep_fs / 1000.0
    target = round(item.config.run.length_ns * 1000.0 / hill_ps)
    rows: dict = {}
    for row in status_table(item.directory):
        if row["state"] == "done" and row["hills"] < target:
            row["state"] = "stopped"
        row["target_ns"] = target * hill_ps / 1000.0
        rows.setdefault(int(row["name"].split("/")[0][3:]), []).append(row)
    return rows


def benchmark_jobs(prepared: Sequence[Prepared], until: str = "run") -> list:
    """What is left to do for the prepared ligands, as the jobs of one call of
    :func:`fmeta.scheduler.run_jobs`.

    Their order is the order in which a free device is given out: the systems of all ligands
    are built, then all are equilibrated, then replica 0 of every ligand runs before any
    replica 1, so that a report over the whole set exists early and gets better. A stage
    waits for the stage before it of its own ligand, and a stage that fails holds back only
    that ligand. The work directories are the truth: a stage that is finished there leaves
    no job, so the same call again continues. ``until`` ends the list with that stage.

    A key, which is also the name of the task, is ``<ligand directory>/<stage>``; the output
    of the task goes to the file of that name with ``.log`` under the work directory."""
    if until not in STAGES:
        raise ValueError(f"until must be one of {', '.join(STAGES)}")
    builds, equilibrations, replicas = [], [], {}
    for item in prepared:
        directory, prefix = item.directory, item.directory.name
        built, equilibrated, rows = _built(directory), _equilibration(directory), _replica_rows(item)
        ran = any(row["state"] != "not started" for group in rows.values() for row in group)
        if (not built and (equilibrated is not None or ran)) or (not equilibrated and ran):
            raise ValueError(f"{directory} has the output of a stage without the stage before it. Remove its "
                             "prepared/, equil/ and rep*/ directories to start this ligand again")
        cli = [sys.executable, "-m", "fmeta.cli"]
        after: list = []
        if not built:
            key = f"{prefix}/prepare"
            builds.append(Job(key, [Task(key, cli + ["benchmark", "build", str(directory)],
                                         directory / "prepare.log")]))
            after = [key]
        if until == "prepare":
            continue
        if not equilibrated:
            key = f"{prefix}/equilibrate"
            equilibrations.append(Job(key, [Task(key, cli + ["equilibrate", str(directory)],
                                                 directory / "equilibrate.log")], after=after))
            after = [key]
        if until == "equilibrate":
            continue
        for job in replica_jobs(item.config, directory):
            number = int(job.key[3:])
            if all(row["state"] == "done" for row in rows[number]):
                continue
            tasks = [Task(f"{prefix}/{task.name}", task.command, task.log) for task in job.tasks]
            replicas.setdefault(number, []).append(Job(f"{prefix}/{job.key}", tasks, devices=job.devices,
                                                       after=after, before_start=job.before_start))
    return builds + equilibrations + [job for number in sorted(replicas) for job in replicas[number]]


def run_set(benchmark: BenchmarkSet, workdir, gpus=None, mode: Optional[str] = None, until: str = "run",
            launcher=None) -> dict:
    """Run what is left of the prepared ligands of ``workdir``: every stage of every ligand
    in one call of the scheduler, on the devices of the run (``gpus`` and ``mode`` stand in
    for ``run.gpus`` and ``run.mode`` of the configuration of the set).

    Every decision is appended to ``workdir/schedule.jsonl``. A stage that failed or was
    stopped is a ``RuntimeError`` after everything that could run has run; the same call
    again continues. Returns the status of the jobs. ``launcher`` replaces the start of a
    child process (for tests)."""
    workdir = Path(workdir).resolve()
    record: list = []
    with contextlib.ExitStack() as held:
        held.enter_context(directory_lock(workdir))
        prepared = prepared_ligands(benchmark, workdir)
        for item in prepared:                                # a 'fmeta run' of one ligand beside this is refused
            held.enter_context(directory_lock(item.directory))
        jobs = benchmark_jobs(prepared, until)
        if not jobs:
            return {}
        run = prepared[0].config.run
        devices = resolve_devices(mode or run.mode, gpus if gpus is not None else run.gpus, run.platform)
        status = run_jobs(jobs, devices, launcher, log=workdir / "schedule.jsonl", events=record.append)
    problem = unfinished_jobs(status, record, workdir)
    if problem:
        raise RuntimeError(problem)
    return status


# ---- status -----------------------------------------------------------------

def _last_call(workdir: Path) -> dict:
    """Job -> how it ended in the last call of ``run_set`` (``running`` when it did not)."""
    path = workdir / "schedule.jsonl"
    states: dict = {}
    for event in (read_schedule(path)[-1] if path.exists() else []):
        if event["event"] == "start":
            states[event["job"]] = "running"
        elif event["event"] == "end":
            states[event["job"]] = event["status"]
        elif event["event"] == "skip":
            states[event["job"]] = "skipped"
    if not is_locked(workdir):                               # that call is over, however it ended
        states = {job: "stopped" if state == "running" else state for job, state in states.items()}
    return states


def status_rows(benchmark: BenchmarkSet, workdir) -> list:
    """Where the set stands: one row per stage of each prepared ligand (``prepare``,
    ``equilibrate``, then each replica, or each walker of it).

    ``state`` is ``done`` when the work directory says so; else what the last call of
    ``run_set`` made of the stage (``running``, ``failed``, ``stopped``, ``skipped``), or
    ``not started``. A replica has its time, the target of the plan, speed and device."""
    workdir = Path(workdir).resolve()
    last = _last_call(workdir)
    rows = []
    for item in prepared_ligands(benchmark, workdir):
        name, prefix = item.ligand.name, item.directory.name
        equilibrated = _equilibration(item.directory)
        states = dict(prepare="done" if _built(item.directory) else last.get(f"{prefix}/prepare", "not started"),
                      equilibrate="done" if equilibrated else last.get(
                          f"{prefix}/equilibrate", "not started" if equilibrated is None else "failed"))
        rows += [dict(ligand=name, stage=stage, state=state, time_ns=None, target_ns=None, speed_ns_day=None,
                      device=None) for stage, state in states.items()]
        for number, group in _replica_rows(item).items():
            for row in group:
                state = row["state"]
                if state in ("stopped", "not started") and last.get(f"{prefix}/rep{number}") in ("failed", "skipped"):
                    state = last[f"{prefix}/rep{number}"]
                rows.append(dict(ligand=name, stage=row["name"], state=state, time_ns=row["time_ns"],
                                 target_ns=row["target_ns"], speed_ns_day=row["speed_ns_day"],
                                 device=row["device"]))
    return rows


def format_status(rows: Sequence[dict]) -> str:
    width = max([len("ligand")] + [len(row["ligand"]) for row in rows])
    lines = [f"{'ligand':{width}s}  {'stage':12s} {'state':12s} {'time / target (ns)':>22s} {'ns/day':>9s}  device"]
    for row in rows:
        progress = "" if row["time_ns"] is None else f"{row['time_ns']:.3f} / {row['target_ns']:.3f}"
        speed = "" if row["speed_ns_day"] is None else f"{row['speed_ns_day']:.1f}"
        lines.append(f"{row['ligand']:{width}s}  {row['stage']:12s} {row['state']:12s} {progress:>22s} "
                     f"{speed:>9s}  {row['device'] or ''}".rstrip())
    runs = [row for row in rows if row["time_ns"] is not None]
    lines.append(f"{len({row['ligand'] for row in rows})} ligand(s): "
                 f"{sum(row['state'] == 'done' for row in runs)} of {len(runs)} run(s) done, "
                 f"{sum(row['time_ns'] for row in runs):.3f} of {sum(row['target_ns'] for row in runs):.3f} ns")
    return "\n".join(lines)


# ---- statistics -------------------------------------------------------------

def _measures(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``MEASURES`` of arrays of shape ``(..., n)``, as an array of shape ``(..., 7)``.

    A correlation of a series without spread is NaN. Kendall's tau is tau-b (ties are taken
    out of the pairs that count)."""
    error = predicted - reference
    centred = error - error.mean(-1, keepdims=True)
    dp = predicted - predicted.mean(-1, keepdims=True)
    dr = reference - reference.mean(-1, keepdims=True)
    sp = np.sign(predicted[..., :, None] - predicted[..., None, :]).astype(np.int8)
    sr = np.sign(reference[..., :, None] - reference[..., None, :]).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = (dp * dr).sum(-1) / np.sqrt((dp ** 2).sum(-1) * (dr ** 2).sum(-1))
        pairs = np.count_nonzero(sp, axis=(-1, -2)) * np.count_nonzero(sr, axis=(-1, -2))
        tau = (sp * sr).sum((-1, -2), dtype=np.int64) / np.sqrt(pairs)
    flat = (np.ptp(predicted, axis=-1) == 0) | (np.ptp(reference, axis=-1) == 0)
    pearson = np.where(flat, np.nan, pearson)                 # 0 / 0 up to rounding otherwise
    return np.stack([np.abs(error).mean(-1), np.sqrt((error ** 2).mean(-1)), error.mean(-1),
                     np.abs(centred).mean(-1), np.sqrt((centred ** 2).mean(-1)), pearson, tau], axis=-1)


def agreement(predicted, reference, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """How ``predicted`` agrees with ``reference``: one value per ligand, both in one unit.

    Returns ``n`` and, for each of ``MEASURES``, ``dict(value, ci95)``. The error of a ligand
    is predicted - reference. ``*_offset_removed`` are the measures after the mean error of
    these same ligands is taken out: what is left when only differences between the ligands
    count, as for relative free energies that were shifted to the experimental mean.

    ``ci95`` is the 2.5 - 97.5 percentile interval of a bootstrap over the ligands (drawn with
    replacement, ``resamples`` times, from ``seed``): it tells how much the measure depends
    on the choice of ligands, not how uncertain each value is. A measure that does not exist
    is None: every interval and both correlations below three ligands, the correlations of a
    series without spread, the offset-removed measures of a single ligand.
    """
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.ndim != 1 or predicted.shape != reference.shape:
        raise ValueError("agreement needs two series of the same length")
    n = len(predicted)
    result: dict = dict(n=n)
    if n == 0:
        result.update({name: dict(value=None, ci95=None) for name in MEASURES})
        return result
    values = _measures(predicted, reference)
    low = high = np.full(len(MEASURES), np.nan)
    if n >= 3 and resamples > 0:
        rng = np.random.default_rng(seed)
        samples = []
        for start in range(0, resamples, 1000):
            index = rng.integers(0, n, size=(min(1000, resamples - start), n))
            samples.append(_measures(predicted[index], reference[index]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a measure that no resample has
            low, high = np.nanpercentile(np.concatenate(samples), [2.5, 97.5], axis=0)
    needs = dict(pearson_r=3, kendall_tau=3, mue_offset_removed=2, rmse_offset_removed=2)
    for i, name in enumerate(MEASURES):
        value = float(values[i]) if n >= needs.get(name, 1) and np.isfinite(values[i]) else None
        interval = [float(low[i]), float(high[i])] if value is not None and np.isfinite(low[i]) else None
        result[name] = dict(value=value, ci95=interval)
    return result
