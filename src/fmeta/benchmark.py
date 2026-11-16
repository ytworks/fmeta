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

import copy
import csv
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .analysis import KCAL
from .config import Config
from .funnel.geometry import KB

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


def set_config(benchmark: BenchmarkSet, protein=None, ligand=None, length_ns: Optional[float] = None,
               replicas: Optional[int] = None) -> Config:
    """The configuration of the set, for one ligand when ``protein`` and ``ligand`` are given.
    Relative paths in it are relative to the set directory. ``length_ns`` and ``replicas``
    stand in for ``run.length_ns`` and ``run.replicas``."""
    data = copy.deepcopy(benchmark.config)
    if protein is not None:
        data["input"] = dict(protein=str(protein), ligand=str(ligand))
    plan = dict(length_ns=length_ns, replicas=replicas)
    if any(value is not None for value in plan.values()):
        data["run"] = dict(data.get("run") or {}, **{key: value for key, value in plan.items() if value is not None})
    return Config.from_dict(data, base_dir=benchmark.directory)


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
