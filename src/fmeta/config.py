"""Configuration: one YAML file, mapped onto the settings dataclasses.

The official defaults are the defaults of the dataclasses. The presets (``presets/*.yaml``,
written by ``fmeta init --preset NAME``) show every key with its default, its unit and what it
does; a configuration may name one (``preset: membrane``) and hold only what it changes. The
reasons for the values are in ``docs/configuration.md`` (FM-D16).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .engine.backend import BACKENDS
from .engine.equilibration import EquilibrationSettings
from .engine.grid import BiasSettings
from .engine.platform import check_devices
from .funnel.autodetect import AutoFunnelSettings
from .system.builder import MembraneSettings, SystemSettings
from .system.ligand import DEFAULT_FORCEFIELD

PRESETS = ("soluble", "membrane")


def preset_text(name) -> str:
    """The preset ``name`` as it is packaged: a full configuration with units and explanations."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from: {', '.join(PRESETS)}")
    return (Path(__file__).parent / "presets" / f"{name}.yaml").read_text()


def on_preset(name, data: dict) -> dict:
    """The configuration ``data`` on top of the preset ``name``: the keys of ``data`` win."""
    preset = yaml.safe_load(preset_text(name))
    merged = dict(data)
    for section, values in preset.items():
        if section not in ("input", "membrane"):      # the file names of a preset are placeholders
            merged[section] = {**values, **(data.get(section) or {})}
    system = data.get("system") or {}
    if "ligand_method" in system and "ligand_forcefield" not in system:
        del merged["system"]["ligand_forcefield"]     # the pinned force field of that method
    # membrane: absent: as the preset; true: the bilayer of the preset, or the default one;
    # a mapping: on top of the bilayer of the preset; false or null: no bilayer
    membrane = data["membrane"] if "membrane" in data else preset.get("membrane")
    if membrane is True:
        membrane = {}
    if isinstance(membrane, dict):
        membrane = {**(preset.get("membrane") or {}), **membrane}
    merged["membrane"] = membrane
    return merged


@dataclass
class InputConfig:
    protein: Optional[str] = None            # PDB / mmCIF of the protein (oriented, if membrane)
    ligand: Optional[str] = None             # posed ligand: SDF / MOL / MOL2 with hydrogens
    # ... or a complex file from which the ligand residue is cut out:
    complex: Optional[str] = None            # PDB / mmCIF holding protein and ligand
    ligand_resname: Optional[str] = None     # residue name of the ligand in the complex
    ligand_smiles: Optional[str] = None      # its SMILES (protonation state to simulate): required
    ligand_chain: Optional[str] = None       # to pick one of several copies
    ligand_resid: Optional[str] = None


@dataclass
class RunConfig:
    length_ns: float = 40.0                  # per replica
    replicas: int = 5                        # independent runs (separate biases), for error bars
    walkers: int = 1                         # walkers sharing one bias within each replica
    walker_sync_hills: int = 1               # a walker reads the others' hills every N of its own
    timestep_fs: float = 4.0                 # needs hydrogen mass repartitioning above 2 fs
    friction: float = 1.0                    # 1/ps
    pace_ps: float = 1.0                     # time between hills
    platform: str = "auto"                   # auto, CUDA, OpenCL, HIP, CPU, Reference
    precision: str = "mixed"
    backend: str = "auto"                    # auto, cupotential, openmm (FM-D18)
    mode: str = "auto"                       # auto, parallel (one replica per GPU), sequential (one GPU)
    gpus: Optional[list | str] = None        # device numbers, or "all": every GPU that is visible
    per_gpu: int = 1                         # replicas that share one GPU at a time (above 1: with NVIDIA MPS)
    checkpoint_ps: float = 250.0
    trajectory_ps: float = 100.0             # 0 disables the trajectory
    trajectory_solute_only: bool = True
    seed: int = 0                            # 0: random; else replica i, walker w: seed + 1000 i + w


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    system: SystemSettings = field(default_factory=SystemSettings)
    membrane: Optional[MembraneSettings] = None
    funnel: AutoFunnelSettings = field(default_factory=AutoFunnelSettings)
    funnel_overrides: dict = field(default_factory=dict)   # e.g. {upper: 4.0, k_wall: 10000}
    metad: BiasSettings = field(default_factory=BiasSettings)
    equilibration: EquilibrationSettings = field(default_factory=EquilibrationSettings)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self):
        self.metad.temperature = self.system.temperature     # one temperature for everything
        dt = self.run.timestep_fs / 1000.0
        self.metad.pace = max(1, round(self.run.pace_ps / dt))
        if self.run.timestep_fs > 2.0 and not self.system.hydrogen_mass:
            raise ValueError("a time step above 2 fs needs system.hydrogen_mass (e.g. 1.5)")
        check_devices(self.run.mode, self.run.gpus, self.run.per_gpu)
        if self.run.backend not in BACKENDS:
            raise ValueError(f"run.backend must be one of {BACKENDS}, not {self.run.backend!r}")
        # the ligand force field is named, so that a saved configuration says which one was used
        if self.system.ligand_forcefield is None:
            self.system.ligand_forcefield = DEFAULT_FORCEFIELD.get(self.system.ligand_method)

    @classmethod
    def from_dict(cls, data: dict, base_dir=None) -> "Config":
        data = dict(data or {})
        if "preset" in data:
            data = on_preset(data.pop("preset"), data)
        sections = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"unknown configuration section(s): {', '.join(sorted(unknown))}")

        def build(kind, values):
            values = dict(values or {})
            names = {f.name for f in dataclasses.fields(kind)}
            extra = set(values) - names
            if extra:
                raise ValueError(f"unknown option(s) for {kind.__name__}: {', '.join(sorted(extra))}")
            return kind(**values)

        membrane = data.get("membrane")
        if membrane is True:
            membrane = {}
        metad = dict(data.get("metad") or {})
        metad.pop("pace", None)              # derived from run.pace_ps
        config = cls(
            input=build(InputConfig, data.get("input")),
            system=build(SystemSettings, data.get("system")),
            membrane=None if membrane in (None, False) else build(MembraneSettings, membrane),
            funnel=build(AutoFunnelSettings, data.get("funnel")),
            funnel_overrides=dict(data.get("funnel_overrides") or {}),
            metad=build(BiasSettings, metad),
            equilibration=build(EquilibrationSettings, data.get("equilibration")),
            run=build(RunConfig, data.get("run")),
        )
        if base_dir is not None:
            config.resolve_paths(Path(base_dir))
        return config

    def resolve_paths(self, base: Path) -> None:
        """Make input paths absolute, relative to the configuration file."""
        def absolute(value):
            return None if value is None else str((base / value).resolve())
        self.input.protein = absolute(self.input.protein)
        self.input.ligand = absolute(self.input.ligand)
        self.input.complex = absolute(self.input.complex)
        self.system.ligand_xml = [absolute(p) for p in self.system.ligand_xml]
        if self.membrane is not None and self.membrane.patch:
            self.membrane.patch = absolute(self.membrane.patch)

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path)
        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle), base_dir=path.parent)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        if self.system.ligand_method not in DEFAULT_FORCEFIELD:     # xml: no generator ran
            data["system"]["ligand_forcefield"] = None
        data["system"]["forcefields"] = list(data["system"]["forcefields"])
        data["system"]["ligand_xml"] = list(data["system"]["ligand_xml"])
        data["system"]["keep_hetero"] = list(data["system"]["keep_hetero"])
        data["equilibration"]["release_factors"] = list(data["equilibration"]["release_factors"])
        if data["funnel"]["exit_direction"] is not None:
            data["funnel"]["exit_direction"] = [float(x) for x in data["funnel"]["exit_direction"]]
        return data

    def save(self, path) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=None)
