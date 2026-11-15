"""Configuration: one YAML file, mapped onto the settings dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .engine.equilibration import EquilibrationSettings
from .engine.grid import BiasSettings
from .funnel.autodetect import AutoFunnelSettings
from .system.builder import MembraneSettings, SystemSettings


@dataclass
class InputConfig:
    protein: Optional[str] = None            # PDB / mmCIF of the protein (oriented, if membrane)
    ligand: Optional[str] = None             # posed ligand: SDF / MOL / MOL2 with hydrogens
    # pre-parameterised alternatives (give exactly one instead of protein + ligand)
    amber: Optional[dict] = None             # {prmtop:, coordinates:}
    gromacs: Optional[dict] = None           # {top:, gro:, include_dir:}
    charmm: Optional[dict] = None            # {psf:, coordinates:, parameters: [...], box: [x, y, z]}
    ligand_selection: Optional[object] = None  # residue name or atom indices, for those inputs


@dataclass
class RunConfig:
    length_ns: float = 500.0                 # per replica
    replicas: int = 5
    timestep_fs: float = 4.0                 # needs hydrogen mass repartitioning above 2 fs
    friction: float = 1.0                    # 1/ps
    pace_ps: float = 1.0                     # time between hills
    platform: str = "auto"                   # auto, CUDA, OpenCL, HIP, CPU, Reference
    precision: str = "mixed"
    gpus: Optional[list] = None              # device indices to spread replicas over
    checkpoint_ps: float = 250.0
    trajectory_ps: float = 100.0             # 0 disables the trajectory
    trajectory_solute_only: bool = True
    seed: int = 0                            # 0: random; otherwise replica i uses seed + i


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

    @classmethod
    def from_dict(cls, data: dict, base_dir=None) -> "Config":
        data = dict(data or {})
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
        for name in ("amber", "gromacs", "charmm"):
            block = getattr(self.input, name)
            if block:
                for key, value in list(block.items()):
                    if key == "parameters":
                        block[key] = [absolute(v) for v in value]
                    elif key != "box" and value is not None:
                        block[key] = absolute(value)
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
        data["system"]["forcefields"] = list(data["system"]["forcefields"])
        data["system"]["ligand_xml"] = list(data["system"]["ligand_xml"])
        data["equilibration"]["release_factors"] = list(data["equilibration"]["release_factors"])
        if data["funnel"]["exit_direction"] is not None:
            data["funnel"]["exit_direction"] = [float(x) for x in data["funnel"]["exit_direction"]]
        return data

    def save(self, path) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=None)


TEMPLATE = """\
# fmeta configuration. Every key is optional except the inputs; values shown are the defaults.
input:
  protein: protein.pdb        # for membrane proteins: oriented with the membrane normal along z
  ligand: ligand.sdf          # bound pose with hydrogens, in the protein's frame
  # Pre-parameterised input instead of protein + ligand (pick one):
  # amber:   {prmtop: complex.prmtop, coordinates: complex.inpcrd}
  # gromacs: {top: topol.top, gro: conf.gro}
  # charmm:  {psf: step5.psf, coordinates: step5.pdb, parameters: [toppar/par_all36m_prot.prm]}
  # ligand_selection: LIG     # ligand residue name in such inputs (auto-detected if unique)

system:
  forcefields: [amber14-all.xml, amber14/tip3p.xml]
  ligand_method: openff       # openff | gaff | xml (then list ligand_xml files)
  ligand_xml: []
  temperature: 300.0
  ionic_strength: 0.15
  padding: 1.2                # nm of solvent (or lipid, in the membrane plane) around the solute
  hydrogen_mass: 1.5          # amu; set to null (and run.timestep_fs: 2) to disable

# membrane: {lipid: POPC, center_z: 0.0}    # uncomment to embed in a bilayer

funnel:
  method: clearance           # automatic placement from the pose
  # exit_direction: [0, 0, 1] # override the exit search
funnel_overrides: {}          # e.g. {upper: 4.0}

metad:
  bias_factor: 10.0
  height: 1.5                 # kJ/mol
  sigma_s: 0.025              # nm, along the funnel axis
  sigma_rho: 0.05             # nm, away from the axis

equilibration: {}             # heating_ps, release_ps, free_ps ...

run:
  length_ns: 500.0            # per replica
  replicas: 5
  timestep_fs: 4.0
  platform: auto              # CUDA if available
  # gpus: [0, 1]              # spread replicas over these devices
"""
