"""The presets, and a configuration on top of a preset."""

from pathlib import Path

import pytest
import yaml

from fmeta.cli import main as cli
from fmeta.config import PRESETS, Config, InputConfig, RunConfig
from fmeta.system.builder import MembraneSettings, SystemSettings

ROOT = Path(__file__).resolve().parents[1]


def defaults(membrane: bool) -> dict:
    """Every key of a configuration file with the default of the code."""
    data = Config(membrane=MembraneSettings() if membrane else None).to_dict()
    del data["metad"]["temperature"], data["metad"]["pace"]     # derived, not keys of a file
    if not membrane:
        del data["membrane"]
    return data


@pytest.mark.parametrize("name", PRESETS)
def test_preset_is_the_defaults(name, tmp_path):
    """``fmeta init --preset`` writes every key with the default of the code, and no other key."""
    path = tmp_path / "fmeta.yaml"
    cli(["init", "--preset", name, str(path)])
    membrane = name == "membrane"
    expected = defaults(membrane)
    expected["input"].update(protein="protein.pdb", ligand="ligand.sdf")     # the placeholders
    assert yaml.safe_load(path.read_text()) == expected

    files = InputConfig(protein=str(tmp_path / "protein.pdb"), ligand=str(tmp_path / "ligand.sdf"))
    reference = Config(input=files, membrane=MembraneSettings() if membrane else None)
    assert Config.load(path).to_dict() == reference.to_dict()


def test_configuration_on_a_preset(tmp_path):
    def load(text):
        return Config.from_dict(yaml.safe_load(text), base_dir=tmp_path)

    # the keys of the file win; the rest is the preset, except its placeholder file names;
    # a ligand method without a force field gets the pinned force field of that method
    config = load("preset: membrane\n"
                  "input: {complex: c.pdb, ligand_resname: BEN}\n"
                  "system: {temperature: 310.0, ligand_method: gaff}\n"
                  "run: {length_ns: 200.0}\n")
    expected = Config(input=InputConfig(complex=str(tmp_path / "c.pdb"), ligand_resname="BEN"),
                      system=SystemSettings(temperature=310.0, ligand_method="gaff",
                                            ligand_forcefield="gaff-2.11"),
                      membrane=MembraneSettings(), run=RunConfig(length_ns=200.0))
    assert config.to_dict() == expected.to_dict()

    # the saved configuration is resolved: it names no preset and loads as the same values
    config.save(tmp_path / "config.yaml")
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert "preset" not in saved and Config.from_dict(saved).to_dict() == config.to_dict()

    assert load("preset: membrane\nmembrane: {lipid: POPE}").membrane == MembraneSettings(lipid="POPE")
    assert load("preset: membrane\nmembrane: false").membrane is None
    assert load("preset: soluble").membrane is None
    assert load("preset: soluble\nmembrane: true").membrane == MembraneSettings()

    with pytest.raises(ValueError, match="unknown preset"):
        load("preset: gpcr")
    with pytest.raises(ValueError, match="unknown option"):
        load("preset: soluble\nrun: {lenght_ns: 1}")


@pytest.mark.parametrize("path", sorted(ROOT.glob("examples/**/fmeta.yaml")),
                         ids=lambda path: path.parent.name)
def test_example_loads(path):
    Config.load(path)
