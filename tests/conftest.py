"""Fixtures shared by the test modules: the small real system is built once."""

import pytest

from fmeta.system.builder import SystemSettings, prepare

from fixtures import BENZENE_XML, write_complex_inputs

LIGAND_SETTINGS = dict(ligand_method="xml", ligand_xml=[BENZENE_XML], ligand_resname="BNZ")


@pytest.fixture(scope="session")
def inputs(tmp_path_factory):
    """Paths of protein.pdb and benzene.sdf."""
    return write_complex_inputs(tmp_path_factory.mktemp("inputs"))


@pytest.fixture(scope="session")
def prepared(inputs):
    """The solvated protein-benzene system with its funnel."""
    return prepare(*inputs, SystemSettings(**LIGAND_SETTINGS))
