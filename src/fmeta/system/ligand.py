"""Ligand parameters for an OpenMM ``ForceField``.

Three routes:

* ``xml``    - the user supplies ffxml file(s) with a residue template for the ligand;
* ``openff`` - OpenFF (SMIRNOFF) parameters generated on the fly;
* ``gaff``   - GAFF parameters generated on the fly.

The last two need ``openmmforcefields`` and ``openff-toolkit`` (conda-forge).
"""

from __future__ import annotations

import openmm.app as app

METHODS = ("xml", "openff", "gaff")


def register_ligand(forcefield: app.ForceField, ligand_path, method: str = "openff",
                    forcefield_name: str | None = None,
                    charge_method: str | None = None) -> None:
    """Make ``forcefield`` able to parameterise the ligand in ``ligand_path``.

    For ``method='xml'`` there is nothing to do here: load the ligand ffxml
    together with the other force-field files. ``charge_method`` (e.g.
    ``"am1bcc"``, ``"gasteiger"``, or a NAGL model name) assigns partial
    charges with the OpenFF toolkit before the generator sees the molecule;
    by default the generator computes AM1-BCC charges itself. The ligand file
    should be an SDF, which carries bond orders and formal charges.
    """
    if method == "xml":
        return
    if method not in METHODS:
        raise ValueError(f"unknown ligand parameter method {method!r}; choose from {METHODS}")
    try:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
    except ImportError as error:
        raise ImportError(
            f"ligand method {method!r} needs openmmforcefields and openff-toolkit "
            "(conda install -c conda-forge openmmforcefields). Alternatively supply an ffxml "
            "file for the ligand (method 'xml') or a pre-parameterised system.") from error
    molecule = Molecule.from_file(str(ligand_path), allow_undefined_stereo=True)
    if isinstance(molecule, list):       # multi-record file: the pose is the first record
        molecule = molecule[0]
    if charge_method is not None:
        molecule.assign_partial_charges(charge_method)
    generator_class = SMIRNOFFTemplateGenerator if method == "openff" else GAFFTemplateGenerator
    kwargs = {} if forcefield_name is None else {"forcefield": forcefield_name}
    generator = generator_class(molecules=molecule, **kwargs)
    forcefield.registerTemplateGenerator(generator.generator)
