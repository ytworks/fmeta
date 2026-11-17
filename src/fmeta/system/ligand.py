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

# The ligand force field when the settings name none (FM-D16). The generators have no default
# of their own (openmmforcefields 0.16 refuses to guess), and a result must not depend on which
# versions happen to be installed. Both prescribe AM1-BCC charges, and without a charge method
# in the settings the generator computes them: in the official image with sqm of AmberTools,
# the only AM1-BCC implementation there (no OpenEye toolkit).
DEFAULT_FORCEFIELD = {"openff": "openff-2.2.1", "gaff": "gaff-2.11"}


def register_ligand(forcefield: app.ForceField, ligand_path, method: str = "openff",
                    forcefield_name: str | None = None,
                    charge_method: str | None = None) -> None:
    """Make ``forcefield`` able to parameterise the ligand in ``ligand_path``.

    For ``method='xml'`` there is nothing to do here: load the ligand ffxml
    together with the other force-field files. ``forcefield_name`` is a name the
    generator knows (``DEFAULT_FORCEFIELD`` of the method when None).
    ``charge_method`` (e.g. ``"am1bcc"``, ``"gasteiger"``, or a NAGL model name)
    assigns partial charges with the OpenFF toolkit before the generator sees the
    molecule; by default the generator assigns the charges its force field
    prescribes (AM1-BCC for both ``DEFAULT_FORCEFIELD``). The ligand file should
    be an SDF, which carries bond orders and formal charges.
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
            f"generating ligand parameters ({method}) needs openmmforcefields and openff-toolkit, "
            "which are conda packages: create the environment from environment.yml, or run "
            "'conda install -c conda-forge openmmforcefields'. Without them, supply an OpenMM "
            "ffxml file for the ligand (--ligand-xml, system.ligand_xml).") from error
    molecule = Molecule.from_file(str(ligand_path), allow_undefined_stereo=True)
    if isinstance(molecule, list):       # multi-record file: the pose is the first record
        molecule = molecule[0]
    if charge_method is not None:
        molecule.assign_partial_charges(charge_method)
    generator_class = SMIRNOFFTemplateGenerator if method == "openff" else GAFFTemplateGenerator
    generator = generator_class(molecules=molecule,
                                forcefield=forcefield_name or DEFAULT_FORCEFIELD[method])
    forcefield.registerTemplateGenerator(generator.generator)
