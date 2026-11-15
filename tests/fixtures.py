"""Test helpers. The small real system itself lives in the package
(``fmeta.testing.systems``) because ``fmeta check`` uses it too."""

from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit

from fmeta.testing.systems import (BENZENE_XML, pocket_point, write_benzene,  # noqa: F401
                                   write_complex_inputs, write_membrane_inputs, write_protein)


def force_field_structure(prepared):
    """The prepared system as a ParmEd structure with real force-field atom types
    (needed to write CHARMM files; harmless for the other formats)."""
    import re

    import openmm.app as app
    import parmed

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml", BENZENE_XML)
    topology = prepared.topology
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, constraints=None,
                                     rigidWater=False)
    structure = parmed.openmm.load_topology(topology, system,
                                            xyz=prepared.positions * unit.nanometer,
                                            condense_atom_types=False)
    data = app.ForceField._SystemData(topology)
    forcefield._matchAllResiduesToTemplates(data, topology, dict(), False)
    names, types = {}, {}
    for atom, topology_atom in zip(structure.atoms, topology.atoms()):
        full = data.atomType[topology_atom]
        if full not in names:
            base = re.sub(r"[^A-Za-z0-9]", "", full.split("-")[-1]).upper()[:4] or "X"
            name, k = base, 0
            while name in names.values():
                k += 1
                name = f"{base[:3]}{k}"
            names[full] = name
            kind = parmed.AtomType(name, len(types) + 1, atom.mass, atom.atomic_number)
            kind.set_lj_params(atom.epsilon, atom.rmin, atom.epsilon_14, atom.rmin_14)
            types[name] = kind
        atom.type = names[full]
        atom.atom_type = types[names[full]]
    return structure


def write_preparameterised(prepared, directory, kind: str, positions=None) -> dict:
    """Write the prepared system in another program's format; returns the
    ``config.input`` block that points at the files."""
    import parmed

    directory = Path(directory)
    structure = force_field_structure(prepared)
    if positions is not None:
        structure.coordinates = np.asarray(positions) * 10.0
    if kind == "amber":
        structure.save(str(directory / "complex.prmtop"))
        structure.save(str(directory / "complex.inpcrd"))
        return {"amber": {"prmtop": "complex.prmtop", "coordinates": "complex.inpcrd"}}
    if kind == "gromacs":
        structure.save(str(directory / "topol.top"))
        structure.save(str(directory / "conf.gro"))
        return {"gromacs": {"top": "topol.top", "gro": "conf.gro"}}
    if kind == "charmm":
        parmed.charmm.CharmmPsfFile.from_structure(structure, copy=True).write_psf(
            str(directory / "complex.psf"))
        parmed.charmm.CharmmParameterSet.from_structure(structure).write(
            par=str(directory / "complex.par"), top=str(directory / "complex.rtf"))
        coords = prepared.positions if positions is None else np.asarray(positions)
        with open(directory / "complex.pdb", "w") as handle:
            app.PDBFile.writeFile(prepared.topology, coords * unit.nanometer, handle)
        return {"charmm": {"psf": "complex.psf", "coordinates": "complex.pdb",
                           "parameters": ["complex.rtf", "complex.par"]}}
    raise ValueError(kind)
