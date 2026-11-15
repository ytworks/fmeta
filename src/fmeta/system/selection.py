"""Classify the atoms of a topology: protein, ligand, lipid, water, ion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import openmm.app as app

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation / tautomer / terminal variants (Amber, CHARMM, GROMACS spellings)
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "HISD", "HISE", "HISH", "CYX", "CYM", "CYS2",
    "ASH", "GLH", "LYN", "ASPP", "GLUP", "MSE", "SEP", "TPO", "PTR",
    "ACE", "NME", "NMA", "NHE", "NH2",
}
WATER_RESIDUES = {"HOH", "WAT", "H2O", "TIP3", "TIP", "TP3", "TIP4", "TIP5", "SOL", "SPC", "OPC"}
ION_RESIDUES = {
    "NA", "CL", "K", "MG", "CA", "ZN", "LI", "RB", "CS", "F", "BR", "I", "MN", "FE", "CU",
    "NA+", "CL-", "K+", "MG2+", "CA2+", "SOD", "CLA", "POT", "CAL", "MG2", "ZN2", "CES", "LIT",
}
LIPID_RESIDUES = {
    # whole-lipid residues (CHARMM, OpenMM's amber lipid17 port, Slipids)
    "POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC", "POPS", "POPG", "POPA", "DOPE",
    "DOPS", "DOPG", "DSPC", "DPPE", "DMPE", "DMPG", "PSM", "SSM", "CHL1", "CHOL", "CLR", "CHL",
    # split-residue Amber Lipid14/17/21 naming
    "PA", "PC", "OL", "PE", "PS", "PGR", "PH-", "LAL", "MY", "ST", "AR", "DHA",
}


@dataclass
class Selection:
    """Atom indices by class. ``other`` holds cofactors and anything unrecognised."""

    protein: np.ndarray
    ligand: np.ndarray
    lipid: np.ndarray
    water: np.ndarray
    ion: np.ndarray
    other: np.ndarray
    heavy: np.ndarray          # boolean mask over all atoms

    def heavy_atoms(self, name: str) -> np.ndarray:
        indices = getattr(self, name)
        return indices[self.heavy[indices]]

    @property
    def solute(self) -> np.ndarray:
        """Everything that is not water or ions."""
        return np.sort(np.concatenate([self.protein, self.ligand, self.lipid, self.other]))


LigandSpec = Union[None, str, int, Sequence[int]]


def _is_heavy(atom) -> bool:
    return atom.element is not None and atom.element.symbol != "H"


def classify(topology: app.Topology, ligand: LigandSpec = None) -> Selection:
    """Sort atoms into classes.

    ``ligand`` may be a residue name, a list of atom indices, or ``None`` to
    pick the single non-polymer, non-lipid, non-solvent residue with at least
    six heavy atoms (an error if there is none or more than one).
    """
    groups = {name: [] for name in ("protein", "lipid", "water", "ion", "other")}
    candidates = []
    explicit = None
    if ligand is not None and not isinstance(ligand, str):
        explicit = set(int(i) for i in ligand)

    heavy = np.zeros(topology.getNumAtoms(), dtype=bool)
    for residue in topology.residues():
        name = residue.name.strip().upper()
        atoms = list(residue.atoms())
        for atom in atoms:
            heavy[atom.index] = _is_heavy(atom)
        indices = [atom.index for atom in atoms]
        if name in WATER_RESIDUES:
            groups["water"] += indices
        elif name in PROTEIN_RESIDUES:
            groups["protein"] += indices
        elif name in LIPID_RESIDUES:
            groups["lipid"] += indices
        elif name in ION_RESIDUES or (len(atoms) == 1 and name not in PROTEIN_RESIDUES):
            groups["ion"] += indices
        else:
            groups["other"] += indices
            candidates.append((residue, indices, sum(heavy[i] for i in indices)))

    if explicit is not None:
        ligand_atoms = sorted(explicit)
    elif isinstance(ligand, str):
        wanted = ligand.strip().upper()
        matches = [idx for res, idx, _ in candidates if res.name.strip().upper() == wanted]
        if not matches:
            # allow selecting a residue that was classified as something else
            matches = [[a.index for a in res.atoms()] for res in topology.residues()
                       if res.name.strip().upper() == wanted]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one residue named {ligand!r}, found {len(matches)}")
        ligand_atoms = matches[0]
    else:
        sizeable = [(res, idx) for res, idx, n_heavy in candidates if n_heavy >= 6]
        if len(sizeable) != 1:
            names = ", ".join(f"{res.name}{res.id}" for res, _ in sizeable) or "none"
            raise ValueError("could not identify the ligand automatically (candidates: "
                             f"{names}); pass its residue name or atom indices")
        ligand_atoms = sizeable[0][1]

    taken = set(ligand_atoms)
    arrays = {name: np.array(sorted(i for i in indices if i not in taken), dtype=int)
              for name, indices in groups.items()}
    return Selection(arrays["protein"], np.array(sorted(taken), dtype=int), arrays["lipid"],
                     arrays["water"], arrays["ion"], arrays["other"], heavy)
