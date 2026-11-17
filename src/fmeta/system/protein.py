"""The protein model taken from an input file: chain pieces, caps and protonation states.

OpenMM reads a structure residue by residue and bonds the consecutive residues of a chain
whatever their distance; it knows a protonation state only from hydrogens whose names fit,
and loses all of them when the hydrogens are rebuilt. This module makes the statements of
the input explicit before the force field sees it (design: FM-D15, ``docs/design/engine.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import Vec3

GAP_NM = 0.2            # C-N of consecutive residues further apart than this: residues are missing
CAP_BOND_NM = 0.17      # an acetyl or N-methyl group this close to a chain end is its cap
# heavy atoms of a new cap nearer than this to the other side of the gap: no room. On the test
# protein the caps of a gap of one residue come within 0.23 nm, of two residues 0.37 nm or more
CAP_CLASH_NM = 0.3
BACKBONE = {"N", "CA", "C"}
WATER = {"HOH", "WAT", "H2O", "TIP3", "TIP", "SOL", "DOD"}
# residue names that state a protonation state: (standard name, variant of Modeller.addHydrogens).
# A cysteine built as CYX without a partner is the thiolate (template CYM).
VARIANT_NAMES = {"HID": ("HIS", "HID"), "HIE": ("HIS", "HIE"), "HIP": ("HIS", "HIP"),
                 "HSD": ("HIS", "HID"), "HSP": ("HIS", "HIP"), "ASH": ("ASP", "ASH"),
                 "GLH": ("GLU", "GLH"), "LYN": ("LYS", "LYN"), "CYX": ("CYS", "CYX"),
                 "CYM": ("CYS", "CYX")}
# residues that are amino acids by their name, whatever atoms the file has of them
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
               "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"} | set(VARIANT_NAMES)
# geometry of the peptide bond to a cap (nm, degrees; the Amber equilibrium values)
_C_N, _N_CH3, _C_O, _C_CH3 = 0.1335, 0.1449, 0.1229, 0.1522
_C_N_C, _C_C_N, _O_C_N = 121.9, 116.6, 122.9


@dataclass
class ProteinInput:
    """A protein file reduced to what will be simulated (see ``inputs.read_protein``)."""
    topology: app.Topology
    positions: np.ndarray                              # nm
    notes: list = field(default_factory=list)          # what was changed, in words
    variants: list = field(default_factory=list)       # per residue: state to build, or None
    info: dict = field(default_factory=dict)           # the same facts for the report (JSON)


def label(residue) -> str:
    return f"{residue.chain.id.strip()}:{residue.name}{residue.id}{residue.insertionCode.strip()}"


def _is_hydrogen(atom) -> bool:
    return atom.element is not None and atom.element.symbol == "H"


def _atom(residue, name):
    return next((atom for atom in residue.atoms() if atom.name == name), None)


def _is_amino(residue) -> bool:
    """An amino acid by its name, or by its backbone atoms under any name."""
    return (residue.name.strip().upper() in AMINO_ACIDS
            or BACKBONE <= {atom.name for atom in residue.atoms()})


def _cap_id(neighbour, shift: int) -> str:
    """The residue number of a cap before (-1) or after (+1) the residue it is bonded to."""
    return str(int(neighbour.id) + shift) if neighbour.id.strip().lstrip("-").isdigit() else neighbour.id


def scan_pdb(path) -> tuple[list[dict], dict]:
    """What OpenMM drops when it reads a PDB file: the alternate locations (it keeps the
    one listed first for a residue) and the residue names that state a protonation state.
    Returns the alternate locations per residue and the names by (chain, number, insertion
    code)."""
    letters: dict[tuple, list[str]] = {}
    names: dict[tuple, str] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        key = (line[21].strip(), line[22:26].strip(), line[26].strip())
        names.setdefault(key, line[17:20].strip().upper())
        if line[16] != " " and line[16] not in letters.setdefault(key, []):
            letters[key].append(line[16])
    altlocs = [dict(residue=f"{chain}:{names[chain, number, code]}{number}{code}", kept=found[0],
                    dropped=found[1:])
               for (chain, number, code), found in letters.items() if len(found) > 1]
    return altlocs, names


# ---- cap geometry ----------------------------------------------------------

def _place(a, b, c, length: float, angle: float, dihedral: float) -> np.ndarray:
    """The point d with |cd| = length, angle b-c-d and dihedral a-b-c-d (degrees)."""
    theta, phi = np.radians(angle), np.radians(dihedral)
    bc = (c - b) / np.linalg.norm(c - b)
    normal = np.cross(b - a, bc)
    normal /= np.linalg.norm(normal)
    return c + length * (-np.cos(theta) * bc + np.sin(theta) * np.cos(phi) * np.cross(normal, bc)
                         + np.sin(theta) * np.sin(phi) * normal)


def _dihedral(a, b, c, d) -> float:
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


def _separation(first: dict, second) -> float:
    """Smallest distance between the atoms of a cap and other points (or another cap)."""
    points = np.array(list(first.values()))
    others = np.array(list(second.values())) if isinstance(second, dict) else second
    return float(np.linalg.norm(points[:, None] - others[None], axis=2).min()) if len(others) else np.inf


def _roomiest(build, residue, fixed, positions: np.ndarray, heavy, avoid=()) -> dict:
    """Of the caps ``build(dihedral)`` (10 degree steps) the one furthest from the heavy atoms
    and the points ``avoid``. The atoms ``fixed`` of the residue do not count: their distance
    to the cap is the same for every dihedral."""
    skipped = {_atom(residue, name).index for name in fixed}
    others = np.array([positions[index] for index in heavy if index not in skipped] + list(avoid))
    return max((build(float(dihedral)) for dihedral in range(-180, 180, 10)),
               key=lambda cap: _separation(cap, others))


def acetyl_position(residue, positions: np.ndarray, heavy, avoid=()) -> dict:
    """Heavy atoms of an ACE in front of ``residue``: trans to CA, in the plane of the amide
    hydrogen (or CD of proline) when the input has it, else where there is most room
    (``avoid``: the cap on the other side of the gap). Two hydrogens on the nitrogen (an
    open end a preparation program filled) do not say which of them is the amide one."""
    n, ca, c = (positions[_atom(residue, name).index] for name in ("N", "CA", "C"))

    def build(carbon):
        return {"CH3": _place(ca, n, carbon, _C_CH3, _C_C_N, 180.0), "C": carbon,
                "O": _place(ca, n, carbon, _C_O, _O_C_N, 0.0)}

    partner = _atom(residue, "CD") if residue.name == "PRO" else _atom(residue, "H")
    if partner is None or _atom(residue, "H2") is not None:
        return _roomiest(lambda dihedral: build(_place(c, ca, n, _C_N, _C_N_C, dihedral)),
                         residue, ("N", "CA"), positions, heavy, avoid)
    bond = positions[partner.index] - n
    away = -(bond / np.linalg.norm(bond) + (ca - n) / np.linalg.norm(ca - n))
    return build(n + _C_N * away / np.linalg.norm(away))


def methylamide_position(residue, positions: np.ndarray, heavy) -> dict:
    """Heavy atoms of an NME behind ``residue``: the nitrogen opposite to the carbonyl oxygen
    (else where there is most room), the methyl trans to CA."""
    n, ca, c = (positions[_atom(residue, name).index] for name in ("N", "CA", "C"))

    def build(dihedral):
        nitrogen = _place(n, ca, c, _C_N, _C_C_N, dihedral)
        return {"N": nitrogen, "C": _place(ca, c, nitrogen, _N_CH3, _C_N_C, 180.0)}

    oxygen = _atom(residue, "O")
    if oxygen is None:
        return _roomiest(build, residue, ("C", "CA"), positions, heavy)
    return build(_dihedral(n, ca, c, positions[oxygen.index]) + 180.0)


# ---- the model -------------------------------------------------------------

@dataclass
class _Piece:
    """Consecutive bonded residues of a chain, with the heavy atoms of their caps."""
    chain: object
    residues: list
    gap: float | None = None            # C-N distance to the piece before it in the chain (nm)
    acetyl: dict | None = None
    methylamide: dict | None = None


def _goes_on(last, residue) -> bool:
    """Whether ``residue`` belongs to the chain of the residue ``last`` before it: the same
    chain of the topology, or the next one under the same identifier with the numbering
    going on (a TER record inside a chain, as files have it at missing residues)."""
    if residue.chain is last.chain:
        return True
    try:
        return (residue.chain.id == last.chain.id
                and (int(residue.id), residue.insertionCode) > (int(last.id), last.insertionCode))
    except ValueError:
        return False


def _apart(positions, atom, other) -> float:
    return float(np.linalg.norm(positions[atom.index] - positions[other.index]))


def _cap_group(residue, positions):
    """``("ACE" | "NME", its heavy atoms by their names there)`` when the residue is no amino
    acid or water and its heavy atoms are those of an acetyl or an N-methyl group, whatever
    the residue and the atoms are called; else None. The acetyl is bonded to the chain by
    its C (the carbon at the oxygen), the N-methyl group by its N."""
    if _is_amino(residue) or residue.name.strip().upper() in WATER:
        return None
    atoms: dict[str, list] = {}
    for atom in residue.atoms():
        if not _is_hydrogen(atom):
            atoms.setdefault("?" if atom.element is None else atom.element.symbol, []).append(atom)
    shape = {symbol: len(group) for symbol, group in atoms.items()}
    if shape == {"C": 2, "O": 1}:
        carbonyl, methyl = sorted(atoms["C"], key=lambda atom: _apart(positions, atom, atoms["O"][0]))
        return "ACE", {"CH3": methyl, "C": carbonyl, "O": atoms["O"][0]}
    if shape == {"C": 1, "N": 1}:
        return "NME", {"N": atoms["N"][0], "C": atoms["C"][0]}
    return None


def _existing_caps(topology, positions, pieces) -> tuple[set, list[str]]:
    """Acetyl and N-methyl groups of the input bonded to a chain end, whatever they are
    called (ACE / NME, or UNK from a preparation program). Returns the residues used and
    what was found. A group nearer to a chain end than atoms without a bond come
    (``CAP_CLASH_NM``) that is not bonded to it is refused: removed with the hetero
    components it would leave a charged terminus the input does not state."""
    used, found = set(), []
    for residue in topology.residues():
        group = _cap_group(residue, positions)
        if group is None:
            continue
        kind, atoms = group
        ends = []
        for piece in pieces:
            end = piece.residues[0] if kind == "ACE" else piece.residues[-1]
            taken = piece.acetyl if kind == "ACE" else piece.methylamide
            gap = _apart(positions, *((atoms["C"], _atom(end, "N")) if kind == "ACE"
                                      else (atoms["N"], _atom(end, "C"))))
            ends.append((gap, label(end)))
            if gap >= CAP_BOND_NM or taken is not None:
                continue
            place = {name: positions[atom.index] for name, atom in atoms.items()}
            if kind == "ACE":
                piece.acetyl = place
                found.append(f"ACE before {label(end)}")
            else:
                piece.methylamide = place
                found.append(f"NME after {label(end)}")
            used.add(residue)
            break
        else:
            gap, end = min(ends, default=(np.inf, ""))
            if gap < CAP_CLASH_NM:
                raise ValueError(
                    f"{label(residue)} has the atoms of an {kind} cap and stands {gap:.2f} nm from the "
                    f"chain end at {end}: nearer than atoms without a bond ({CAP_CLASH_NM} nm), not "
                    f"bonded as a cap is ({CAP_BOND_NM} nm). Removed with the hetero components it would "
                    "leave a charged terminus that the input does not state. Put the cap at its bond "
                    "length, or delete it from the file")
    return used, found


def incomplete_backbone(topology: app.Topology) -> list[tuple]:
    """The residues named as amino acids that lack one of N, CA and C, each with the names of
    the atoms it lacks."""
    found = []
    for residue in topology.residues():
        if residue.name.strip().upper() in AMINO_ACIDS:
            missing = BACKBONE - {atom.name for atom in residue.atoms()}
            if missing:
                found.append((residue, sorted(missing)))
    return found


def backbone_listing(incomplete) -> list[str]:
    return [f"{label(residue)} ({', '.join(missing)})" for residue, missing in incomplete]


# How far apart (nm) two backbone atoms of consecutive residues can be when a peptide bond joins
# the residues, by the number of bonds from one atom to the other: C-N; C-CA and CA-N; CA-CA, N-N
# and C-C; N-CA and CA-C; N-C. Measured on 1192 bonded pairs of four structures: at most 0.138,
# 0.254, 0.391, 0.505 and 0.617 nm. About 0.05 nm is added (the bond itself: GAP_NM).
REACH_NM = {1: GAP_NM, 2: 0.30, 3: 0.44, 4: 0.55, 5: 0.67}
_ALONG = {"N": 0, "CA": 1, "C": 2}


def _can_be_bonded(last, residue, positions: np.ndarray) -> bool:
    """Whether a peptide bond from ``last`` to ``residue`` is possible, judged from the backbone
    atoms that are there: the last one of ``last`` and the first one of ``residue``."""
    ends = [pick((atom for atom in group.atoms() if atom.name in _ALONG),
                 key=lambda atom: _ALONG[atom.name], default=None)
            for pick, group in ((max, last), (min, residue))]
    if None in ends:
        return True
    reach = REACH_NM[3 + _ALONG[ends[1].name] - _ALONG[ends[0].name]]
    return float(np.linalg.norm(positions[ends[0].index] - positions[ends[1].index])) <= reach


def split_at_breaks(topology: app.Topology, positions: np.ndarray) -> app.Topology:
    """A copy of the topology in which a chain ends where two consecutive amino acids cannot be
    bonded (residues are missing between them), without the bond OpenMM makes over the gap.

    For the repair of backbone atoms, which are missing when ``build_model`` would find the
    gap from C and N: an atom built in a chain that runs over a gap is placed towards the
    residue behind it. The C of a residue stood 0.28 nm from the N behind a gap of 0.447 nm,
    and no cap had room (review of PR #1)."""
    new, copies, cut = app.Topology(), {}, set()
    new.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    for chain in topology.chains():
        target, last = new.addChain(chain.id), None
        for residue in chain.residues():
            if (last is not None and _is_amino(last) and _is_amino(residue)
                    and not _can_be_bonded(last, residue, positions)):
                target = new.addChain(chain.id)
                cut.add(frozenset((last, residue)))
            copy = new.addResidue(residue.name, target, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                copies[atom] = new.addAtom(atom.name, atom.element, copy, atom.id)
            last = residue
    for bond in topology.bonds():
        if frozenset((bond[0].residue, bond[1].residue)) not in cut:
            new.addBond(copies[bond[0]], copies[bond[1]], bond.type, bond.order)
    return new


def caps_in_place(topology: app.Topology, positions: np.ndarray) -> tuple[app.Topology, np.ndarray]:
    """A copy of the structure in which the acetyl and N-methyl groups bonded to a chain end
    stand in the chain as ACE and NME (their heavy atoms), wherever the file has them and
    whatever it calls them.

    For the repair of backbone atoms. PDBFixer builds a missing N or C with its bond to a cap
    it knows; a cap written as UNK after the chain it does not know. The N of the first
    residue was then built 0.198 nm from the acetyl carbon (0.146 nm with the cap in place):
    no bond for ``build_model``, which removed the cap with the hetero components and left a
    charged terminus the input does not state (review of PR #1). A cap is found from the
    backbone atom of the end residue that is nearest to it along the chain: N or C
    (``CAP_BOND_NM``), or CA when that atom is the missing one (two bonds, ``REACH_NM``).
    Over three bonds a cap cannot be told from a molecule that lies next to the residue:
    refused."""
    before, after, moved = {}, {}, set()
    chain_residues = [residue for residue in topology.residues() if _is_amino(residue)]
    for residue in topology.residues():
        group = _cap_group(residue, positions)
        if group is None:
            continue
        kind, atoms = group
        pick, link, slot = (min, "C", before) if kind == "ACE" else (max, "N", after)
        for end in chain_residues:
            anchor = pick((atom for atom in end.atoms() if atom.name in _ALONG),
                          key=lambda atom: _ALONG[atom.name], default=None)
            if anchor is None or end in slot:
                continue
            bonds = 1 + (_ALONG[anchor.name] if kind == "ACE" else 2 - _ALONG[anchor.name])
            gap = _apart(positions, atoms[link], anchor)
            if gap >= (CAP_BOND_NM if bonds == 1 else REACH_NM[bonds]):
                continue
            if bonds == 3:
                raise ValueError(
                    f"{label(residue)} has the atoms of an {kind} cap and stands {gap:.2f} nm from "
                    f"{anchor.name} of {label(end)}, which lacks the backbone atoms between the two: "
                    "whether it is the cap of the residue cannot be told. Complete the residue, or "
                    "delete the residue or the group from the file")
            slot[end] = atoms
            moved.add(residue)
            break

    new, coordinates, copies = app.Topology(), [], {}
    new.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())

    def cap(name, chain, neighbour, shift, atoms):
        target = new.addResidue(name, chain, _cap_id(neighbour, shift))
        for atom_name, atom in atoms.items():
            new.addAtom(atom_name, atom.element, target)
            coordinates.append(positions[atom.index])

    for chain in topology.chains():
        target = None
        for residue in chain.residues():
            if residue in moved:
                continue
            target = target or new.addChain(chain.id)
            if residue in before:
                cap("ACE", target, residue, -1, before[residue])
            copy = new.addResidue(residue.name, target, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                copies[atom] = new.addAtom(atom.name, atom.element, copy, atom.id)
                coordinates.append(positions[atom.index])
            if residue in after:
                cap("NME", target, residue, 1, after[residue])
    for bond in topology.bonds():
        if bond[0] in copies and bond[1] in copies:
            new.addBond(copies[bond[0]], copies[bond[1]], bond.type, bond.order)
    return new, np.array(coordinates, dtype=float).reshape(-1, 3)


def build_model(topology: app.Topology, positions: np.ndarray, keep_hetero=(),
                keep_water: bool = False, names: dict | None = None) -> ProteinInput:
    """The protein of a structure as it will be simulated.

    Amino-acid residues (N, CA and C present) are kept; of the rest only water with
    ``keep_water`` and the residues named in ``keep_hetero``, in chains of their own. A
    residue named as an amino acid without one of the three atoms is refused: removed with
    the rest it would shorten or cut the chain (``read_protein`` rebuilds its atoms first). A
    chain is split where the C-N distance of consecutive residues is above ``GAP_NM``
    (residues missing from the structure), with or without a TER record there: the two ends
    are capped with NME and ACE. Where the input states a terminus nothing is added: after
    an OXT, and before a residue with H2 and H3, a new chain starts. Acetyl and N-methyl
    groups of the input stay as ACE and NME. Residue names that state a protonation state
    become the standard name plus a variant. ``names`` are the residue names of the file
    where OpenMM renamed them (``scan_pdb``).
    """
    incomplete = incomplete_backbone(topology)
    if incomplete:
        raise ValueError(
            "amino-acid residues without their backbone atoms N, CA and C: "
            f"{', '.join(backbone_listing(incomplete))}. "
            "A residue like this cannot be simulated, and without it the chain would end or be cut "
            "there. PDBFixer rebuilds the atoms (install it, or set system.fix_protein); or delete the "
            "residue from the file, and the chain ends, or is split and capped, at that place")
    names = names or {}
    keep = {name.strip().upper() for name in keep_hetero}
    heavy = [atom.index for atom in topology.atoms() if not _is_hydrogen(atom)]
    amino = {residue for residue in topology.residues()
             if BACKBONE <= {atom.name for atom in residue.atoms()}}

    pieces: list[_Piece] = []
    for chain in topology.chains():
        for residue in chain.residues():
            if residue not in amino:
                continue
            last = pieces[-1].residues[-1] if pieces and _goes_on(pieces[-1].residues[-1], residue) else None
            gap = None if last is None else float(np.linalg.norm(
                positions[_atom(last, "C").index] - positions[_atom(residue, "N").index]))
            if gap is not None and gap <= GAP_NM:
                pieces[-1].residues.append(residue)
                continue
            if last is not None and (_atom(last, "OXT") is not None
                                     or {"H2", "H3"} <= {atom.name for atom in residue.atoms()}):
                gap = None                                 # a terminus as stated: no gap
            pieces.append(_Piece(chain, [residue], gap))
    caps, caps_kept = _existing_caps(topology, positions, pieces)

    gaps = []
    for before, piece in zip(pieces, pieces[1:]):
        if piece.gap is None:                              # the piece starts a chain
            continue
        last, first = before.residues[-1], piece.residues[0]
        added = []
        if before.methylamide is None:
            before.methylamide = methylamide_position(last, positions, heavy)
            added.append("NME")
        if piece.acetyl is None:
            piece.acetyl = acetyl_position(first, positions, heavy, list(before.methylamide.values()))
            added.append("ACE")
        if added:
            # the caps stand where the missing residues were: with a single residue missing
            # (or none: a stretched bond) the two would overlap
            room = min([_separation(before.methylamide, piece.acetyl)] + [
                _separation(cap, positions[[atom.index for atom in other.atoms() if not _is_hydrogen(atom)]])
                for cap, other in ((before.methylamide, first), (piece.acetyl, last))])
            if room < CAP_CLASH_NM:
                raise ValueError(
                    f"{label(last)} and {label(first)} are not bonded (C-N {piece.gap:.2f} nm) and "
                    f"there is no room for caps between them (atoms {room:.2f} nm apart): at most "
                    "one residue is missing. Add it to the input, or remove one of the two residues")
        gaps.append(dict(after=label(last), before=label(first), c_n_nm=round(piece.gap, 3), caps=added))

    new = app.Topology()
    new.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    coordinates, copies, hints = [], {}, []
    removed: dict[str, int] = {}
    kept: dict[str, int] = {}

    def copy(residue, chain):
        key = (residue.chain.id.strip(), residue.id.strip(), residue.insertionCode.strip())
        stated = names.get(key, residue.name).strip().upper()
        standard, variant = VARIANT_NAMES.get(stated, (None, None))
        renamed = standard is not None and residue in amino and residue.name in (standard, stated)
        target = new.addResidue(standard if renamed else residue.name, chain, residue.id,
                                residue.insertionCode)
        hints.append(variant if renamed else None)
        for atom in residue.atoms():
            copies[atom] = new.addAtom(atom.name, atom.element, target)
            coordinates.append(positions[atom.index])

    def cap(name, chain, neighbour, shift, atoms):
        target = new.addResidue(name, chain, _cap_id(neighbour, shift))
        hints.append(None)
        for atom_name, position in atoms.items():
            new.addAtom(atom_name, app.Element.getBySymbol(atom_name[0]), target)
            coordinates.append(position)

    for chain in topology.chains():
        for piece in pieces:
            if piece.chain is not chain:
                continue
            target = new.addChain(chain.id)
            if piece.acetyl is not None:
                cap("ACE", target, piece.residues[0], -1, piece.acetyl)
            for residue in piece.residues:
                copy(residue, target)
            if piece.methylamide is not None:
                cap("NME", target, piece.residues[-1], 1, piece.methylamide)
        target = None
        for residue in chain.residues():
            if residue in amino or residue in caps:
                continue
            name = residue.name.strip().upper()
            if name in keep or (keep_water and name in WATER):
                target = target or new.addChain(chain.id)
                copy(residue, target)
                kept[name] = kept.get(name, 0) + 1
            else:
                removed[name] = removed.get(name, 0) + 1

    # bonds: the standard ones of the pieces, disulfides, and what else the input states
    # (CONECT records of kept components); never a bond twice, never one across a gap
    points = np.array(coordinates, dtype=float).reshape(-1, 3)
    new.createStandardBonds()
    new.createDisulfideBonds([Vec3(*point) for point in points] * unit.nanometer)
    present = {frozenset((first.index, second.index)) for first, second in new.bonds()}
    for first, second in topology.bonds():
        if first not in copies or second not in copies:
            continue
        if (first.residue is not second.residue and first.residue in amino
                and second.residue in amino and {first.name, second.name} == {"C", "N"}):
            continue
        pair = frozenset((copies[first].index, copies[second].index))
        if pair not in present:
            new.addBond(copies[first], copies[second])
            present.add(pair)

    def listing(counts):
        return ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))

    notes = []
    if removed:
        notes.append(f"removed from the protein file: {listing(removed)}")
    if kept:
        notes.append(f"kept from the protein file: {listing(kept)}")
    if caps_kept:
        notes.append("caps of the input kept: " + ", ".join(caps_kept))
    for gap in gaps:
        what = " and ".join(gap["caps"]) or "the caps of the input stay"
        notes.append(f"residues are missing between {gap['after']} and {gap['before']} "
                     f"(C-N {gap['c_n_nm']:.2f} nm): chain split there ({what})")
    info = dict(removed=removed, kept_hetero=kept, caps_kept=caps_kept, gaps_capped=gaps)
    return ProteinInput(new, points, notes, hints, info)


_HISTIDINE = {(True, True): "HIP", (True, False): "HID", (False, True): "HIE"}


def read_states(protein: ProteinInput) -> None:
    """Take the protonation states the input states into ``protein.variants``.

    The hydrogens of a residue decide (histidine: HD1 / HE2; ASP and GLU: a hydrogen on the
    acid; lysine: two or three HZ; cysteine: HG, and without it a disulfide or the thiolate);
    a residue without hydrogens keeps the state its name stated, if any. The states go to
    ``Modeller.addHydrogens``, which otherwise decides by pH and loses them whenever the
    hydrogens are rebuilt.
    """
    topology = protein.topology
    bridged = {atom.residue for bond in topology.bonds() for atom in bond
               if bond[0].name == "SG" and bond[1].name == "SG"}
    taken: dict[str, list[str]] = {}
    for index, residue in enumerate(topology.residues()):
        hydrogens = sorted(atom.name for atom in residue.atoms() if _is_hydrogen(atom))
        state = None
        if hydrogens and residue.name == "HIS":
            state = _HISTIDINE.get(("HD1" in hydrogens, "HE2" in hydrogens))
        elif hydrogens and residue.name == "ASP":
            state = "ASH" if {"HD1", "HD2"} & set(hydrogens) else "ASP"
        elif hydrogens and residue.name == "GLU":
            state = "GLH" if {"HE1", "HE2"} & set(hydrogens) else "GLU"
        elif hydrogens and residue.name == "LYS":
            amine = [name for name in hydrogens if name.startswith("HZ")]
            state = {3: "LYS", 2: "LYN"}.get(len(amine))
            if state == "LYN":                              # the neutral amine holds HZ1 and HZ2
                for atom in residue.atoms():
                    if atom.name in amine:
                        atom.name = ("HZ1", "HZ2")[amine.index(atom.name)]
        elif hydrogens and residue.name == "CYS":
            state = "CYS" if "HG" in hydrogens else "CYX"
        state = state or protein.variants[index]
        protein.variants[index] = state
        shown = "CYM" if state == "CYX" and residue not in bridged else state
        if shown in ("HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYM"):
            taken.setdefault(shown, []).append(label(residue))
    protein.info["protonation_from_input"] = taken
    if taken:
        protein.notes.append("protonation states taken from the input: " + ", ".join(
            f"{state} x{len(found)}" for state, found in sorted(taken.items())))


def model_states(topology: app.Topology) -> dict:
    """What the finished model holds: the histidine tautomers and the disulfide bonds."""
    histidines: dict[str, list[str]] = {}
    for residue in topology.residues():
        if residue.name == "HIS":
            names = {atom.name for atom in residue.atoms()}
            state = _HISTIDINE.get(("HD1" in names, "HE2" in names))
            if state is not None:
                histidines.setdefault(state, []).append(label(residue))
    disulfides = [[label(residue) for residue in sorted((first.residue, second.residue),
                                                         key=lambda residue: residue.index)]
                  for first, second in topology.bonds() if first.name == "SG" and second.name == "SG"]
    return dict(histidines=histidines, disulfides=disulfides)
