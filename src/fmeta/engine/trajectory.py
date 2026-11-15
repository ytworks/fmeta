"""DCD output for an atom subset, with exact truncation for restarts."""

from __future__ import annotations

import os
import struct
from typing import Optional, Sequence

import numpy as np
import openmm.app as app
import openmm.unit as unit


def subset_topology(topology: app.Topology, atoms: Sequence[int]) -> app.Topology:
    """A topology holding only ``atoms`` (bonds are dropped; DCD does not store them)."""
    keep = set(int(i) for i in atoms)
    out = app.Topology()
    out.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    for chain in topology.chains():
        new_chain = None
        for residue in chain.residues():
            new_residue = None
            for atom in residue.atoms():
                if atom.index not in keep:
                    continue
                if new_chain is None:
                    new_chain = out.addChain(chain.id)
                if new_residue is None:
                    new_residue = out.addResidue(residue.name, new_chain, residue.id)
                out.addAtom(atom.name, atom.element, new_residue)
    return out


def _dcd_layout(path) -> tuple[int, int]:
    """Return (header size, bytes per frame) of a DCD written by OpenMM."""
    with open(path, "rb") as handle:
        first = struct.unpack("<i", handle.read(4))[0]
        handle.seek(48)
        has_box = struct.unpack("<i", handle.read(4))[0] != 0
        handle.seek(first + 8)
        title = struct.unpack("<i", handle.read(4))[0]
        atoms_block = first + 8 + title + 8
        handle.seek(atoms_block + 4)
        n_atoms = struct.unpack("<i", handle.read(4))[0]
    header = atoms_block + 12
    frame = (56 if has_box else 0) + 3 * (4 * n_atoms + 8)
    return header, frame


def count_dcd_frames(path) -> int:
    header, frame = _dcd_layout(path)
    return (os.path.getsize(path) - header) // frame


def truncate_dcd(path, n_frames: int, first_step: int, interval: int) -> None:
    """Cut a DCD back to its first ``n_frames`` frames and fix the header."""
    header, frame = _dcd_layout(path)
    with open(path, "r+b") as handle:
        handle.truncate(header + n_frames * frame)
        handle.seek(8)
        handle.write(struct.pack("<i", n_frames))
        handle.seek(20)
        handle.write(struct.pack("<i", first_step + n_frames * interval))


class TrajectoryWriter:
    """Writes selected atoms to a DCD, one frame per call to :meth:`write`."""

    def __init__(self, path, topology: app.Topology, timestep_ps: float, interval_steps: int,
                 atoms: Optional[Sequence[int]] = None, resume_frames: Optional[int] = None):
        self.path = str(path)
        self.atoms = None if atoms is None else np.asarray(sorted(int(i) for i in atoms))
        top = topology if self.atoms is None else subset_topology(topology, self.atoms)
        self._periodic = topology.getPeriodicBoxVectors() is not None
        append = resume_frames is not None and os.path.exists(self.path)
        if append:
            truncate_dcd(self.path, resume_frames, 0, interval_steps)
        self._handle = open(self.path, "r+b" if append else "wb")
        self._dcd = app.DCDFile(self._handle, top, timestep_ps * unit.picosecond, 0,
                                interval_steps, append)
        self.n_frames = resume_frames if append else 0

    def write(self, state) -> None:
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if self.atoms is not None:
            positions = positions[self.atoms]
        box = state.getPeriodicBoxVectors() if self._periodic else None
        self._dcd.writeModel(positions, periodicBoxVectors=box)
        self.n_frames += 1

    def flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()
