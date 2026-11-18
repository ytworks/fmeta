"""The saved DCD stride, coordinates and periodic cell are required for contact checks."""

import runpy
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit


def test_structural_reader_uses_only_requested_prefix_and_preserves_tilted_box(tmp_path):
    dcd_frames = runpy.run_path(str(Path(__file__).parents[1] / "scripts/paper_regions.py"))["dcd_frames"]
    top = app.Topology()
    residue = top.addResidue("LIG", top.addChain())
    top.addAtom("C1", app.element.carbon, residue)
    top.addAtom("C2", app.element.carbon, residue)
    box = [[2., 0., 0.], [0.27, 2.3, 0.], [-0.14, 0.33, 2.7]]
    vectors = tuple(mm.Vec3(*v) for v in box) * unit.nanometer
    top.setPeriodicBoxVectors(vectors)
    coordinates = np.array([[0.91, -0.12, 0.03], [1.04, 0.03, -0.09]])
    path = tmp_path / "traj.dcd"
    with path.open("wb") as handle:
        dcd = app.DCDFile(handle, top, 0.004 * unit.picosecond, 0, 25000)
        for offset in (0., 0.15):
            dcd.writeModel(coordinates + offset, periodicBoxVectors=vectors)
    [(result, parsed_box)] = list(dcd_frames(path, 2, 1))
    np.testing.assert_allclose(result, coordinates, atol=1e-7)
    np.testing.assert_allclose(parsed_box, box, atol=1e-7)
    with pytest.raises(ValueError, match="incomplete periodic DCD"):
        list(dcd_frames(path, 2, 3))
    with pytest.raises(ValueError, match="DCD layout"):
        list(dcd_frames(path, 3, 1))
