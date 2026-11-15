"""Write the funnel as pseudo-atoms so it can be inspected in any molecular viewer."""

from __future__ import annotations

import numpy as np

from .geometry import Funnel, funnel_frame
from .validate import funnel_points


def write_funnel_pdb(path, funnel: Funnel, positions: np.ndarray, spacing: float = 0.2,
                     n_ring: int = 16) -> int:
    """Write rings of pseudo-atoms on the funnel wall, and the axis, to a PDB file.

    Residue ``FUN`` holds the wall (element He), ``AXS`` the axis from p0 to the
    upper wall, and ``WAL`` the lower and upper walls. Load it together with
    the structure; coordinates are in the same frame. Returns the atom count.
    """
    positions = np.asarray(positions, dtype=float)
    origin, axis, _ = funnel_frame(funnel, positions)
    s_values = np.arange(funnel.lower, funnel.upper + 0.5 * spacing, spacing)
    wall = funnel_points(funnel, positions, s_values, n_ring=n_ring, include_axis=False)
    axis_points = origin + np.arange(0.0, funnel.upper + 0.5 * spacing, spacing)[:, None] * axis
    caps = funnel_points(funnel, positions, [funnel.lower, funnel.upper], n_ring=n_ring)

    lines, serial = ["REMARK  funnel pseudo-atoms written by fmeta"], 0
    for residue, element, points in (("FUN", "HE", wall), ("AXS", "NE", axis_points),
                                     ("WAL", "AR", caps)):
        for x, y, z in points * 10.0:    # nm -> angstrom
            serial += 1
            lines.append(f"HETATM{serial % 100000:5d} {element:<4s} {residue} X{1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
    lines.append("END")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return serial
