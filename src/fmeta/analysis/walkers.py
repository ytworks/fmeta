"""The bias each walker of a shared bias had, rebuilt from the HILLS files and the SYNC records.

Walkers read each other's hills as wall-clock time allows, so at one simulation time two
walkers hold different sums of hills. A sample is therefore weighted with the bias of its own
walker: its own hills so far, plus the rows of every other HILLS file that its SYNC record says
were on its grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..engine.grid import BiasGrid
from ..engine.hills import read_sync, row_checksums


def hills_on_grid(sources: list, parts: list) -> Optional[list]:
    """For each walker an integer table ``[k, j]``: how many hills of walker ``j`` were on its
    grid after ``k`` hills of its own (``k`` from 0 to its hill count).

    ``sources`` are the walker directories and ``parts`` their hills. An entry is -1 where the
    bias cannot be rebuilt: the rows the walker had read are no longer in the file (a restart
    removed them and other hills took their place), or they were read before the record began.
    None when a walker kept no record at all (a run from before the record existed). A run
    without walkers needs no record: its table is the count of its own hills."""
    names = [Path(directory).name for directory in sources]
    sums = [row_checksums(Path(directory) / "HILLS") for directory in sources] if len(sources) > 1 else []
    tables = []
    for a, (directory, own) in enumerate(zip(sources, parts)):
        table = np.repeat(np.arange(len(own) + 1)[:, None], len(sources), axis=1)
        if len(sources) > 1:
            record = read_sync(Path(directory) / "SYNC")
            if record is None:
                return None
            n_hills, seen = record
            # The line in force after k own hills: the last one written at or before them. A
            # restart cuts the record to its checkpoint and goes on from there, so the counts
            # never decrease along the file.
            line = np.searchsorted(n_hills, np.arange(len(own) + 1), side="right") - 1
            for j, name in enumerate(names):
                if j != a:
                    rows = np.array([_rows_on_disk(entry.get(name), sums[j]) for entry in seen] + [-1], dtype=int)
                    table[:, j] = rows[line]        # line -1 (before the record began) takes the last entry
        tables.append(table)
    return tables


def _rows_on_disk(entry: Optional[tuple], checksums: np.ndarray) -> int:
    """The rows of a record entry when the file still starts with them, else -1. A walker that
    a line does not name had no hills on the grid."""
    rows, crc = entry or (0, 0)
    if rows == 0:
        return 0
    return rows if rows <= checksums.size and checksums[rows - 1] == crc else -1


class SharedBias:
    """The grid of one walker as hills are counted in: ``at(counts)`` holds the first
    ``counts[j]`` hills of every walker ``j``. Counts that only grow reuse the sum at hand."""

    def __init__(self, grid: BiasGrid, parts: list):
        self._grid, self._parts = grid, parts
        self._work, self._done = grid.empty_copy(), [0] * len(parts)

    def at(self, counts) -> BiasGrid:
        if any(count < done for count, done in zip(counts, self._done)):
            self._work, self._done = self._grid.empty_copy(), [0] * len(self._parts)
        for j, count in enumerate(counts):
            if count > self._done[j]:
                part = self._parts[j][self._done[j]:int(count)]
                self._work.add_hills(part.s, part.rho, part.height)
                self._done[j] = int(count)
        return self._work
