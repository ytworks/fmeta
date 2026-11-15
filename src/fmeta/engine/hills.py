"""HILLS and COLVAR files in PLUMED's text layout.

HILLS columns: time(ps) proj ext sigma_proj sigma_ext height biasf.
As in PLUMED, the stored height of a well-tempered hill is multiplied by
``biasf/(biasf-1)`` so that summing the stored Gaussians gives -F directly;
``biasf = -1`` marks plain metadynamics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid import BiasGrid

HILLS_HEADER = ("#! FIELDS time proj ext sigma_proj sigma_ext height biasf\n"
                "#! SET multivariate false\n"
                "#! SET kerneltype gaussian\n")
COLVAR_HEADER = "#! FIELDS time proj ext bias wall\n"


def _stored_scale(bias_factor: Optional[float]) -> float:
    return 1.0 if bias_factor is None else bias_factor / (bias_factor - 1.0)


class HillsWriter:
    def __init__(self, path, bias_factor: Optional[float], append: bool = False):
        self.path = str(path)
        self.bias_factor = bias_factor
        new = not (append and os.path.exists(self.path))
        self._handle = open(self.path, "w" if new else "a")
        if new:
            self._handle.write(HILLS_HEADER)
            self._handle.flush()

    def write(self, time: float, s: float, rho: float, sigma_s: float, sigma_rho: float,
              height: float) -> None:
        biasf = -1.0 if self.bias_factor is None else self.bias_factor
        stored = height * _stored_scale(self.bias_factor)
        self._handle.write(f"{time:.6f} {s:.10g} {rho:.10g} {sigma_s:.10g} {sigma_rho:.10g} "
                           f"{stored:.12g} {biasf:.10g}\n")

    def release(self) -> None:
        """Hand buffered rows to the operating system so other processes can read them."""
        self._handle.flush()

    def flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


class ColvarWriter:
    def __init__(self, path, append: bool = False):
        self.path = str(path)
        new = not (append and os.path.exists(self.path))
        self._handle = open(self.path, "w" if new else "a")
        if new:
            self._handle.write(COLVAR_HEADER)
            self._handle.flush()

    def write(self, time: float, s: float, rho: float, bias: float, wall: float) -> None:
        self._handle.write(f"{time:.6f} {s:.8f} {rho:.8f} {bias:.8f} {wall:.8f}\n")

    def flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


@dataclass
class Hills:
    """Hills as read from disk; ``height`` is the true deposited height."""

    time: np.ndarray
    s: np.ndarray
    rho: np.ndarray
    sigma_s: np.ndarray
    sigma_rho: np.ndarray
    height: np.ndarray
    bias_factor: Optional[float]

    def __len__(self) -> int:
        return self.time.size

    def __getitem__(self, item) -> "Hills":
        return Hills(self.time[item], self.s[item], self.rho[item], self.sigma_s[item],
                     self.sigma_rho[item], self.height[item], self.bias_factor)


def _data_lines(path) -> list[str]:
    with open(path) as handle:
        return [line for line in handle if line.strip() and not line.startswith("#")]


def read_hills(path) -> Hills:
    rows = [line.split() for line in _data_lines(path)]
    # A crash can leave a partial last line; drop it.
    if rows and len(rows[-1]) != 7:
        rows.pop()
    data = np.array(rows, dtype=float).reshape(-1, 7)
    biasf = float(data[0, 6]) if data.shape[0] else -1.0
    bias_factor = None if biasf <= 0 else biasf
    height = data[:, 5] / _stored_scale(bias_factor)
    return Hills(data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], height, bias_factor)


def read_colvar(path) -> dict:
    rows = [line.split() for line in _data_lines(path)]
    if rows and len(rows[-1]) != 5:
        rows.pop()
    data = np.array(rows, dtype=float).reshape(-1, 5)
    return dict(time=data[:, 0], s=data[:, 1], rho=data[:, 2], bias=data[:, 3], wall=data[:, 4])


def sum_hills(hills: Hills, grid: BiasGrid, n_hills: Optional[int] = None) -> BiasGrid:
    """Return a copy of ``grid`` (emptied) holding the sum of the first ``n_hills`` hills."""
    out = grid.empty_copy()
    part = hills if n_hills is None else hills[:n_hills]
    if len(part):
        out.add_hills(part.s, part.rho, part.height)
    return out


def truncate_data_file(path, keep_rows: int = None, max_time: float = None) -> int:
    """Keep the header plus the first ``keep_rows`` data rows (or rows with
    ``time <= max_time``). Returns the number of rows kept."""
    header, rows = [], []
    with open(path) as handle:
        for line in handle:
            (header if line.startswith("#") else rows).append(line)
    rows = [r for r in rows if r.endswith("\n") and r.strip()]
    if keep_rows is not None:
        rows = rows[:keep_rows]
    if max_time is not None:
        rows = [r for r in rows if float(r.split()[0]) <= max_time + 1e-9]
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as handle:
        handle.writelines(header + rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(rows)


class HillsFollower:
    """Reads the hills another walker appends to its HILLS file, a batch at a time."""

    def __init__(self, path):
        self.path = str(path)
        self.offset = 0
        self.rows = 0

    def read_new(self) -> Optional[Hills]:
        """Hills written since the last call (complete lines only), or ``None``."""
        if not os.path.exists(self.path):
            return None
        size = os.path.getsize(self.path)
        if size < self.offset:        # the file was rolled back under us: do not re-read
            self.offset = size
            return None
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        end = data.rfind(b"\n")
        if end < 0:
            return None
        self.offset += end + 1
        rows = [line.split() for line in data[:end + 1].decode().splitlines()
                if line.strip() and not line.startswith("#")]
        rows = [row for row in rows if len(row) == 7]
        if not rows:
            return None
        data = np.array(rows, dtype=float)
        biasf = float(data[0, 6])
        bias_factor = None if biasf <= 0 else biasf
        self.rows += len(rows)
        return Hills(data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4],
                     data[:, 5] / _stored_scale(bias_factor), bias_factor)


def merge_hills(parts: list) -> Hills:
    """Hills of several walkers as one time-ordered series."""
    parts = [p for p in parts if len(p)]
    if not parts:
        empty = np.zeros(0)
        return Hills(empty, empty, empty, empty, empty, empty, None)
    time = np.concatenate([p.time for p in parts])
    order = np.argsort(time, kind="stable")
    stack = lambda name: np.concatenate([getattr(p, name) for p in parts])[order]   # noqa: E731
    return Hills(time[order], stack("s"), stack("rho"), stack("sigma_s"), stack("sigma_rho"),
                 stack("height"), parts[0].bias_factor)
