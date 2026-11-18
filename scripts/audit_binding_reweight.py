"""Replay the finite-site primary binding estimate independently from HILLS, SYNC and COLVAR.

The published midpoint-slice and thinning rules are fixed inputs to the comparison. The audit
uses SciPy natural-spline node derivatives, fixed-point MBAR, and numerical radial quadrature.
It imports no fmeta code. All internal energies are in kJ/mol, reported binding values in kcal/mol.
"""

import argparse
import hashlib
import json
import os
import zlib
from pathlib import Path

import numpy as np
import yaml
from scipy.constants import Avogadro, R
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from scipy.special import expit, logsumexp


def histories(paths):
    hills, checksums = [], []
    for path in paths:
        rows, crc, checks = [], 0, [0]
        with (path / 'HILLS').open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    break
                crc = zlib.crc32(line, crc)
                if line.strip() and not line.startswith(b'#'):
                    rows.append(line.split())
                    checks.append(crc)
        h = np.asarray(rows, dtype=float)
        if h.shape[1] != 7 or not np.isfinite(h).all() or np.any(np.diff(h[:, 0]) <= 0):
            raise ValueError(f'invalid HILLS in {path.name}')
        h[:, 5] *= np.where(h[:, 6] > 1, (h[:, 6] - 1) / h[:, 6], 1)
        hills.append(h)
        checksums.append(np.array(checks, dtype=np.uint32))
    return hills, checksums


def sync_table(path, own, paths, hills, checksums):
    cycles = np.arange(len(hills[own]) + 1)
    if len(paths) == 1:
        return cycles[:, None]
    names = {p.name: j for j, p in enumerate(paths)}
    records, vectors = [], []
    with (path / 'SYNC').open() as stream:
        for line in stream:
            if not line.endswith('\n') or line.startswith('#') or not line.strip():
                continue
            cycle, *peers = line.split()
            vector = np.zeros(len(paths), dtype=int)
            for entry in peers:
                name, count, crc = entry.split(':')
                j, n = names[name], int(count)
                if n and (n >= checksums[j].size or int(checksums[j][n]) != int(crc)):
                    vector[j] = -1
                else:
                    vector[j] = n
            records.append(int(cycle))
            vectors.append(vector)
    if not records or np.any(np.diff(records) < 0):
        raise ValueError(f'absent or unordered SYNC in {path.name}')
    vectors.append(np.full(len(paths), -1))
    active = np.searchsorted(records, cycles, side='right') - 1
    table = np.asarray(vectors)[active]
    table[:, own] = cycles
    return table


def hermite(surface, x_nodes, y_nodes, s, rho):
    """Tensor-product natural cubic, using derivatives at grid nodes rather than B-spline coefficients."""
    dx = CubicSpline(x_nodes, surface, axis=1, bc_type='natural')(x_nodes, 1)
    dy = CubicSpline(y_nodes, surface, axis=0, bc_type='natural')(y_nodes, 1)
    dxy = CubicSpline(y_nodes, dx, axis=0, bc_type='natural')(y_nodes, 1)

    def weights(nodes, coordinate):
        i = np.clip(np.searchsorted(nodes, coordinate, side='right') - 1, 0, len(nodes) - 2)
        h = nodes[i + 1] - nodes[i]
        u = (np.clip(coordinate, nodes[0], nodes[-1]) - nodes[i]) / h
        return i, ((1 - u)**2 * (1 + 2*u), u*u * (3 - 2*u)), (h*u*(1 - u)**2, h*u*u*(u - 1))

    ix, hx, gx = weights(x_nodes, s)
    iy, hy, gy = weights(y_nodes, rho)
    value = np.zeros(len(s))
    for a in (0, 1):
        for b in (0, 1):
            index = (iy+b, ix+a)
            value += (hx[a] * hy[b] * surface[index] + gx[a] * hy[b] * dx[index]
                      + hx[a] * gy[b] * dy[index] + gx[a] * gy[b] * dxy[index])
    return value


def raw_grids(hills, prefixes, x, y, reflect):
    surfaces = np.zeros((len(prefixes), len(y), len(x)))
    for source, h in enumerate(hills):
        last, subtotal = 0, np.zeros((len(y), len(x)))
        for index in np.argsort(prefixes[:, source]):
            stop = prefixes[index, source]
            for start in range(last, stop, 512):
                part = h[start:min(start+512, stop)]
                along = np.exp(-0.5*((x - part[:, 1, None])/part[:, 3, None])**2)
                across = np.exp(-0.5*((y - part[:, 2, None])/part[:, 4, None])**2)
                if reflect:
                    across += np.exp(-0.5*((y + part[:, 2, None])/part[:, 4, None])**2)
                subtotal += (across * part[:, 5, None]).T @ along
            surfaces[index] += subtotal
            last = stop
    return surfaces


def fixed_point_weights(bias, counts):
    offset = bias.min(axis=0)
    kernel = np.exp(offset - bias)

    def update(free):
        total = (counts * np.exp(free-free.max())) @ kernel
        updated = -np.log(kernel @ (1 / total))
        return updated - updated[0]

    free = np.zeros(len(counts))
    for _ in range(120):
        new = update(free)
        if np.max(np.abs(new - free)) < 1.e-10:
            break
        free = new
    else:
        # Accelerate the same self-consistency equations in the f[0]=0 gauge.
        fit = root(lambda v: (update(np.r_[0., v])[1:] - v), free[1:], method='krylov',
                   options={'fatol': 1.e-10, 'maxiter': 300})
        free = np.r_[0., fit.x]
    residual = float(np.max(np.abs(update(free) - free)))
    if not np.isfinite(residual) or residual > 1.e-8:
        raise ArithmeticError(f'fixed-point MBAR residual {residual:g}')
    return offset - np.log((counts * np.exp(free - free.max())) @ kernel), residual


def radial_area(s, funnel, kt, eps):
    radius = funnel['buffer'] + funnel['width'] * expit(-funnel['steepness']*(s - funnel['inflection']))
    stiffness = funnel['k_wall'] / kt

    def radial_integrand(r):
        penalty = 0.5 * stiffness * np.maximum(np.hypot(r, eps) - radius, 0)**2
        return 2*np.pi*r*np.exp(-penalty)

    area, _ = quad_vec(radial_integrand, 0, float(radius.max() + 16/np.sqrt(stiffness)),
                       epsabs=1.e-11, epsrel=1.e-10)
    return area


def binding(s, weights, regions, funnel, kt, eps):
    blo, bhi = regions['site']
    ulo, uhi = regions['unbound']
    site, bulk = (s >= blo) & (s < bhi), (s >= ulo) & (s < uhi)
    if not np.any(site) or not np.any(bulk):
        return None
    area = radial_area(s[bulk], funnel, kt, eps)
    log_volume = logsumexp(weights[site]) - logsumexp(weights[bulk] - np.log(area)) + np.log(uhi-ulo)
    return float(-kt * (log_volume - np.log(1.e24 / Avogadro)) / 4.184)


def audit_replica(paths, regions, cut, discard, expected):
    hills, crc = histories(paths)
    metas = [yaml.safe_load((p / 'run.yaml').read_text()) for p in paths]
    meta = metas[0]
    for m in metas[1:]:
        if any(m[k] != meta[k] for k in ('bias', 'funnel', 'grid', 'temperature', 'timestep_ps', 'backend') if k in m):
            raise ValueError('recorded walker conditions differ')
    for h, m in zip(hills, metas, strict=True):
        if not (np.allclose(h[:, 3], m['bias']['sigma_s']) and np.allclose(h[:, 4], m['bias']['sigma_rho'])
                and np.allclose(h[:, 6], m['bias']['bias_factor']) and h[-1, 0] >= cut*1000):
            raise ValueError('HILLS geometry, tempering factor or endpoint differs')
    prefixes, probes, probe_cv, selected = [], [], [], []
    n_slices = max(1, min(int(np.ceil(4*(cut-discard))), 300//len(paths)))
    edges = np.linspace(1000*discard, 1000*cut, n_slices+1)
    mids = (edges[:-1] + edges[1:]) / 2
    unknown = postcut = 0
    limits = [np.searchsorted(h[:, 0], edges[-1]+1.e-6, side='right') for h in hills]
    for i, path in enumerate(paths):
        table = sync_table(path, i, paths, hills, crc)
        data = np.loadtxt(path / 'COLVAR', comments='#', ndmin=2)
        if data[-1, 0] < edges[-1] or np.any(np.diff(data[:, 0]) <= 0) or not np.isfinite(data).all():
            raise ValueError('invalid COLVAR prefix')
        rows = data[(data[:, 0] > edges[0]) & (data[:, 0] <= edges[-1]+1.e-6)]
        actual = table[np.searchsorted(hills[i][:, 0], rows[:, 0], side='left')]
        unknown += int((actual < 0).any(axis=1).sum())
        postcut += int((actual > limits).any(axis=1).sum())
        states = table[np.searchsorted(hills[i][:, 0], mids, side='right')]
        if unknown or (states < 0).any():
            raise ValueError('unknown SYNC prefix; cannot certify this replica')
        ids = np.searchsorted(edges[1:], rows[:, 0], side='left').clip(0, n_slices-1)
        selected.append(np.column_stack([rows[:, 1:3], ids + len(prefixes)]))
        prefixes.extend(states)
        at = np.linspace(0, len(rows)-1, 4, dtype=int)
        probes.extend(actual[at])
        probe_cv.extend(rows[at, 1:4])
    samples = np.concatenate(selected)
    samples = samples[::int(np.ceil(len(samples)/40000))]
    occupied, counts = np.unique(samples[:, 2].astype(int), return_counts=True)
    prefixes = np.concatenate([np.asarray(prefixes)[occupied], probes])
    grid = meta['grid']
    x = np.linspace(grid['s_min'], grid['s_max'], grid['n_s'])
    y = np.linspace(0, grid['rho_max'], grid['n_rho'])
    surfaces = raw_grids(hills, prefixes, x, y, grid['reflect'])
    kt = R * meta['bias']['temperature'] / 1000
    reduced = np.empty((len(counts), len(samples)))
    for i, surface in enumerate(surfaces[:len(counts)]):
        reduced[i] = hermite(surface, x, y, samples[:, 0], samples[:, 1]) / kt
    errors = []
    for surface, (s, rho, logged) in zip(surfaces[len(counts):], probe_cv, strict=True):
        errors.append(float(hermite(surface, x, y, np.array([s]), np.array([rho]))[0] - logged))
    del surfaces
    weights, residual = fixed_point_weights(reduced, counts)
    dg = binding(samples[:, 0], weights, regions, meta['funnel'], kt, 0.)
    regularized = binding(samples[:, 0], weights, regions, meta['funnel'], kt, meta['funnel']['eps'])
    reference = expected['dG'] / 4.184 if expected['dG'] is not None else None
    gap = dg - reference if dg is not None and reference is not None else None
    if ((dg is None) != (reference is None) or expected['n_mbar_samples'] != len(samples)
            or expected['n_mbar_states'] != len(counts) or expected['peer_postcut_samples'] != postcut
            or expected['unknown_samples'] != unknown):
        raise ValueError('the two analyses used different numeric support or slices')
    return dict(samples=len(samples), states=len(counts), unknown_sync_samples=unknown, postcut_sync_samples=postcut,
                logged_bias_probes=len(errors), max_abs_logged_bias_error_kj_mol=float(np.max(np.abs(errors))),
                fixed_point_residual=residual, reported_kcal_mol=reference, independent_kcal_mol=dg,
                difference_kcal_mol=gap, physical_radial_regularization_kcal_mol=regularized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--expected', required=True, type=Path, help='paper analysis JSON of the same frozen prefix')
    parser.add_argument('--regions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text())
    regions = json.loads(args.regions.read_text())
    run = yaml.safe_load((args.workdir / 'config.yaml').read_text())['run']
    if expected['regions'] != regions or expected['base_seed'] != run['seed']:
        raise ValueError('region definitions or seed differ')
    inputs = {'system': 'prepared/system.xml', 'positions': 'prepared/positions.npy', 'funnel': 'prepared/funnel.yaml',
              'topology': 'prepared/topology.cif', 'equil_state': 'equil/state.xml'}
    hashes = {f'{key}_sha256': hashlib.sha256((args.workdir / path).read_bytes()).hexdigest()
              for key, path in inputs.items()}
    if any(regions[k] != digest for k, digest in hashes.items()) or expected['n_replicas'] != run['replicas']:
        raise ValueError('prepared input digests or replica counts differ')
    records = []
    for i, source in enumerate(expected['replicas']):
        rep = args.workdir / f'rep{i}'
        paths = [rep / f'w{j}' for j in range(run['walkers'])] if run['walkers'] > 1 else [rep]
        if source['n_walkers'] != len(paths):
            raise ValueError('walker count differs')
        records.append(audit_replica(paths, regions, expected['cut_ns'], expected['discard_ns'], source))
    report = dict(audit_commit=os.environ.get('FMETA_CODE_COMMIT'), cut_ns=expected['cut_ns'],
                  discard_ns=expected['discard_ns'], base_seed=run['seed'], prepared_hashes=hashes,
                  independent_replicas=len(records), walkers_per_replica=run['walkers'], replicas=records)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
