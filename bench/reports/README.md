# Benchmark reports

What `fmeta benchmark report <set> -o <workdir>` writes (design: FM-D17, `docs/design/engine.md`):

- `<UTC date>-<commit>-<set>.json`: per ligand the binding free energy of fmeta (mean of the
  replicas, 95% interval), the experimental and the FEP value, the rebinding events and the
  diagnostics of the replicas; the agreement with experiment (MUE, RMSE, mean signed error, the
  same with the mean offset removed, Pearson r, Kendall tau, each with a bootstrap interval)
  over every ligand with a value of fmeta, with FEP on the same ligands; the commit, the image
  and the lock of the code that wrote it and of the calls that ran the simulations.
- the same as text, `.md`, in kcal/mol and in kJ/mol.
- `index.json`: one entry per report.

A report states what was measured. It has no pass / fail field: the criteria of the framework
are the GPU gates, whose evidence is in `bench/results/`. `scripts/verify_bench_results.py`
(CI) checks both directories. A replica of the default length does not converge on its own, so
a report has no "converged" column either: the replicas of a ligand together are the
measurement, and the interval says what their scatter leaves open (FM-D9, second amendment).

The set definitions are in `examples/benchmarks/<set>/`.

## Pilot reports

Three ligands per set ran before the scan of the bias gave the defaults: 5 replicas of 20 ns
with the hill widths that were the default then (`metad.sigma_s` 0.025 and `sigma_rho`
0.05 nm), on commits eeb5302, c41c08b and f2ea653. They measured the throughput and showed
defects. Nothing was selected from them, and they are not results of the official defaults.

| Report | Set | Ligands |
|---|---|---|
| `2026-11-16-45055cc99-tyk2` | TYK2 | ejm_44, ejm_31, ejm_46 |
| `2026-11-16-80bf2209b-a2a` | A2A | 4q, 4d, 4e D |

The set definition of a pilot report is that of `examples/benchmarks/<set>/` with one
difference, `config: {metad: {sigma_s: 0.025, sigma_rho: 0.05}}`: the `set.yaml` hash in the
report is therefore not that of the repository. The analysis is that of the commit the report
names.
