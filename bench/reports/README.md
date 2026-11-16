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
