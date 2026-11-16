# Benchmark reports

What `fmeta benchmark report <set> -o <workdir>` writes (design: FM-D17, `docs/design/engine.md`):

- `<UTC date>-<commit>-<set>.json`: per ligand the binding free energy of fmeta (mean of the
  replicas, 95% interval), the experimental and the FEP value, the rebinding events and what the
  analysis found; the agreement with experiment (MUE, RMSE, mean signed error, the same with the
  mean offset removed, Pearson r, Kendall tau, each with a bootstrap interval) over all ligands
  and over the converged ones; the commit, the image and the lock of the code that wrote it and
  of the calls that ran the simulations.
- the same as text, `.md`, in kcal/mol and in kJ/mol.
- `index.json`: one entry per report.

A report states what was measured. It has no pass / fail field: the criteria of the framework
are the GPU gates, whose evidence is in `bench/results/`. `scripts/verify_bench_results.py`
(CI) checks both directories.

The set definitions are in `examples/benchmarks/<set>/`.
