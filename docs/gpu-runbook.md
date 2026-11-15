# GPU runbook

How to verify a change on a GPU machine. CI is CPU-only (lint, PROGRESS.md budget, evidence
index, CPU tests); everything that needs a GPU or the conda-only packages runs here. Criteria:
`.claude/rules/numerics-invariants.md`.

## Prerequisites

- An NVIDIA GPU with a working driver (or another GPU that OpenMM's OpenCL / HIP platform supports).
- The conda environment: `conda env create -f environment.yml && conda activate fmeta &&
  pip install -e ".[dev]"`. It brings the CUDA build of OpenMM and the ligand-parameter
  generators (`openmmforcefields`, `openff-toolkit`, `ambertools`).
- The GPUs must be free: `nvidia-smi --query-compute-apps=pid,process_name --format=csv` shows
  no other jobs.

## Procedure

1. Commit the change (the gates record the commit). The record is flagged dirty when `src/`,
   `tests/`, `scripts/`, `examples/`, `pyproject.toml` or `environment.yml` differ from it.
2. `python scripts/gpu_gates.py` (10–20 minutes expected; add `--gpus 0,1` to exercise the
   multi-GPU launcher):
   - `gpu-tests`: `pytest -m gpu`;
   - the items of `fmeta check` (`src/fmeta/check.py`, criteria in `CRITERIA` there): `platform`,
     `forces`, `toy_restart`, `toy_free_energy`, `pdb_sdf_openff`, `pdb_sdf_gaff`, `prepare`,
     `periodic_images`, `equilibrate`, `production`, `walkers`, `analysis`, `benchmark`,
     `membrane_build`, `membrane_run`.
   A skipped item is not a pass: the gates need the conda environment.
   While iterating use `fmeta check --quick` or `scripts/gpu_gates.py --quick` (not recorded).
3. On ALL PASSED the script writes `bench/results/<date>-<commit>-gpu-gates.json` and updates
   `bench/results/index.json`. Commit both (CI validates the index), and update `PROGRESS.md`
   → Current Status if the gate status changed.
4. On failure: `gates_work/gpu-gates-not-recorded.json` lists every item with its detail, and
   `gates_work/<item>.traceback.txt` holds the traceback. Never relax a criterion to pass; a
   criterion changes only by a design amendment recorded in `docs/design/` and `PROGRESS.md`.

## Diagnostics

Wall time and the `benchmark` item (ns/day with and without the bias, and the cost of a bias
table update) are recorded but never decide a verdict. `fmeta doctor` prints the platforms,
the optional packages and the table-update cost without running anything long.

## Production runs (reported, not gated)

Binding free energies of real systems (`examples/<name>/`) need hundreds of nanoseconds per
replica and are not analytic: they are reported, not gated. Run them on a clean commit:

```bash
fmeta prepare examples/trypsin_benzamidine/fmeta.yaml -o run && fmeta funnel run
fmeta equilibrate run && fmeta run run --gpus 0,1,2,3 && fmeta analyze run --plot
```

Look at `run/prepared/funnel.pdb` with the structure before spending GPU time. Record the
outcome (`run/summary.json`, the commit, what converged and what did not) in the example's
README and in `PROGRESS.md`.
