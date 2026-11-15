# Archived Change Log

Older `PROGRESS.md` Change Log entries, newest first. Frozen: insert only at the marker
(procedure: `docs/progress/archiving.md`).

<!-- ARCHIVE-INSERT-POINT -->

### 2026-11-16 — [amendment] FM-D2, FM-D5, FM-D6: CV forces without periodic boundaries; gate `periodic_images` redefined; runs stop before recording NaN
- What: the forces of the CVs, the walls and the bias no longer use periodic boundaries (the `periodic` argument is gone); the box rule is unchanged. Before: 5 of 40 biased GPU runs of 800 ps on the test system of `fmeta check` ended in NaN. After: the 5 failing seeds pass and so do 40 new runs; started with a corner of the cell inside the complex, the complex changed cell 57 times in 200 ps with the CVs within 8.9e-07 nm of numpy. The gate `periodic_images` now runs dynamics from two placements and probes the upper wall along the narrowest direction of the box (CV tolerance 2e-3 nm unchanged; wall energy within 0.1 kJ/mol): it fails on the periodic forces (1.2e6 kJ/mol), which the old static gate passed. `forces` loses its seventh position, a periodic image of another one. A run now stops with an error before a value that is not finite reaches HILLS, COLVAR or the grid. `fmeta check --quick` on CUDA: 15 passed, 0 failed, 0 skipped (not evidence).
- Why: two facts from the first GPU runs. OpenMM's periodic wrap of a displacement is the nearest image only within half of `a.x`, `b.y` and `c.z`, which is 0.354 L along z in the rhombic dodecahedron, so the CVs jumped (`s` from 2.75 to −4.00 nm in one step) in a box that satisfies FM-D6. And a context holds the ligand and the protein as one molecule, because the centroid forces join them, so a platform moves them between cells together. Rejected: periodic CVs in a cube (41% more volume).
- Refs: FM-D2, FM-D5, FM-D6; `.claude/rules/numerics-invariants.md`; `src/fmeta/funnel/forces.py`, `src/fmeta/check.py`, `src/fmeta/engine/runner.py`; `tests/test_engine.py`.

### 2026-11-16 — [amendment] FM-D4 (roadmap R4): the bias table holds spline coefficients; two force forms by platform
- What: fmeta fits the B-spline coefficients of the bias itself after every hill, and the energy expression evaluates the bicubic spline from a `Discrete2DFunction`. The bias is the same function as before (test against a `Continuous2DFunction`: energy 1e-10, forces 1e-8). Measured on the test system of `fmeta check` (24062 atoms, CUDA, mixed precision, 4 fs, one hill per ps): 73.4% of the speed of plain MD before, 96–97% after (1.84 ms per hill). GPU platforms get a `CustomCentroidBondForce` plus a zero-energy force that reads the CVs outside the integration; Reference and CPU get one `CustomCVForce`; a context of the other kind is refused. `benchmark` compares the same stretch with and without the bias and reports the cost of a checkpoint.
- Why: the old form was below the 80% the plan set for starting R4. Two forms because of three measurements on OpenMM 8.6.1: a centroid force with a table and an energy parameter derivative returns zero on CUDA / OpenCL without an error; a `CustomCVForce` in every step costs about 20% on a GPU; a table update of a centroid force takes 0.7–0.9 s on Reference / CPU.
- Refs: FM-D4; `docs/roadmap.md` (R4 done); `src/fmeta/engine/bias.py`, `src/fmeta/engine/grid.py`, `src/fmeta/check.py`; `tests/test_engine.py`, `tests/test_platform.py`.

### 2026-11-15 — [amendment] FM-D2: the forces compute the CVs in the angle form
- What: `s = d cos θ`, `rho = sqrt((d sin θ)² + eps²)` from the distance ligand–p0 and the angle ligand–p0–p1, instead of three distances; the numpy reference uses the dot and the cross product of the same two vectors. Gate item `forces` on CUDA (mixed precision), worst CV deviation from Reference: 3.6e-05 nm before (criterion 2e-5 nm: failed), 3.8e-07 nm after. The criterion is unchanged.
- Why: the first run of the item on a GPU. Near the axis `dL0² − s²` is a difference of nearly equal numbers and loses its digits in single precision.
- Refs: FM-D2; `src/fmeta/funnel/forces.py`, `src/fmeta/funnel/geometry.py`; `tests/test_forces.py`.

### 2026-11-15 — [design] FM-D13: everything except CI runs in one official image built from a lock
- What: `Dockerfile` (base pinned by digest, the environment of `conda-linux-64.lock`, fmeta, the test and lint tools), `scripts/in_docker.sh` (repository mounted, calling user, only the GPUs named), `scripts/update_lock.sh`, rule `.claude/rules/docker-execution.md`; `CLAUDE.md`, the GPU runbook and the README give Docker commands. Gate evidence is written only in the image whose lock is the repository's, and names the image, the lock and the GPU UUIDs. Roadmap: R7–R10 added (run modes, real inputs, presets, examples and benchmarks). The CI job `docker` and a publish workflow (tags and manual runs only) follow in the next commit.
- Why: owner rule of 2026-11-15 (execution and tests run on Docker, except CI). `environment.yml` pins nothing, so evidence taken in a conda environment does not describe what a later user gets.
- Refs: FM-D13; `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`; a4b91eb, fb6bc95, 383b5ec; PR #1.

### 2026-11-15 — [amendment] FM-D11: fmeta names the ligand force field (`openff-2.2.1`, `gaff-2.11`)
- What: when the settings name no ligand force field, fmeta uses `openff-2.2.1` or `gaff-2.11`, with AM1-BCC charges. Before this change, in the official image (the first run with the real ligand packages): `pytest` 74 passed and `pytest -m slow` 6 passed, but `fmeta check --platform CPU --quick` gave 13 passed, 2 failed (`pdb_sdf_openff`, `pdb_sdf_gaff`). With this change: `pytest` 76 passed; `fmeta check --platform CPU --quick` 15 passed, 0 failed, 0 skipped.
- Why: the generators of `openmmforcefields` 0.16 have no default force field, so a PDB and an SDF alone failed; a default taken from the installed packages would change the parameters with the environment.
- Refs: FM-D11; `src/fmeta/system/ligand.py`; `tests/test_inputs.py` (new: `test_default_settings_work_with_the_real_generators`, which fails for both generators on the code before; the two tests that hide the ligand packages also hide the submodule it imports).

### 2026-11-15 — [progress] Harness introduced: rules, progress log, CPU CI, GPU gates with recorded evidence
- What: `CLAUDE.md` / `AGENTS.md`, `.claude/rules/` (progress log, invariants), SessionStart hook, `PROGRESS.md` with its size check, `docs/` (design, roadmap, plans, GPU runbook, archive), ruff, `.github/workflows/ci.yml`, `scripts/gpu_gates.py` and `scripts/verify_bench_results.py`.
- Why: owner request to follow the harness of ytworks/pypic and ytworks/kairyu, so that rules, status and GPU evidence live in the repository rather than in one session.
- Refs: PR #1; FM-D12; `docs/gpu-runbook.md`.

### 2026-11-15 — [amendment] FM-D11: OpenMM only; a protein PDB and a ligand SDF are enough
- What: readers for AMBER / GROMACS / CHARMM input removed. The protein file is reduced to its amino-acid residues, repaired with PDBFixer when installed and protonated; an SDF without hydrogens gets them from RDKit; a ligand not in contact with the protein is refused before building. `fmeta check` gained `pdb_sdf_openff` / `pdb_sdf_gaff`.
- Why: owner decisions of 2026-11-15 ("GROMACS etc. are not needed", "runnable from the protein PDB and the ligand SDF alone; the SDF is in the pocket").
- Refs: FM-D11; 4139aae, 5464ef1, 6d6e85e; PR #1.

### 2026-11-15 — [progress] M7: walkers, membrane orientation, figures, empty-funnel reference, `fmeta check`
- What: shared-bias walkers through HILLS files with rollback before restart; membrane-plane fit (slab search, scored against shuffled sequences) for `orient: check | auto`; lipids-in-funnel check; charged ligands get a longer funnel; results within 1 kT of an empty funnel are reported as no binding; figures; one-command acceptance checks. CPU dry run of `fmeta check`: 13 passed, 2 skipped.
- Refs: FM-D8, FM-D9, FM-D10, FM-D12; a46f616, 32e5776; PR #1.

### 2026-11-15 — [progress] M0–M6 built and verified on CPU
- What: CVs and walls; bias engine with exact restart; analysis; automatic funnel; system building in water and in a bilayer; workflow and command line. Toy system: plateau errors −0.15..+0.50 kJ/mol (4 × 40 ns), reweighted errors −0.12..−0.27 kJ/mol (4 × 20 ns). One commit (47798fc) was pushed with a failing test because a pipe hid the exit status; fixed in 332b15c, and the rule is now in `CLAUDE.md`.
- Refs: FM-D1..D9; 7c0898a..c9465d5; PR #1.

### 2026-11-15 — [design] FM-D1..D12 Accepted: initial design
- What: OpenMM custom forces for CVs, walls and bias; own bias engine with clamped lookup; funnel placed before solvation and the box sized from it; dG0 from the axis profile with the effective cross-section; verification through a solvable toy system, CPU CI and GPU gates.
- Why: the reference recipe gives wrong results in three places if copied as is (bias zero outside its grid, box not sized from the funnel, bound-state length fixed at 1 Å); the development machine has no GPU, so correctness is established on exact answers on CPU and the platform-dependent part is gated on the GPU.
- Refs: `docs/design/engine.md`; `docs/plans/2026-11-15-initial-framework.md`; PR #1.
