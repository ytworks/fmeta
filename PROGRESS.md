# Progress

Cross-session memory. Rules: `.claude/rules/progress-log.md`.
Older Change Log entries: `docs/progress/archive/change-log.md`.

## Product

**Goal** (owner request 2026-11-15; roadmap `docs/roadmap.md`): funnel metadynamics on OpenMM
that runs on GPUs, end to end, from two files.

- Inputs: a protein PDB and a ligand SDF in its bound pose. Everything else is derived:
  protein clean-up and hydrogens, ligand parameters (OpenFF / GAFF), funnel, box.
- Soluble proteins in a rhombic dodecahedron; membrane proteins (GPCRs) in a bilayer built by
  `Modeller.addMembrane`, with the funnel pointing into water.
- Engine: well-tempered metadynamics on the funnel CVs (projection `s`, extent `rho`) with
  OpenMM custom forces; replicas for error bars, shared-bias walkers for speed; exact restart.
- Result: standard binding free energy (plateau mean, 95 % confidence interval over replicas)
  with convergence diagnostics that say NOT CONVERGED or "no binding" when that is the case.
- OpenMM only: no input from other MD programs.
- Planned (owner request 2026-11-15, roadmap R7–R10): replicas side by side on several GPUs or
  one after another on one GPU; official presets; a kinase and a GPCR example; benchmarks
  against experiment and FEP.
- Hardware: an 8-GPU machine (since 2026-11-15). Everything except CI runs in the official
  Docker image (FM-D13).

## Current Status

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D13 Accepted; FM-D2, D4,
D5, D6, D7 and D11 amended since, see the Change Log). Everything below is on PR #1 (not merged).
Plan in progress: `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md`.

| Milestone | Status |
|---|---|
| M0 Scaffold | Complete |
| M1 CVs and funnel walls | Complete; equal to numpy references and finite differences |
| M2 Bias engine, HILLS / COLVAR, restart | Complete; table equal to `openmm.app.Metadynamics`, restart byte-identical |
| M3 Analysis and the solvable toy system | Complete; toy dG0 within 0.5 kJ/mol of exact (4 × 40 ns) |
| M4 Automatic funnel, validation | Complete on synthetic pockets (direction ≤ 10°) |
| M5 System building, equilibration, workflow, CLI | Complete; PDB + SDF alone, with automatic clean-up |
| M6 Membrane | Complete on CPU: real `addMembrane` build (POPC, 108 lipids) and a short run; membrane-plane fit on synthetic helix bundles only |
| M7 Replicas, walkers, figures, `fmeta check` | Complete on CPU |
| Harness: CPU CI + GPU gates | Introduced 2026-11-15 (PR #1) |
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine; the CI job `docker` and the publish workflow have never run |
| GPU plan M0: first gate record (R1) | Gate record done (all 16 gates on 2c4ad4e). Remaining: a run with the fmeta inside the image, without the mounted source |
| GPU plan M1–M7: run modes, real inputs, presets, examples, benchmarks (R2, R3, R7–R10) | Not started |

- GPU gates: **all 16 passed** on clean 2c4ad4e (CUDA, mixed precision, 2 GPUs, in the image):
  `bench/results/2026-11-15-2c4ad4e50-gpu-gates.json` (dated in UTC). The commits after
  2c4ad4e change no gated path (documents and evidence only).
- CPU evidence, in the image (lock d061e45a…, the source of 2c4ad4e): `ruff check .` passes;
  `pytest` 81 passed; `pytest -m slow` 6 passed. CI has not run on the commits after aadd0ad
  (see Blockers); it was green on 171220a.
- Not verified yet:
  - a binding free energy of a real protein-ligand system; input from a real PDB entry;
    the membrane-plane estimate on a real membrane protein;
  - reading HILLS with PLUMED itself;
  - the CI job `docker` and the publish workflow (never run).
- Known gaps: ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`; a
  directory prepared before 2026-11-16 keeps periodic funnel walls in `prepared/system.xml` and
  must be prepared again; the gate `toy_free_energy` passes with 0.04 kJ/mol to spare (its
  seed is the worst of 12 on CUDA mixed, see the Change Log), so a change of the trajectory
  (another GPU, another OpenMM) can fail it without a defect.
- Blockers: pushes from the GPU machine fail (the session has no git credentials), so the
  commits after aadd0ad are local: CI and PR #1 do not see them.

## Change Log

### 2026-11-16 — [progress] First GPU gate record (R1): all 16 gates passed on 2c4ad4e
- What: `scripts/gpu_gates.py` in the official image (image 6f68772d…, lock d061e45a…) on two RTX PRO 6000 Blackwell (host GPUs 6 and 7), CUDA, mixed precision, clean 2c4ad4e, 211 s: `pytest -m gpu` 2 passed and the 15 items of `fmeta check` passed. Measured: forces against Reference CV 3.8e-07 nm, energy 7.2e-06, force 5.6e-06; `periodic_images` CVs 2.3e-06 nm, wall energy 5.1e-03 kJ/mol; toy 10 ns reweighted 1.96 and plateau 1.49 kJ/mol from exact (criteria 2.0 and 3.0), 24 rebinding events; 24062 atoms at 1343 ns/day plain and 1293 with the bias; membrane (35500 atoms) 803 ns/day. Because the toy margin is 0.04 kJ/mol, the same 10 ns run was repeated with 12 seeds per platform (diagnostic, not evidence). Error of the reweighted estimate, mean ± sd and largest: CUDA mixed +0.96 ± 0.64, 1.96; CUDA double +0.61 ± 0.52, 1.18; Reference +0.44 ± 0.59, 1.52 kJ/mol. The gate's seed is the worst of the 12 on CUDA mixed. The criterion is unchanged. `pytest` 81 passed, `pytest -m slow` 6 passed on the same source.
- Refs: `bench/results/2026-11-15-2c4ad4e50-gpu-gates.json` (dated in UTC), `bench/results/index.json`; `docs/roadmap.md` (R1 done); `README.md` (検証状況).

### 2026-11-16 — [amendment] FM-D7: after dynamics the ligand is judged by the funnel walls, not atom by atom
- What: the first full gate run (clean 81efa0d, CUDA, 2 GPUs, 193 s) gave 14 passed, 1 failed (`membrane_run`); nothing was recorded. The stage `equilibrate` had stopped with "ligand atoms lie 0.02 nm outside the funnel wall". Now `validate_funnel(..., sample_kT=...)` judges a structure from dynamics by the wall energy of the ligand centre: an error above 10 kT (`WALL_BREACH_KT`); atoms beyond the wall and a centre within 0.2 nm of the lower wall are warnings. The checks of the input pose at preparation and the gate criteria are unchanged. `pytest` 81 passed; the two new tests fail on the code before. Gates not rerun yet.
- Why: measured on the failed run. The benzene test ligand had diffused 0.37 nm in 70 ps (protein CA RMSD 0.08 nm); its centre was 0.11 nm inside the wall and one atom 0.017 nm outside; the walls did 0.088 kJ/mol on average (0.035 kT). The walls act on the centroid, so the atom rule on one snapshot was decided by diffusion, not by the funnel. Rejected: a wider funnel or a shorter equilibration in the gate (a user's run would still stop); warnings only (a ligand the walls do not hold must stop the pipeline).
- Refs: FM-D7; `.claude/rules/numerics-invariants.md`; `src/fmeta/funnel/validate.py`, `src/fmeta/workflow.py`; `tests/test_autofunnel.py`, `tests/test_system.py`.

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
