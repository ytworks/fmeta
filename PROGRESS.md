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

Snapshot date: 2026-11-16. Decisions: `docs/design/engine.md` (FM-D1..D13 Accepted; FM-D11
amended twice). Everything below is on PR #1 (not merged).
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
| Official image, Docker-only execution (FM-D13) | In use on the GPU machine; the CI job `docker` and the publish workflow follow |
| GPU plan M0: first gate record (R1) | In progress |
| GPU plan M1–M7: run modes, real inputs, presets, examples, benchmarks (R2, R3, R7–R10) | Not started |

- GPU gates: **none recorded** (`bench/results/` is empty). The gate script and its criteria
  exist (`scripts/gpu_gates.py`, `docs/gpu-runbook.md`).
- CPU evidence, in the image (lock d061e45a…, on the source of this commit before it was
  committed): `ruff check .` passes; `pytest` 76 passed. `fmeta check --platform CPU --quick`
  15 passed, 0 failed, 0 skipped on 20c43af (`src/` is unchanged since). `pytest -m slow` not
  run on this commit (6 passed on aadd0ad, in the image). CI has not run on the commits after
  aadd0ad (see Blockers); it was green on 171220a.
- Not verified anywhere yet:
  - any GPU platform (CUDA / OpenCL): the first runs on this machine showed failures, and
    their fixes follow in the next commits; nothing is recorded;
  - a binding free energy of a real protein-ligand system; input from a real PDB entry;
    the membrane-plane estimate on a real membrane protein;
  - reading HILLS with PLUMED itself.
- Known gaps: the bias table is refit on the CPU at every hill (about 0.5 µs per grid point;
  R4); ligand protonation is taken from the SDF as drawn (R5); unresolved loops are not
  modelled; non-protein components are removed unless listed in `system.keep_hetero`.
- Blockers: pushes from the GPU machine fail (the session has no git credentials), so the
  commits after aadd0ad are local: CI and PR #1 do not see them.

## Change Log

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
