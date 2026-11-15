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
- Hardware: development container is CPU-only; production and the GPU gates run on the owner's
  GPU machine.

## Current Status

Snapshot date: 2026-11-15. Decisions: `docs/design/engine.md` (FM-D1..D12 Accepted).
Everything below is on PR #1 (not merged).

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
| M8 GPU gates, real-system validation (R1–R3) | Not started: needs a GPU machine |

- GPU gates: **none recorded** (`bench/results/` is empty). The gate script and its criteria
  exist (`scripts/gpu_gates.py`, `docs/gpu-runbook.md`).
- CPU evidence: CI green on 171220a (Python 3.10 and 3.12: lint, PROGRESS budget, evidence
  index, `pytest`, 73 tests). `fmeta check --platform CPU --quick` on the same source tree:
  13 passed, 0 failed, 2 skipped (`pdb_sdf_openff`, `pdb_sdf_gaff`: generators not installed).
  `pytest -m slow` (6 tests) passed on 6d6e85e; the head differs from it by lint fixes, CLI
  error reporting and the harness, and has not repeated them yet.
- Not verified anywhere yet:
  - any GPU platform (CUDA / OpenCL), including the periodic-image handling of the CVs there;
  - ligand parameters from the real OpenFF / GAFF generators — the conda-only packages could
    not be installed in the development container; fmeta's side is tested with stand-ins and
    the pipeline with a hand-written benzene ffxml;
  - a binding free energy of a real protein-ligand system; input from a real PDB entry;
    the membrane-plane estimate on a real membrane protein (no access to structure databases);
  - reading HILLS with PLUMED itself.
- Known gaps: the bias table is refit on the CPU at every hill (about 0.5 µs per grid point;
  overhead on the GPU unmeasured, R4); ligand protonation is taken from the SDF as drawn (R5);
  unresolved loops are not modelled; non-protein components are removed unless listed in
  `system.keep_hetero`.
- Blockers: none for development. R1 (first GPU gate record) needs the owner's GPU machine.

## Change Log

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
