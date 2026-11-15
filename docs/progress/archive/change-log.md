# Archived Change Log

Older `PROGRESS.md` Change Log entries, newest first. Frozen: insert only at the marker
(procedure: `docs/progress/archiving.md`).

<!-- ARCHIVE-INSERT-POINT -->

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
