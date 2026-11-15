# fmeta

fmeta is a funnel-metadynamics framework on top of OpenMM. From a protein PDB and a ligand SDF
in its bound pose it places the funnel, builds the system (water, or a lipid bilayer with
`Modeller.addMembrane`), equilibrates, runs well-tempered metadynamics on the GPU and reports the
standard binding free energy. Package: `src/fmeta/`.

## Session start

ALWAYS read `PROGRESS.md` first — it is the cross-session memory of design changes,
milestone status, gate evidence and blockers.

## Progress log rules

@.claude/rules/progress-log.md

## Free-energy and funnel invariants

@.claude/rules/numerics-invariants.md

## Test policy

Apply these rules whenever a change adds, removes, moves or reclassifies behavior or tests.

1. Keep only tests that protect current behavior from a concrete regression: necessary and
   sufficient, not maximal. Do not mirror constants or restate an invariant that product code
   already enforces at construction (e.g. the checks in `Funnel.__post_init__`, unknown
   configuration keys).
2. Test observable results: agreement with the exact binding free energy of the toy system,
   with numpy references and finite differences, with `openmm.app.Metadynamics`, bit-exact
   restarts, end-to-end behavior of prepare / equilibrate / run / analyze.
3. `pytest` (the default selection) must pass on a CPU-only machine without conda-only
   packages: that is what CI runs. `-m slow` holds the long CPU runs (multi-replica toy runs,
   the full pipeline, the real `addMembrane` build); run the ones a change touches before
   merging. `-m gpu` needs a GPU and is verified with the GPU gates, not by CI.
4. A verdict depends only on the code under test: never on wall-clock time, speed ratios or
   runner load. Use deterministic quantities (fixed seeds, exact identities, counts); use
   timeouts only to detect hangs; record timing as a diagnostic (the `benchmark` item of
   `fmeta check` never decides a verdict).
5. Statistical checks use fixed seeds and explicit tolerances taken from measured scatter
   (see the toy free-energy tests and the numbers in `docs/design/engine.md`, FM-D12).
6. A regression test for a bug must fail on the code before the fix; check that once.
7. When a feature is deleted, delete its tests and helpers with it; file moves are not coverage.

## Where things live

- Package: `src/fmeta/` — `funnel/` (CVs, walls, automatic placement, validation), `engine/`
  (bias, runner, equilibration, platform), `system/` (inputs, building, membrane orientation),
  `analysis/`, `workflow.py` and `cli.py` (the stages), `check.py` (acceptance checks)
- The solvable toy system and the small real test system: `src/fmeta/testing/`
- Example configurations for real targets: `examples/<name>/`
- Design decisions (D-IDs, `Status:` lines): `docs/design/*.md`
- Roadmap: `docs/roadmap.md`; implementation plans: `docs/plans/YYYY-MM-DD-<topic>.md`
- GPU gate procedure: `docs/gpu-runbook.md`; recorded evidence: `bench/results/` (`index.json`)
- Archived progress history: `docs/progress/archive/`

## Dev commands

- CPU (any machine): `pip install -e ".[dev,prep,plots]"`, then `ruff check .`,
  `python scripts/check_progress_size.py`, `python scripts/verify_bench_results.py`, `pytest`
  (what CI runs). Long CPU tests: `pytest -m slow`.
- GPU (conda environment from `environment.yml`): `python scripts/gpu_gates.py`
  (see `docs/gpu-runbook.md`).
- Before a commit: run the commands above and read the exit status of the command itself.
  A pipe (`pytest | tail`) reports the status of its last stage and hides a failure.
- Commit in small steps and push each one; the PR is the review unit.
