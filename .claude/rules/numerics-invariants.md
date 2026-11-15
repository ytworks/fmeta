# Free-energy and funnel invariants

The framework's value is that its binding free energies can be trusted. Every change to
`src/fmeta/` must keep these contracts; the tests and the GPU gates (`scripts/gpu_gates.py`,
procedure in `docs/gpu-runbook.md`) check them.

## Contracts (test and gate criteria)

| Quantity | Criterion | Checked by |
|---|---|---|
| CVs `s`, `rho` vs the numpy reference (`compute_cvs`) | ≤ 1e-9 nm on Reference | `tests/test_forces.py` |
| Wall and bias energy, forces, CVs on the run platform vs Reference | CV ≤ 2e-5 nm; energy ≤ 1e-4, force ≤ 2e-3 (relative) | gate `forces`, `tests/test_platform.py` (`gpu`) |
| CVs and wall energy of a solvated complex while the run platform moves it between periodic cells (dynamics from two placements, and the ligand at the upper wall along the narrowest direction of the box) | ligand and anchors in one molecule of the context; CVs ≤ 2e-3 nm from the numpy CVs of the reported coordinates; wall energy ≤ 0.1 kJ/mol from the walls at those CVs | gate `periodic_images` |
| Bias table vs `openmm.app.Metadynamics` for the same hills | ≤ 1e-6 relative | `tests/test_engine.py` |
| Bias force (both forms) vs a `Continuous2DFunction` table of the same grid values | energy ≤ 1e-10 relative, forces ≤ 1e-8 (Reference) | `tests/test_engine.py` |
| Restart: N + N hills vs 2N in one run | HILLS, COLVAR, DCD frames identical (Reference) | `tests/test_engine.py` |
| Walkers: each walker's grid vs the sum of all HILLS files | ≤ 1e-7 | `tests/test_walkers.py` |
| Toy system: dG0 vs the exact value (quadrature) | 10 ns: reweighted ≤ 2.0, plateau ≤ 3.0 kJ/mol (gate); 4 × 40 ns: each ≤ 1.5, mean ≤ 0.75 (`slow`) | gate `toy_free_energy`, `tests/test_analysis.py` |
| Toy system without a well | reported as no binding (within 1 kT of the empty funnel) | `tests/test_analysis.py` |
| Automatic funnel on synthetic pockets with a known exit | direction ≤ 10°; with a membrane slab the exit is on the water side | `tests/test_autofunnel.py` |

Never relax a criterion to make a run pass. A criterion changes only by a design amendment
recorded in `docs/design/` and `PROGRESS.md`, with its reason.

## Rules that keep the estimate right (each one came from a failure or a measurement)

- The bias lookup is clamped to the grid. An OpenMM tabulated function is zero outside its
  range, so an unclamped bias vanishes the moment a CV overshoots the grid.
- CV forces (CVs, walls, bias) never use periodic boundaries. OpenMM's periodic wrap of a
  displacement is the nearest image only within half of `a.x`, `b.y` and `c.z` (0.354 L along z
  in the rhombic dodecahedron); beyond that the CVs jumped to another image and 5 of 40 short
  GPU runs ended in NaN. The complex stays in one piece because the centroid forces make the
  ligand and the protein one molecule of the context, which a platform moves between cells as a
  whole. Compute CVs from coordinates as reported or made whole, never wrapped atom by atom.
- The box has an image distance ≥ 2 × (reach + 0.3 nm) for a tumbling solute, or per axis for
  a membrane system: the ligand in the funnel is then nearer to its own protein than to any
  periodic copy. The funnel is therefore placed BEFORE solvation and the box is sized from it;
  never size a box from the solute alone.
- The walls act on the ligand centroid. "Every ligand atom inside the wall" is a rule for the
  input pose, which sets the funnel width. A structure from dynamics is judged by its wall
  energy (an error above 10 kT, `WALL_BREACH_KT`): with the atom rule there, the first full
  gate run failed on a test ligand that had diffused 0.37 nm while the walls did 0.035 kT.
- Group weights of the CVs are explicit, never particle masses: hydrogen mass repartitioning
  changes masses, and the toy system's anchors are massless.
- `rho` is regularised at the axis (`eps`) and hills are mirrored at `rho = 0` by default.
- dG0 uses the effective cross-section `A_eff` (hard disc plus the thermally populated skin of
  the harmonic wall). The free-energy surface over `(s, rho)` already contains the `2 pi rho`
  measure: do not apply the Jacobian a second time.
- The headline is the plateau mean of dG0(t), not the last value. Reweighting uses the bias a
  sample felt at its own time and the offset c(t); reweighting with the final bias was wrong by
  several kJ/mol on the toy system.
- A result within 1 kT of the empty-funnel value (`empty_funnel_free_energy`) is reported as
  "no binding", never as a binding free energy.
- HILLS is the record of the bias (heights stored times `gamma / (gamma - 1)`, as PLUMED does).
  The checkpoint bundle is replaced atomically; a restart rolls HILLS / COLVAR / DCD back to
  it. With walkers, every walker is rolled back before any of them restarts. A run stops with
  an error before a value that is not finite is written to HILLS or COLVAR or added to the grid.

## Change gate

- A change under `src/fmeta/` to forces, bias, runner, system building or platform handling is
  verified on a GPU machine before merge: `python scripts/gpu_gates.py` on a clean commit.
  Commit the result file and the index entry (`bench/results/`) and note a status change in
  `PROGRESS.md`. CI cannot run these; say so in the PR if it has not been done yet.
- A new CV, restraint or estimator gets a case on the toy system, where the exact answer is
  known, before it is used on a real system.

## Scope (owner decisions)

- OpenMM is the only engine and the only input side: no readers for other MD programs'
  formats (2026-11-15).
- The inputs are a protein PDB and a ligand SDF in its bound pose; everything else is derived.
  Ligand parameters come from the OpenFF / GAFF generators (`openmmforcefields`) or from an
  ffxml the user supplies. Never invent force-field parameters.
- BioSimSpace is GPL v3: its ideas may be reimplemented from the papers and the documented
  behavior; its code is not copied.

## Framework vs examples

`src/fmeta/` stays target-agnostic. A particular protein-ligand system (its files, lengths,
funnel overrides) lives in `examples/<name>/` as configuration; move a helper into the package
only when a second example needs it.
