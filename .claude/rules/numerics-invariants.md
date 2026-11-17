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
| Nonbonded energy and forces of the equilibrated test systems (rhombic dodecahedron; membrane box), with OpenMM's PME and with the back end of the runs, vs PME at an Ewald tolerance of 1e-6 in double precision; at the equilibrated coordinates and with whole molecules moved by lattice vectors | energy ≤ 1e-4, force ≤ 2e-3 (relative) | gates `backend`, `membrane_backend` |
| CVs, wall and bias energy and forces with the cuPotential back end vs OpenMM's at the same coordinates; `run.backend: auto` where the plugin can take part | CV ≤ 2e-5 nm; energy ≤ 1e-4, force ≤ 2e-3 (relative); `auto` gives cuPotential | gates `backend`, `membrane_backend` |
| Bias table vs `openmm.app.Metadynamics` for the same hills | ≤ 1e-6 relative | `tests/test_engine.py` |
| Bias force (both forms) vs a `Continuous2DFunction` table of the same grid values | energy ≤ 1e-10 relative, forces ≤ 1e-8 (Reference) | `tests/test_engine.py` |
| Restart: N + N hills vs 2N in one run | HILLS, COLVAR, DCD frames identical (Reference) | `tests/test_engine.py` |
| Walkers: each walker's grid vs the sum of all HILLS files | ≤ 1e-7 | `tests/test_walkers.py` |
| Walkers out of step: the bias of every sample, rebuilt from the HILLS files and the walker's `SYNC` record, vs the bias column of COLVAR | ≤ 1e-5 kJ/mol (Reference); one walker after the other, 4 pairs of 2 × 10 ns: `dG_reweighted` each ≤ 2.0, mean ≤ 1.0 kJ/mol from the exact value (`slow`) | `tests/test_walkers.py` |
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
- The headline of a replica is measured as funnel metadynamics is published: dG0(t) from the
  accumulated bias (surface = bias times `-gamma / (gamma - 1)`), averaged over the second half
  of the run; the first half, where the bias changes fastest, is rejected. `dG_error` is the
  standard error of that mean from 5 blocks (owner, 2026-11-16; FM-D9, third amendment).
  Cross-checks from the samples: after the first 2 ns (`discard_ns`; of a run shorter than
  4 ns, its second half) and of the second half alone. A sample is weighted with the bias of
  its own time slice, and the slices are combined by solving for their free energies;
  reweighting with the final bias was wrong by several kJ/mol on the toy system. A difference
  above 4 kJ/mol from the headline is written as a diagnostic, never a reason to pick one.
- With walkers, a sample is weighted with the bias its own walker had: its own hills plus the
  rows of the other HILLS files that its `SYNC` record names (rows read and their checksum, one
  line per read). Walkers read each other as wall-clock time allows; the sum of all hills up to
  the time of a sample was off by 1.1 and 3.3 kJ/mol for toy walkers out of step (review of
  PR #1; FM-D9, fourth amendment). A sample whose bias cannot be rebuilt is left out and
  counted; walkers without a record get no estimate from the samples.
- The measurement is the set of independent replicas: the mean of their dG0 with its standard
  error and 95% interval. A replica of the default length does not converge on its own, so the
  analysis gives no converged / not converged verdict (owner, 2026-11-16; FM-D9, second
  amendment). The checks of a replica (rebinding events, drift, cross-checks, walls, grid) keep
  their thresholds and are reported as diagnostics; they never decide which replicas count.
  On trypsin the replicas of one setting lay between −1.7 and −78 kJ/mol after 40 ns.
- A result within 1 kT of the empty-funnel value (`empty_funnel_free_energy`) is reported as
  "no binding", never as a binding free energy.
- HILLS is the record of the bias (heights stored times `gamma / (gamma - 1)`, as PLUMED does).
  The checkpoint bundle is replaced atomically; a restart rolls HILLS / COLVAR / DCD back to
  it. With walkers, every walker is rolled back before any of them restarts, a walker
  without a checkpoint goes back to the start, and `SYNC` is cut or removed with HILLS. A run stops with an error before a value that is
  not finite is written to HILLS or COLVAR or added to the grid.
- One directory holds one run: a restart under another funnel, bias, grid, time step, system
  or trajectory layout than its `run.yaml` states is refused before a file is touched. Every
  row of HILLS is decoded with the bias factor of the first; with the factor changed from 10 to
  2 in a run, the bias rebuilt from the file was off by 1.09 kJ/mol (review of PR #1).
- The analysis adds every HILLS file of a replica directory, so a run never leaves one out: the
  set of walkers may only grow, and a run that would leave out a walker with hills (or switch
  between walkers and none) is refused before a file is touched. With 3 walkers cut to 2, the
  bias of the analysis was off by 2.74 kJ/mol from the bias of the run (review of PR #1).

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
