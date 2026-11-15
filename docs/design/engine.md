# Engine design

Status: **Accepted** (2026-11-15, PR #1). Since then: FM-D11 amended and FM-D13 added
(2026-11-15: owner decisions, and the first run with the real ligand-parameter packages);
FM-D2, FM-D4, FM-D5, FM-D6 and FM-D7 amended (2026-11-15 and 16: the first runs on a GPU);
FM-D12 amended (2026-11-16: the scatter of the toy gate is on record, criteria unchanged; and by
the owner: operation on a CPU platform is not guaranteed); FM-D14 added (2026-11-16: run modes
and the scheduler).
Applies to: `src/fmeta/` (all of it), `examples/`, the official image and how the project is run.

## Requirements (owner, 2026-11-15)

A funnel-metadynamics framework with OpenMM as its engine; it must run on GPUs. Given a ligand
whose pose and position are known, the funnel is set up automatically (reference: the
BioSimSpace funnel-metadynamics material). Membrane proteins such as GPCRs run with a membrane
built by OpenMM's own membrane builder. Later the same day: OpenMM only (no other MD programs'
formats); a protein PDB and a ligand SDF, with the SDF already placed in the pocket, must be
enough to run.

## Decisions

### FM-D1 — OpenMM is the only engine; funnel and bias are OpenMM custom forces
CVs, walls and bias are `CustomCentroidBondForce` / `CustomCVForce` objects, so they run on the
CUDA and OpenCL platforms with everything else. No PLUMED plugin.
Why: one dependency, the bias stays on the GPU, and CPU CI can check every expression on the
Reference platform.

### FM-D2 — Two CVs from centroid distances, explicit weights, no periodic boundaries
`s = (dL0² + d01² − dL1²) / (2 d01)`, `rho = sqrt(max(dL0² − s², 0) + eps²)` over the groups
ligand / p0 / p1. Group weights are stored in the funnel (ligand heavy atoms by element mass,
anchors as given), never taken from particle masses.
Amendment (2026-11-15): the forces compute the same CVs in the angle form, `s = d cos θ`,
`rho = sqrt((d sin θ)² + eps²)`, with `d` the distance ligand–p0 and `θ` the angle
ligand–p0–p1 (`funnel/forces.py`). The first run of the gate `forces` on a GPU (CUDA, mixed
precision) gave a `rho` 3.6e-05 nm off Reference near the axis, above the criterion of
2e-5 nm: `dL0² − s²` is a difference of nearly equal numbers there and loses its digits in
single precision. The angle form has no such difference (3.8e-07 nm at worst over the same
positions), and `eps` bounds its gradient on the axis. The criterion is unchanged.
Amendment (2026-11-16): the forces of the CVs, the walls and the bias do not use periodic
boundaries. Until now they did, for the reason this decision gave: "GPU platforms wrap
molecules, so the distances must be minimum-image". Two facts from the first runs on a GPU
changed that. (1) OpenMM wraps a displacement one box vector at a time, which gives the nearest
image only while its components stay within half of `a.x`, `b.y` and `c.z`. In the rhombic
dodecahedron `c.z / 2 = 0.354 L`, less than half the image distance, so the box rule of FM-D6
did not protect the CVs. On the test system of `fmeta check` (L = 7.0 nm, upper wall at
2.77 nm) the CVs jumped from `s` = 2.75 nm to −4.00 nm in one step when the ligand passed
2.475 nm along z; the walls answered with 4.7e5 kJ/mol, and 5 of 40 runs of 800 ps ended in
NaN. (2) A context holds the ligand and the protein as one molecule, because the centroid
forces join their groups, so a platform moves them between periodic cells together. Measured
without periodic boundaries (CUDA, mixed precision): in 200 ps started with a corner of the
cell inside the complex the complex changed cell 57 times, never apart from the ligand, with
the CVs within 8.9e-07 nm of the numpy values; the five seeds that failed pass, and so do
40 new runs of 800 ps. Rejected: periodic CVs in a cube of side 2 (reach + 0.3 nm), which
is 41% more volume to simulate.
The gate `periodic_images` is redefined with this. Before, it compared the CVs of three static
placements with the values at build time: that passed on the forces that failed in dynamics,
and its question (do periodic CVs survive a move to another image) no longer exists. Now it
asserts that the ligand and the anchors are in one molecule of the context, runs dynamics from
two placements and puts the ligand at the upper wall along the narrowest direction of the box:
the CVs must be within 2e-3 nm (unchanged) of the numpy CVs of the reported coordinates, and
the wall energy within 0.1 kJ/mol of the walls at those CVs. It fails on the periodic forces
(1.2e6 kJ/mol at the upper wall). The item `forces` loses its seventh position, a periodic
image of another one; its tolerances are unchanged.
Why: the same CVs as `d cos(angle)` / `d sin(angle)` but with a bounded gradient on the axis;
explicit weights survive hydrogen mass repartitioning and massless test anchors; without
periodic boundaries the CVs do not depend on the shape of the box, and the platform keeps the
complex in one piece.

### FM-D3 — Walls: one force for the sigmoid side wall and the lower / upper walls
`R(s) = width / (1 + exp(steepness (s − inflection))) + buffer`; harmonic penalties, default
force constants 10⁴ kJ/mol/nm². The walls live in their own force group.
Why: the BioSimSpace funnel shape, so parameters carry over; a stiff wall keeps the soft-wall
correction of FM-D9 small and the CVs inside the bias grid.

### FM-D4 — Own bias engine on the mechanism of `openmm.app.Metadynamics`
A 2-D table (`Continuous2DFunction`) refreshed after every hill, with the lookup clamped to
the grid and hills mirrored at `rho = 0`; 3 grid points per sigma.
Amendment (2026-11-16, roadmap R4): the table holds the B-spline coefficients of the bias
(`Discrete2DFunction`, (n_s + 3) × (n_rho + 3) entries) and the energy expression evaluates the
bicubic spline from 16 of them. fmeta fits the coefficients itself after every hill (two banded
solves, `engine/bias.py`). The bias is the same function as before, the natural bicubic spline
through the grid values that a `Continuous2DFunction` gives: a test holds the two together
(energy to 1e-10, forces to 1e-8). Measured on the test system of `fmeta check` (24062 atoms,
CUDA, mixed precision, 4 fs, one hill per ps, 19198 grid points): the old form ran at 73.4% of
the speed of plain MD, below the 80% the plan set for starting R4; the new form runs at 96–97%
(1.84 ms per hill).
The force that carries the expression has two forms, chosen from the platform it is built for,
because three measurements (OpenMM 8.6.1) leave no single one. On CUDA and OpenCL a
`CustomCentroidBondForce` with a tabulated function and an energy parameter derivative returns
zero energy and zero forces, without an error. A `CustomCVForce` in every step costs about 20%
of the speed there. On Reference and CPU a table update of a `CustomCentroidBondForce` takes
0.7–0.9 s, against 0.01 ms and 2 ms for a `CustomCVForce`. So GPU platforms get one
`CustomCentroidBondForce` and a second, zero-energy force that only reads the CVs and is left
out of the integration; Reference and CPU get one `CustomCVForce`. A context of the other kind
is refused.
Why: OpenMM tables are zero outside their range (the bias would vanish when a CV overshoots);
hill logging, exact restart and bias sharing need access the stock class does not give. A test
keeps the table equal to the stock class for the same hills. Every hill costs a fit and a
transfer of the whole table, hence the coarse grid (with a `Continuous2DFunction` OpenMM refits
the spline on the CPU at every update, about 0.5 µs per grid point measured: that was the
cost R4 removed).

### FM-D5 — HILLS is the record; one atomic checkpoint bundle; exact restart
PLUMED-layout HILLS / COLVAR. A checkpoint is one zip (binary checkpoint, State XML, bias
grid, counters) replaced atomically; a restart rolls HILLS / COLVAR / DCD back to it.
Amendment (2026-11-16): a run stops with an error before a value that is not finite (a CV, the
bias or the wall energy) is written to COLVAR or HILLS or added to the grid. The first
breakdown on a GPU wrote NaN rows and failed only later, inside the spline fit. Now the files
stay readable and a restart continues from the last checkpoint.
Why: a crash between files must not leave a mixed state; on the Reference platform a split run
reproduces the continuous one byte for byte, which makes restart testable.

### FM-D6 — The funnel is placed before solvation and the box is sized from it
Tumbling solute: image distance ≥ max(bounding sphere + padding, funnel reach + solute radius +
cutoff, 2 (reach + 0.3 nm)), rhombic dodecahedron. Membrane: box height from the funnel, xy
from the lipid patch.
Amendment (2026-11-16): the rule is unchanged; its reason is restated because the CVs are no
longer periodic (FM-D2). The original reason was: "with minimum-image CVs a box smaller than
twice the reach lets the ligand change image before the upper wall".
Why: wherever the walls let the ligand go it is within reach + 0.3 nm of its own protein, and
with an image distance of twice that every periodic copy of the protein is farther away. So
the ligand does not meet a copy of the protein inside the funnel, and analysis that puts a
trajectory together by nearest images finds the ligand in the funnel. A box sized from the
solute alone does not know the funnel.

### FM-D7 — Automatic placement by path clearance, weighted alpha-carbon anchors
Straight paths from the ligand are scored by the solute (and membrane) they cross and by how
soon they reach bulk; anchors are alpha carbons weighted so that the axis lies on that
direction; inflection, walls and width follow the pocket geometry. A BioSimSpace-style grid
average is kept as `method: bss`.
Amendment (2026-11-16): the check of the ligand against the walls depends on where the
structure comes from. The input pose defines the funnel: each of its atoms must be inside the
wall (the width is chosen so that it is) and its centre 0.2 nm clear of the lower wall. That
is unchanged. A structure that has been through dynamics, as in the check after equilibration,
is a thermal sample of the funnel, and the walls act on the ligand centroid: it fails only
when its wall energy is above 10 kT (`WALL_BREACH_KT`); atoms beyond the wall and a centre
near the lower wall are warnings there. Until now the atom rule was applied to that structure
as well. The first full gate run (commit 81efa0d, CUDA) failed on it in `membrane_run`, with
the other 14 items passed: after 70 ps without restraints the benzene of the test system,
which is not a binding mode, had moved 0.37 nm sideways in the protein frame (protein CA RMSD
0.08 nm). Its centroid was 0.11 nm inside the wall (rho 0.47 nm, wall at 0.58 nm) and one atom
0.017 nm outside, while the walls had done 0.088 kJ/mol on average (0.035 kT) and 2.6 kJ/mol
at most. The soluble system of the same run ended 0.14 nm inside the wall. For a ligand that
is free in the funnel the atom rule is decided by where diffusion has put it at the last step,
not by the code: a centroid that fills a disc of 0.58 nm is in the wall's skin about 7% of the
time. The wall energy is decided by the code: 10 kT is 0.07 nm beyond a default wall
(10⁴ kJ/mol/nm²), which a harmonic wall lets a ligand reach with a probability of about 1e-5,
and a ligand 0.1 nm outside is at 50 kJ/mol. Rejected: a wider funnel or a shorter
equilibration in the gate (the outcome would still be chance), and warnings only (a ligand
that the walls do not hold must stop the stage).
Why: the grid average assumes a shallow pocket and treats "no protein" as "water", which fails
for buried sites and inside a bilayer.

### FM-D8 — Membranes with `Modeller.addMembrane`; slab before the build; orientation check
Before the build the bilayer is a slab obstacle; afterwards the lipid atoms are. The box height
is preset from the funnel. The membrane plane is estimated from the protein's hydrophobic belt
to check the input orientation (`check`, default) or to set it (`auto`).
Why: owner requirement to use OpenMM's builder; the funnel must point into water and its
unbound end must clear the head groups; `addMembrane` trusts the input orientation.

### FM-D9 — dG0 from the axis profile with the effective cross-section
`K_b = ∫_site e^{−W/kT} ds / ⟨e^{−W/kT} / A_eff⟩_unbound`, `dG0 = −kT ln(K_b / 1.6605 nm³)`,
`A_eff = πR² + 2πR sqrt(πkT/2k) + 2πkT/k`. Headline: plateau mean of dG0(t). Cross-check:
time-dependent reweighting. Reference: the empty-funnel value.
Why: the usual shortcut (well depth + area / 1660 Å³) fixes the bound-state length at 1 Å and
ignores the soft wall (≈ 20 % of the area at default settings).

### FM-D10 — Replicas for the error estimate, walkers for speed
Independent replicas (default 5) give the confidence interval. Walkers of one replica share
the bias by reading each other's HILLS files.
Why: replicas are the honest error bar; walkers shorten wall time on several GPUs without a
message-passing layer.

### FM-D11 — Inputs: a protein PDB and a ligand SDF; nothing else is required
The protein file is reduced to its amino-acid residues, repaired with PDBFixer when installed,
and protonated; an SDF without hydrogens gets them from RDKit; a ligand not in contact with the
protein is refused. Ligand parameters come from the OpenFF or GAFF generators, or a user ffxml.
Amendment (2026-11-15, owner): readers for AMBER / GROMACS / CHARMM input were removed.
Amendment (2026-11-15): fmeta names the ligand force field itself when the settings name none:
`openff-2.2.1` for `openff`, `gaff-2.11` for `gaff`, both with AM1-BCC charges
(`system/ligand.py`). The first run with the real packages showed that the generators of
`openmmforcefields` 0.16 have no default (two files alone failed), and a default taken from the
installed packages would change the parameters with the environment.
Why: owner decisions — OpenMM only, and two files must be enough.

### FM-D12 — Verification: a solvable toy system, CPU CI, GPU gates with recorded evidence
The toy system (one particle, Gaussian well, exact K_b by quadrature) validates bias, estimator
and standard state end to end: four 40 ns runs gave plateau errors of −0.15 to +0.50 kJ/mol,
four 20 ns runs reweighted errors of −0.12 to −0.27 kJ/mol. `fmeta check` /
`scripts/gpu_gates.py` repeat the platform-dependent checks on the GPU and record them in
`bench/results/`.
Amendment (2026-11-16): the criteria are unchanged; the scatter of the gate's 10 ns toy run is
on record. The run of the item `toy_free_energy` was repeated with 48 other seeds (100–147),
and a 40 ns run with 12 seeds, on CUDA in mixed and in double precision and on Reference
(a diagnostic in the image, not gate evidence). Error against the exact value, mean ± standard
deviation, kJ/mol:

| Run | Estimate | CUDA mixed | CUDA double | Reference | Largest | Beyond the gate criterion |
|---|---|---|---|---|---|---|
| 10 ns (48 each) | reweighted | +0.48 ± 0.83 | +0.51 ± 0.83 | +0.46 ± 0.83 | 2.62 | 2, 2 and 1 of 48 (criterion 2.0) |
| 10 ns (48 each) | plateau | +0.19 ± 0.87 | +0.15 ± 0.88 | +0.33 ± 0.86 | 2.62 | none (criterion 3.0) |
| 40 ns (12 each) | reweighted | +0.18 ± 0.25 | +0.13 ± 0.39 | +0.05 ± 0.35 | 0.87 | — |
| 40 ns (12 each) | plateau | +0.50 ± 0.44 | +0.33 ± 0.58 | +0.36 ± 0.50 | 1.28 | — |

The platforms do not differ (a difference seen with 12 seeds was scatter), and mixed precision
adds no offset. The positive mean of the reweighted estimate belongs to the short run: it is
+0.48 ± 0.07 at 10 ns and +0.12 ± 0.05 at 40 ns (mean ± standard error over the platforms).
The gate's own seed gives +1.96 (reweighted) and +1.49 (plateau) on CUDA mixed, the same
numbers in both full gate runs so far: the verdict is repeatable while the toy trajectory is
unchanged. A change that alters the trajectory (the bias code, OpenMM, the GPU) draws a new
value, and about one 10 ns run in 30 misses the criterion with a correct implementation. When
the item fails after such a change, repeat the run over seeds and compare with this table
first. The tolerance is not widened for a run; another form of the criterion (the mean over
several seeds, say) would be a new amendment.
Amendment (2026-11-16, owner): operation on a CPU platform is not guaranteed, and no time is
spent on confirming it. From FM-D14 on, the pipeline (prepare, equilibrate, run, analyze, the
run modes) is verified by the GPU gates alone: `fmeta check --platform CPU` and the pipeline
runs of `pytest -m slow` are not rerun as verification, and a new feature gets no test that
runs dynamics on a CPU platform. The default `pytest` stays what it is, the test of logic and
numerics (toy system, numpy references, Reference platform, the order of the scheduler's
decisions). Not decided by this amendment, and unchanged until the owner decides: the sentence
of the test policy that the default `pytest` passes on a CPU-only machine, and the `cpu` job
of CI.
Why: the development machine has no GPU; correctness has to be established where it can be
(exact answers on CPU) and the rest gated on the machine that runs production.

### FM-D13 — Everything except CI runs in one official image built from a lock
One image (`Dockerfile`): a micromamba base pinned by digest, the environment of
`conda-linux-64.lock` (an explicit lock solved from `environment.yml` in a container; OpenMM
built against CUDA 12.9), fmeta, and the test and lint tools. fmeta commands, simulations,
tests, lint, the scripts, the GPU gates and the benchmarks run in a container of it, through
`scripts/in_docker.sh`: the repository is mounted and its source is the one that runs, as the
calling user, with only the GPUs named. No Python environment is made on the host. Exceptions:
CI (the `cpu` job installs the core with pip and keeps it usable without the conda-only
packages; the `docker` job builds the image and tests in it) and the SessionStart size check.
`scripts/gpu_gates.py` records evidence only when the lock installed in the image is the lock
of the repository; the record names the image, the hash of the lock and the UUIDs of the GPUs,
and `scripts/verify_bench_results.py` requires that. Rule: `.claude/rules/docker-execution.md`.
Why: owner rule of 2026-11-15 (execution and tests run on Docker, except CI). `environment.yml`
pins nothing, so it resolves differently over time: evidence taken in a conda environment does
not describe the environment a later user gets. With one locked image, the environment that is
verified is the one that is distributed, and a record says which one it was.

### FM-D14 — Run modes, and one scheduler for the child processes of a run
`run.mode` (`--mode` for one call) says how the replicas use the devices. `parallel`: side by
side, one per GPU of `run.gpus` (every GPU that `nvidia-smi` lists when none is named, or with
`gpus: all`); with more replicas than GPUs the next one starts on the GPU that becomes free.
`sequential`: one after another on one GPU. `auto` (default): `parallel` when GPUs are named,
one device otherwise, which is what fmeta did before. A sequential run with several GPUs, a GPU
named twice and a GPU that is not there are errors (`resolve_devices`, `engine/platform.py`).
"All GPUs" are those the container sees.
One scheduler (`src/fmeta/scheduler.py`) runs the jobs of a call. A job is the child processes
that must run together: a replica, or the walkers of one replica. Its rules: a device is used
by one job at a time; a job takes as many devices as it asks for (all of them, if there are
fewer); jobs start in list order and free devices are handed out in list order, by one loop in
the calling thread, so the same jobs on the same devices are always assigned the same way; a
task that fails stops the other tasks of its job, and the other jobs go on; after SIGINT or
SIGTERM nothing starts, every child is terminated with its process group (killed after 5 s or
at a second signal) and the call returns when the last one has ended. Every decision is
appended to `schedule.jsonl`; `verify_schedule` checks such a record against the rules.
A replica is always a child process (`fmeta run WORKDIR --replica I`) that reads the saved
`config.yaml`; mode, GPUs and length are passed per call. `run_stage` holds the directory of
its run with `flock` (a second run of it is refused; a process that dies leaves no lock) and
writes `device.json`: the device asked for, the device the platform reports, and the memory in
use on that GPU before and after the context was made. Preparation and equilibration use the
first device of the run. `fmeta status WORKDIR` shows state, time reached and device per replica.
Gate items `replicas_sequential` and `replicas_parallel`: the record of a real run breaks no
rule, each replica reaches its hills, its `device.json` names the assigned device and the
memory in use on that GPU grew. `replicas_parallel` needs two GPUs, so evidence is no longer
recorded on a machine with one. Both items are skipped off a GPU platform.
Not done: extending the device pool of `run_replicas` (one pool per call and no signal
handling: `docker stop` left child processes behind, and a benchmark could not share GPUs over
its ligands); starting a replica in the calling process (the gates then never exercised the
launcher); a key for several replicas on one GPU (its effect is not measured, and a saved
configuration with a key that is later withdrawn no longer loads).
Why: owner request of 2026-11-15 for both modes. A stopped container must leave nothing on a
GPU, and what was run where has to be checkable afterwards from a record, on the GPU machine.
