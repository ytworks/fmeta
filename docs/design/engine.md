# Engine design

Status: **Accepted** (2026-11-15, PR #1). Since then: FM-D11 amended and FM-D13 added
(2026-11-15: owner decisions, and the first run with the real ligand-parameter packages);
FM-D2, FM-D4, FM-D5, FM-D6 and FM-D7 amended (2026-11-15 and 16: the first runs on a GPU);
FM-D12 amended (2026-11-16: the scatter of the toy gate is on record, criteria unchanged; and by
the owner: operation on a CPU platform is not guaranteed); FM-D14 added (2026-11-16: run modes
and the scheduler); FM-D15 added in two parts (2026-11-16: real inputs, the membrane part and
the protein part); FM-D16 added (2026-11-16: official defaults as presets) and amended the same
day (owner: the run length of 500 ns withdrawn); FM-D17 added (2026-11-16: benchmark sets).
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
launcher); a key for several replicas on one GPU (a saved configuration with a key that is
later withdrawn no longer loads, and the gain is small. Measured on 2026-11-15 UTC: TYK2 with
ejm_46, 88476 atoms, one RTX PRO 6000 Blackwell, processes sharing the GPU without MPS, speed
from the time between checkpoints: one run 569 ns/day; two runs 322 + 296 = 618; three runs
206 + 202 + 207 = 615. That is 9% more in total at best. With MPS: not measured).
Why: owner request of 2026-11-15 for both modes. A stopped container must leave nothing on a
GPU, and what was run where has to be checkable afterwards from a record, on the GPU machine.

### FM-D15 — Real inputs. Membrane part: parts dropped first, kept components are receptor
Accepted 2026-11-16.
Dropping. `fmeta split --drop CHAIN[:FIRST-LAST]` (repeatable) removes a chain or a residue
range of it before the ligand is searched and the two input files are written, so no later step
sees it: not the orientation estimate, not the funnel placement. The case is a fusion partner,
the BRIL of A2A-StaR2-bRIL (residues 1001-1106 in place of 209-218 in the RCSB entries 5MZJ and
5OLZ): a soluble domain in the hydrophobic-belt fit, and an obstacle that is not in the
receptor. Residue numbers are compared as integers and the insertion code breaks ties; a bound
without a code stands for every residue of that number; a selection takes every residue in its
range, protein or not; one that matches nothing is an error that lists the chains of the file.
Kept components. A non-protein residue kept by `system.keep_hetero` (the Na+ of A2A, the Ca2+
of trypsin) is part of the receptor. It is found on the dry complex by its residue name,
followed through solvation and `addMembrane` by its position, recorded in `prepared/meta.json`
(`kept_atoms`) and given the class `other`, not `ion`. The rest follows from the class, without
a second mechanism: it is an obstacle for the automatic placement and the validation of the
funnel, it counts for the size of the box, it is restrained in equilibration and released with
the protein's heavy atoms, and it is in the solute trajectory. An ion of the same name that the
builder adds stays solvent. Until now a kept ion was a solvent ion: free from the first step of
heating, and invisible to the funnel.
Orientation check. `membrane.orient: check` says "ok" only when the input was compared with a
confident fit of the hydrophobic belt. When the fit is not confident the report is now a warning
that the orientation could not be checked; it used to read "ok". The build goes on: a file from
OPM is right whatever the estimate finds, and `orient: input` states that without a check.
Box height. A membrane box is built 8% taller than the rule of FM-D6 asks
(`MEMBRANE_NPT_ALLOWANCE`); a water box keeps its 3%. The rule itself is unchanged. One
measurement stands behind the number: A2A with theophylline from the OPM file of 5MZJ in 484
POPC, 210 892 atoms, built at 12.74 x 13.38 x 12.63 nm with 3%. After 1 ns (100 ps heating,
400 ps release, 500 ps free, GPU) x and y had changed by 0.2% and z by 3.5% (to 12.18 nm): the
solvent is placed less dense than it becomes, and with the bilayer holding the area the whole
change goes into the height. In the same time the height the rule asks for rose from 12.26 to
12.41 nm, because the funnel axis had turned towards z with the protein. The stage stopped on
`box_cv`, as it must. The two effects are 4.9%; 8% leaves room for a longer equilibration and
another bilayer and costs 5% more atoms. Built with 8% (221 143 atoms, 13.24 nm high) the same
system lost 3.7% in the same protocol and passed: 12.76 nm against 12.28 nm asked. Two runs of
1 ns on one system: the 10 ns equilibration of the A2A example is to confirm the number.
Not done: restraining a kept water (it stays mobile water and is no obstacle); a configuration
key for dropping (the `input.complex` route cannot drop yet); a finer orientation estimate. On
the OPM-oriented A2A the estimate is 14 deg and 0.2 nm from the OPM plane, so the 25 deg limit
of the check stays, a model that is 9.5 deg off (the prepared A2A structure of the Schrödinger
benchmark set) passes as "ok", and `auto` would not improve it: such a model is superposed onto
its OPM entry instead.
Why: the three were found on the way to the first real membrane target. What the receptor is
made of has to be decided once, before the funnel and the box are derived from it; and a check
that checked nothing must not read as a pass.

### FM-D15 — Real inputs. Protein part: gaps, caps, protonation states, and what is reported
Accepted 2026-11-16. The protein model of the input (`src/fmeta/system/protein.py`, called by
`read_protein`); kept components and membranes are the part above.
The model is made from what the file states, before PDBFixer and the template matching.
Gaps: consecutive residues of a chain whose C-N distance is above 0.2 nm (`GAP_NM`; a peptide
bond is 0.133 nm) have residues missing between them. The chain is split there and the two
ends get NME and ACE (the Amber templates), with their heavy atoms where the backbone of the
missing neighbours was: the nitrogen opposite to the carbonyl oxygen, the acetyl carbon in the
plane of the amide hydrogen, and without those atoms where there is most room. A TER record at
the gap changes nothing: the next chain of the file under the same identifier, with the
numbering going on, is the same chain. Where the input states a terminus, a chain starts and
nothing is added: after an OXT, and before a residue with H2 and H3. If a cap would stand
within 0.3 nm (`CAP_CLASH_NM`) of the other cap or of the residue across the gap, `prepare`
stops: at most one residue is missing and there is no model without it (measured on the test
protein, heavy atoms: one residue missing leaves at most 0.23 nm, two at least 0.37 nm).
Caps of the input stay: an acetyl group (two C, one O) or an N-methyl group (one N, one C)
within 0.17 nm of a chain end is its ACE or NME under any residue name (preparation programs
write `UNK` residues after the chain).
Protonation states: the hydrogens of a residue state its form (histidine: HD1 / HE2; ASP and
GLU: a hydrogen on the acid; LYS: two or three HZ; CYS: HG, and without it a disulfide or the
thiolate). A residue without hydrogens keeps the form its name states (HID, HIE, HIP, HSD,
HSP, ASH, GLH, LYN, CYX, CYM), which OpenMM drops when it reads the file. The forms go to
`Modeller.addHydrogens` as its `variants`, so they hold against `system.ph` and when the
hydrogens of the file fit no template and are all rebuilt. Everything else is decided by
OpenMM at `system.ph`, as before.
Kept non-protein residues (`keep_hetero`, water) go to a chain of their own behind the protein
of their chain: an ion listed after the last residue without TER otherwise hides the C terminus.
Report: `info.protein_input` of `prepared/report.json` and the check `protein_input` hold what
was removed and kept, the gaps capped (residues, C-N distance, caps added), the caps kept, the
alternate locations (OpenMM takes the first listed one of a residue; PDB files only), the
protonation states taken from the input, the histidine forms of the model and its disulfides.
No second report file.
Not done: building loops (a modelled loop is an input the user did not give; a cap is a
statement about what is there); capping the ends of the file (they stay charged termini unless
the file has caps; no NHE); neutral chain ends (a hydrogen that a preparation program put on an
open end, HXT or a second one on N, fits no template: the hydrogens are then rebuilt in the
forms read); pKa prediction; the name HSE (CHARMM's HIE, but a homoserine in the PDB); which
oxygen of an acid holds the proton after a rebuild. Two chains under one identifier with the
numbering going on and neither OXT nor H2 / H3 are read as a gap and capped.
Verified (2026-11-16, GPU 6, files not committed): 3PTB (6 disulfides), 4GIH and the prepared
TYK2 protein of Schrödinger's `public_binding_free_energy_benchmark` (commit a65d86b; its UNK
caps kept, HID x5, HIE x7, HIP x1 from its hydrogens) through `prepare` and a 200 ps
equilibration with zero wall energy, also 4GIH with residues 1056-1059 removed (capped). The
protein model alone on A2A: OPM 5MZJ with the sodium kept (no TER: gap LEU208-GLU219 capped,
ion in its own chain) and the prepared `a2a_hip278` protein of the same benchmark (ends filled
with hydrogens: capped, hydrogens rebuilt, HIE x4 and HIP x2 as its hydrogens state them).
Why: plan milestone M2 (2026-11-16). OpenMM bonded LEU208 to GLU219 over 1.15 nm and the force
field accepted it; caps of a prepared file were removed as hetero components and the ends
became charged; a rebuild turned HIP and HIE of a prepared file into HID. Each of these changes
the charge or the shape of the protein at the pocket without a message, and the binding free
energy with it. Parameters are never invented: only templates of the force field are used.

### FM-D16 — Official defaults: presets in the package, and one reference of the keys
Accepted 2026-11-16. The official defaults are the defaults of the settings dataclasses, and
they are written nowhere else by hand. Two presets, `soluble` and `membrane`, are files of the
package (`src/fmeta/presets/`): every key with its default, its unit and one line on what it
does. `membrane` is `soluble` plus the section `membrane`. `fmeta init --preset NAME` writes
one; the template text of `config.py` is removed. `docs/configuration.md` lists every key with
its unit, its default and where the value comes from. Tests hold the three together: a preset
has exactly the keys and the defaults of the code, and the reference has a row for every key
and for no other, with the same default.
A configuration may name a preset and hold only what it changes (`preset: membrane`). It is
read as the defaults of the code, then the preset, then the keys of the file. The file names
of a preset are placeholders and are not taken over; `membrane: false` removes the bilayer of
a preset and a mapping changes it; a file that names a ligand method without a force field
gets the pinned force field of that method. `preset` is an instruction for reading, not a
setting: the configuration saved in a work directory is resolved, has no `preset` key and
names the ligand force field that was used. Unknown keys and sections are errors as before.
Ligand parameters: the pins of FM-D11 stay (`openff-2.2.1`, `gaff-2.11`), and
`ligand_charge_method: null` stays the default and means the charges that the force field
prescribes. Measured in the official image (openmmforcefields 0.16.0, openff-toolkit 0.19.0,
AmberTools 26.0, no OpenEye toolkit): `openff-2.2.1` prescribes AM1-BCC and `openff-2.3.0`
NAGL charges; AM1-BCC is computed by `sqm` of AmberTools, the only implementation registered
there; for benzamidinium and theophylline, with both pins, the charges of the default path and
charges assigned beforehand as `am1bcc` are identical (largest difference 0). No default of
system building changes with this decision.
Every default in the reference has its origin: a source that was read, a measurement of this
project, a decision of this file, or the statement that it has none (`convention`, or `M4`:
to be confirmed on trypsin-benzamidine from convergence diagnostics, before any benchmark, and
never tuned against experimental values). No default was changed from the literature. Where
the sources differ, the reference lists it: hills every 1 ps against 2 ps, a hill width in
`rho` of 0.05 nm against 0.03 nm, 4 fs with hydrogens of 1.5 amu against 2 fs, the wall force
constants, the ligand force field. These are the first questions of M4.
Not done: `ligand_charge_method: am1bcc` as the default (the same charges; with charges
assigned beforehand the SMIRNOFF generator warns at every build and accepts non-integral
charges, and the setting would replace the NAGL charges of `openff-2.3.0` for a user who picks
that force field); a second copy of the defaults as a template; presets for classes of targets
such as kinases or GPCRs (the values of a target are an example, `examples/<name>/`); a key
whose effect is not established; a default changed because a paper uses another value.
Why: owner request of 2026-11-15 for a configuration with official defaults. A default that
is written in two places drifts apart; a saved configuration has to say what was run without
the code that wrote it; a value without a stated origin cannot be reviewed, and a value tuned
on the benchmark would make the benchmark worthless.
Amendment (2026-11-16, owner): the default run length of 500 ns per replica is withdrawn. The
owner decided that it is too long. Its origin was the literature that was read, not a
measurement of this project. The new `run.length_ns` comes from the official
trypsin-benzamidine run, by a rule written when that run was at 41 to 44 ns
(`examples/trypsin_benzamidine/README.md`, "Run length"): the shortest of 100, 150 and 200 ns
at which all five replicas pass the convergence criteria; 200 ns if they pass at none, and the
reference then says that the default did not converge there. Five replicas stay and no
criterion changes. The scan of the other defaults is compared at the same length. Until the
rule has given the value, the code and the presets hold 500.
Why: the owner's decision, and two measurements with the default bias: on trypsin 3 to 5
rebinding events per replica within 40 ns, and on TYK2 with ejm_46 (4.8 nM) the ligand left
the site within 5 ns and was back within 20 ns. A length chosen from one system's convergence
diagnostics has a stated origin; a number picked by hand would not.

### FM-D17 — Benchmark sets: the ordinary stages for every ligand, one scheduler call, a report without a verdict
Status: **Accepted** (2026-11-16).
A benchmark set is a directory, `examples/benchmarks/<set>/`, with `set.yaml`, `reference.csv`
and a `README.md`; `src/fmeta/benchmark.py` knows no target. `set.yaml` names the source
repository at a commit (40 hexadecimal digits), the files (`protein`, and `ligands`: one SDF
with a record per ligand) each with its sha256, the ligands with their formal charge, the pilot
ligands, and `config`: the differences from the default configuration, the same for every
ligand. Structure files are not committed. `fmeta benchmark prepare SET -o WORKDIR` fetches
each file into `WORKDIR/inputs`, refuses content whose sha256 is not the one named, and does
not fetch again a file that is there and matches; what is fetched is read as data and never
executed. The record of a ligand must be there once and carry the formal charge of `set.yaml`.
`superpose_on` names a structure (URL, sha256) on which the protein is put before the ligand
directories are written: a rigid least-squares fit of the alpha carbons of the same residue
number, refused above `max_rmsd`, and the same rotation and translation for every ligand
record. That is how a membrane protein gets the orientation of its OPM entry (FM-D8).
`reference.csv` holds per ligand the experimental value (in kcal/mol as its source gives it, or
a Ki with the temperature of its measurement, converted at that temperature), the FEP value
with its error and its kind (`relative`: known up to one constant; `absolute`), an absolute
alchemical value when one is published, and the DOI of each.
Every ligand gets an ordinary fmeta work directory (`config.yaml`, `ligand.sdf`) and a marker
with a hash of everything that decides its free energy: the configuration, the files it names
by content, the files of the set and the superposition. The devices and the sampling plan
(`run.length_ns`, `run.replicas`) are not in the hash: a set can be run longer and stay one
set. A work directory of another hash, or a changed `config.yaml`, is refused, as are ligands
with different plans. `prepare` takes `pilot`, `all` or names, and a later call adds ligands.
`fmeta benchmark run` gives what is left to one call of the scheduler (FM-D14), on the devices
of the run: the build and the equilibration of each ligand are jobs like its replicas, and a
stage waits for the stage before it of its own ligand. The list order is: all builds, all
equilibrations, replica 0 of every ligand, then replica 1 of every ligand, and so on; on one
device this is the order of execution, on several a later replica of a ready ligand uses a
device that would otherwise wait. A report over the whole set therefore exists early and gets
better. The work directories are the truth: a finished stage leaves no job, so the same call
continues, a failed stage holds back only its ligand, and the call ends with an error that
names it. Every call that starts jobs appends the commit, the image and the lock to
`runs.jsonl`. `fmeta benchmark status` shows every stage of every ligand.
`fmeta benchmark report` analyses every prepared ligand with `analyze_replicas` and writes
`bench/reports/<UTC date>-<commit>-<set>.json`, the same as text (`.md`) and an entry in
`bench/reports/index.json`. Per ligand: dG0 as the mean of the plateau values of the analysed
replicas with its 95% Student-t interval, the experimental and the FEP value, the rebinding
events, what the analysis found, and how many replicas within 1 kT of the empty funnel ("no
binding", not a binding free energy). Over the set: MUE, RMSE, mean signed error, MUE and RMSE
after the mean error on the same ligands is removed, Pearson r and Kendall tau, each with a
bootstrap interval over the ligands (10000 resamples, seed 20261116; none below three ligands),
for fmeta and for FEP on the same ligands, side by side for all ligands with a value and for
those whose analysed replicas all passed the checks. No ligand is dropped: one without a value
is listed with the reason. Energies are given in kcal/mol and in kJ/mol, the standard state is
1 mol/L, and experimental values are not corrected to the temperature of the simulation. A
report names the commit, the image and the lock hash of the code that wrote it and of the
calls that ran the simulations. It has no pass / fail field: `scripts/verify_bench_results.py`
checks index, file names, image and lock, and rejects a verdict key. `bench/results/` stays
gate evidence only.
Sets: `tyk2` (ten of the sixteen ligands of the JACS set, a kinase in water) and `a2a` (eight
ligands of the A2A set with the protonated His278, a GPCR in POPC, superposed on OPM 5OLZ),
both from the public FEP+ benchmark of Ross et al. 2023 at one pinned commit; the reasons for
the choice of ligands are in the `README.md` of each set.
Not done: a criterion on the agreement (what fmeta guarantees is in the gates; how well a
force field reproduces an experiment is a measurement, and a threshold on it would invite
tuning to the benchmark); dropping ligands that did not converge (the statistic would then
describe the easy ligands only; both columns are given instead); correcting experimental values
to 300 K (it needs the enthalpy, which is not known); absolute alchemical values for these sets
(none was found under a licence that allows it: the columns are empty); committing structure
files (a pinned commit and a sha256 say the same in two lines, and the licence question does
not arise).
Why: owner request of 2026-11-15 for benchmarks with experimental and FEP values, run and
compared. With one scheduler call the GPUs are shared over the ligands and nothing waits for a
ligand that failed; with replica 0 first the first full comparison comes after one replica per
ligand, not after the first ligand has all of its replicas. The FEP values of the source are
shifted so that their mean is the experimental mean, so the offset-free measures are the ones
in which the two methods can be compared.
