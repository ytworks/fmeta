# Engine design

Status: **Accepted** (2026-11-15, PR #1). Since then: FM-D11 amended and FM-D13 added
(2026-11-15: owner decisions, and the first run with the real ligand-parameter packages);
FM-D2, FM-D4, FM-D5, FM-D6 and FM-D7 amended (2026-11-15 and 16: the first runs on a GPU);
FM-D12 amended (2026-11-16: the scatter of the toy gate is on record, criteria unchanged; and by
the owner: operation on a CPU platform is not guaranteed); FM-D14 added (2026-11-16: run modes
and the scheduler); FM-D15 added in two parts (2026-11-16: real inputs, the membrane part and
the protein part); FM-D16 added (2026-11-16: official defaults as presets) and amended the same
day (owner: the run length of 500 ns withdrawn; then: 20 ns pilots first and a shorter
equilibration); FM-D17 added (2026-11-16: benchmark sets); FM-D14 amended (2026-11-16: several
replicas on one GPU with NVIDIA MPS, `run.per_gpu`; one math-library thread per replica); FM-D9
amended three times (2026-11-16, owner: the first 2 ns are not used; then: a replica does not
converge on its own, the set of replicas is the measurement and there is no verdict; then: the
value of a replica is measured by the published procedure, from the bias), with FM-D16 and
FM-D17 amended to follow; amendments from the review of PR #1 (2026-11-16 and 17): FM-D5 (a
restart under other conditions is refused), FM-D11 (a complex is split only with the SMILES of
the ligand), FM-D15 (an amino acid without a backbone atom is rebuilt before the chains are
split, never dropped), FM-D9 and FM-D10 (a sample of a walker is weighted with the bias that
walker had, rebuilt from a record of the hills it had read); from its second round (2026-11-17):
FM-D5 and FM-D10 (the set of walkers of a replica may only grow), FM-D15 (a chain ends at a gap
before a missing backbone atom is rebuilt); from its third round (2026-11-17): FM-D15 (the caps
of the input stand in the chain before a missing backbone atom is rebuilt); FM-D13 amended
(2026-11-17, owner request: the image holds NVIDIA's cuPotential plugin, with OpenMM 8.4.0 on
CUDA 13.4 and the CUDA compatibility libraries; and again the same day: the image holds the MPS
programs of the driver of those libraries, so NVIDIA MPS works on a host driver below R615);
FM-D18 added (2026-11-17, owner request: the cuPotential plugin as the back end of a run, an
option until its configuration and its accuracy are measured); FM-D12 amended (2026-11-17: the
tolerances of the default 20 ns toy test come from its measured scatter over 48 seeds, after a
CI failure by 0.0015 kJ/mol on another CPU; the criteria of the gate and of the `slow` test are
unchanged); FM-D13 amended a third time (2026-11-17: the MPS daemon runs with protocol 3, after
the daemon of the default protocol stopped answering at about 50 client starts).
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
Amendment (2026-11-16, review of PR #1): a restart is refused, before any file of the run is
touched, when its conditions differ from those in `run.yaml`: the funnel, the bias settings
(the temperature is one of them), the grid, the time step, the system (particle count, barostat
and pressure: the new block `system`) and the layout of the trajectory (interval and atoms: a
DCD file has one header). The length, the device and platform, the checkpoint interval and the
walker settings may change. An entry that an older `run.yaml` lacks is not judged. With walkers
the bias settings and the time step of every walker are checked before any walker is rolled
back; each walker then makes the whole check. The reviewed code continued whenever a checkpoint
existed: with `metad.bias_factor` changed from 10 to 2 after 3 hills, the next hills went into
the same HILLS file, and because every row is decoded with the factor of the first, the bias
rebuilt from the file was off by up to 1.09 kJ/mol while `run.yaml` still said 10.
Amendment (2026-11-17, second review of PR #1): of the walker settings only how often walkers
read each other may change freely. The set of walkers may only grow. A run is refused, before a
file is touched, when its replica directory holds hills that the run would leave out of its
bias: a walker directory with hills beyond `run.walkers`, walker directories under a run
without walkers, or the `HILLS` of a run without walkers under a run with walkers
(`check_walker_set`, at the start of `fmeta run` and again before walkers are rolled back). The
directory is judged, not `run.yaml`, which a restart does not rewrite. This corrects the
sentence above ("the walker settings may change"): with `run.walkers` cut from 3 to 2 the two
walkers continued without the hills of `w2`, while the analysis adds every HILLS file of the
replica; the bias of the analysis was off by up to 2.74 kJ/mol from the bias of the run.
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
Amendment (2026-11-16, owner): the first 1 to 2 ns of a run change the most and are not used.
The headline is now the estimate from the samples after the first 2 ns (`discard_ns` of
`analyze_run`, `fmeta analyze --discard-ns`; of a run shorter than 4 ns, its second half). The
part that is kept is cut into time slices of 250 ps (at most 300 slices), each with the bias of
its middle. Every sample is evaluated under the bias of every slice, and the free energies of
the slices follow from the self-consistent equations of the multistate Bennett acceptance ratio
(Shirts and Chodera, J. Chem. Phys. 129, 124105, 2008); the weights give the profile along `s`,
and `K_b` and `dG0` as above. The hills of the discarded part still define the bias: only its
samples are left out. Kept as cross-checks, each a convergence problem above 4 kJ/mol
(`MAX_REWEIGHT_GAP`, unchanged): the same estimate from the second half alone (new), and the
plateau mean of dG0(t) from the bias (the former headline). The former cross-check, weights
`exp((V − c(t))/kT)` on the last 80% of the samples, is removed. When the samples do not cover
the unbound region the headline falls back to the plateau mean and the replica is reported as
not converged. No threshold changed.
Measured on the toy (8 seeds of 20 ns each, error of the mean ± standard error and RMSE against
the exact value, kJ/mol): well of 20 kJ/mol, new headline −0.03 ± 0.15 and 0.40, plateau mean
+0.31 ± 0.15 and 0.52; well of 40 kJ/mol, −0.11 ± 0.08 and 0.25 against +0.41 ± 0.15 and 0.58;
all 16 runs pass every check. Without a well all 8 runs are reported as no binding (−0.91 ± 0.15
against −0.69 of the empty funnel). The second-half check is there because the gap to the
plateau mean alone no longer shows what the old cross-check showed on trypsin (FM-D16): with
the default bias at 40 ns the new headline and the plateau mean agree (−28.0 and −29.1) while
the estimate from the second half alone is 11 to 45 kJ/mol higher in every replica.
Why: the owner's instruction. At the start the ligand sits in the site while the bias fills it,
which is not a sample of the equilibrium the estimate assumes.
Second amendment (2026-11-16, owner): a funnel metadynamics replica of the default length is
not a simulation that converges, so the analysis no longer says that one did or did not. What
holds now:
- The measurement is the set of independent replicas (FM-D10). Its value is the mean of their
  dG0; its uncertainty is the standard error and the 95% Student-t interval of that mean. One
  replica gives a value without an interval.
- There is no verdict. `converged` and `problems` are gone from `result.json` and
  `summary.json`, and `fmeta analyze` no longer prints NOT CONVERGED. The checks of a replica
  (rebinding events, drift and scatter of the plateau, the two cross-checks, the upper wall,
  the bias grid) keep their thresholds and are written as `diagnostics`: notes that tell where
  the scatter comes from. A difference between two estimates is still never a reason to pick
  one of them.
- One statement stays a decision: `no_binding`, a value within 1 kT of the empty funnel, for a
  replica and for the value of the set.
- The estimate of a replica does not change (the amendment above). Three ways to the value of
  the set were compared first, by a rule written before the comparison: (C) the mean of the
  per-replica estimates, as it is; (R) the mean of per-replica mean-force estimates; (P) one
  mean-force estimate from the samples of all replicas, with a jackknife over the replicas. A
  mean-force estimate assumes only local equilibrium under the bias of the moment:
  `grad F = −kT grad ln n − ⟨grad V_t⟩` in bins of half a hill width, integrated to the
  weights of the samples. Rule: on the toy, 4 replicas of 10 ns within 2.0 kJ/mol of the exact
  value for a shallow and a deep well in two sets of seeds, and no binding without a well; of
  the candidates that meet this, the smallest spread of the value between the six bias
  settings of the trypsin scan at 40 ns (FM-D16); experiment decides nothing. All three meet
  the toy part (errors of at most 0.80 kJ/mol; below 5 ns R and P give no value). The spread
  on trypsin is 25.1 kJ/mol for C, 32.8 for R and 42.0 for P: C stays, and no mean-force
  estimator is in the package. The numbers are in FM-D16, third amendment.
Why: the owner's instruction, and what the scan showed (FM-D16). After 40 ns a replica has had
0 to 17 rebinding events, the replicas of one setting lie between −1.7 and −78 kJ/mol, and no
setting had five replicas without a note. A verdict that every run fails says nothing; the
interval over the replicas says how much is open.
Third amendment (2026-11-16, owner): the measurement follows the published procedure of funnel
metadynamics; settings for accuracy come after everything is shown to work. What holds now:
- The headline of a replica is the estimate from the bias (Limongelli, Bonomi and Parrinello,
  PNAS 110, 6358, 2013): the free-energy surface is the accumulated bias times `−γ/(γ−1)`, its
  profile along the axis gives dG0(t) by the formula above, the first half of the run is
  rejected and dG0(t) is averaged over the second half (`plateau_fraction` 0.5, 20 points).
  `dG` is `dG_plateau`.
- `dG_error` (new) is the standard error of that mean from 5 consecutive blocks of the kept
  half. dG0(t) comes from the bias accumulated since the start, so the blocks are not
  independent: the error says how far the value moved within the kept part of this run. What a
  run of this length leaves open is the scatter between the replicas (second amendment,
  unchanged: mean, standard error and 95% interval of the set, no verdict).
- The estimates from the samples (first amendment) are the cross-checks: after the first 2 ns
  (`dG_reweighted`) and of the second half (`dG_second_half`), each a diagnostic above
  4 kJ/mol from the headline (`MAX_REWEIGHT_GAP`, unchanged). When the samples do not cover
  the unbound region the note says that the value has no cross-check. `dG_source` is gone from
  `result.json`; `no_binding` is decided with the headline.
- No threshold and no criterion of a test or a gate changed: the toy gate checks the estimate
  from the samples (2.0 kJ/mol) and the one from the bias (3.0 kJ/mol) as before.
Measured on the toy runs of the first amendment (8 seeds of 20 ns, kJ/mol): well of 20 kJ/mol,
error of the headline +0.32 ± 0.15, RMSE 0.52, scatter between the seeds 0.43, `dG_error` 0.34
on average (0.19 to 0.55); well of 40 kJ/mol, +0.41 ± 0.15, RMSE 0.58, scatter 0.43, `dG_error`
0.35 (0.14 to 0.60); without a well −0.69 ± 0.12 (empty funnel −0.69, scatter 0.35, `dG_error`
0.28) and 8 of 8 runs reported as no binding. The block error is about 0.8 of the scatter
between seeds there. On real systems it is far below the scatter between replicas: trypsin with
the default bias at 40 ns, `dG_error` 1.1 to 16.2 with replicas between −4.6 and −78.2 (value
of the set −28.28, 95% interval −66.93 to 10.37; with the former headline −27.35); TYK2 with
ejm_46, 2.0 to 7.1 with replicas between −4.0 and −29.0 (−14.85, 95% interval −26.60 to −3.11;
former headline −10.05). The six bias settings of the trypsin scan at 40 ns give −14.9 to
−29.1 kJ/mol with this headline (the column "analysis before the amendment" of the table in
FM-D16), against −2.9 to −28.0 with the former one.
Why: the owner's instruction. The published method reads the value from the surface that the
bias builds, after a rejected first part, and gives its error from block averages; an estimate
from the samples is not part of it. The estimate from the samples also had no value for a
replica whose samples do not reach the unbound region (9 of the 15 replicas of the A2A pilot),
so the headline was of two kinds within one set. Deferred by the same instruction, until
operation is shown: shared-bias walkers in the defaults, the run length and the bias settings
(the published runs are 250 ns or longer on one surface).
Fourth amendment (2026-11-17, review of PR #1; the owner chose the record over giving no
estimate): with walkers, a sample is weighted with the bias its own walker had, not with the
sum of all hills up to its time.
- Walkers read each other's files as wall-clock time allows, so at one simulation time they
  hold different sums. The bias of a walker after `k` hills of its own is its first `k` hills
  plus the rows of the other HILLS files that its `SYNC` record names (FM-D10).
- Each walker is a sequence of time slices on its own time axis, and the slices of all walkers
  are combined as before (`MAX_SLICES` is shared between them).
- A sample whose bias cannot be rebuilt is left out and counted in a note. Walkers without a
  record (a run from before it existed) get neither estimate from the samples, with a note.
- The headline does not change: it is read from the sum of all hills. The estimates of a run
  without walkers are the same numbers as before (two replicas of the examples: identical to
  the last digit).
Measured on the toy system (Reference). Two walkers out of step (one deposits 30 hills before
the other starts): the bias of every sample rebuilt against the bias column of COLVAR,
3e-7 kJ/mol; with all hills up to the time of the sample, 1.1 and 3.3 kJ/mol. One walker ending
its 10 ns before the other starts, 12 pairs of seeds, error of `dG_reweighted` against the exact
value in kJ/mol: −0.17 ± 0.19 (mean ± standard error), standard deviation 0.67, largest 1.87;
with the former rule +0.11 ± 0.47, standard deviation 1.63, largest 2.63.
Why: the review. The former rule holds only for walkers in lock step, which a set of processes
on several GPUs is not. No committed result came from a run with walkers.

### FM-D10 — Replicas for the error estimate, walkers for speed
Independent replicas (default 5) give the confidence interval. Walkers of one replica share
the bias by reading each other's HILLS files.
Amendment (2026-11-17, review of PR #1): each walker records what it has on its grid from the
others, in the file `SYNC` of its directory. One line after each read of the other HILLS files
(at the start or a restart, then after every `sync_hills` hills of its own): its own hill
count, and for every other walker the number of rows read and the CRC-32 of the bytes they were
read from. The record is flushed with the checkpoint, cut to the checkpoint with HILLS and
COLVAR at a restart, and removed with them when a walker without a checkpoint starts again.
The analysis accepts an entry when the file of the other walker still begins with those rows
(the same checksum). When a restart removed rows that had been read and other hills took their
place, the bias of the samples in between cannot be rebuilt (FM-D9, fourth amendment). The
contract of the grids is unchanged: each walker's grid against the sum of all HILLS files, 1e-7.
The files alone do not say when a walker read which rows; a count with a checksum is the
smallest record that does, and it stays right when another walker is rolled back.
Amendment (2026-11-17, second review of PR #1): a replica continues with the walkers it was
run with or with more, never with fewer (FM-D5, amendment of the same date). A walker that is
added reads the hills that are there and the others read its hills; the `SYNC` records cover
it (test: after 3 walkers became 4, every grid is the sum of the four HILLS files, 1e-7).
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
Amendment (2026-11-16, review of PR #1): a ligand is cut out of a complex (`fmeta split`,
`input.complex`) only with its SMILES; without it the split stops before a file is written. The
route that wrote the ligand from the bonds of the structure file is removed. A structure file
states neither bond orders nor formal charges, with or without hydrogens: a benzene with its
six hydrogens was written with single bonds, and the parameter generator read it as a
cyclohexane short of six hydrogens. The protein of a split PDB complex keeps the residue names
that state a protonation state (HIP, ASH, ...), as the direct route does (FM-D15): OpenMM reads
them as the standard names, and the split had written those.
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
Amendment (2026-11-17): the tolerances of the default 20 ns toy test are taken from its
measured scatter. The criteria of the gate (10 ns) and of the `slow` test (4 × 40 ns) are
unchanged. `tests/test_analysis.py::test_toy_binding_free_energy_end_to_end` (20 ns, seed 12,
Reference platform) failed in the `docker` job of CI on cc789b3 and on 565b359: −5.955 kJ/mol
from the samples against the exact −4.954, an error of 1.0015 against the tolerance of 1.0. The
same code in the image of the same lock gives −5.295 on the GPU machine. The trajectory of a
fixed seed depends on the CPU, so the verdict of the test on another machine is a new draw. The
run was repeated with seeds 1 to 48 on the GPU machine (Reference, the present estimators; a
diagnostic in the image). Error against the exact value, kJ/mol:

| 20 ns, 48 seeds | Mean ± standard deviation | Smallest and largest | Beyond 1.0 | Beyond 1.5 |
|---|---|---|---|---|
| from the samples (`dG_reweighted`) | −0.28 ± 0.52 | −1.32, +1.06 | 6 of 48 | none |
| from the bias (`dG_plateau`) | +0.10 ± 0.72 | −1.49, +1.34 | 9 of 48 | none |

Every run had 51 to 70 rebinding events, none was reported as no binding and none had a
diagnostic. The former tolerances were 1.0 from the samples and 2.0 from the bias: 6 of the 48
runs of a correct implementation miss the first. The test now has the pair of the gate's toy
item, 2.0 kJ/mol from the samples and 3.0 from the bias: 3.3 and 4.0 standard deviations from
the measured mean on the nearer side (a miss in about 1 draw of 2000, for errors that are
normally distributed). The tight check of the estimate stays the mean over replicas of the
`slow` test. Not measured: the scatter of the present estimate from the samples at the 10 ns
of the gate (the table of the first amendment is of the former reweighting).
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
The NVIDIA MPS daemon of FM-D14 also runs in a container of the image (`scripts/mps.sh`), and
`scripts/in_docker.sh --mps` starts a container as its client: nothing is installed on the host.
Why: owner rule of 2026-11-15 (execution and tests run on Docker, except CI). `environment.yml`
pins nothing, so it resolves differently over time: evidence taken in a conda environment does
not describe the environment a later user gets. With one locked image, the environment that is
verified is the one that is distributed, and a record says which one it was.
Amendment (2026-11-17, owner request: NVIDIA cuPotential as the default back end; plan
`docs/plans/2026-11-17-cupotential-backend.md`, its M0): the image holds NVIDIA's cuPotential
plugin for OpenMM (`nv-openmm-plugin` 0.3.5), and the environment follows what the plugin is
built for: OpenMM 8.4.0 on CUDA 13.4 (before: 8.6.1 on CUDA 12.9), Python 3.12. No code under
`src/fmeta/` changed with it; the use of the plugin by a run is a later decision.
- What the lock describes. The plugin is a conda package of NVIDIA's release, not of a channel:
  `docker/nvidia.pins` names the release file and its sha256, `scripts/update_lock.sh` solves
  against it and writes the URL of the package into the lock. The CUDA forward-compatibility
  libraries (`cuda-compat-13-4` 615.71.10, a `.deb` of NVIDIA's repository) are named with their
  sha256 in a comment line of the lock, and the image build reads them from there
  (`docker/install_nvidia.sh`). The hash of the lock, by which an image is judged official,
  covers both.
- The driver of the host. CUDA 13.4 needs a driver of R615 or newer; the GPU machine has
  595.99.02. The compatibility libraries lie in `/usr/local/cuda/compat`, where the NVIDIA
  Container Toolkit (1.19.1 here) finds them: it puts them on the loader path of a container
  when they are newer than the driver of the host, so with R615 or newer they are not used. A
  gate record names the host driver, the `libcuda` the process loaded, the CUDA runtime and the
  plugin of the lock (block `environment`): it shows in which of the two ways it was taken.
- Licences. The plugin and the compatibility libraries are NVIDIA's software under NVIDIA's
  licences, whose texts are in the image under `/opt/fmeta/licenses`. A speed measured with the
  plugin is not written to a tracked file, a commit message or GitHub until NVIDIA permits it
  (owner, 2026-11-17: the licence of the plugin forbids the disclosure of performance data
  without written permission). Speeds of stock OpenMM are recorded as before.
- The ligand tools go down with OpenMM: `openmmforcefields` 0.16.0 needs OpenMM 8.5.1 or newer,
  so the image has 0.15.1, and with it AmberTools 24.8 (before: 26.0). `openff-toolkit` 0.19.0,
  `openff-interchange` 0.5.5 and `openff-forcefields` 2026.11.0 are the same. Measured on the
  TYK2 ligand ejm_46 through `register_ligand`, old image against new: the charge, sigma and
  epsilon of every atom are identical (largest difference 0) and the bonds, angles and torsions
  are the same (counts and parameter sums), for `openff` (38 / 64 / 159) and for `gaff`
  (38 / 64 / 105). The results on record keep their parameterisation.
- Measured in the new image on this host (image f7d97cf1…, lock 0b6c344f…, the compatibility
  `libcuda` 615.71.10 loaded): `python -m openmm.testInstallation` computes forces on all four
  platforms; directories prepared and equilibrated with OpenMM 8.6.1 (a soluble system in the
  rhombic dodecahedron, 24062 atoms, and a membrane system, 37401 atoms) load and run; a
  Context with the plugin steps on both of them with the funnel forces and the bias in the
  System (the plugin's force alone, with its integrator, with its barostat for the soluble
  system, and with the stock `MonteCarloMembraneBarostat` for the membrane), every energy
  finite, the potential energy at the starting state 1.3e-5 and 1.1e-5 (relative) from the
  stock PME of the same System (a first reading, not a criterion).
- Open: NVIDIA MPS with a host driver older than R615. With the daemon of `scripts/mps.sh` on
  driver 595.99.02, a client that loads the compatibility `libcuda` does not return from
  `cuInit`, whether the MPS server loaded the compatibility library (4 clients, stopped by
  hand) or the library of the host (stopped by a timeout). A client forced onto the host's
  `libcuda` 595.99.02 initialises under MPS, but cannot load the kernels of the CUDA 13.4
  runtime (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). So on this host a process of the new image
  has a GPU context of its own, and several replicas on one GPU at the same time (`run.per_gpu`
  above 1, FM-D14) need a host driver of R615 or newer. Not tried: the MPS programs of driver
  R615 run in a container on the R595 kernel module. (Tried the same day: the next amendment.)
Why: the plugin has no build for another OpenMM or CUDA version, and one image stays the rule:
the environment moves as a whole, and its tests and gates are taken again in it.
Amendment (2026-11-17, owner: the driver of the host is not raised for now, and the MPS programs
of driver R615 may be tried in a container): the image holds the two NVIDIA MPS programs of the
driver its compatibility libraries come from (`nvidia-cuda-mps-control` and
`nvidia-cuda-mps-server` of `libnvidia-compute` 615.71.10, in `/opt/fmeta/mps/bin`), and a
container uses them exactly when it uses the compatibility libraries. This closes the open point
above: `run.per_gpu` above 1 works with a host driver older than R615.
- The rule: an MPS daemon, its server and the CUDA driver library of every client are of one
  driver version. The entry of the image (`docker/entrypoint.sh`) puts the programs of the image
  first on `PATH` when the loader resolves `libcuda.so.1` to the compatibility libraries, and
  `scripts/mps.sh start` starts the daemon through the entry. With a host driver of R615 or
  newer the compatibility libraries are not used, and the programs are the host's, as before.
- An MPS client has no OpenCL platform there. Under the R615 daemon a process that loads
  NVIDIA's OpenCL library of the host (595.99.02) does not return from the start of OpenCL,
  and `import openmm` loads the OpenCL platform: a client waited for ever. The entry hides the
  OpenCL vendors (`OCL_ICD_VENDORS` names an empty directory) when the compatibility libraries
  are in use and the control socket of an MPS daemon is in the container. Such a client has the
  Reference, CPU and CUDA platforms; a container that is no MPS client keeps OpenCL. A command
  that bypasses the entry (`docker exec`, another `--entrypoint`) gets neither change.
- What the lock describes: `docker/nvidia.pins` names the package and its sha256, the lock holds
  it as a second comment line (`scripts/update_lock.sh --pins` rewrites these lines without a
  new solve: the packages of the lock are those of the gate record of abd4548), and the image
  build takes the two programs and the licence text from it and refuses a package whose version
  is not that of the compatibility libraries. The licence of the package permits the
  distribution of its unmodified binaries together with its text
  (`/opt/fmeta/licenses/libnvidia-compute`).
- Measured on this host (driver 595.99.02). First with the programs mounted from the unpacked
  package into a container of the former image: the daemon and its server start on the R595
  kernel module, the server loads the compatibility `libcuda` 615.71.10, and 4 replicas of the
  soluble test system ran at the same time on one GPU as its clients (`fmeta run --per-gpu 4`
  with the cuPotential back end, a run and its continuation to 3 ns, both with exit 0).
  Then with the image built from the lock of this amendment (image 648480d3…, lock ee32ef34…).
  A container with a GPU finds the two programs of the image on `PATH` and has the platforms
  Reference, CPU, CUDA and OpenCL; a container without a GPU finds no MPS program. The daemon
  was started by `scripts/mps.sh start 6,7`. In a client (`scripts/in_docker.sh --mps`)
  `OCL_ICD_VENDORS` names the empty directory, `import openmm` returns and the platforms are
  Reference, CPU and CUDA. 4 replicas of the soluble test system (24062 atoms) ran at the same
  time on one GPU as clients of that daemon, once with the cuPotential back end and once with
  stock OpenMM (`fmeta run --per-gpu 4`, 0.5 ns each: `nvidia-smi` listed the 4 processes of
  each GPU as MPS clients, every HILLS has 503 lines, both runs with exit 0; stock OpenMM at 925
  to 934 ns/day per replica). In that image `ruff check .` passes and `pytest` 193 passed.
- No test covers this: CI has no GPU and no MPS. The check is the runs above.
Why: several replicas on one GPU are where the throughput of a GPU is (FM-D14), and the driver of
a host is not the project's to change. The other ways were measured and fail: the daemon of the
host with a client on the compatibility library, and a client on the library of the host.
Amendment (2026-11-17, third: the daemon runs with MPS protocol 3): `scripts/mps.sh start` starts
the daemon as `nvidia-cuda-mps-control -f --protocol 3`. The image, the lock and the clients are
unchanged, and a client needs no setting for it.
- What failed: on one daemon of the second amendment (the default protocol of the control
  program, 2) the measurements of FM-D18 stopped after about 50 client starts. New clients waited
  at their start for ever with the GPUs idle, and the daemon answered no command.
- The mechanism, read from the stopped processes: under protocol 2 the control program of
  driver 615.71.10 stops reading the pipe of its log (`log` in the pipe directory) at the first
  record of a client. Every client that starts adds records; when the pipe is full (64 KiB) the
  daemon and every process that writes to the pipe block for good. It is a count of client
  starts, not a load: 4 replicas at once worked (the second amendment) because they were the
  first clients of their daemon.
- With protocol 3 the pipe is read: the log of the daemon and of its servers is `docker logs` of
  the daemon's container (`fmeta-mps`).
- Measured on this host (driver 595.99.02, image 648480d3…, the daemon of `scripts/mps.sh start`
  on all 8 GPUs): 20 calls of `fmeta run` with 4, 6 or 8 replicas on one GPU (soluble 88479
  atoms and membrane 221068 atoms, both back ends, 0.5 or 1 ns each), 8 calls and up to 64
  replicas at the same time: 120 replica processes, 480 client contexts in the log of the daemon,
  every call with exit 0 and every replica at its end. During the 64 replicas and after the last
  one the pipe held 0 of 65536 bytes and the control program had 2 threads. The log holds, for
  every client context that ends, the two error-level lines "Server failed to receive command …
  assuming client … has exited" and "Message receive failed, assuming the other end is dead"
  (480 each), and once, 4 ms before the start of the first server, "Dropping connection with
  failed handshake": no run failed with them.
- Not verified: a host with a driver of R615 or newer, where `scripts/mps.sh` starts the MPS
  programs of the host: a control program that does not know `--protocol` would not start.
- No test covers this: CI has no GPU and no MPS. The check is the runs above.
Why: a run with `run.per_gpu` above 1 starts one client per replica, a benchmark set starts
hundreds on one daemon, and a daemon that stops answering after about 50 of them takes every
run on the machine with it.

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
Amendment (2026-11-16): several replicas may share a GPU, and the "not done" above no longer
holds for that key. `run.per_gpu` (`--per-gpu` for one call; default 1) is the number of
replicas on one GPU at a time. The devices of the scheduler are places: a place is used by one
job at a time, and a device that is named twice carries two jobs side by side.
`resolve_devices` goes through the devices `per_gpu` times (`[0, 1]` with 2 gives
`[0, 1, 0, 1]`), so the first replicas each get a GPU of their own. `sequential` with `per_gpu`
above 1 is an error, `auto` without GPUs is one device with `per_gpu` places, and a device
named twice in `run.gpus` stays an error. Like mode and GPUs, the key is outside the
configuration hash of a benchmark. Above 1 the processes need NVIDIA MPS: `scripts/mps.sh
start` runs the daemon in a container of the official image, and `scripts/in_docker.sh --mps`
starts a container as its client (the same user, the daemon's pipe directory and its shared
memory). Only those containers are clients; any other process keeps the GPU as before.
Why: the measurement that was missing, asked for by the owner. TYK2 with ejm_44, 88k atoms,
4 fs, default bias, one RTX PRO 6000 Blackwell per configuration, speed from the hills written
in about 300 s:

| On one GPU | Total (ns/day) | Per replica (ns/day) | Total against one | Replica against alone |
|---|---|---|---|---|
| 1 replica | 606 | 606 | 1.00 | 1.00 |
| 2, MPS | 776 | 388 | 1.28 | 0.64 |
| 3, MPS | 1066 | 355 | 1.76 | 0.59 |
| 4, MPS | 1283 | 321 | 2.12 | 0.53 |
| 6, MPS | 1541 | 257 | 2.54 | 0.42 |
| 8, MPS | 1620 | 203 | 2.67 | 0.33 |
| 2, no MPS | 579 | 289 | 0.96 | 0.48 |

The single replica is the one of the pilot run on its own GPU (3501 ps of hills in 499 s); in
the 300 s window of this measurement a single replica gave 426 ns/day, which the pilot does not
confirm, so the factors are against 606. One run leaves the GPU at 80%: hills are fitted on the
CPU between the steps. A benchmark of many replicas has 2.1 times the throughput at 4 per GPU
and 2.5 to 2.7 times at 6 to 8; time slicing without MPS gains nothing, as the first
measurement said.
Not done: a default above 1 (alone a replica is fastest, and MPS is a daemon that somebody has
to start); starting the daemon from fmeta, or a check for it in the client (without MPS the
replicas still run, time-sliced); a gate item, since the scheduler's rules for places are
tested on the CPU and MPS is NVIDIA's. An MPS server that fails ends every client on it; the
runs continue from their checkpoints.
Amendment (2026-11-16): the scheduler starts every replica process with one math-library
thread (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` set to 1 unless the caller
has set them). The default of the libraries is a thread per CPU core in every process: 41
replicas on a machine with 128 cores had a load average above 1200. Measured on the running
production, total ns/day before and after the limit: TYK2 set, 12 replicas on 2 GPUs, 2749 and
3164; TYK2 example, 5 on one GPU, 1375 and 1512; trypsin, one replica, 544 to 593 and 708; the
three A2A runs 3 to 4% more. The load average fell to 46.
Why: the work of a replica is on the GPU; its CPU part (fitting hills, writing files) does not
gain from threads, and the spinning threads of the other replicas took the cores from it.

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
1 ns on one system. The free stage of a membrane system is 2 ns by default (FM-D16, second
amendment) and `equil/equil.json` holds its box area and height in ten blocks (`free_blocks`):
those records of the A2A example and the A2A set are to confirm the number.
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
Amendment (2026-11-17, review of PR #1): a residue named as an amino acid (the 20 standard
names and the names that state a protonation state) that lacks N, CA or C is never removed with
the non-protein residues. With PDBFixer its atoms are built on the structure as read, before the
model is made, so gaps and caps are judged on the repaired backbone. That pass builds only the
missing atoms of the residues present: no terminal oxygen (it would read as a chain end the
input states) and no replacement of non-standard residues; both stay in the pass after the
model. Without PDBFixer `prepare` stops and names the residues and their missing atoms. The
notes and `info.protein_input.backbone_rebuilt` list what was rebuilt. A file without such a
residue is read exactly as before. Not covered: a non-standard residue (MSE, ...) without its
backbone atoms is not known by its name and is removed as before, named in the note
"removed from the protein file". Measured on the test protein, one backbone atom deleted from
each of five residues: the rebuilt atom lies 0.004 to 0.022 nm from the atom of the complete
file, and the model is the one of the complete file in chains, residues and atoms (1, 35, 582).
Amendment (2026-11-17, second review of PR #1): before that pass a chain is ended where two
consecutive amino acids cannot be bonded (`split_at_breaks`), and the bond OpenMM makes between
neighbours of a chain is left out there. The question is answered from the backbone atoms that
are present: the last one of the first residue and the first one of the second, against a limit
by the number of bonds between them (`REACH_NM`: 0.20, 0.30, 0.44, 0.55, 0.67 nm for 1 to 5
bonds; measured on 1192 bonded pairs of four structures: at most 0.138, 0.254, 0.391, 0.505,
0.617 nm). The residue at such an end is built with its terminal oxygen, which is taken away
again: PDBFixer has bonds and angles only for a residue that matches one of its templates, and
an end without OXT matches none. The reviewed code gave the pass the chain as read: ARG14
without its C, residues 15 to 17 missing, and the C was built 0.28 nm from the N of ALA18,
where 0.447 nm lie between the two; `prepare` stopped with "no room for caps". With the chain
ended there the C lay 0.107 nm from its place (no template) and the caps had no room next to
the hydrogens; with the template 0.040 nm, and the model has 32 amino acids, NME and ACE.
Amendment (2026-11-17, third review of PR #1): before that pass the acetyl and N-methyl groups of
the input that are bonded to a chain end are put into the chain as ACE and NME, wherever the
file has them and whatever it calls them (`caps_in_place`). PDBFixer builds a missing N or C with
its bond to a cap it knows; a cap written as UNK after the chain it does not know. The reviewed
code rebuilt the N of the first residue 0.198 nm from the acetyl carbon: above the bond rule
(`CAP_BOND_NM`, 0.17 nm), so the cap was removed with the hetero components and the terminus
became charged (net charge +2 instead of +1, note "removed from the protein file: UNK x1").
With the cap in the chain the N lies 0.146 nm from the acetyl carbon and 0.021 nm from its place
(C at the other end: 0.131 and 0.005 nm), both caps are kept and the charge is +1. A cap is
found from the backbone atom of the end residue nearest to it along the chain: N or C within
`CAP_BOND_NM`, or CA within 0.30 nm (`REACH_NM`, two bonds) when N or C is the missing atom.
Two refusals instead of a silent removal: a cap-shaped group that only the far backbone atom
can place (three bonds, both atoms between missing), and, in `build_model`, a cap-shaped group
not bonded to a chain end but nearer to it than atoms without a bond come (`CAP_CLASH_NM`,
0.3 nm). No criterion changed. The second case of the review (C of the last residue missing)
kept its cap on this machine before the change, by 0.008 nm (0.162 nm); it is a test all the same.
Why: a residue counted as an amino acid only with all three atoms. LEU1 without its CA was
dropped as a hetero residue (34 residues, note "removed from the protein file: LEU x1"), and
ALA16 without its C stopped `prepare` with a message about SER15 and PHE17, although PDBFixer,
which ran afterwards, rebuilds the atom. A chain that is shorter or cut without an error changes
the protein the free energy is computed for.

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
Second amendment (2026-11-16, owner): the rule above is replaced before it gave a value. The
owner decided that 100 to 200 ns per replica is still too long: run 20 ns first and judge from
that, the shorter the better, and shorten the equilibration with it. What holds now:
- The 100 / 150 / 200 ns rule is withdrawn. The trypsin run was stopped at 75 to 81 ns.
- The pilots of both benchmark sets and the examples run 5 × 20 ns. `run.length_ns` and the
  default bias are set by the scan below; until then the code and the presets hold 500 and
  every run names its length (`--length-ns`). A run can be extended exactly (FM-D5).
- `equilibration.free_ps: null` is 2000 ps with a membrane (it was 10000) and 1000 ps without
  (unchanged). `equil/equil.json` gains `free_blocks`: the box area and height of the free
  stage in ten blocks, so a box that is still relaxing shows as a trend. The warning for an
  area change above 3% between the halves of the stage stays.
- Five replicas stay and no criterion changes: a run that is too short for the criteria is
  reported as not converged, with its numbers.

What 20 ns gives on trypsin-benzamidine (the official run, 5 replicas, default bias, analysed
at cut lengths; a replica passes when it has no convergence problem):

| Cut (ns) | dG0 (kJ/mol) | 95% interval (kJ/mol) | Rebinding events per replica | Replicas that pass |
|---|---|---|---|---|
| 10 | −67.8 | −87.4 to −48.3 | 1, 1, 1, 1, 0 | 0 |
| 20 | −50.6 | −65.7 to −35.5 | 2, 4, 1, 2, 2 | 0 |
| 30 | −37.9 | −51.5 to −24.2 | 2, 5, 3, 2, 3 | 0 |
| 40 | −29.1 | −43.8 to −14.3 | 4, 5, 3, 4, 3 | 0 |
| 60 | −19.6 | −30.6 to −8.6 | 7, 6, 4, 6, 4 | 0 |
| 75 | −16.0 | −24.2 to −7.8 | 9, 10, 5, 8, 5 | 2 |

At 20 ns four replicas have fewer than three rebinding events, the drift of the plateau is +8
to +33 kJ/mol and the reweighted estimate is 6 to 35 kJ/mol away. The mean still rises at
75 ns. So with the default bias 20 ns is not enough on this system, and the pilots are
expected to report the same: a short run needs a faster bias.

Scan of the bias (rule written before a scan run started). Five settings on the equilibrated
trypsin system, 5 replicas each, started for 40 ns and analysed at cut lengths; the default
bias above is the baseline:

| Setting | `bias_factor` | `height` (kJ/mol) | `sigma_s` / `sigma_rho` (nm) |
|---|---|---|---|
| default | 10 | 1.5 | 0.025 / 0.05 |
| `bf20` | 20 | 1.5 | 0.025 / 0.05 |
| `h3` | 10 | 3.0 | 0.025 / 0.05 |
| `sig2` | 10 | 1.5 | 0.05 / 0.10 |
| `bf20_h3` | 20 | 3.0 | 0.025 / 0.05 |
| `bf20_h3_sig2` | 20 | 3.0 | 0.05 / 0.10 |

- The lengths 10, 15, 20, 30 and 40 ns are taken in that order. The first length at which a
  setting has all five replicas without a convergence problem becomes `run.length_ns`, and
  that setting the default bias; of several at the same length, the one with the smallest
  standard error of the five.
- If no setting has five at 40 ns: the setting with the most passing replicas at 40 ns and
  40 ns become the defaults, and the documentation states the count. Nothing runs longer.
- Selection uses the convergence diagnostics on trypsin only. The distance to experiment is
  reported beside it and decides nothing; no benchmark is tuned; no criterion changes.
- Walkers are not in the scan: a benchmark is bound by the total throughput of the GPUs, which
  walkers do not change.
- This replaces the scan of `dt2`, `pace2` and `srho03` (first amendment), which compared at
  100 ns or more: `dt2` was stopped at 15 to 18 ns without a result, the other two never ran.

Why: the owner's decision. The cost it removes: at the measured speeds (FM-D14, amendment) a
20 ns replica of 88k atoms takes 48 min alone and 90 min with four on a GPU, so a set of 50
replicas is hours on 8 GPUs, where 100 ns per replica without sharing was days.

Result of the scan (2026-11-16; all 25 replicas at 40 ns, analysed at the cut lengths). No
setting has five replicas without a convergence problem at 10, 15, 20, 30 or 40 ns: 0 of 5
for every setting up to 30 ns, and at 40 ns the table below. This holds with the analysis of
the first amendment of FM-D9 (first 2 ns not used; the product analysis when the scan was read) and with the analysis before
it. By the second point of the rule the default bias is `sig2` (`sigma_s` 0.05 nm,
`sigma_rho` 0.1 nm; `bias_factor` 10 and `height` 1.5 kJ/mol stay) and `run.length_ns` is 40:
2 of 5 replicas pass at 40 ns.

| Setting at 40 ns | dG0 (kJ/mol) | SEM | 95% interval (kJ/mol) | Replicas that pass | Analysis before the amendment: dG0, SEM, pass | Rebinding events per replica |
|---|---|---|---|---|---|---|
| default (until now) | −27.95 | 5.94 | −44.43 to −11.47 | 0 | −29.05, 5.33, 0 | 4, 5, 3, 4, 3 |
| `bf20` | −8.07 | 5.14 | −22.34 to 6.20 | 1 | −15.12, 5.78, 0 | 7, 2, 14, 16, 3 |
| `h3` | −12.68 | 6.96 | −32.00 to 6.64 | 0 | −14.94, 6.06, 0 | 4, 8, 7, 5, 4 |
| `sig2` | −27.35 | 14.88 | −68.67 to 13.98 | 2 | −28.28, 13.92, 2 | 0, 3, 5, 5, 5 |
| `bf20_h3` | −15.67 | 11.05 | −46.35 to 15.02 | 0 | −22.51, 12.94, 0 | 6, 17, 1, 17, 5 |
| `bf20_h3_sig2` | −2.89 | 1.42 | −6.84 to 1.06 | 0 | −15.82, 4.10, 0 | 11, 17, 4, 8, 7 |

What the count does not say:
- `sig2` has the widest scatter of the six. Its replicas give −78.2, −1.7, −44.3, −3.4 and
  −9.1 kJ/mol; the first left the site once, at 27.5 ns, and did not come back (corrected on
  2026-11-16: this line said that it never left), and the two that pass are the last two.
  The rule counts passing replicas and that is what was applied; the scatter is reported here.
- The defaults do not give a converged binding free energy of trypsin-benzamidine in 40 ns: a
  run with them is expected to end as NOT CONVERGED, which is what the diagnostics are for.
  Roadmap R2 ("five replicas to convergence") is not met within the length the owner allows.
- With the faster settings the ligand leaves early, the bias at the site overfills, and the
  samples after the first 2 ns rarely visit the site: the estimate from the samples is
  shallow (1 to 4 replicas per setting within 1 kT of the empty funnel) while the bias sum is
  deep. Their difference is one of the convergence checks, and it fires.
- With the bias that was the default until now the ligand stays in the site well beyond 2 ns
  (first exit at 4.5 to 10 ns), so the estimate without the first 2 ns follows the plateau
  mean: −66.7, −48.6, −36.5, −28.0, −18.4 and −14.3 kJ/mol at 10, 20, 30, 40, 60 and 75 ns
  (0 of 5 up to 60 ns, 1 of 5 at 75 ns). The table of that run above holds the numbers of the
  analysis before the amendment.
- Experiment is −26.8 to −30.5 kJ/mol (`examples/trypsin_benzamidine/README.md`). It decided
  nothing.

Third amendment (2026-11-16, owner; FM-D9, second amendment): the scan read without a verdict.
"Replicas that pass" above is the count the rule used. It stays as the record of how the
defaults were chosen; the analysis no longer reports such a count.
- The defaults stay: `sig2` and 40 ns. No criterion that leaves experiment out makes another
  setting better. The smallest interval does not: `bf20_h3_sig2` has it (SEM 1.42) with a
  value of −2.89 kJ/mol, within 1 kT of the empty funnel (−0.78 kJ/mol), which would report
  trypsin-benzamidine as no binding.
- A run with the defaults gives on this system a value with a wide interval: −27.35 kJ/mol,
  95% interval −68.67 to 13.98 (−28.28, −66.93 to 10.37 with the headline of the third
  amendment of FM-D9).
- The value depends on the bias setting, with every estimator that was tried (FM-D9, second
  amendment). dG0 of the set at 40 ns in kJ/mol, with the standard error of the mean (C, R) or
  of the jackknife (P):

| Setting | C: mean of the per-replica estimates | R: mean of per-replica mean-force estimates | P: pooled mean-force estimate |
|---|---|---|---|
| default until 47e84f7 | −27.95 (5.94) | −36.89 (9.39) | −41.67 (6.79) |
| `bf20` | −8.07 (5.14) | −13.94 (7.68) | −5.24 (10.06) |
| `h3` | −12.68 (6.96) | −16.88 (9.31) | −12.73 (13.00) |
| `sig2` | −27.35 (14.88) | −4.06 (2.31), 3 replicas with a value | −30.61 (16.16) |
| `bf20_h3` | −15.67 (11.05) | −11.39 (7.99) | −1.62 (7.26) |
| `bf20_h3_sig2` | −2.89 (1.42) | −4.50 (2.32) | 0.37 (0.85) |
| Spread (max − min) | 25.1 | 32.8 | 42.0 |

  The bias of the mean-force estimates was rebuilt from HILLS and agrees with the COLVAR column
  within 0.56 kJ/mol. Without the fallback to the bias sum for the replica of `sig2` that never
  left the site, C is −14.64 (10.01) there, and the spread is the same.
- Where the dependence comes from: the replicas fall in two groups, about 0 to −9 kJ/mol and
  −30 to −78 kJ/mol. A replica whose ligand left early over an overfilled site reads shallow,
  one whose ligand stayed reads deep, and 40 ns hold too few crossings to average the two.
  That is a limit of the sampling at this length. It is not removed by the estimator, and it
  is what the interval shows.
- Roadmap R2 is reworded: the result of a real system is its value with the interval over the
  replicas, not "five replicas to convergence".

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
`bench/reports/index.json`. Per ligand: dG0 as the mean of the dG0 of the analysed replicas
(FM-D9, as amended) with its 95% Student-t interval, the experimental and the FEP value, the rebinding
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
Amendment (2026-11-16; FM-D9, second amendment): the report follows the analysis without a
verdict. There is one set of statistics, over every ligand with a value of fmeta, with FEP on
the same ligands; the columns for the ligands "whose replicas all passed" are gone. A ligand
is marked "no binding" when its value (the mean of its replicas) is within 1 kT of the empty
funnel; it stays in the statistics. The notes on single replicas are listed as diagnostics.
Why: choosing ligands by a check that almost no replica of 40 ns passes would describe a
subset picked by noise.

### FM-D18 — The back end of a run: NVIDIA's cuPotential plugin in place of OpenMM's nonbonded force and integrator
Status: **Accepted** (2026-11-17, owner request). In force as an option: `run.backend` is
`openmm` unless a configuration says otherwise. Open until measured (plan
`docs/plans/2026-11-17-cupotential-backend.md`, M2 to M4): which parts the plugin provides for
each kind of system, the accuracy criteria of a gate item for the back end, and `auto` as the
default.
`run.backend` names what computes the nonbonded forces and moves the atoms of a production run:
`openmm` (OpenMM's `NonbondedForce` and `LangevinMiddleIntegrator`), `cupotential` (the classes
of `nv_openmm_plugin`, NVIDIA's plugin of OpenMM's CUDA platform, in their place) or `auto`
(cuPotential where its conditions hold, OpenMM otherwise). FM-D1 stands: the engine is OpenMM,
and the plugin replaces objects inside an OpenMM System. `src/fmeta/engine/backend.py` does
that in memory, just before the Context of a run is made: `prepared/system.xml` stays as OpenMM
wrote it, the forces of the funnel (CVs, walls, bias; FM-D2 to FM-D4) are OpenMM's own under
every back end, and the build, the equilibration and the analysis do not use the plugin. Every
place that makes the Context of a run goes through this module: `fmeta run`, and the items of
`fmeta check` that run dynamics on a solvated system.
Conditions, checked by name before a run (`unmet`): the CUDA platform; single or mixed
precision; the plugin is installed and its libraries load; exactly one `NonbondedForce`, with
PME, an Ewald error tolerance of 5e-4 and no mesh set by hand; a box of at least 3 cutoffs
along x, y and z. With `cupotential` an unmet condition is an error before a file is touched.
With `auto` the run is made with OpenMM, and the condition is written to the log, to `run.yaml`
and to `device.json` (`backend_reason`). An error of the plugin that has no name here stops the
run: it is never turned into a run with OpenMM.
What the plugin provides is one set of parts for each kind of system (soluble, membrane), out
of the force, the integrator and the isotropic barostat (`CONFIGURATION`); it has no
replacement of the membrane barostat. Its integrator removes the motion of the centre of mass
itself, so the `CMMotionRemover` leaves the System with it. The plugin sizes its mesh and its
cell list when the Context is made, from the default box of the System: the box of the
equilibrated state is made that box before the Context exists, also when a run continues, so
every Context of a run has the same mesh. The probe of the CVs (FM-D4) is taken out of the
integration before the Context exists.
One run keeps the back end and the parts it began with: its hills and samples are then those
of one set of forces. `run.yaml` and `device.json` name them (`backend`, `backend_parts`); a
`run.yaml` without the entry is a run with OpenMM, and what `run.yaml` states about the system
(particles, barostat) is read before forces are replaced. A run that continues under `auto`
follows its record; another back end by name is refused before a device is chosen or a file is
touched (`settle`), and with walkers before any walker is rolled back. A new replica under
`auto` follows the replicas that are in the work directory. `fmeta status` shows the back end
beside the device; `fmeta run --backend` and `fmeta check --backend` set it for one call.
Benchmark sets (FM-D17): the back end is part of the hash of the configuration; with `openmm`
the hash is the one from before the key existed, so the work directories of the reported sets
go on.
Evidence: a gate record is labelled with the back end its test runs used, and the two back ends
of one commit are two records (`<date>-<commit>-gpu-gates.json` and
`<date>-<commit>-cupotential-gpu-gates.json`). The record of cuPotential holds the verdicts and
the accuracy, and neither a speed nor a duration: NVIDIA's licence of the plugin forbids the
disclosure of benchmark results without permission (clause 2.12; owner, 2026-11-17: no speed
measured with the plugin in a tracked file, a commit message or a text on GitHub).
`scripts/verify_bench_results.py` rejects such a record with a speed, in CI; with NVIDIA's
permission the switch is `PLUGIN_WITHHELD` there.
Not done: the plugin in the equilibration or inside `addMembrane` (1 to 2.5% of the GPU time
of a calculation, and the restraints and stages there would need their own verification);
changing the back end of a run that has begun (its bias would be a sum of hills from two sets
of forces); a silent run with OpenMM where cuPotential was asked for (a calculation slower than
planned without a word, and a default that could pass every gate without using the plugin).
Why: owner request of 2026-11-17, to make fmeta faster with cuPotential and to make it the
default. The plugin is early-access software under NVIDIA's licence, which may end: `openmm`
stays a complete path with its own gate record, and `auto` gives OpenMM, with the reason, when
the plugin is not there.
