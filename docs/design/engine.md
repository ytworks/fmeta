# Engine design

Status: **Accepted** (2026-11-15, PR #1; FM-D11 amended twice the same day: an owner decision,
and the first run with the real ligand-parameter packages).
Applies to: `src/fmeta/` (all of it), `examples/`.

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

### FM-D2 — Two CVs from centroid distances, explicit weights, periodic boundaries on
`s = (dL0² + d01² − dL1²) / (2 d01)`, `rho = sqrt(max(dL0² − s², 0) + eps²)` over the groups
ligand / p0 / p1. Group weights are stored in the funnel (ligand heavy atoms by element mass,
anchors as given), never taken from particle masses.
Why: the same CVs as `d cos(angle)` / `d sin(angle)` but with a bounded gradient on the axis;
explicit weights survive hydrogen mass repartitioning and massless test anchors; GPU platforms
wrap molecules, so the distances must be minimum-image.

### FM-D3 — Walls: one force for the sigmoid side wall and the lower / upper walls
`R(s) = width / (1 + exp(steepness (s − inflection))) + buffer`; harmonic penalties, default
force constants 10⁴ kJ/mol/nm². The walls live in their own force group.
Why: the BioSimSpace funnel shape, so parameters carry over; a stiff wall keeps the soft-wall
correction of FM-D9 small and the CVs inside the bias grid.

### FM-D4 — Own bias engine on the mechanism of `openmm.app.Metadynamics`
A 2-D table (`Continuous2DFunction`) refreshed after every hill, with the lookup clamped to
the grid and hills mirrored at `rho = 0`; 3 grid points per sigma.
Why: OpenMM tables are zero outside their range (the bias would vanish when a CV overshoots);
hill logging, exact restart and bias sharing need access the stock class does not give. A test
keeps the table equal to the stock class for the same hills. OpenMM refits the spline on the CPU
at every update (about 0.5 µs per grid point measured), hence the coarse grid.

### FM-D5 — HILLS is the record; one atomic checkpoint bundle; exact restart
PLUMED-layout HILLS / COLVAR. A checkpoint is one zip (binary checkpoint, State XML, bias
grid, counters) replaced atomically; a restart rolls HILLS / COLVAR / DCD back to it.
Why: a crash between files must not leave a mixed state; on the Reference platform a split run
reproduces the continuous one byte for byte, which makes restart testable.

### FM-D6 — The funnel is placed before solvation and the box is sized from it
Tumbling solute: image distance ≥ max(bounding sphere + padding, funnel reach + solute radius +
cutoff, 2 (reach + 0.3 nm)), rhombic dodecahedron. Membrane: box height from the funnel, xy
from the lipid patch.
Why: with minimum-image CVs a box smaller than twice the reach lets the ligand change image
before the upper wall; a box sized from the solute alone does not know the funnel.

### FM-D7 — Automatic placement by path clearance, weighted alpha-carbon anchors
Straight paths from the ligand are scored by the solute (and membrane) they cross and by how
soon they reach bulk; anchors are alpha carbons weighted so that the axis lies on that
direction; inflection, walls and width follow the pocket geometry. A BioSimSpace-style grid
average is kept as `method: bss`.
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
Why: the development machine has no GPU; correctness has to be established where it can be
(exact answers on CPU) and the rest gated on the machine that runs production.
