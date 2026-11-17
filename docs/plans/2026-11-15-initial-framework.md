# Initial framework

Status: Done (PR #1), on CPU. GPU gates and real-system validation are roadmap R1–R3.
Refs: FM-D1..FM-D12

## Context

The repository was empty. The owner asked for a funnel-metadynamics framework with OpenMM as
the engine, running on GPUs, with automatic funnel set-up from a known ligand pose (reference:
BioSimSpace) and membrane support through OpenMM's membrane builder; then for OpenMM-only
input and a route that needs nothing but a protein PDB and a ligand SDF.

## Approach

Core (funnel, engine, analysis) depending only on openmm / numpy / scipy / pyyaml; system
preparation with optional packages. The funnel is placed on the dry complex and the periodic
box is sized from it. Three things in the reference recipe were changed on purpose: the bias
table is clamped (it is zero outside its grid), the box follows the funnel, and dG0 integrates
the bound basin with the effective cross-section (FM-D4, FM-D6, FM-D9).

Milestones as built: M0 scaffold; M1 CVs and walls; M2 bias engine, logs, restart; M3 analysis
and the solvable toy system; M4 automatic funnel; M5 system building, equilibration, workflow,
command line; M6 membrane; M7 replicas, walkers, acceptance checks; harness.

## Invariants

All of `.claude/rules/numerics-invariants.md` were established by this work.

## Verification

- CPU: `pytest` and `pytest -m slow`; toy system against its exact binding constant; restart
  byte-identical; table equal to `openmm.app.Metadynamics`; full pipeline on a 35-residue
  protein with benzene, in water and in a POPC bilayer built by `addMembrane`.
- Not done here (no GPU, no conda, no access to structure databases): the GPU gates, the
  OpenFF / GAFF route with the real packages, runs on real protein-ligand systems.
