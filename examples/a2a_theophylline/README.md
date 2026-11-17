# Adenosine A2A receptor with theophylline

A membrane target with a weak ligand: the human adenosine A2A receptor in POPC with
theophylline, from the crystal structure 5MZJ. **State of 2026-11-16:** the steps below ran
to a binding free energy with the official defaults: dG0 = −18.6 kJ/mol (−4.45 kcal/mol), 95%
interval −33.3 to −4.0 kJ/mol, from five replicas of 40 ns. The measurement on the simulated
construct is −33.0 kJ/mol (−7.89 kcal/mol) ("Result" below).

Numbers in this file come from four places, and are marked: **[entry]** the header of the
RCSB entry 5MZJ, **[file]** the OPM file and fmeta runs on it (both read or run on 2026-11-16),
**[ChEMBL]** activity records of ChEMBL 37 (read on 2026-11-15 UTC; the papers behind them were
not read), and **[unconfirmed]** a note from planning whose source nobody has read yet.

## The system

- Entry 5MZJ: "Crystal structure of stabilized A2A adenosine receptor A2AR-StaR2-bRIL in
  complex with theophylline at 2.0 Å resolution", X-ray, 2.00 Å, deposited 2017-01-31. Cheng,
  Segala, Robertson, Deflorian, Doré, Errey, Fiez-Vandal, Marshall, Cooke, *Structure* 25, 1275
  (2017), doi:10.1016/j.str.2017.06.012. **[entry]**
- Construct: human A2A (UniProt P29274) residues 2-208 and 219-318, with cytochrome b562
  (BRIL, residues 1001-1106) between them in place of residues 209-218 of the third
  intracellular loop. **[entry]**
- Source of the coordinates: the file oriented by OPM,
  `https://opm-assets.storage.googleapis.com/pdb/5mzj.pdb` (352 269 bytes, sha256
  `4be1818c64db2fa8d2f63b3b2ec74fe25111f50b1d18dbc06b8d72f42ffd3d04`). The ligand is cut from
  the same file, because OPM and RCSB coordinates are in different frames. **[file]**
- In the OPM file: chain A, residues 0-208 and 219-305; the sodium ion `NA 2400`;
  theophylline `TEP 2401` (13 atoms); 3 cholesterol, 22 oleic acid, 7 monoolein, 116 waters;
  1282 dummy atoms for the membrane planes at z = ±1.61 nm. The BRIL is not in it, so
  `fmeta split --drop` is not needed here. With an RCSB file of this construct it is
  `--drop A:1001-1106`. **[file]**

## What is simulated, and what a comparison with wild type means

The receptor is simulated as crystallised. Nine engineered mutations lie in the modelled
part: A54L, T88A, R107A, K122A, N154A, L202A, L235A, V239A, S277A (a tenth, G318A, is beyond
the last modelled residue). **[entry]** The BRIL is absent and residues 209-218 are not
rebuilt: the chain is split between LEU 208 and GLU 219 and capped there (N-methylamide and
acetyl), so no charged ends are made inside the receptor. The sodium ion is kept
(`system.keep_hetero`) and treated as part of the receptor; crystal waters, cholesterol and
the crystal's lipids are removed.

Closest heavy-atom distance of the mutated residues in the OPM file **[file]**:

| Residue | to theophylline | to the sodium ion |
|---|---|---|
| S277A | 6.6 Å | 6.6 Å |
| T88A | 7.4 Å | 5.4 Å |
| A54L | 12.0 Å | 8.5 Å |
| V239A | 16.3 Å | 9.7 Å |
| N154A, L235A, K122A, L202A, R107A | 17.9 to 35.1 Å | 12.0 to 33.0 Å |

For a comparison with measurements this means:

- The computed dG0 is the affinity of theophylline for this stabilised construct, not for
  the wild-type receptor. No mutated side chain touches the ligand, but two are in the second
  shell of the site and of the sodium ion, and the construct was made to stabilise the receptor
  (the entry's title; in which state, and by how much the affinity of an antagonist changes,
  was not read for this example). An indirect effect on the affinity is possible and is not
  estimated here.
- A wild-type measurement is therefore a reference with a caveat. A difference of the size
  of the effect of the mutations cannot be put down to the method. The clean reference is a
  measurement on the StaR2 receptor: the first row of the table below.
- Mutating back to wild type in the model is not done: it would put wild-type side chains on
  a backbone that was determined for the mutant.
- The simulation is at 300 K (the official default) with POPC only. The conditions of a
  measurement (temperature, sodium concentration, membrane preparation) have to be read with
  its value.

## Experimental values

Theophylline (CHEMBL190) at the human A2A receptor (CHEMBL251). **[ChEMBL]** The free energy is
RT ln K at 298.15 K.

| Measurement | Value | ΔG (kcal/mol) | Document of the record |
|---|---|---|---|
| Kd, surface plasmon resonance, purified A2A-StaR2 in lipid discs: the stabilised receptor of the crystal | 1.64 µM | −7.89 | K4DD kinetics data set, doi:10.6019/CHEMBL3885741 |
| Kd, surface plasmon resonance, a stabilised human A2A (StaR) on the chip | 3.16 µM | −7.50 | Zhukov et al., J. Med. Chem. 54, 4312 (2011), doi:10.1021/jm2003798 |
| Kd, surface plasmon resonance, wild-type human A2A | 3.63 µM | −7.42 | Shepherd et al., ACS Med. Chem. Lett. 13, 1172 (2022), doi:10.1021/acsmedchemlett.2c00099 |
| Ki, radioligand displacement, human A2A in cell membranes, six documents | 1.7 to 18.5 µM | −7.87 to −6.46 | doi:10.1039/C5MD00573F, 10.1016/j.bmcl.2012.10.112, 10.1021/jm900413e, 10.1021/jm051268+, 10.1021/jm0504149, 10.1016/j.bmc.2010.02.014 |
| Median of the 24 Ki records with an exact value | pKi 5.17 (6.8 µM) | −7.05 | ChEMBL 37 |

- The measurements span 1.4 kcal/mol. A computed value is compared with the range, and with the
  first row for the construct that is simulated; no single number is "the" experimental value.
- Left out of the range: one record of 0.6 nM (Bioorg. Med. Chem. Lett. 15, 609 (2005)), four
  orders of magnitude from every other, and records filed under the human target whose assay is
  bovine striatum (21 to 25 µM).
- Not read: the papers themselves, and whether the paper of the structure gives an affinity of
  theophylline for the construct.
- **[unconfirmed]** A funnel-metadynamics value of −6.89 ± 0.14 kcal/mol from a preprint was
  noted during planning. Its source and conditions were not read: it is not a reference here.

## Steps

Everything runs in the official image, from the repository root. `runs/` is ignored by git:
the structure files stay out of the repository.

```bash
mkdir -p runs/a2a_theophylline
cp examples/a2a_theophylline/fmeta.yaml runs/a2a_theophylline/

# the OPM file (checks the checksum above)
scripts/in_docker.sh python examples/a2a_theophylline/fetch.py runs/a2a_theophylline/download

# protein.pdb (everything but the ligand) and ligand.sdf (theophylline with hydrogens)
scripts/in_docker.sh fmeta split runs/a2a_theophylline/download/5mzj_opm.pdb \
    --resname TEP --smiles "Cn1c(=O)c2[nH]cnc2n(C)c1=O" -o runs/a2a_theophylline/inputs

# build, look at the funnel, equilibrate (GPU 0 of the host)
scripts/in_docker.sh --gpus 0 fmeta prepare runs/a2a_theophylline/fmeta.yaml -o runs/a2a_theophylline/work
scripts/in_docker.sh fmeta funnel runs/a2a_theophylline/work
scripts/in_docker.sh --gpus 0 fmeta equilibrate runs/a2a_theophylline/work --device 0

# production and analysis
scripts/in_docker.sh --gpus 0,1,2,3 --detach --name a2a fmeta run runs/a2a_theophylline/work --gpus 0,1,2,3
scripts/in_docker.sh fmeta status runs/a2a_theophylline/work
scripts/in_docker.sh fmeta analyze runs/a2a_theophylline/work --plot
```

The SMILES is neutral theophylline with the hydrogen on N7: a choice of this example.
`docker stop a2a` stops the run; the same `fmeta run` command continues it from the last
checkpoints.

In the official image the production run is made with NVIDIA's cuPotential plugin of OpenMM:
`run.backend` is `auto` by default (FM-D18, `docs/design/engine.md`). With a membrane the
plugin provides the nonbonded force and the integrator, and the membrane barostat is OpenMM's.
`fmeta status` names the back end of every replica; `run: {backend: openmm}` in `fmeta.yaml`
gives OpenMM's own forces and integrator.

## Result

Measured on 2026-11-16 (UTC): the steps above with the official defaults (5 replicas of 40 ns)
on commit 47e84f7 in the official image. One difference from the `fmeta run` line above: the
five replicas ran at the same time on one GPU with NVIDIA MPS
(`scripts/in_docker.sh --mps --gpus 4 ... fmeta run runs/a2a_theophylline/work --gpus 0 --per-gpu 5`),
at 106 to 107 ns/day each (221 101 atoms, 4 fs). The run took 9 h 5 min. It was stopped once
with `docker stop` after 35 min, to give each replica one CPU thread, and the same `fmeta run`
command continued the five replicas from their checkpoints (`schedule.jsonl`); each ended with
40 000 hills. The numbers are those of `fmeta analyze runs/a2a_theophylline/work --plot` with
the analysis of the third amendment of FM-D9 (`docs/design/engine.md`, code of c59ec99): the
value of a replica is the mean over the second half of the run of dG(t) from the accumulated
bias, with a block error.

The run was made with OpenMM's own forces and integrator: commit 47e84f7 is from before
`run.backend` existed, and today that is `run.backend: openmm`. A run of the full length with
the cuPotential back end has not been made.

| Quantity | fmeta | Experiment |
|---|---|---|
| dG0 | −18.64 kJ/mol (−4.45 kcal/mol) | −33.0 kJ/mol (−7.89 kcal/mol) on the simulated construct; all measurements −33.0 to −27.0 kJ/mol (−7.89 to −6.46 kcal/mol) |
| 95% interval of the five replicas | −33.31 to −3.97 kJ/mol (standard error 5.28) | |
| Each replica, with its block error | −27.40 ± 2.01, −4.60 ± 1.05, −33.60 ± 7.00, −11.17 ± 0.82, −16.41 ± 4.99 kJ/mol | |
| From the samples after the first 2 ns (cross-check) | −22.33, −4.47, −28.40, −4.19, −16.96 kJ/mol | |
| Rebinding events per replica | 2, 7, 1, 3, 9 | |
| Replicas with a note in the diagnostics | 5 of 5 | |

The value is 14.4 kJ/mol (3.4 kcal/mol) above the measurement on the simulated construct. The
experimental values lie inside the 95% interval, at its lower end; the interval is 29 kJ/mol
wide. It is reported as measured: nothing was tuned on this system, and the five replicas
together are the measurement (FM-D9, second amendment).

What the diagnostics of `fmeta analyze` say:

- The site is closed for a 0.3 nm probe in the crystal structure ("Preparation" below), and
  theophylline still left and came back along this funnel in every replica: s was above
  3.0 nm for the first time at 0.8 to 1.9 ns (`COLVAR`), and the ligand returned from the
  unbound region to the site 1 to 9 times in 40 ns. Two replicas have fewer than 3 returns.
- One replica (−4.60 kJ/mol) is within 1 kT of what an empty funnel gives (−2.18 kJ/mol): for
  that replica alone the analysis reports no binding beyond the funnel volume. The value of the
  five together is not within 1 kT of it.
- In four of the five replicas an estimate from the samples (after the first 2 ns, or of the
  second half) differs from the value by 5.1 to 25.0 kJ/mol.
- In four of the five replicas dG(t) from the bias changed by 4.7 to 41.8 kJ/mol over the last
  half of the run. The bias fills the free-energy surface for as long as the run lasts, and
  the difference between the site and the unbound region moves with it
  (`runs/a2a_theophylline/work/summary.png`).

The run can be extended exactly (`fmeta run runs/a2a_theophylline/work --length-ns 80`); that
was not done, because nothing runs longer than 40 ns by the owner's decision of 2026-11-16.

## State (2026-11-16)

- `fmeta prepare` builds the system from the files as `fmeta split` writes them: no editing of
  the protein file is needed (the code before b54292f stopped at SER 305, and bonded across the chain break).
- The steps ran from the download to dG0 ("Result" above).
- Open choices, none examined yet:
  - The 116 crystal waters are removed (`system.keep_water` is off), also those near the ligand
    and the sodium ion.
  - Protonation states are those of the hydrogen placement at pH 7, because the OPM file has no
    hydrogens: all six histidines are neutral with the hydrogen on ND1. The prepared structure
    of the benchmark set (`examples/benchmarks/a2a`, from 5OLZ) has HIS 264 and HIS 278 doubly
    protonated instead.
  - Run length, replicas and bias are the official defaults. Settings for a narrower interval
    are deferred by the owner until everything is shown to work.

## Preparation (2026-11-16)

The input as `fmeta split` writes it, this `fmeta.yaml`, commit b54292f, image `fmeta:dev`.
**[file]**

| Quantity | Value |
|---|---|
| What the build did with the protein file | removed 3 CLR, 22 OLA, 7 OLC, 116 HOH and the 1282 dummy atoms; kept the sodium ion; split the chain between LEU 208 and GLU 219 (C-N 1.15 nm) and capped it there (N-methylamide on LEU 208, acetyl on GLU 219); rebuilt 1 missing heavy atom; added 2353 hydrogens at pH 7; four disulfide bonds (71-159, 74-146, 77-166, 259-262) |
| Atoms | 221 110: 484 POPC (64 856 atoms), 50 437 waters, the receptor with its sodium ion, theophylline (21), the ions of the solvent |
| Box as built | 12.74 x 13.37 x 13.24 nm |
| Funnel | exit direction (-0.09, 0.26, 0.96): 16 deg from +z, into the extracellular water. Lower wall 0.70, bound state at s = 1.20, inflection 3.20, upper wall 5.00 nm |
| Validation | no errors. Two warnings: no straight open path from the site to bulk (least obstructed direction, blocking 0.40), and the site is closed for a 0.3 nm probe in this structure: unbinding will need the protein to open |
| Pose and funnel | theophylline 0.43 nm inside the wall at s = 1.20 nm; unbound region 1.04 nm clear of the solute and 1.90 nm above the membrane; no lipid in the funnel |
| Box rule | z 13.24 nm against the 12.26 nm asked for, x and y against 5.00 nm; the unbound ligand is 4.00 nm from the nearest periodic image |

The bilayer of `Modeller.addMembrane` is made of whole lipid patches of 6.37 x 6.69 nm. In
this frame it is 2 x 2 patches with `system.padding: 1.5` and also with the preset's 1.2 nm
(221 176 atoms, measured with eeb5302), so here the padding does not decide the cost. The
receptor of the benchmark set, in the frame of 5OLZ, fits 2 x 1 patches with 1.2 nm (about
116 000 atoms).

The equilibration of the run of "Result" (commit 47e84f7, 300 K; 100 ps heating, 400 ps
release, 2 ns free; 2 fs, NPT). The build of that commit has 221 101 atoms. **[file]**

| Quantity | Value |
|---|---|
| Wall energy on the bound ligand, free stage | mean 0.000, max 0.000 kJ/mol; ligand 0.46 nm inside the wall at s = 1.36 nm |
| Box after the equilibration | 12.57 x 13.19 x 12.99 nm. Area per lipid 0.685 nm² (box area over 242 lipids per leaflet, receptor included); box area 168.2 nm² in the first 200 ps of the free stage and 166.2 nm² in the last |
| Funnel checks on the equilibrated structure | no errors: unbound region 1.21 nm clear of the solute and 1.79 nm above the membrane; no lipid in the funnel; box z 12.99 nm against 12.28 nm; the unbound ligand is 4.05 nm from the nearest periodic image |

Earlier measurements on a placeholder model of the same file (charged ends at the break,
310 K, 221 143 atoms):

| Quantity | Value |
|---|---|
| fmeta's own orientation estimate | 14 deg from the OPM normal, midplane 0.2 nm off (confident): `orient: check` would pass it, `orient: auto` would turn a correct input by 14 deg |
| Equilibration, 1 ns (100 ps heating, 400 ps release, 500 ps free) | 558 s; about 167 ns/day at 2 fs, NPT, one NVIDIA RTX PRO 6000 Blackwell |

The warning about the closed site was the point to watch in the first run. Theophylline left
along this funnel and came back in every replica within 40 ns ("Result" above).
