# Adenosine A2A receptor with theophylline (draft)

A membrane target with a weak ligand: the human adenosine A2A receptor in POPC with
theophylline, from the crystal structure 5MZJ. **Draft of 2026-11-16.** The inputs can be made
and the system builds and equilibrates, but only with the workaround of "State of the draft";
no metadynamics has been run and no free energy is reported.

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
rebuilt: the chain is broken between LEU 208 and GLU 219. The sodium ion is kept
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
  was not read for this draft). An indirect effect on the affinity is possible and is not
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

# production and analysis (not run yet for this system)
scripts/in_docker.sh --gpus 0,1,2,3 --detach --name a2a fmeta run runs/a2a_theophylline/work --gpus 0,1,2,3
scripts/in_docker.sh fmeta analyze runs/a2a_theophylline/work --plot
```

The SMILES is neutral theophylline with the hydrogen on N7: a choice of this draft.

## State of the draft

With the code of 2026-11-16 `fmeta prepare` stops on the input as `fmeta split` writes it:

```
fmeta: error: the protein does not fit the force field's residue templates (No template found
for residue 295 (SER). The atoms and bonds in the residue match SER, but the set of externally
bonded atoms is missing 1 C atom. Is the chain missing a terminal capping group? ...
```

Two things in the protein model are open, both belonging to the protein-model part of FM-D15
(`docs/design/engine.md`), not to this example:

1. In the OPM file the sodium ion follows SER 305 without a `TER` record. It is read into the
   protein's chain, so SER 305 is no chain end and gets no terminal oxygen: the error above.
2. Without a `TER` at the chain break, LEU 208 C and GLU 219 N are bonded across 1.15 nm and
   nothing stops the build. The intended model has caps (ACE / NME) at the break.

The numbers below were made with a placeholder: a `TER` record at the break and before the
hetero atoms, which gives two chains with charged ends at the break. Insert it after `fmeta
split` (the result is the input of the runs below, byte for byte):

```bash
awk 'substr($0,1,4)=="ATOM" && substr($0,23,4)+0==219 && !gap {print "TER"; gap=1}
     substr($0,1,6)=="HETATM" && !het {print "TER"; het=1} {print}' \
    runs/a2a_theophylline/inputs/protein.pdb > runs/a2a_theophylline/inputs/protein_ter.pdb
mv runs/a2a_theophylline/inputs/protein_ter.pdb runs/a2a_theophylline/inputs/protein.pdb
```

Open choices, none examined yet: the 116 crystal waters are removed (`system.keep_water` is
off), also those near the ligand and the sodium ion; protonation states are the defaults at
pH 7; run length and replicas are the official defaults (5 × 500 ns), and the pilot run says
whether that is enough here.

## First preparation (2026-11-16)

Placeholder input as above, the configuration of that day (this `fmeta.yaml` with
`system.temperature: 310.0` and `equilibration.free_ps: 500`), one NVIDIA RTX PRO 6000
Blackwell, image `fmeta:dev`. **[file]** Not repeated at 300 K yet.

| Quantity | Value |
|---|---|
| Atoms | 221 143: protein 4630, theophylline 21, 484 POPC (64 856), 50 451 waters, 137 Na+ and 145 Cl- of the solvent, the kept Na+ |
| Box as built | 12.74 x 13.38 x 13.24 nm |
| Time | `prepare` 101 s, of which the membrane build 64 s |
| Funnel | exit direction (-0.09, 0.26, 0.96): 16 deg from +z, into the extracellular water. Lower wall 0.70, bound state at s = 1.20, inflection 3.20, upper wall 5.00 nm |
| Validation | no errors. Two warnings: no straight open path from the site to bulk (least obstructed direction, blocking 0.40), and the site is closed for a 0.3 nm probe in this structure: unbinding will need the protein to open. Unbound region 1.04 nm clear of the solute and 1.90 nm above the membrane; no lipid in the funnel |
| fmeta's own orientation estimate | 14 deg from the OPM normal, midplane 0.2 nm off (confident): `orient: check` would pass it, `orient: auto` would turn a correct input by 14 deg |
| Equilibration, 1 ns (100 ps heating, 400 ps release, 500 ps free) | 558 s; about 167 ns/day at 2 fs, NPT |
| Wall energy on the bound ligand, free stage | mean 0.000, max 0.000 kJ/mol; ligand 0.40 nm inside the wall at s = 1.21 nm |
| Box after 1 ns | 12.74 x 13.38 x 12.76 nm (height -3.7%); the box rule asks for z >= 12.28 nm. Area per lipid 0.70 nm² (box area over 242 lipids per leaflet, receptor included) |

The warning about the closed site is the point to watch in the pilot run: whether theophylline
leaves along this funnel within the run length has not been tried.
