# Benchmark set: adenosine A2A receptor

Eight ligands of the A2A series of the GPCR set, a receptor in a POPC bilayer.

```bash
scripts/in_docker.sh fmeta benchmark prepare examples/benchmarks/a2a -o runs/bench-a2a
scripts/in_docker.sh --gpus 0,1,2,3 --detach --name bench-a2a \
    fmeta benchmark run examples/benchmarks/a2a -o runs/bench-a2a --gpus 0,1,2,3
scripts/in_docker.sh fmeta benchmark status examples/benchmarks/a2a -o runs/bench-a2a
scripts/in_docker.sh fmeta benchmark report examples/benchmarks/a2a -o runs/bench-a2a
```

`prepare` takes the three pilot ligands (`4q`, `4d`, `4e D`: the weakest, a middle and the
strongest binder); `--ligands all` adds the other five to the same work directory.

## Source and licence

Structures and reference values come from the Schrödinger
[public binding free energy benchmark](https://github.com/schrodinger/public_binding_free_energy_benchmark),
commit `a65d86b782a678c75388566ce00446e45283ee63` (`main` and tag `v2.0` on 2026-11-16),
MIT licence (Copyright (c) 2023 Gregory A. Ross). No file of it is stored here: `set.yaml`
names each file at that commit with its sha256, and `fmeta benchmark prepare` downloads and
verifies them.

| File | Content |
|---|---|
| `fep_benchmark_inputs/structure_inputs/gpcrs/a2a_hip278_protein.pdb` | chain A, residues -1 to 305 without 209-218, with hydrogens, HIS 278 doubly protonated, one sodium ion; from PDB 5OLZ |
| `fep_benchmark_inputs/structure_inputs/gpcrs/a2a_hip278_ligands.sdf` | 29 records: the ligands in their bound poses, some in two rotamer or protonation states |
| `21_4_results/ligand_predictions/gpcrs/a2a_hip278_sbpkacorr_out.csv` | experimental and FEP+ values, one row per compound |

The orientation in the membrane comes from the entry of 5OLZ in the
[OPM database](https://opm.phar.umich.edu/) (`https://opm-assets.storage.googleapis.com/pdb/5olz.pdb`,
half thickness of the hydrophobic slab 15.7 Å, midplane at z = 0). `prepare` superposes the
CA atoms of the benchmark structure on those of the same residue number in that file and
moves every ligand with the protein. The OPM file is not stored here either; OPM has no
versions, so the sha256 in `set.yaml` is what pins it. Its terms of use were not confirmed
while this set was written: check them before redistributing the file.

## The names of the ligands: `D` and `U`

The series was prepared by Deflorian et al., J. Chem. Inf. Model. 60, 5563 (2020),
doi:10.1021/acs.jcim.0c00449. The supplementary information of Ross et al. (below) says:
"Deflorian et al. prepared seven of the ligands with 2 rotamer states for one of the
R-groups" (4b, 4e, 4j, 4k, 4l, 4m, 4r), and of one of them that it "was modeled by Deflorian
et al. in two orientations: "up" and "down"". `U` and `D` are these two states, up and down:
the same compound with one substituted ring turned over (in the SDF file `4b D` and `4b U`
differ in the positions of 8 atoms; the chlorine moves by 5.3 Å). The prediction table has one
row per compound, named after one of its states, and its FEP+ value belongs to the compound.
This set starts funnel metadynamics from the `D` pose of 4b and 4e; a bound state that is
sampled long enough holds both.

For the phenols 4e, 4f and 4o the source also has a deprotonated form (charge -1) beside the
doubly protonated HIS 278, and the FEP+ value of these compounds includes a pKa correction.
This set uses the neutral form (`4e D`).

## Reference values (`reference.csv`)

The values are those of `a2a_hip278_sbpkacorr_out.csv`, rounded to four decimals (the file
holds single-precision numbers); the same experimental values are in the `r_exp_dg` field of
the SDF records.

- **Experiment.** The source gives a free energy per ligand in kcal/mol, not the measured Ki,
  and not the temperature of the measurement: `exp_ki_nm` and `exp_temperature_k` are empty,
  and the value is used as given. `exp_doi` is the reference the benchmark names for the
  series (Deflorian et al., above); the papers of the measurements were not read for this set.
- **FEP.** FEP+ predictions of Ross et al., Commun. Chem. 6, 222 (2023),
  doi:10.1038/s42004-023-01019-9. They are relative free energies (`fep_kind: relative`):
  `21_4_results/README.md` of the repository says "the CSV files contained here have predicted
  _absolute_ binding free energies that are known up to an unknown additive constant. The
  additive constant was set such that the mean of the predicted absolute binding free energies
  equals the mean of experimental binding free energies." The shift was made over the 20 rows
  of the file (means: experiment -10.9689, prediction -10.9660 kcal/mol). On the eight
  ligands of this set the two means differ, and an absolute method such as funnel
  metadynamics has no such shift at all: a report therefore gives every measure a second time
  after the mean error on the same ligands is removed, and that is the fair comparison
  with FEP+.
- **Absolute FEP** (`afe_*`): empty; no absolute values for this series were looked up.

The whole series as its source reports it: 17 compounds, 28 perturbations, pairwise RMSE 1.58
and MUE 1.21 kcal/mol, R² 0.50, Kendall tau 0.49. Its largest error among the ligands of this
set is 4a (experiment -9.41, FEP+ -7.15 kcal/mol).

## Configuration

`config` in `set.yaml`: a POPC bilayer around z = 0 with `orient: input` (the frame is that
of the superposition), the sodium ion kept (`keep_hetero: [NA]`), 10 ns of free
equilibration; everything else is the default of fmeta, at 300 K. The standard state of
every value is 1 mol/L.

## What is not settled

- The temperature of the measurements is not in the source; a report compares the values as
  given, without a correction to 300 K.
- **Do not run this set for results yet.** Measured on 2026-11-16 with the input handling of
  the commit that added this set (build of `4q`, 110352 atoms, 228 POPC): the protein is not
  simulated as the file gives it, and `prepare.log` reports it as `[ok ] protein_input`.
  - The file ends the chain at the gap with hydrogens (`HXT` on LEU 208, `H1`/`H2` on
    GLU 219) and has no `TER` there. LEU 208 and GLU 219 become neighbours of one chain and
    get a peptide bond: 1.57 nm long in the prepared system (r0 0.1335 nm, 4.2e5 kJ/mol).
  - Because those hydrogens fit no residue template, all 2354 hydrogens of the file are
    removed and 2349 are built at pH 7: HIS 264 and HIS 278 (both doubly protonated in the
    file) become neutral, and HIS 75, 155 and 250 change from the epsilon to the delta form.
  - SER 305 (`HXT`, no `OXT` in the file) gets a charged carboxylate.
  - The sodium ion is kept.
  The set needs an input handling that keeps the ends of a chain at a gap and the hydrogens
  of the file; then check `prepare.log` and `prepared/report.json` of a ligand again.
- The hydrophobic thickness of OPM (2 x 15.7 Å) is not used: the bilayer is built with the
  default `membrane.half_thickness`.
