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

With the defaults a new work directory runs 5 replicas of 100 ns per ligand (FM-D16, fourth
amendment, 2026-11-25). The result below was made with 40 ns, the default of then:
`fmeta benchmark prepare ... --length-ns 40` repeats that plan. A work directory keeps the
length its ligands were prepared with.

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
  metadynamics has no such shift at all: a report therefore gives the MUE and the RMSE a second time,
  after the mean error on the same ligands is removed, and that is the fair comparison
  with FEP+.
- **Absolute FEP** (`afe_*`): empty; no absolute values for this series were looked up.

The whole series as its source reports it: 17 compounds, 28 perturbations, pairwise RMSE 1.58
and MUE 1.21 kcal/mol, R² 0.50, Kendall tau 0.49. Its largest error among the ligands of this
set is 4a (experiment -9.41, FEP+ -7.15 kcal/mol).

## Configuration

`config` in `set.yaml` is the preset `membrane` (a POPC bilayer around z = 0, 2 ns of free
equilibration) with two differences: `orient: input` (the frame is that of the
superposition) and the sodium ion kept (`keep_hetero: [NA]`). Everything else is the official
default, at 300 K. The standard state of every value is 1 mol/L.

The back end of the runs is the default as well (`run.backend: auto`, FM-D18 in
`docs/design/engine.md`): in the official image a new work directory runs with NVIDIA's
cuPotential plugin of OpenMM (the nonbonded force and the integrator; the membrane barostat is
OpenMM's), and a report names the back end of its replicas. The result below was made with
OpenMM's own forces and integrator, before `run.backend` existed. A work directory of that time
has another configuration than the set asks for today, and `fmeta benchmark` refuses to go on
in it; with `run: {backend: openmm}` added to `config` in `set.yaml` the set is the one it was
prepared for.

`Modeller.addMembrane` builds the bilayer from whole lipid patches of 6.37 x 6.69 nm, so the
size of the system moves in steps. For `4q` in this frame:

| `system.padding` | Membrane | Box (nm) | Atoms |
|---|---|---|---|
| 1.2 nm (the preset, used here) | 2 x 1 patches, 228 POPC | 12.74 x 6.69 x 13.92 | 115 625 (c41c08b) |
| 1.5 nm (the set until 2026-11-16) | 2 x 2 patches | 12.74 x 13.37 x 13.92 | 231 482 (eeb5302) |

Both pass the box rule (the unbound ligand is 2.54 nm, or 4.31 nm, from the nearest periodic
image). The set was changed to 1.2 nm before any ligand was equilibrated: the larger value
doubled the cost and no rule asks for it.

The three pilot ligands as built on c41c08b (`fmeta benchmark run --until prepare`):

| Ligand | Atoms | Box height (nm) | Asked for by the box rule (nm) | Pose inside the wall (nm) | Nearest periodic image of the unbound ligand (nm) |
|---|---|---|---|---|---|
| `4q` | 115 625 | 13.92 | 12.89 | 0.34 | 2.54 |
| `4d` | 117 494 | 14.11 | 13.06 | 0.23 | 2.50 |
| `4e D` | 116 949 | 14.07 | 13.02 | 0.32 | 2.43 |

No error in the validation of any of them; each has the two warnings of the example (no
straight open path from the site to bulk, site closed for a 0.3 nm probe). The unbound region
is 2.17 to 2.24 nm above the membrane.

## The protein file as fmeta reads it

Measured on 2026-11-16 with commit eeb5302 (`fmeta benchmark prepare`, then
`fmeta benchmark run --until prepare` for `4q`; `inputs/superposition.json`, `4q/prepare.log`
and `4q/prepared/report.json` of the work directory).

| Point | Result |
|---|---|
| Superposition on the OPM file | 297 CA atoms, RMSD 0.112 nm (the limit in `set.yaml` is 0.2 nm) |
| Histidines | read from the hydrogens of the file: HIS 264 and HIS 278 doubly protonated, HIS 75, 155, 230 and 250 with the hydrogen on NE2 |
| Sodium ion | kept (`NA 2400`), 13.5 Å from the nearest heavy atom of `4q` |
| Break between LEU 208 and GLU 219 (C-N 1.57 nm) | chain split there, N-methylamide on LEU 208 and acetyl on GLU 219 |
| Disulfide bonds | four: 71-159, 74-146, 77-166, 259-262 |
| Hydrogens | the 2354 hydrogens of the file did not match the force field's names and were rebuilt, in the protonation states read before |
| System of `4q` | built with `system.padding: 1.5`, the value of the set at that commit: 231 482 atoms; no error in the validation. The systems of the set as it is now: "Configuration" above |

Two differences from the model of the FEP+ input remain:

- In the source file the ends at the break and the last residue are neutral: LEU 208 and
  SER 305 end in an aldehyde (`HXT` on the carbonyl carbon, no `OXT`), GLU 219 starts with a
  neutral amine. The force field has no such end groups: the break gets the two caps above
  (neutral as well), and SER 305 gets the missing `OXT`, a charged carboxylate. SER 305 is
  35 Å from the nearest heavy atom of `4q`. GLY -1 is a charged ammonium in both models.
- Hydrogen positions are fmeta's, not those of the file (heavy atoms are unchanged).

## Result

The whole set with the configuration above: 8 ligands, 5 replicas of 40 ns each, every replica
to its end. The simulations ran from commit 47e84f7 on 2026-11-16 (06:48 to 18:51 UTC, beside
the TYK2 set and the examples on 8 GPUs, several replicas per GPU with NVIDIA MPS). Report:
`bench/reports/2026-11-16-7386d2881-a2a.md`, written by the code of 7386d28 (the dG0 of a
replica from the accumulated bias; FM-D9, third amendment). Nothing was tuned for the set, and
the report holds no verdict. Values in kcal/mol:

| Ligand | dG0 fmeta | 95% interval | Experiment | FEP+ | fmeta − experiment | Rebinding events |
|---|---|---|---|---|---|---|
| 4q | −10.56 | −14.08 to −7.05 | −8.89 | −8.96 | −1.68 | 6 |
| 4a | −7.42 | −12.57 to −2.28 | −9.41 | −7.15 | +1.99 | 4 |
| 4b D | −13.31 | −27.76 to 1.15 | −9.96 | −10.06 | −3.35 | 6 |
| 4d | −22.50 | −33.46 to −11.54 | −10.50 | −9.98 | −12.00 | 2 |
| 4h | −12.76 | −21.83 to −3.69 | −10.65 | −11.13 | −2.10 | 0 |
| 4g | −14.94 | −22.47 to −7.40 | −11.06 | −11.83 | −3.87 | 1 |
| 4c | −11.05 | −23.57 to 1.46 | −11.46 | −12.09 | +0.41 | 9 |
| 4e D | −21.39 | −32.81 to −9.98 | −12.07 | −11.01 | −9.32 | 0 |

Agreement with experiment over the 8 ligands (in brackets the 2.5 to 97.5 percentile of a
bootstrap over the ligands):

| Measure | fmeta | FEP+ |
|---|---|---|
| MUE (kcal/mol) | 4.34 [1.98, 7.21] | 0.74 [0.35, 1.23] |
| RMSE (kcal/mol) | 5.79 [2.24, 8.41] | 0.98 [0.45, 1.49] |
| Mean signed error (kcal/mol) | −3.74 [−6.94, −0.93] | +0.23 [−0.37, 0.94] |
| MUE, offset removed (kcal/mol) | 3.49 [1.30, 5.28] | 0.79 [0.27, 1.17] |
| RMSE, offset removed (kcal/mol) | 4.42 [1.67, 5.66] | 0.96 [0.33, 1.26] |
| Pearson r | 0.57 [−0.08, 0.94] | 0.79 [0.51, 0.99] |
| Kendall tau | 0.36 [−0.33, 0.90] | 0.64 [0.20, 1.00] |

What the numbers say:

- Six of the eight values of fmeta are below experiment (stronger binding), 4d and 4e D by 12.0
  and 9.3 kcal/mol; 4a and 4c are above it by 2.0 and 0.4.
- The experimental value is inside the 95% interval for 7 of the 8 ligands (not for 4d). The
  intervals are 7.0 to 28.9 kcal/mol wide (29 to 121 kJ/mol); the experimental values of the
  set span 3.2 kcal/mol. The order of the ligands is not resolved at this length and number of
  replicas.
- The FEP+ values are relative free energies placed on the mean of experiment by their source,
  so their MUE holds no error of the absolute scale. On the rows with the offset removed and on
  the correlations, which compare like with like, FEP+ is closer to experiment than fmeta.
- In 25 of the 40 replicas the ligand did not come back to the site within 40 ns: in 23 it left
  once and stayed out, in 2 (one of 4d, one of 4e D) it did not reach the unbound region. No
  replica of 4h or 4e D has a rebinding event. In the TYK2 set, in water, 47 of the 50 replicas
  have one or more. A surface built from one passage out holds the site as the bias had filled
  it by then, and the two ligands farthest below experiment (4d, 4e D) are among those with the
  fewest returns.

## What is not settled

- The temperature of the measurements is not in the source; a report compares the values as
  given, without a correction to 300 K.
- A value with a narrow interval, and a run in which every ligand returns to the site several
  times. The settings for accuracy (walkers, run length, bias) are not chosen yet: the owner
  deferred them until every function is shown to work.
- The hydrophobic thickness of OPM (2 x 15.7 Å) is not used: the bilayer is built with the
  default `membrane.half_thickness`.
