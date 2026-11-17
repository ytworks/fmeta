# Configuration reference

One YAML file configures a calculation. Its sections map onto the settings dataclasses of
`src/fmeta/`, and the defaults of those dataclasses are the official defaults (FM-D16,
`docs/design/engine.md`). This file lists every key with its unit, its default, what it does
and where the value comes from. A test (`tests/test_config.py`) keeps the tables equal to the
code: a key that is added, removed or given another default fails it until this file follows.

## Presets

```bash
fmeta init --preset soluble  fmeta.yaml     # a protein-ligand complex in water (the default)
fmeta init --preset membrane fmeta.yaml     # the same, in a lipid bilayer
```

`fmeta init` writes a preset: a complete configuration with every key, its default, its unit
and one line on what it does. The presets are files of the package
(`src/fmeta/presets/*.yaml`), and a test keeps them equal to the defaults of the code.
`membrane` is `soluble` plus the section `membrane`; no other value differs.

A configuration may also hold only what it changes:

```yaml
preset: membrane
input: {protein: receptor.pdb, ligand: ligand.sdf}
system: {temperature: 310.0}
run: {length_ns: 1000.0}
```

How such a file is read:

1. The values are the defaults of the code, then the preset, then the keys of the file: the
   file wins, key by key within a section.
2. The file names of a preset (`protein.pdb`, `ligand.sdf`) are placeholders and are never
   taken over: `input` comes from the file alone.
3. `membrane` absent: as in the preset. `membrane: {lipid: POPE}`: on top of the bilayer of
   the preset. `membrane: true`: the bilayer of the preset, or the default bilayer when the
   preset has none. `membrane: false` or `null`: no bilayer.
4. A file that names `system.ligand_method` without `system.ligand_forcefield` gets the pinned
   force field of that method, not the one of the preset.
5. A key or a section that does not exist is an error, with or without a preset.

Without `preset:` a file is read on top of the defaults of the code, which are the values of
`soluble`. Paths (`input.*`, `system.ligand_xml`, `membrane.patch`) are relative to the
configuration file.

`fmeta prepare` saves the configuration it used as `config.yaml` in the work directory. That
file is resolved: it has no `preset` key, it holds every value, the paths are absolute and the
ligand force field is named. The later stages read it, so it is the record of what was run.
Two values in it are derived and are not keys of a configuration: `metad.temperature` (always
`system.temperature`) and `metad.pace` (the number of steps between hills,
`run.pace_ps / run.timestep_fs`, 250 with the defaults).

## How to read the last column

Every default has one of these origins. Nothing is cited that was not read for this file; the
sources are listed at the end.

- **A source in brackets**, e.g. [E20]: the source uses or states this value.
- **measured**: a measurement of this project; the place of the record is named.
- **M4**: no source. Milestone M4 of
  `docs/plans/2026-11-15-gpu-validation-modes-benchmarks.md` set the hill widths and the run
  length on trypsin-benzamidine, from convergence diagnostics and before any benchmark run
  (the scan of FM-D16). A value that still carries this tag was not in that scan and stands
  unconfirmed. Nothing is tuned against experimental values.
- **convention**: no source. Common practice, not checked against the literature here.
- **design**: follows from a decision of `docs/design/engine.md`, which is named.

## input

Either `protein` and `ligand`, or `complex`, `ligand_resname` and `ligand_smiles`.

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `input.protein` | path | `null` | The protein, PDB or mmCIF. It is reduced to its amino-acid residues, repaired with PDBFixer when that is installed, and its hydrogens are completed (design: FM-D11). With a membrane: the membrane normal along z (see `membrane.orient`). |
| `input.ligand` | path | `null` | The ligand in its bound pose, in the frame of the protein: SDF, which carries bond orders and formal charges (MOL and MOL2 are read as well). A ligand that is not in contact with the protein is refused (design: FM-D11). |
| `input.complex` | path | `null` | Instead of the two files: one PDB or mmCIF with both. The ligand residue is cut out into its own file. |
| `input.ligand_resname` | - | `null` | With `complex`: the residue name of the ligand. |
| `input.ligand_smiles` | - | `null` | With `complex`, required: the SMILES of the ligand in the protonation state to simulate. A structure file states neither bond orders nor formal charges, with or without hydrogens: both come from the SMILES, and the hydrogens are added to the heavy atoms of the file. Without it the build stops (design: FM-D11). |
| `input.ligand_chain` | - | `null` | With `complex`: the chain of the ligand, when there are several copies. |
| `input.ligand_resid` | - | `null` | With `complex`: the residue number of the ligand, when there are several copies. |

## system

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `system.forcefields` | - | `[amber14-all.xml, amber14/tip3p.xml]` | OpenMM force-field files. `amber14-all.xml` holds ff14SB for proteins and Lipid17 for lipids, `amber14/tip3p.xml` TIP3P water with ions [OMM]. The same protein force field and water as the BioSimSpace funnel metadynamics tutorial [BSS-T]; [E20] used a99SB-disp. The papers of the force fields were not read: convention. |
| `system.water_model` | - | `tip3p` | The water that is added around the solute. It has to be the model of the water file in `forcefields`. |
| `system.ligand_method` | - | `openff` | Where the ligand parameters come from: `openff` or `gaff` (a generator of `openmmforcefields` builds them from the ligand file) or `xml` (the files of `ligand_xml`). fmeta never invents parameters. Convention; the funnel metadynamics literature read here used GAFF2 ([E20] with RESP charges, [BSS-T]), which `gaff` gives with AM1-BCC charges. |
| `system.ligand_forcefield` | - | `openff-2.2.1` | The force field of the generator. `null` means the pinned one of the method: `openff-2.2.1` for `openff`, `gaff-2.11` for `gaff`. Design: FM-D11 and FM-D16: the generators have no default, and a result must not depend on the versions that happen to be installed. Both pins prescribe AM1-BCC charges; `openff-2.3.0`, which the image also offers, prescribes NAGL charges (measured, FM-D16). With `ligand_method: xml` it is not used. |
| `system.ligand_charge_method` | - | `null` | `null`: the generator assigns the charges its force field prescribes. For both pins that is AM1-BCC, computed in the official image by `sqm` of AmberTools 26.0 (measured, FM-D16). Otherwise a charge method of the OpenFF toolkit (`am1bcc`, `gasteiger`, a NAGL model name): the charges are then assigned before the generator sees the molecule. |
| `system.fix_protein` | - | `null` | Repair the protein with PDBFixer (missing atoms, non-standard residues). `null`: whenever PDBFixer is installed, as it is in the official image (design: FM-D11). |
| `system.keep_hetero` | - | `[]` | Residue names of non-protein groups to keep, e.g. `[ZN]`. The files of `forcefields` must cover them. A kept group is part of the receptor: restrained in equilibration, an obstacle for the funnel, counted for the box (design: FM-D15). Everything else that is not an amino acid is removed. |
| `system.ligand_xml` | path | `[]` | ffxml file(s) with the residue template of the ligand, for `ligand_method: xml`. |
| `system.ligand_resname` | - | `LIG` | The residue name the ligand gets in the system. |
| `system.keep_water` | - | `false` | Keep the water molecules of the protein file. Convention. |
| `system.add_hydrogens` | - | `true` | Complete the hydrogens of the protein with the force field. |
| `system.ph` | pH | `7.0` | The pH for the protonation states of the protein when hydrogens are added. The ligand keeps the protonation state of its file (roadmap R5). Convention. |
| `system.ionic_strength` | mol/L | `0.15` | Sodium chloride, in addition to the ions that neutralise the system. Convention. |
| `system.padding` | nm | `1.2` | Solvent around the solute; with a membrane, lipid around the protein in the membrane plane and water above and below. A lower bound: the box is sized from the funnel as well (design: FM-D6). Convention. |
| `system.box_shape` | - | `dodecahedron` | `dodecahedron` or `cube`; not used with a membrane. A rhombic dodecahedron has 71% of the volume of the cube with the same image distance (geometry). |
| `system.cutoff` | nm | `1.0` | Cutoff of the direct-space non-bonded interactions; the long range is particle mesh Ewald. The value of the OpenMM user guide [OMM]. |
| `system.hydrogen_mass` | amu | `1.5` | Mass of the hydrogens; the difference is taken from the heavy atom they are bonded to. `null`: real masses, and `run.timestep_fs` can then be at most 2. Bonds to hydrogen are constrained and water is rigid. Heavier hydrogens allow a longer time step [OMM]; the value 1.5 has no source: M4. |
| `system.temperature` | K | `300.0` | Temperature of the equilibration, the production and the bias. The default of the BioSimSpace protocol [BSS-P]. Convention. |
| `system.pressure` | bar | `1.0` | Pressure of the Monte Carlo barostat (with a membrane: the membrane barostat, x and y coupled, z free, no surface tension). Convention ([BSS-P]: 1 atm). |

## membrane

Absent or `null`: no bilayer. The section exists in the preset `membrane` only.

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `membrane.lipid` | - | `POPC` | Lipid of the bilayer that `Modeller.addMembrane` builds: POPC, POPE, DLPC, DLPE, DMPC, DOPC or DPPC (design: FM-D8). Convention. For DPPC, DMPC and DLPE fmeta warns when the temperature is less than 10 K above the gel transition. |
| `membrane.center_z` | nm | `0.0` | z of the membrane midplane in the frame of the input files. Convention: the protein is given with the membrane normal along z and the midplane at z = 0. |
| `membrane.half_thickness` | nm | `2.2` | Half the thickness of the slab that stands for the bilayer while the funnel is placed, before the lipids exist (design: FM-D8). No source: convention. |
| `membrane.patch` | path | `null` | A pre-equilibrated bilayer patch (PDB) that is used instead of `lipid`. |
| `membrane.orient` | - | `check` | `check`: warn when the hydrophobic belt of the protein is tilted more than 25 degrees or its midplane is more than 1.0 nm from `center_z`, and also when no belt is found with confidence, so that nothing was checked. `auto`: re-orient the complex onto the plane estimated from the belt (an error when no belt is found with confidence). `input`: trust the files. Design: FM-D8, FM-D15. |

## funnel

The automatic placement (design: FM-D7). Its lengths are the construction of this project, not
values from the literature: they are tested on synthetic pockets with a known exit
(`tests/test_autofunnel.py`) and are confirmed on real systems in M4 and roadmap R3. The
BioSimSpace paper says the same about any automatic funnel: its radius has to be found case
by case, and the placement should be looked at before a run [L23]. `fmeta funnel WORKDIR`
writes the funnel for that.

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `funnel.method` | - | `clearance` | `clearance`: the exit is the straight path from the ligand that crosses the least solute (and membrane) and reaches bulk soonest; the walls and the width follow the pocket. `bss`: the average of the grid points around the ligand that hold no protein, with the fixed lengths of the table below, reimplemented from the documented behaviour of BioSimSpace [BSS-F]. Design: FM-D7: the grid average fails for buried sites and inside a bilayer. |
| `funnel.n_directions` | - | `1000` | Directions tried for the exit, spread evenly over the sphere. M4. |
| `funnel.max_path` | nm | `4.5` | How far from the binding site a path is followed. M4. |
| `funnel.step` | nm | `0.05` | Step along a path. M4. |
| `funnel.probe` | nm | `0.3` | A path nearer than this to an atom is squeezing through, which counts against it. M4. |
| `funnel.contact_clearance` | nm | `0.6` | From this distance to the solute on, the ligand centre has left the surface; that point of the path is the inflection of the wall. M4. |
| `funnel.bulk_margin` | nm | `1.0` | Distance between ligand atoms and solute that counts as bulk. The unbound region of the analysis has to lie beyond it, and the upper wall is placed so that it does. M4. |
| `funnel.charged_bulk_margin` | nm | `1.5` | `bulk_margin` for a ligand with a net charge above 0.5 e, because of the long-range attraction; the larger of the two is used. M4. |
| `funnel.anchor_min_atoms` | - | `10` | Least number of alpha carbons in an anchor group (p0, p1). M4. |
| `funnel.anchor_radius` | nm | `1.0` | Radius around an anchor point in which alpha carbons are taken. M4. |
| `funnel.anchor_separation` | nm | `1.2` | Distance between the anchor points p0 and p1. M4. |
| `funnel.pose_margin` | nm | `0.2` | Kept between the atoms of the bound pose and the wall: it sets the width. M4. |
| `funnel.min_width` | nm | `0.6` | Lower limit of the funnel width; the BioSimSpace default width [BSS-F]. |
| `funnel.max_width` | nm | `1.5` | Upper limit of the funnel width. M4. |
| `funnel.exit_direction` | - | `null` | A vector such as `[0, 0, 1]` sets the exit by hand; `null`: search. |
| `funnel.check_channel` | - | `true` | Warn when a probe of 0.3 nm cannot reach bulk from the site. |

## funnel_overrides

Funnel parameters set by hand, e.g. `funnel_overrides: {upper: 4.0}`. They replace the value
of the placement, whatever the method. The default is the empty mapping. The column Default
shows the fixed value that `method: bss` uses; with `clearance`, `width`, `inflection`,
`lower` and `upper` are derived from the pocket and the others are these values.

Wall energy, with `s` along the axis and `rho` away from it (design: FM-D3):
`0.5 k_wall max(rho - R(s), 0)^2 + 0.5 k_lower max(lower - s, 0)^2 + 0.5 k_upper max(s - upper, 0)^2`,
`R(s) = width / (1 + exp(steepness (s - inflection))) + buffer`.

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `funnel_overrides.width` | nm | `0.6` | Radius of the wide part, without `buffer`. [BSS-F]. |
| `funnel_overrides.buffer` | nm | `0.15` | Radius of the narrow part, far from the site. [BSS-F]. |
| `funnel_overrides.steepness` | 1/nm | `1.5` | How fast the wall narrows around `inflection`. [BSS-F]. |
| `funnel_overrides.inflection` | nm | `2.0` | Where along the axis the wall narrows. [BSS-F]. |
| `funnel_overrides.lower` | nm | `0.5` | Lower wall in `s`. [BSS-F]. |
| `funnel_overrides.upper` | nm | `4.0` | Upper wall in `s`. [BSS-F]; its tutorial uses 3.5 nm [BSS-T], and [E20] 4.5 nm with the unbound state beyond 3.0 nm. |
| `funnel_overrides.k_wall` | kJ/mol/nm^2 | `10000.0` | Force constant of the side wall. Design: FM-D3: a stiff wall keeps the soft-wall term of the free energy small and the CVs inside the bias grid. No source for the value; the BioSimSpace bounds have 2000.0 (the page gives no unit) [BSS-F], the PLUMED example 35100 [PLUMED]. |
| `funnel_overrides.k_lower` | kJ/mol/nm^2 | `10000.0` | Force constant of the lower wall. As `k_wall`. |
| `funnel_overrides.k_upper` | kJ/mol/nm^2 | `10000.0` | Force constant of the upper wall. As `k_wall`. |
| `funnel_overrides.eps` | nm | `0.001` | `rho` is `sqrt(rho^2 + eps^2)`, which bounds its gradient on the axis (design: FM-D2). |

## metad

Well-tempered metadynamics on `s` and `rho`.

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `metad.bias_factor` | - | `10.0` | Well-tempered bias factor; `null`: plain metadynamics. [E20] (for all its systems) and [BSS-T]. |
| `metad.height` | kJ/mol | `1.5` | Height of the first hills. [E20] and [BSS-T]. [E20] deposits one every 2 ps, fmeta every 1 ps (`run.pace_ps`). |
| `metad.sigma_s` | nm | `0.05` | Hill width along the axis. Measured: the scan of FM-D16 on trypsin-benzamidine, where this width and `sigma_rho` 0.1 nm had the most replicas passing the convergence check of the rule at 40 ns (2 of 5; none with the same bias and 0.025 nm, the value of [BSS-F] and [E20]). The analysis has no such verdict any more, and the value of the set depends on the bias setting (FM-D16, third amendment). |
| `metad.sigma_rho` | nm | `0.1` | Hill width away from the axis. Measured: the same scan. [BSS-F] has 0.05 nm; [E20] used 0.03 nm. |
| `metad.points_per_sigma` | - | `3.0` | Grid points per hill width. Measured: the cubic spline through the grid reproduces a Gaussian to about 2e-4 of its height at 3 points per sigma; a finer grid only costs time (`src/fmeta/engine/grid.py`, design: FM-D4). |
| `metad.grid_margin` | nm | `0.3` | How far the grid extends past the walls. The bias is clamped at the edge of the grid, so the CVs have to stay inside: 0.3 nm beyond a default wall costs 450 kJ/mol. Convention. |
| `metad.reflect` | - | `true` | Mirror the hills at `rho = 0`, so that a hill next to the axis does not lose the part that falls below zero (rule of `.claude/rules/numerics-invariants.md`). |
| `metad.s_min` | nm | `null` | Lower grid limit in `s`; `null`: the lower wall minus `grid_margin`. |
| `metad.s_max` | nm | `null` | Upper grid limit in `s`; `null`: the upper wall plus `grid_margin`. |
| `metad.rho_max` | nm | `null` | Upper grid limit in `rho`; `null`: the widest wall radius plus `grid_margin`. |

## equilibration

Minimisation; heating at constant volume with position restraints on the heavy atoms of the
protein, the ligand and the kept hetero groups (with a membrane also on the phosphorus atoms
of the lipids, along z); release of the restraints at constant pressure; then a free stage
with the funnel walls on. The funnel is accepted when the mean wall energy of the free stage
is below 0.1 kT. The schedule has no source: convention, and M4 looks at it.

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `equilibration.minimize_iterations` | - | `5000` | Most steps of the energy minimisation. Convention. |
| `equilibration.restraint_k` | kJ/mol/nm^2 | `1000.0` | Force constant of the position restraints. Convention. |
| `equilibration.heating_ps` | ps | `100.0` | Heating from 10 K to `system.temperature` at constant volume, restrained. Convention. |
| `equilibration.release_ps` | ps | `400.0` | Constant pressure, split evenly over the stages of `release_factors`. Convention. |
| `equilibration.release_factors` | - | `[1.0, 0.5, 0.2, 0.05]` | The restraint is scaled by these factors, one stage each. Convention. |
| `equilibration.free_ps` | ps | `null` | Without restraints, funnel walls on. `null`: 1000, with a membrane 2000 (the bilayer relaxes around the protein). Convention; `equil/equil.json` holds the box area and height of the stage in ten blocks (`free_blocks`), and a change of the area above 3% between its halves is a warning: extend the stage then. The membrane value is checked in roadmap R3. |
| `equilibration.timestep_fs` | fs | `2.0` | Time step of the equilibration. Convention (the production time step of [E20] and of [BSS-P]). |
| `equilibration.friction` | 1/ps | `1.0` | Friction of the Langevin thermostat. [OMM] (its example), [BSS-P] (time constant 1 ps). |
| `equilibration.seed` | - | `0` | Seed of the velocities and the thermostat; `0`: random. |

## run

| Key | Unit | Default | What it does; where the value comes from |
|---|---|---|---|
| `run.length_ns` | ns | `40.0` | Production length of each replica (of each walker, with `walkers` above 1). Measured, with the owner's limit of 40 ns: the longest the scan of FM-D16 on trypsin-benzamidine ran. A replica of this length does not converge on its own: the result is the mean of the replicas with its interval (FM-D9, second amendment), and on trypsin that interval is wide with the defaults (−68.7 to 14.0 kJ/mol around −27.4). A run is extended exactly by running `fmeta run` again with a larger `--length-ns` (design: FM-D5). [E20] ran 500 ns per simulation; BioSimSpace recommends "of the order of 500 - 2000 ns" [L23]. |
| `run.replicas` | - | `5` | Independent runs, each with its own bias; their spread is the error estimate (design: FM-D10). Five repeats in [L23] and [BSS-T]. |
| `run.walkers` | - | `1` | Walkers that share one bias within a replica, for wall time on several GPUs (design: FM-D10). A replica that has run continues with the same number or more: a run that would leave out a walker with hills is refused. |
| `run.walker_sync_hills` | - | `1` | A walker reads the hills of the others after every N of its own. |
| `run.timestep_fs` | fs | `4.0` | Time step of the production. Above 2 fs it needs `system.hydrogen_mass`. With constrained bonds to hydrogen about 4 fs is usable with Langevin dynamics [OMM]; [E20] and [BSS-P] use 2 fs. The GPU measurements of FM-D4 ran at 4 fs; what it does to a free energy is not measured yet: M4. |
| `run.friction` | 1/ps | `1.0` | Friction of the Langevin thermostat. [OMM] (its example), [BSS-P] (time constant 1 ps). |
| `run.pace_ps` | ps | `1.0` | Time between hills. No source: [E20] and [BSS-P] deposit every 1000 steps of 2 fs, which is 2 ps. M4. |
| `run.platform` | - | `auto` | `auto`: the first usable of CUDA, HIP, OpenCL, CPU, Reference; or one of these names. |
| `run.precision` | - | `mixed` | Precision of a GPU platform: `single`, `mixed` or `double`. The measurements behind FM-D2 and FM-D4 were taken in mixed precision. Convention. |
| `run.backend` | - | `auto` | What computes the nonbonded forces and moves the atoms of the production. `auto`: NVIDIA's cuPotential plugin of the CUDA platform where it can take part, OpenMM itself otherwise, with the reason in the log, `run.yaml` and `device.json`. `cupotential`: the plugin, and an error where it cannot take part. `openmm`: OpenMM itself. The plugin needs CUDA in single or mixed precision, PME at an Ewald error tolerance of 5e-4 with the mesh left to it, and a box of at least 3 cutoffs; it is in the official image and not in an installation with pip, where `auto` gives OpenMM. It takes the place of the nonbonded force and the integrator, and in a soluble system of the barostat; a membrane keeps OpenMM's membrane barostat, and the forces of the funnel and the bias are OpenMM's with every back end. A run keeps the back end it began with, and under `auto` a new replica takes the back end of the replicas that are there. `fmeta status` shows the back end of every replica (design: FM-D18). |
| `run.mode` | - | `auto` | `parallel`: one replica per GPU, side by side; `sequential`: one after another on one GPU; `auto`: parallel when `gpus` is set (design: FM-D14). |
| `run.gpus` | - | `null` | The GPUs of the run: device numbers such as `[0, 1]`, or `all`; `null`: one device, picked by the platform (design: FM-D14). |
| `run.per_gpu` | - | `1` | Replicas that share one GPU at a time. Above 1 the container needs NVIDIA MPS (`scripts/mps.sh start`, then `scripts/in_docker.sh --mps`): measured on one RTX PRO 6000 Blackwell with an 88k-atom system, 2, 3, 4, 6 and 8 replicas give 1.3, 1.8, 2.1, 2.5 and 2.7 times the total speed of one, each replica running at 0.64, 0.59, 0.53, 0.42 and 0.33 of its speed alone; without MPS two replicas give no more than one (0.96 times) (design: FM-D14). The default leaves a GPU to one replica: the shortest wall time of a single run. |
| `run.checkpoint_ps` | ps | `250.0` | Time between checkpoints; a restart continues from the last one (design: FM-D5). A run that has a checkpoint continues only under the conditions of its `run.yaml`: the length, the device, this interval and `walker_sync_hills` may change and `walkers` may grow, while fewer walkers than the replica has hills of, another bias, time step, temperature, pressure or trajectory setting stops the restart with an error before a file is touched (use a new work directory). Convention. |
| `run.trajectory_ps` | ps | `100.0` | Time between trajectory frames; `0`: no trajectory. Convention. |
| `run.trajectory_solute_only` | - | `true` | Write only the solute (protein, ligand, kept hetero groups) to the trajectory. |
| `run.seed` | - | `0` | `0`: random. Otherwise replica i, walker w use `seed + 1000 i + w`. |

## Where the defaults differ from the literature that was read

Two of these were set by measurement: the hill widths and the run length, by the scan of the
bias on trypsin-benzamidine (FM-D16 in `docs/design/engine.md` has the settings, the rule that
was written before the runs, and the result). The others stand as they were chosen for this
file. None was tuned against experimental values.

- **Hill widths.** 0.05 nm along the axis and 0.1 nm in rho, from the scan. The BioSimSpace
  defaults are 0.025 and 0.05 nm [BSS-F]; [E20] used 0.03 nm for its second variable.
- **Length.** 40 ns per replica, the longest the scan ran. [E20] ran 500 ns per simulation and
  [L23] recommends 500 to 2000 ns. On trypsin-benzamidine five replicas of 40 ns with the
  defaults give −28.3 kJ/mol with a 95% interval of −66.9 to 10.4.
- **Deposition rate.** Hills of 1.5 kJ/mol every 1 ps; [E20] and the BioSimSpace protocol
  default [BSS-P] deposit every 2 ps (1000 steps of 2 fs). At the same height the bias starts
  to grow twice as fast. Hills of 3.0 kJ/mol and a bias factor of 20 were in the scan and
  passed fewer replicas.
- **Time step.** 4 fs with hydrogens of 1.5 amu; [E20] and [BSS-P] use 2 fs with real masses.
  Not in the scan.
- **Wall force constants.** 10000 kJ/mol/nm^2 in `0.5 k x^2`; the BioSimSpace bounds have
  `force_constant=2000.0` with exponent 2 and no unit on the page [BSS-F]; the PLUMED example
  has `KAPPA=35100` [PLUMED].
- **Ligand parameters.** OpenFF 2.2.1 with AM1-BCC charges; [E20] used GAFF2 with RESP
  charges and [BSS-T] GAFF2. `ligand_method: gaff` gives GAFF 2.11 with AM1-BCC charges.
- **Protein force field.** ff14SB with TIP3P as in [BSS-T]; [E20] used a99SB-disp.
- **Shape and lengths of the funnel.** One sigmoid wall that moves with the protein, as in
  BioSimSpace [L23]; the original method and PLUMED use a cone attached to a cylinder
  [PLUMED]. [E20] fixed the unbound state beyond 3.0 nm and the upper limit at 4.5 nm; fmeta
  derives both from the clearance of the pocket (design: FM-D7).

## Sources

Read for this file:

- [E20] Evans, Hovan, Tribello, Cossins, Estarellas, Gervasio, "Combining Machine Learning
  and Enhanced Sampling Techniques for Efficient and Accurate Calculation of Absolute Binding
  Free Energies", J. Chem. Theory Comput. 16, 4641-4654 (2020).
  doi:10.1021/acs.jctc.0c00075
- [L23] Hedges et al., "A Suite of Tutorials for the BioSimSpace Framework for Interoperable
  Biomolecular Simulation", Living J. Comput. Mol. Sci. 5(1), 2375 (2023), section 3.2.
  doi:10.33011/livecoms.5.1.2375
- [BSS-F] BioSimSpace API, `BioSimSpace.Metadynamics.CollectiveVariable.Funnel`.
  https://biosimspace.openbiosim.org/api/generated/BioSimSpace.Metadynamics.CollectiveVariable.Funnel.html
- [BSS-P] BioSimSpace API, `BioSimSpace.Protocol.Metadynamics`.
  https://biosimspace.openbiosim.org/api/generated/BioSimSpace.Protocol.Metadynamics.html
- [BSS-T] BioSimSpace tutorials, funnel metadynamics (`02_funnel_metad`, set-up and analysis).
  https://github.com/OpenBioSim/biosimspace_tutorials
- [OMM] OpenMM user guide, "Running Simulations".
  https://docs.openmm.org/latest/userguide/application/02_running_sims.html
- [PLUMED] PLUMED 2.9, `FUNNEL`.
  https://www.plumed.org/doc-v2.9/user-doc/html/_f_u_n_n_e_l.html

The method itself, of which only the abstract and the record were read, so no value above
rests on them: Limongelli, Bonomi, Parrinello, Proc. Natl. Acad. Sci. USA 110, 6358-6363
(2013), doi:10.1073/pnas.1303186110; Raniolo, Limongelli, Nat. Protoc. 15, 2837-2866 (2020),
doi:10.1038/s41596-020-0342-4.

Not read, and therefore not cited for any value: the papers of well-tempered metadynamics, of
hydrogen mass repartitioning, of ff14SB, TIP3P, Lipid17, OpenFF 2.x, GAFF and AM1-BCC.

## Adding a key

A new key is a field of a settings dataclass with its default. Add it to both presets with its
unit and one line, and to the table of its section here with where the value comes from; the
tests name what is missing. Add no key whose effect is not established: a saved configuration
with a key that is later withdrawn no longer loads.
