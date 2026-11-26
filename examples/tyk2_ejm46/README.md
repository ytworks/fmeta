# TYK2 kinase domain with ejm_46 (PDB entry 4GIH)

A soluble kinase with a tight neutral inhibitor: the JH1 (kinase) domain of human TYK2 with
2,6-dichloro-N-{2-[(cyclopropylcarbonyl)amino]pyridin-4-yl}benzamide (ligand 0X5 of the entry,
23 heavy atoms, compound 46 of the paper, `ejm_46` in the free-energy benchmark sets). The
same ligand is part of the TYK2 benchmark (`examples/benchmarks/tyk2`).

| Quantity | Value | Source |
|---|---|---|
| Experimental Ki | 4.8 nM | Liang et al., Eur. J. Med. Chem. 67, 175 (2013), doi:10.1016/j.ejmech.2013.03.070, as listed for the entry by BindingDB |
| Experimental binding free energy | −11.3 kcal/mol | RT ln Ki (−11.35 at 298.15 K; the benchmark tables list −11.31) |
| fmeta, default back end (cuPotential) | −6.12 kcal/mol, 95% interval −11.26 to −0.99 | 5 replicas of 100 ns with the defaults (2026-11-25); see "Result at 100 ns" below |
| fmeta, OpenMM's own forces | −3.55 kcal/mol, 95% interval −6.36 to −0.74 | 5 replicas of 40 ns with the defaults of then (2026-11-16); see "Result" below |

The crystal structure is 2.0 Å, one chain, residues 890 to 1177 of the kinase domain with the
engineered mutations C936A, Q969A, E971A, K972A, D1023N and C1142A. The missing residues are at
the two ends only, so the chain has no break. The mutations are kept as they are in the entry.
Which construct the Ki was measured on has not been checked here.

## Inputs

The structure files are not part of the repository. Everything runs in the official image
(`scripts/in_docker.sh`, from the root of the repository):

```bash
mkdir -p runs/tyk2_dl runs/tyk2_in
scripts/in_docker.sh python -I -c "import urllib.request; urllib.request.urlretrieve('https://files.rcsb.org/download/4GIH.pdb', '/work/runs/tyk2_dl/4gih.pdb')"
scripts/in_docker.sh fmeta split runs/tyk2_dl/4gih.pdb --resname 0X5 --smiles "Clc1cccc(Cl)c1C(=O)Nc2ccnc(NC(=O)C3CC3)c2" -o runs/tyk2_in
cp runs/tyk2_dl/4gih.pdb runs/tyk2_in/4gih.pdb
mv runs/tyk2_in/ligand.sdf runs/tyk2_in/ejm_46.sdf
cp examples/tyk2_ejm46/fmeta.yaml runs/tyk2_in/fmeta.yaml
```

`fmeta split` cuts the ligand out of the entry and gives it bond orders and hydrogens from the
SMILES (the neutral form). The protein file is the entry as downloaded: `fmeta prepare` removes
the waters and the crystal ligand, rebuilds the missing side-chain atom and adds the hydrogens.

## Run

```bash
scripts/in_docker.sh --gpus 0 fmeta prepare runs/tyk2_in/fmeta.yaml -o runs/tyk2
scripts/in_docker.sh fmeta funnel runs/tyk2                # look at runs/tyk2/prepared/funnel.pdb
scripts/in_docker.sh --gpus 0 fmeta equilibrate runs/tyk2
scripts/in_docker.sh --gpus 0,1,2,3,4 --detach --name fmeta-tyk2 fmeta run runs/tyk2 --mode parallel
scripts/in_docker.sh fmeta status runs/tyk2                # time reached, ns/day and GPU per replica
scripts/in_docker.sh fmeta analyze runs/tyk2 --plot        # after the run, or during it
```

`docker stop fmeta-tyk2` stops the run; the same `fmeta run` command continues it from the last
checkpoints. With one GPU use `--gpus 0` and `--mode sequential`: the replicas then run one
after another.

In the official image the production run is made with NVIDIA's cuPotential plugin of OpenMM:
`run.backend` is `auto` by default (FM-D18, `docs/design/engine.md`). `fmeta status` names the
back end of every replica; `run: {backend: openmm}` in `fmeta.yaml` gives OpenMM's own forces
and integrator.

## What the preparation gives (measured on 2026-11-15 UTC, commit 8d1f10a)

- Removed from the entry: 0X5 × 1, HOH × 119. One missing heavy atom rebuilt, 2307 hydrogens
  added at pH 7. Six atoms of the entry have two alternate locations.
- Pose: 22 of the 23 ligand heavy atoms are within 0.45 nm of the protein.
- Funnel: found by the automatic placement (open path to bulk 2.65 nm long, axis within 0.0° of
  the exit direction); lower wall at s = 0.70 nm, upper wall at s = 4.65 nm, width 0.6 nm; the
  bound pose is at s = 1.20 nm. The unbound region is 1.05 nm clear of the solute.
- Box: rhombic dodecahedron with an image distance of 10.92 nm, 88476 atoms (OpenFF ligand
  parameters, 0.15 M NaCl).
- Equilibration: mean funnel wall energy during the free stage 0.000 kJ/mol.

## Result

Measured on 2026-11-16 (UTC): the steps above with the official defaults (5 replicas of 40 ns)
on commit 47e84f7 in the official image. One difference from the `fmeta run` line above: the
five replicas ran at the same time on one GPU with NVIDIA MPS
(`scripts/in_docker.sh --mps --gpus 3 ... fmeta run runs/tyk2 --gpus 0 --per-gpu 5`), at about
300 ns/day each. The numbers are those of `fmeta analyze runs/tyk2 --plot` with the analysis of
the third amendment of FM-D9 (`docs/design/engine.md`): the value of a replica is the mean over
the second half of the run of dG(t) from the accumulated bias, with a block error.

The run was made with OpenMM's own forces and integrator: commit 47e84f7 is from before
`run.backend` existed, and today that is `run.backend: openmm`. The run of the full length with
the cuPotential back end is in "Result at 100 ns" below.

| Quantity | fmeta | Experiment |
|---|---|---|
| dG0 | −14.85 kJ/mol (−3.55 kcal/mol) | −47.3 kJ/mol (−11.31 kcal/mol) |
| 95% interval of the five replicas | −26.60 to −3.11 kJ/mol (standard error 4.23) | |
| Each replica, with its block error | −18.48 ± 2.04, −12.51 ± 3.58, −3.97 ± 2.82, −10.31 ± 3.07, −29.00 ± 7.13 kJ/mol | |
| From the samples after the first 2 ns (cross-check) | −4.65, −9.51, −6.80, 0.58, −29.89 kJ/mol | |
| Rebinding events per replica | 5, 0, 1, 1, 1 | |
| Replicas with a note in the diagnostics | 5 of 5 | |

The value is 32 kJ/mol (7.8 kcal/mol) above the experimental one, and the experimental value is
outside the interval. It is reported as measured: nothing was tuned on this system, and the
five replicas together are the measurement (FM-D9, second amendment).

What the diagnostics of `fmeta analyze` say:

- Four of the five replicas have fewer than 3 rebinding events. In every replica the bias
  pushed the ligand out early (`COLVAR`: s above 3.0 nm for the first time at 1.0 to 5.1 ns);
  in those four it then came back from the unbound region to the site once or not at all.
- In four of the five replicas an estimate from the samples (after the first 2 ns, or of the
  second half) differs from the value by 6.6 to 20.6 kJ/mol.
- In every replica dG(t) from the bias changed by 12 to 43 kJ/mol over the last half of the
  run. The bias fills the free-energy surface for as long as the run lasts, and the difference
  between the site and the unbound region moves with it: from −84 kJ/mol at 2 ns to −14 kJ/mol
  at 40 ns in the mean of the five (`runs/tyk2/summary.png`).

A run is extended exactly by a larger length (`fmeta run runs/tyk2 --length-ns 100`). The limit of
40 ns that held when this run was made is removed and the default is 100 ns (owner,
2026-11-25; FM-D16, fourth amendment).

## Result at 100 ns with the default back end

Measured on 2026-11-25 (UTC): the steps above with the official defaults (5 replicas of 100 ns,
`run.backend: auto`) on commit b3a4a26 in the official image (image d8b9c513… of lock
ee32ef34…, OpenMM 8.4.0), 88488 atoms. Every replica ran with the cuPotential plugin for the
nonbonded force, the integrator and the barostat (`device.json`, no fallback reason), from its
start to 100 000 hills without a stop, return code 0 (`schedule.jsonl`; the container of
`fmeta run` ended with 0). One difference from the `fmeta run` line above: the five replicas ran
on three GPUs with NVIDIA MPS, two, two and one per GPU
(`scripts/in_docker.sh --mps --gpus 5,6,7 ... fmeta run runs/tyk2_100 --gpus 0,1,2 --per-gpu 2`),
while the A2A example ran on the other five (owner, 2026-11-26: the eight GPUs of the machine
for both examples). The values at 40, 60 and 80 ns are from copies of the same runs cut at
those times (`truncate_data_file`, the rollback of a restart) and analysed by `fmeta analyze`
as runs of that length. Speeds measured with the plugin are not published (NVIDIA's licence).

| Length | dG0 (kJ/mol) | 95% interval (width) | Each replica, with its block error | From the samples after 2 ns | Rebinding events |
|---|---|---|---|---|---|
| 100 ns | −25.63 (−6.12 kcal/mol), standard error 7.74 | −47.12 to −4.13 (43.0) | −29.39 ± 4.41, −32.24 ± 5.15, −1.09 ± 1.66, −17.90 ± 2.52, −47.51 ± 0.81 | −17.87, −24.98, −2.44, −9.09, −43.73 | 3, 3, 2, 3, 2 |
| 80 ns | −25.97, standard error 9.38 | −52.02 to 0.08 (52.1) | −21.60, −39.22, −0.62, −14.36, −54.06 | −14.84, −25.77, −2.79, −6.85, −42.73 | 2, 3, 12, 7, 2 |
| 60 ns | −28.36, standard error 10.06 | −56.30 to −0.41 (55.9) | −15.49, −46.57, −7.93, −13.56, −58.23 | −15.59, −28.86, −4.18, −5.10, −40.54 | 4, 2, 13, 2, 2 |
| 40 ns | −28.52, standard error 5.63 | −44.15 to −12.89 (31.3) | −19.81, −35.87, −23.24, −16.76, −46.91 | −13.42, −29.43, −9.04, −5.17, −36.75 | 4, 1, 1, 1, 1 |

The rebinding events of a length are counted against the bound region of the surface at that
length (its minimum plus 0.3 nm), so the counts of two lengths are not cumulative.

- Experiment: −47.3 kJ/mol. The value at 100 ns is 21.7 kJ/mol (5.2 kcal/mol) above it; the
  experimental value lies 0.2 kJ/mol below the lower end of the interval. The replicas reach
  from −47.51 to −1.09 kJ/mol.
- Against the run with OpenMM's own forces above, at the same length of 40 ns: −28.52
  (standard error 5.63) against −14.85 (4.23), a difference of −13.67 kJ/mol with a standard
  error of 7.04 (Welch, 7.4 degrees of freedom: 95% interval −30.13 to +2.79). The two runs
  differ in more than the back end: the image (OpenMM 8.6.1 then, 8.4.0 now), the build and the
  equilibration (88476 and 88488 atoms), and the random numbers.
- From 40 to 100 ns the interval of the five went from 31.3 to 43.0 kJ/mol wide; at 60 and 80 ns
  it was 55.9 and 52.1. The surface is still being filled: in four replicas dG(t) from the bias
  changed by 6.9 to 27.5 kJ/mol over the last half of the run, up and down.
- Diagnostics of `fmeta analyze` at 100 ns: replica 2 (−1.09 kJ/mol) is within 1 kT of what an
  empty funnel gives (−1.32 kJ/mol) and is reported as no binding; the value of the five
  together is not. Two replicas have fewer than 3 rebinding events. In four replicas an estimate
  from the samples differs from the value by 7.3 to 16.9 kJ/mol.

It is reported as measured: nothing was tuned on this system, and the five replicas together
are the measurement (FM-D9, second amendment).
