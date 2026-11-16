# Trypsin-benzamidine (PDB entry 3PTB)

The classic test case of funnel metadynamics: a small charged ligand (benzamidinium, +1 e) in
the S1 pocket of bovine beta-trypsin. No membrane.

| Quantity | Value | Source |
|---|---|---|
| Experimental binding free energy | −6.4 to −7.3 kcal/mol | the measurements quoted by Limongelli et al. (their references 40 and 41) |
| Funnel metadynamics, published | −8.5 ± 0.7 kcal/mol | Limongelli, Bonomi, Parrinello, PNAS 110, 6358 (2013), doi:10.1073/pnas.1303186110 |
| fmeta | not available yet | see "Result" below |

## Inputs

The structure files are not part of the repository. Everything runs in the official image
(`scripts/in_docker.sh`, from the root of the repository):

```bash
mkdir -p runs/tb_dl runs/tb_in
scripts/in_docker.sh python -I -c "import urllib.request; urllib.request.urlretrieve('https://files.rcsb.org/download/3PTB.pdb', '/work/runs/tb_dl/3ptb.pdb')"
scripts/in_docker.sh fmeta split runs/tb_dl/3ptb.pdb --resname BEN --smiles "NC(=[NH2+])c1ccccc1" -o runs/tb_in
cp runs/tb_dl/3ptb.pdb runs/tb_in/3ptb.pdb
mv runs/tb_in/ligand.sdf runs/tb_in/benzamidine.sdf
cp examples/trypsin_benzamidine/fmeta.yaml runs/tb_in/fmeta.yaml
```

`fmeta split` cuts the ligand out of the entry and gives it the hydrogens of the charged
(amidinium) form from the SMILES. The protein file is the entry as downloaded: `fmeta prepare`
removes the waters, the calcium ion and the crystal ligand and adds the hydrogens.

## Run

```bash
scripts/in_docker.sh --gpus 0 fmeta prepare runs/tb_in/fmeta.yaml -o runs/tb
scripts/in_docker.sh fmeta funnel runs/tb                  # look at runs/tb/prepared/funnel.pdb
scripts/in_docker.sh --gpus 0 fmeta equilibrate runs/tb
scripts/in_docker.sh --gpus 0,1,2,3,4 --detach --name fmeta-tb fmeta run runs/tb --mode parallel
scripts/in_docker.sh fmeta status runs/tb                  # time reached, ns/day and GPU per replica
scripts/in_docker.sh fmeta analyze runs/tb --plot          # after the run, or during it
```

`docker stop fmeta-tb` stops the run; the same `fmeta run` command continues it from the last
checkpoints. With one GPU use `--gpus 0` and `--mode sequential`.

## What the preparation gives (measured on 2026-11-15 UTC, commit 8d1f10a)

- Removed from the entry: BEN × 1, CA × 1, HOH × 62. 1591 hydrogens added at pH 7. The six
  disulfide bonds of trypsin are in the topology, and the chain has no break.
- Funnel: found by the automatic placement (open path to bulk 2.30 nm long, axis within 0.0° of
  the exit direction); lower wall at s = 0.70 nm, upper wall at s = 4.30 nm, width 0.6 nm.
  Because the ligand is charged, the unbound region is 1.5 nm from the solute instead of 1.0 nm.
- Box: rhombic dodecahedron with an image distance of 10.20 nm, 72434 atoms (OpenFF ligand
  parameters, 0.15 M NaCl).
- Equilibration: mean funnel wall energy during the free stage 0.000 kJ/mol.
- Speed: 660 to 700 ns/day per replica with the bias (RTX PRO 6000 Blackwell, 4 fs, mixed
  precision), five replicas on five GPUs at the same time.

## Result

Not available yet. The run of five replicas with the official defaults was started on
2026-11-15 (UTC) on commit 8d1f10a in the official image, configured for the default length of
that day, 500 ns per replica. It ends at the length that "Run length" below gives. This section
will hold the standard binding free energy with its confidence interval, the rebinding counts
and the convergence verdict, whatever they turn out to be.

The calcium ion is removed by default. Keeping it (`system.keep_hetero`) is a variation that
has not been run.

## Run length

The default was 500 ns per replica, taken from the literature that was read
(`docs/configuration.md`). On 2026-11-16 the owner decided that this is too long. The new
default comes from this run, by a rule written at about 01:00 UTC on 2026-11-16, when the five
replicas were at 41 to 44 ns:

- The run is analysed with `fmeta analyze` as it stood at 100, 150 and 200 ns (HILLS and COLVAR
  cut at that time; the trajectory does not depend on the configured length).
- The default `run.length_ns` becomes the shortest of the three at which the verdict over the
  five replicas is "converged": every replica passes every convergence criterion of the product
  (rebinding events, drift and roughness of dG(t), gap to the reweighted estimate).
- If the verdict is "not converged" at all three, the default becomes 200 ns, and this file and
  `docs/configuration.md` say that the default did not converge on this system.
- Five replicas stay. No criterion changes. Agreement with experiment is not looked at.
- The run is stopped at the chosen length, and the result above is the analysis of that length.

Known when the rule was written: this run analysed at 20 ns and at 40 ns, not converged at
either (at 40 ns 3 to 5 rebinding events per replica, a drift of dG(t) of +12 to +22 kJ/mol
over the plateau window); and replica 0 of `examples/tyk2_ejm46`, where the ligand left the
site within 5 ns and was back within 20 ns.

## Scan of the defaults

Three official defaults differ from the literature in a way that a run can test
(`docs/configuration.md`, "Where the defaults differ from the literature that was read"). Each
is run here with that one value changed, three replicas of 200 ns, and compared with the first
200 ns of the five replicas of the run above.

| Variant | Changed | Official default |
|---|---|---|
| `pace2` | `run.pace_ps: 2.0` | 1.0 ps |
| `srho03` | `metad.sigma_rho: 0.03` | 0.05 nm |
| `dt2` | `run.timestep_fs: 2.0` with `system.hydrogen_mass: null` | 4 fs with hydrogens of 1.5 amu |

`pace2` and `srho03` start from the prepared and equilibrated system of the run above; `dt2` is
prepared and equilibrated again, because the masses are part of the system.

The rule, written on 2026-11-16 before any of these runs had started:

- Compared per variant: rebinding events per replica, drift and roughness of dG(t), the gap to
  the reweighted estimate, the convergence verdict, the plateau mean of each replica, their mean
  and their spread.
- Three replicas of 200 ns show a gross difference and nothing finer. A default changes only if
  1. for `dt2`: its mean lies outside the 95% confidence interval of the five replicas and
     their mean outside the interval of `dt2`. The time step would then change the free energy,
     and the default would become 2 fs with real masses;
  2. for `pace2` and `srho03`: every replica of the variant passes the convergence criteria at
     200 ns while fewer than half of the five replicas do. The two bias settings have the same
     free energy as their limit, so a difference of the means at 200 ns says only that one of
     them has not converged: it is reported and changes nothing.
  Otherwise the defaults stay and the numbers are reported as measured.
- Agreement with experiment is not a criterion, here or on any benchmark.

Amendment of 2026-11-16, about 01:00 UTC, with "Run length". The `dt2` replicas were at 6 to
8 ns, `pace2` and `srho03` had not started, and no variant had been analysed.

- The comparison is made at the chosen length L instead of 200 ns: three replicas of L per
  variant against the first L of the five replicas. Conditions 1 and 2 are otherwise unchanged.
- Condition 2 can only be met when fewer than half of the five replicas pass at L. `pace2` and
  `srho03` are run only in that case. Otherwise they cannot change a default: they are not run,
  and are reported as not run. The reason for leaving them out is the time: the owner asked for
  the shortest way to the results.
- `dt2` is run to L in every case.

Not scanned: the wall force constants, the force fields and the shape of the funnel. Each would
be a scan of its own; they are listed with their sources in `docs/configuration.md`.

Result: not available yet.
