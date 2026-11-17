# One setting for the kinase and the membrane example: accurate in the shortest run

Status: In progress
Refs: FM-D19 (to be written with the result); FM-D9, FM-D10, FM-D16

## Context

Owner request (2026-11-17): find and make the setting of the examples that gives a solid
accuracy in the shortest run time. The examples of this work are the membrane system
(`examples/a2a_theophylline`) and the kinase (`examples/tyk2_ejm46`) only, and the two have one
common setting. `examples/trypsin_benzamidine` is not part of it and stays as it is.

What the official defaults gave on the two (5 replicas of 40 ns, 200 ns of simulation each):

| Example | fmeta, kJ/mol | 95% interval | Experiment |
|---|---|---|---|
| TYK2 with ejm_46 | −14.85 | −26.60 to −3.11 | −47.3 |
| A2A with theophylline | −18.64 | −33.31 to −3.97 | −33.0 (all measurements −33.0 to −27.0) |

What those replicas did (read again from their HILLS and COLVAR files): the bias in the site
reached 74 to 150 kJ/mol within 2 to 10 ns, before the ligand left for the first time; after
that the bias outside the site grew from 0 to 60–83 kJ/mol while the ligand came back to the
site a few times or never (0 to 9 returns in 40 ns). The value of a replica is then set by
where its ligand stayed last, not by exchange between the site and the unbound region.

The picture that the scan tests:

- In well-tempered metadynamics the bias of a region grows with the logarithm of the time
  spent there, by `kT (gamma - 1) ln 10` per decade: 51 kJ/mol with the bias factor 10 of the
  defaults, 23 with 5, 11.5 with 3. A lower bias factor limits how far the region of the
  moment is overfilled. The hill height sets how fast the first filling is and little after it.
- The estimate needs many passages between the site and the unbound region on one surface.
  Walkers that share a bias (`run.walkers`, FM-D10) give them at the same time. FM-D16 scanned
  neither a bias factor below 10 nor walkers.

## Approach

**Terms.** Accuracy is (a) reproducibility, the 95% interval of independent replicas, and
(b) agreement with experiment. Run time is the simulated time of the set, replicas × walkers ×
length: on a machine that the set fills, the wall time follows from it.

**Scan directories.** The prepared and equilibrated directories of the two examples
(`runs/tyk2`, `runs/a2a_theophylline/work`) are copied with another `metad` and `run` section
of `config.yaml`; funnel, system and equilibration are those of the examples. Runs use the
default back end, random seeds and 4 processes per GPU with NVIDIA MPS; since the GPU fault
below, one MPS daemon per pair of GPUs with one run on it, started again when it ends with a
CUDA error.

**Round 1, a screen.** One replica of 8 walkers × 20 ns per arm.

| Arm | `metad.bias_factor` | `metad.height` (kJ/mol) | TYK2 | A2A |
|---|---|---|---|---|
| A | 10 | 1.5 (the defaults) | yes | yes |
| B | 5 | 0.5 | yes | yes |
| C | 5 | 1.5 | yes | |
| D | 3 | 1.5 | yes | |
| F (added) | 10 | 0.2 | yes | yes |
| G (added) | 10 | 0.05 | yes | |

One replica has no interval. An arm goes on by what its surface shows (passages back into the
site summed over the walkers, the bias in the site against the bias in the unbound window, how
the value moves between 10 and 20 ns), with the experimental value in view. This is a
judgment; it is recorded with the numbers of every arm. A further arm (another bias factor,
height or number of walkers) is added when round 1 points to it, and recorded the same way.

**Round 1, judgments and added arms** (2026-11-17; numbers: largest bias in the site, kJ/mol).

- Arm D (TYK2) stopped at 7.25 ns: no walker left the site; the bias was 49 at 0.5 ns, 60 at
  5 ns and 61 at 7 ns. With the bias factor 3 it grows too slowly to reach the level at which
  walkers of the other arms left.
- Arm C (TYK2) stopped at 7.5 ns: one walker of 8 left (at 2.2 ns, bias 90) and reached the
  unbound window at 7 ns; the seven others stayed while the bias went from 81 (0.5 ns) to 97.
- The first walkers left at a lower bias where the bias grew more slowly. TYK2: at 90 in arm C
  (2.2 ns), at 115 to 125 in arm A (0.2 to 0.5 ns). A2A: at 45 to 51 in arm B (0.3 to 0.7 ns),
  at 68 to 82 in arm A (0.1 to 0.3 ns). In arm A the bias is 104 (TYK2) and 68 (A2A) after the
  first 0.1 ns: 8 walkers add hills to the site 8 times as fast as one walker, faster than the
  ligand moves. What is put into the site beyond the level of leaving has to be made up outside
  before the surface gives a difference.
- Added for this reason, with the bias factor 10 of the defaults and a lower height:
  arm F (`height` 0.2, about the rate of one walker with 1.5) and arm G (`height` 0.05).
  Expected from the growth of arm A: G has the bias of arm A's first 0.1 ns after about 3 ns.
- Arm G (TYK2) stopped at 4.75 ns: walkers left the site from 2.3 ns on, at a bias of 88 to 91
  (the level of arm C), and stayed within 0.5 nm of the site for the next 2.5 ns; none reached
  the unbound window. The bias outside grows with the height as well: in arm A it follows
  `22.4 ln(1 + 5.3 t)` kJ/mol, t in ns after the first walker reached the window (34.7 at
  0.7 ns, 61.6 at 2.7 ns). With the height 0.05 that is about 17 ns for 31 kJ/mol, the bias
  outside at which a site filled to 93 would give the experimental value. G cannot be the
  shortest; its pair of GPUs went to arm F on TYK2.
- The bias factor 5 is not continued (TYK2 arm B not resumed from 1.75 ns, A2A arm B stopped
  at 4.5 ns), because the setting is common: on TYK2 one walker of 8 left in 7.5 ns with the
  bias factor 5 and 3 times the height of arm B (arm C), and the bias outside would then grow
  by 10 instead of 22.4 kJ/mol per factor e of time. A2A arm B itself: walkers left from 0.7 ns
  on at a bias of 45 to 62, one reached the unbound window (3 ns), the headline was −48.7 at
  4.5 ns. Its pair of GPUs went to arm F on A2A.
- Arm A, the bias of the defaults with 8 walkers (TYK2 to its 20 ns, A2A stopped at 10.1 ns).
  Headline, the bias in the site and the bias in the unbound window, kJ/mol:

  | Length, ns | TYK2 headline | site | window | A2A headline | site | window |
  |---|---|---|---|---|---|---|
  | 2 | −135.2 | 143.9 | 0 | −70.6 | 107.3 | 38.3 |
  | 5 | −92.9 | 153.0 | 61.6 | −36.9 | 118.5 | 70.7 |
  | 10 | −72.0 | 158.6 | 77.1 | −35.8 | 129.5 | 83.2 |
  | 15 | −63.4 | 159.8 | 85.1 | | | |
  | 20 | −54.2 | 162.2 | 102.2 | | | |

  A2A is within 3 kJ/mol of the experimental −33.0 from 5 ns on (−36.9, −35.4 at 7.5 ns,
  −35.8). TYK2 moves towards the experimental −47.3 through the whole run, by 9 kJ/mol in the
  last 5 ns, while the walkers in the site go from 3 to 1 of 8.
- Arm F stopped (TYK2 at 12.4 ns, A2A at 4.8 ns): behind arm A at every length on both
  systems. TYK2: headline −96.8 at 5 ns, −80.6 at 10 ns, −78.1 at 12 ns; the site at 118.6 and
  the window at 29.9 at 10 ns, a difference of 88.7 against 81.5 in arm A. A2A: −67.5 at
  4.5 ns, the first walker in the window at 3.3 ns. The lower height leaves the site 40 kJ/mol
  lower, and the bias in the window then grows at a third of the rate: the two cancel, as the
  growth with the logarithm of time says (the height multiplies the time in the site and
  outside alike). The hill height does not shorten the run.
- No passage from the unbound window back into the site in any arm (TYK2 arm A: 4 passages
  out in 160 ns of walkers, none back). Walkers that went 0.5 nm or more beyond the site did
  not come back into it within the runs; walkers that returned had been less far.
- Round 2 is made with the setting of arm A alone: bias factor 10, height 1.5 kJ/mol,
  8 walkers. No arm is ahead of it in any quantity that was read, and a second setting would
  double the 3 hours of the measurement.
- Faults: none after the first one (six runs of 13 to 62 minutes on the pair daemons).
  Round 2 has 32 walkers of one work directory (one `fmeta run` per directory), so it is one
  run on one MPS daemon over the 8 GPUs, started again on a CUDA error as before.

**Round 2, the measurement.** 4 independent replicas of 8 walkers of the setting (or two
settings) from round 1, on both systems. The first N ns of a run are the run of N ns, so the
lengths 5, 10, 15 and 20 ns are read from the same files; the runs go to 10 ns first and are
extended (`fmeta run --length-ns`, exact) only if the rule below is not met by then.

**Selection rule** (fixed here, before a round 2 run is read):

1. A setting with a length qualifies when on both systems, with the 4 replicas, the half-width
   of the 95% interval is at most 8.4 kJ/mol (2 kcal/mol) and the mean is within 8.4 kJ/mol of
   experiment (A2A: of the range −33.0 to −27.0 kJ/mol).
2. Of the qualifying ones, the least simulated time wins; lengths are taken in the order 5,
   10, 15, 20 ns.
3. If none qualifies up to 20 ns: the one with the smallest half-width (the larger of the two
   systems) is the example setting, and the READMEs say that the rule was not met.
4. Added 2026-11-17, before a round 2 run was started, from a fact of round 1: replicas in
   which no walker had left the site gave a headline with a block error of 0.7 to 2 kJ/mol
   (arms B, C, D, G). That value is the level to which the site is filled; it moves with the
   length of the run, and a set of such replicas would have a narrow interval. So a length
   qualifies under point 1 only if in every replica of both systems walkers have gone from the
   site to the unbound window and back into the site by that length (at least one passage each
   way, summed over the walkers and counted as the product analysis counts them). Point 3
   chooses among the lengths at which every replica has a walker that reached the unbound
   window.
5. Amended 2026-11-17 on the owner's word, given after the TYK2 reading at 10 ns below and
   before any other round 2 reading: the length is not limited to 20 ns; accuracy comes first,
   and of the lengths that give it the shorter is the better. Points 1 and 4 stay as they are,
   thresholds included. The lengths of point 2 go on in steps of 5 ns to 40 ns, and point 3
   applies if none qualifies up to 40 ns. 40 ns is the owner's limit of 2026-11-16 for any run;
   a longer one needs their word. The runs go on in steps of 10 ns (TYK2, then A2A) and stop
   at the first length that qualifies.
6. Amended 2026-11-17 on the owner's second word, with no round 2 reading made in between: the
   limit of point 5 is 100 ns. The lengths of point 2 go on in steps of 5 ns to 100 ns, and
   point 3 applies if none qualifies up to 100 ns. Nothing else of the rule changes.

Values are the headline of the product analysis (`fmeta analyze`; FM-D9, third amendment).
The checks of a replica stay diagnostics.

**Round 2, readings** (2026-11-17). Work directories `runs/r2/tyk2` and `runs/r2/a2a`: the
equilibrated directories of the examples with `run.replicas: 4`, `run.walkers: 8`, everything
else the defaults. The numbers at a cut length come from the functions of the product analysis
on the first N ns of the files; at the length of the run they are the output of
`fmeta analyze` (checked at 10 ns on TYK2: the same mean, interval and replica values).

- TYK2 to 10 ns, no fault. kJ/mol; experiment −47.3:

  | Length, ns | Mean | 95% interval | Half-width | From experiment | Replicas | Passages out / back |
  |---|---|---|---|---|---|---|
  | 5 | −93.39 | −121.76 to −65.02 | 28.37 | 46.09 | −108.9, −70.2, −105.7, −88.8 | 2/0, 3/0, 1/0, 3/0 |
  | 7.5 | −82.19 | −108.58 to −55.80 | 26.39 | 34.89 | −97.2, −64.3, −95.4, −71.9 | 2/0, 3/0, 2/0, 4/0 |
  | 10 | −73.26 | −96.89 to −49.63 | 23.63 | 25.96 | −91.8, −59.4, −78.5, −63.3 | 3/0, 4/0, 2/0, 4/0 |

  No part of the rule is met on TYK2 at 5 or at 10 ns (half-width, distance, passages back),
  so no length up to 10 ns qualifies whatever A2A gives: both systems go on to 20 ns, TYK2
  after the 10 ns of A2A, then A2A. As in round 1, no walker came back from the unbound window
  into the site. The replicas differ in both levels of the surface: at 10 ns the largest bias
  in the site is 178.4, 164.4, 157.6 and 157.5 kJ/mol, and the bias in the unbound window
  76.3, 91.0, 78.1 and 81.9.

- A2A to 10 ns, no fault. kJ/mol; experiment −33.0 on the simulated construct, all
  measurements −33.0 to −27.0:

  | Length, ns | Mean | 95% interval | Half-width | From the experimental range | Replicas | Passages out / back |
  |---|---|---|---|---|---|---|
  | 2.5 | −71.71 | −88.61 to −54.81 | 16.90 | 38.71 | −74.6, −58.4, −69.8, −84.0 | 2/0, 2/0, 1/0, 1/0 |
  | 5 | −46.41 | −58.89 to −33.93 | 12.48 | 13.41 | −47.7, −35.1, −50.1, −52.8 | 3/0, 1/0, 3/0, 3/0 |
  | 7.5 | −35.64 | −52.67 to −18.61 | 17.03 | 2.64 | −37.7, −20.9, −46.6, −37.4 | 3/0, 1/0, 3/0, 4/0 |
  | 10 | −30.98 | −50.51 to −11.45 | 19.53 | 0.00 | −35.9, −15.9, −44.7, −27.5 | 3/0, 4/1, 3/0, 4/0 |

  `fmeta analyze` at 10 ns gives the same mean and interval. The mean is inside the
  experimental range at 10 ns. The rule is not met at any length: the half-width is 12.48 to
  19.53 against 8.4, and one replica of four has a passage back into the site (between 7.5 and
  10 ns; the first of round 2). The half-width is smallest at 5 ns and grows after it: the
  replica values are 17.7 apart at 5 ns and 28.8 apart at 10 ns. Both systems go on in steps
  of 10 ns (points 5 and 6).

- TYK2 to 20 ns, no fault:

  | Length, ns | Mean | 95% interval | Half-width | From experiment | Replicas | Passages out / back |
  |---|---|---|---|---|---|---|
  | 12.5 | −69.24 | −91.11 to −47.37 | 21.87 | 21.94 | −88.7, −57.8, −68.7, −61.7 | 3/0, 4/0, 2/0, 5/1 |
  | 15 | −66.20 | −87.12 to −45.27 | 20.93 | 18.90 | −85.5, −58.0, −63.5, −57.8 | 3/1, 4/0, 3/0, 5/1 |
  | 17.5 | −62.96 | −81.18 to −44.75 | 18.21 | 15.66 | −79.8, −57.3, −60.0, −54.7 | 4/1, 4/0, 3/0, 5/1 |
  | 20 | −60.61 | −77.00 to −44.23 | 16.39 | 13.31 | −75.4, −56.9, −58.5, −51.6 | 4/1, 4/0, 3/0, 5/3 |

  No part of the rule is met at 15 or at 20 ns, so no length up to 20 ns qualifies whatever
  A2A gives. Walkers came back into the site for the first time on TYK2, in two replicas of
  four (the first between 10 and 12.5 ns). The mean moves towards experiment by 2.3 to 4.0
  per 2.5 ns and the half-width goes down with it.

- A2A to 20 ns, no fault:

  | Length, ns | Mean | 95% interval | Half-width | From the experimental range | Replicas | Passages out / back |
  |---|---|---|---|---|---|---|
  | 12.5 | −29.91 | −48.75 to −11.08 | 18.84 | 0.00 | −36.4, −18.3, −43.2, −21.8 | 4/0, 4/1, 3/0, 4/1 |
  | 15 | −28.47 | −47.92 to −9.02 | 19.45 | 0.00 | −35.8, −19.5, −41.8, −16.8 | 4/0, 4/1, 3/0, 4/1 |
  | 17.5 | −26.81 | −46.73 to −6.88 | 19.93 | 0.19 | −34.7, −20.3, −39.7, −12.6 | 4/0, 4/0, 3/0, 6/2 |
  | 20 | −25.25 | −45.64 to −4.86 | 20.39 | 1.75 | −33.6, −19.7, −37.7, −9.9 | 4/1, 4/0, 3/0, 6/2 |

  The rule is not met at 15 or at 20 ns: the half-width stays at 18.8 to 20.4 (8.4 asked
  for), and two replicas of four have a passage back. The mean is inside the experimental
  range from 10 to 15 ns and 1.75 above it at 20 ns; it rises by 1.1 to 1.7 per 2.5 ns, and
  one replica (−9.9 at 20 ns) rises faster than the other three. Both systems go on to 30 ns.
- TYK2 to 30 ns, no fault:

  | Length, ns | Mean | 95% interval | Half-width | From experiment | Replicas | Passages out / back |
  |---|---|---|---|---|---|---|
  | 22.5 | −58.56 | −74.24 to −42.88 | 15.68 | 11.26 | −72.2, −56.6, −56.9, −48.6 | 4/1, 4/0, 4/1, 5/3 |
  | 25 | −57.09 | −72.24 to −41.95 | 15.15 | 9.79 | −69.5, −56.8, −55.7, −46.4 | 4/1, 4/0, 4/1, 6/3 |
  | 27.5 | −55.86 | −70.00 to −41.73 | 14.14 | 8.56 | −66.4, −57.1, −55.1, −44.8 | 4/1, 4/0, 4/1, 6/3 |
  | 30 | −55.11 | −68.83 to −41.39 | 13.72 | 7.81 | −63.9, −57.8, −55.4, −43.3 | 4/1, 4/0, 4/1, 7/3 |

  At 30 ns the mean is within 8.4 of experiment for the first time. The half-width (13.72)
  and the passages back (three replicas of four) do not meet the rule at 25 or at 30 ns. The
  two middle replicas have stayed within 1.6 of their values of 20 ns; the lowest rises
  (−75.4 to −63.9) and the highest rises (−51.6 to −43.3).
- A2A to 30 ns, no fault:

  | Length, ns | Mean | 95% interval | Half-width | From the experimental range | Replicas | Passages out / back |
  |---|---|---|---|---|---|---|
  | 22.5 | −23.71 | −43.95 to −3.48 | 20.23 | 3.29 | −32.3, −19.4, −35.4, −7.7 | 4/1, 4/0, 3/0, 7/2 |
  | 25 | −22.76 | −43.07 to −2.44 | 20.32 | 4.24 | −30.9, −19.7, −34.4, −6.1 | 4/1, 5/0, 3/0, 7/2 |
  | 27.5 | −22.45 | −41.62 to −3.28 | 19.17 | 4.55 | −29.6, −20.5, −33.4, −6.3 | 4/1, 5/0, 5/0, 7/2 |
  | 30 | −21.95 | −39.03 to −4.87 | 17.08 | 5.05 | −28.4, −20.7, −31.3, −7.3 | 4/1, 6/0, 5/0, 7/2 |

  The rule is not met at 25 or at 30 ns: half-width 17.08 at 30 ns, two replicas of four
  with a passage back. The mean is 5.05 above the experimental range at 30 ns (within the
  8.4 of point 1) and rises by 0.3 to 0.5 per 2.5 ns after 25 ns, a third of the rate before.
  The half-width goes down after 25 ns: the highest replica has stopped rising (−6.1 at 25 ns,
  −7.3 at 30 ns) and the two lowest go on rising. Both systems go on to 40 ns.
- The count of passages at a length is not the count at a shorter length plus the new ones.
  The analysis calls the ligand bound below the minimum of the profile along `s` plus 0.3 nm,
  and that minimum moves while the surface fills (A2A, second replica: threshold 1.70 nm at
  10 ns, 1.50 nm at 20 ns; fourth replica: 1.52 and 1.77 nm). The passage back of the second
  replica at 10 to 15 ns is none with the threshold of 17.5 and 20 ns. Point 4 takes the count
  at the length that is judged, as the rule says.

**What the choice means.** The setting is chosen on these two systems with their experimental
values in view. Its agreement with experiment there is the outcome of the choice, not a
validation of the method, and the READMEs and FM-D19 say so. The package defaults do not
change (FM-D16: never tuned against experiment); whether they should follow is a separate
question for the owner, with this evidence.

**Into the repository.** The same `metad` and `run` values in the two `fmeta.yaml` files (a
test keeps them equal), the READMEs with the new result and the run command (walkers,
`--per-gpu`, MPS), FM-D19 with the table of every arm, `PROGRESS.md`, `README.md` and
`docs/roadmap.md` where they give the results of the two examples. No speed measured with the
cuPotential plugin goes to a tracked file.

## GPU fault at the first start of round 1 (2026-11-17)

- What happened: the first four arms (TYK2 A, B and A2A A, B: 32 walker processes with the
  cuPotential back end, clients of one MPS daemon over the 8 GPUs) ended in the same second,
  4.5 minutes after their start, each with `CUDA_ERROR_LAUNCH_FAILED` (TYK2 walkers at 1.8 to
  1.95 ns, A2A walkers at 0.7 ns).
- Why all of them: the log of the MPS server has one fatal GPU error, after which it ended every
  client on every GPU. Clients in containers all have process number 0 for the server (their
  own PID namespaces), so it cannot end the faulting client alone. The daemon went on and took
  new clients.
- What faulted: not known. The process that wrote its error first is walker 1 of A2A arm B
  (host GPU 7, first in the server's list); its last samples are ordinary (`s` 1.2 to 1.3 nm,
  no wall energy). The plugin names a kernel that died on the device (one of its checks, or a
  memory fault); the kernel log of the host cannot be read from this account. The earlier long
  runs under MPS (90 replicas of 40 ns and the two examples) used OpenMM's own forces and had no
  such end; this is the first run of this length with the plugin.
- What is done: every run is a client of a daemon of its own GPU pair (`FMETA_MPS_NAME`,
  `FMETA_MPS_DIR` of `scripts/mps.sh` and `scripts/in_docker.sh`), so a fault ends one run; a
  run that ends with a CUDA error is started again by the same command and continues from its
  checkpoints (at most 250 ps back, every walker of the replica rolled back together); the
  faults are counted per arm and go to FM-D19. The fault ends a run with an error, it does not
  change a file that the analysis reads.
- When the back end changes: if the faults cost more time than the plugin gains over OpenMM's
  own forces, the remaining runs are made with `--backend openmm` in new work directories (a
  run keeps its back end), and FM-D19 says which arm ran with which.

## Completion

| Part of the request | Done when |
|---|---|
| Search | rounds 1 and 2 have ended and every arm is in the table of FM-D19 with its numbers |
| A common, accurate, short setting | the rule has chosen it (or its third point has, stated as such); both `fmeta.yaml` hold it; the configuration that each example file gives equals the `config.yaml` of the round 2 runs that measured it |
| Documents | READMEs of the two examples, FM-D19, `PROGRESS.md`, `README.md`, `docs/roadmap.md` agree with the measured numbers |
| Checks | `ruff`, the progress size, `verify_bench_results`, `pytest` pass in the image; pushed; CI green |

## Invariants

No file under `src/fmeta/` changes: no criterion of `.claude/rules/numerics-invariants.md` is
touched and the change gate (GPU gates) does not apply. Walkers are an existing mechanism with
its own contracts (rows 9 and 10 of the table there). If the scan shows that the package has
to change, that change gets its own plan entry and the gates of both back ends.

## Verification

- `tests/test_config.py`: both example files load; their `metad` and `run` values are equal.
- The round 2 runs of the chosen setting are the evidence of the READMEs: work directories,
  commit, back end and the output of `fmeta analyze` are named there.
