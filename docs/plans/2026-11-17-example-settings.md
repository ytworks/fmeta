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

One replica has no interval. An arm goes on by what its surface shows (passages back into the
site summed over the walkers, the bias in the site against the bias in the unbound window, how
the value moves between 10 and 20 ns), with the experimental value in view. This is a
judgment; it is recorded with the numbers of every arm. A further arm (another bias factor,
height or number of walkers) is added when round 1 points to it, and recorded the same way.

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

Values are the headline of the product analysis (`fmeta analyze`; FM-D9, third amendment).
The checks of a replica stay diagnostics.

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
