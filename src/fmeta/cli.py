"""Command line interface: ``fmeta <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import PRESETS, Config, preset_text
from .engine.platform import MODES, check_devices, resolve_devices


def _gpus(text: str):
    """The value of ``--gpus``: device numbers such as ``0,1``, or ``all``."""
    if text.strip() == "all":
        return "all"
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"give device numbers such as 0,1, or all (not {text!r})") from None


def _first_device(config: Config, device=None):
    """The device of a stage that uses one: the one given, or the first device of the run."""
    if device is not None:
        return device
    return resolve_devices(config.run.mode, config.run.gpus, config.run.platform)[0]


def _config_from_args(args) -> Config:
    if args.config:
        config = Config.load(args.config)
    else:
        if args.complex and args.ligand_resname:
            data = {"input": {"complex": args.complex, "ligand_resname": args.ligand_resname,
                              "ligand_smiles": args.ligand_smiles}}
        elif args.protein and args.ligand:
            data = {"input": {"protein": args.protein, "ligand": args.ligand}}
        else:
            sys.exit("give a configuration file, --protein and --ligand, or --complex and "
                     "--ligand-resname")
        config = Config.from_dict(data, base_dir=Path.cwd())
    if args.membrane:
        from .system.builder import MembraneSettings
        config.membrane = config.membrane or MembraneSettings()
        config.membrane.lipid = args.membrane
    if args.ligand_xml:
        config.system.ligand_method = "xml"
        config.system.ligand_xml = [str(Path(p).resolve()) for p in args.ligand_xml]
    if args.length_ns is not None:
        config.run.length_ns = args.length_ns
    if args.replicas is not None:
        config.run.replicas = args.replicas
    if args.gpus is not None:
        config.run.gpus = args.gpus
    if args.mode:
        config.run.mode = args.mode
    if args.platform:
        config.run.platform = args.platform
    check_devices(config.run.mode, config.run.gpus)
    return config


def _add_setup_options(parser) -> None:
    parser.add_argument("config", nargs="?", help="YAML configuration (see 'fmeta init')")
    parser.add_argument("--protein", help="protein structure (PDB or mmCIF)")
    parser.add_argument("--ligand", help="posed ligand with hydrogens (SDF, MOL, MOL2)")
    parser.add_argument("--complex", help="structure holding protein and ligand (PDB or mmCIF)")
    parser.add_argument("--ligand-resname", help="residue name of the ligand in --complex")
    parser.add_argument("--ligand-smiles", help="ligand SMILES, to add its hydrogens (needs RDKit)")
    parser.add_argument("--membrane", nargs="?", const="POPC", metavar="LIPID",
                        help="embed in a lipid bilayer (default POPC); the protein must be "
                             "oriented with the membrane normal along z")
    parser.add_argument("--ligand-xml", nargs="+", help="OpenMM ffxml file(s) with the ligand template")
    parser.add_argument("-o", "--out", default="fmeta_run", help="work directory")
    parser.add_argument("--length-ns", type=float, help="production length per replica")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--gpus", type=_gpus, help="the GPUs of the run: device numbers such as 0,1, or all")
    parser.add_argument("--mode", choices=MODES,
                        help="parallel: one replica per GPU, side by side; sequential: one replica after "
                             "another on one GPU; auto (default): parallel if GPUs are given")
    parser.add_argument("--platform", help="OpenMM platform (default: auto)")


def cmd_init(args) -> None:
    path = Path(args.path)
    if path.exists():
        sys.exit(f"{path} already exists")
    path.write_text(preset_text(args.preset))
    print(f"wrote {path} (preset {args.preset})")


def cmd_split(args) -> None:
    from .system.inputs import split_complex
    notes: list[str] = []
    protein, ligand = split_complex(args.complex, args.resname, args.out, args.smiles,
                                    args.chain, args.resid, drop=args.drop, notes=notes)
    for note in notes:
        print(note)
    print(f"wrote {protein} and {ligand}")


def cmd_prepare(args) -> None:
    from .workflow import prepare_stage
    config = _config_from_args(args)
    prepared = prepare_stage(config, args.out, _first_device(config))
    print(prepared.report)
    print(f"\n{prepared.topology.getNumAtoms()} atoms; files in {Path(args.out) / 'prepared'}")


def cmd_funnel(args) -> None:
    report = json.loads((Path(args.workdir) / "prepared" / "report.json").read_text())
    for check in report["checks"]:
        print(f"[{check['status']:7s}] {check['name']}: {check['message']}")
    print(f"\nfunnel definition: {Path(args.workdir) / 'prepared' / 'funnel.yaml'}")
    print(f"view with the structure: {Path(args.workdir) / 'prepared' / 'funnel.pdb'}")


def cmd_equilibrate(args) -> None:
    from .workflow import equilibrate_stage
    config = Config.load(Path(args.workdir) / "config.yaml")
    summary = equilibrate_stage(config, args.workdir, _first_device(config, args.device))
    print(f"funnel wall energy during the free stage: mean {summary['wall_energy_mean']:.3f} kJ/mol"
          f" ({'ok' if summary['wall_ok'] else 'TOO HIGH: widen the funnel'})")


def cmd_run(args) -> None:
    from .workflow import run_replicas, run_stage
    config = Config.load(Path(args.workdir) / "config.yaml")
    if args.replica is not None:
        run_stage(config, args.workdir, args.replica, args.device, args.length_ns, args.walker)
    else:
        run_replicas(config, args.workdir, args.gpus, args.length_ns, args.mode)


def cmd_status(args) -> None:
    from .workflow import format_status, status_table
    print(format_status(status_table(args.workdir)))


def cmd_analyze(args) -> None:
    from .workflow import analyze_stage, format_summary
    print(format_summary(analyze_stage(args.workdir, plot=args.plot)))
    if args.plot:
        print(f"figures: {Path(args.workdir) / 'summary.png'} and rep*/analysis.png")


def cmd_auto(args) -> None:
    from .workflow import format_summary, run_all
    print(format_summary(run_all(_config_from_args(args), args.out)))


def cmd_check(args) -> None:
    from .check import Checker
    gpus = args.gpus
    if gpus == "all":
        gpus = resolve_devices("parallel", "all", args.platform or "auto")
    checker = Checker(args.out, args.platform or "auto", args.precision, args.device, gpus,
                      args.quick)
    if not checker.run_all():
        sys.exit(1)


def cmd_doctor(args) -> None:
    import numpy as np
    import openmm as mm

    from .engine import BiasSettings, FunnelBias
    from .engine.platform import available_platforms, select_platform
    from .testing.toy import make_toy

    print(f"OpenMM {mm.__version__}; platforms: {', '.join(available_platforms())}")
    platform, properties = select_platform(args.platform or "auto")
    print(f"selected: {platform.getName()} {properties or ''}")
    try:
        import openmmforcefields  # noqa: F401
        print("openmmforcefields: available (automatic ligand parameters)")
    except ImportError:
        print("openmmforcefields: not installed (use ligand ffxml or pre-parameterised input)")

    # the funnel forces and the bias table on the selected platform, against Reference
    values = {}
    for name in dict.fromkeys(["Reference", platform.getName()]):
        toy = make_toy()
        bias = FunnelBias(toy.system, toy.funnel, BiasSettings(), platform=name)
        bias.grid.add_hill(1.4, 0.1, 5.0)
        selected = mm.Platform.getPlatformByName(name)
        context = mm.Context(toy.system, mm.VerletIntegrator(0.001), selected,
                             properties if name == platform.getName() else {})
        positions = toy.positions.copy()
        positions[0] += np.array([0.12, -0.05, 0.4])
        context.setPositions(positions)
        bias.push(context)
        state = context.getState(getEnergy=True, getForces=True)
        values[name] = (state.getPotentialEnergy()._value, np.array(state.getForces(asNumpy=True)._value))
        if name == platform.getName():
            start = time.time()
            for _ in range(20):
                bias.push(context)
            cost = (time.time() - start) / 20
            print(f"bias table update: {1000 * cost:.1f} ms for {bias.grid.values.size} grid points")
    reference, selected = values["Reference"], values[platform.getName()]
    print(f"funnel + bias energy {selected[0]:.4f} kJ/mol; difference from Reference: "
          f"{abs(selected[0] - reference[0]):.2e} kJ/mol, "
          f"max force difference {np.abs(selected[1] - reference[1]).max():.2e} kJ/mol/nm")


# ---- benchmark sets (FM-D17): fmeta benchmark prepare | run | status | report ----

def cmd_benchmark_prepare(args) -> None:
    from .benchmark import load_set, prepare_set
    print("\n".join(prepare_set(load_set(args.set), args.out, args.ligands, args.length_ns, args.replicas)))
    print(f"next: fmeta benchmark run {args.set} -o {args.out}")


def cmd_benchmark_build(args) -> None:
    """One job of 'benchmark run': the prepare stage of one ligand, on the device the scheduler gave."""
    from .workflow import prepare_stage
    config = Config.load(Path(args.workdir) / "config.yaml")
    prepared = prepare_stage(config, args.workdir, _first_device(config, args.device))
    print(prepared.report)
    print(f"\n{prepared.topology.getNumAtoms()} atoms; files in {Path(args.workdir) / 'prepared'}")


def cmd_benchmark_run(args) -> None:
    from .benchmark import format_status, load_set, run_set, status_rows
    benchmark = load_set(args.set)
    try:
        status = run_set(benchmark, args.out, args.gpus, args.mode, args.until)
    finally:
        try:
            print(format_status(status_rows(benchmark, args.out)))
        except ValueError:                   # nothing is prepared: the error of run_set says so
            pass
    if not status:
        print(f"nothing left to do up to the stage '{args.until}'")


def cmd_benchmark_status(args) -> None:
    from .benchmark import format_status, load_set, status_rows
    print(format_status(status_rows(load_set(args.set), args.out)))


def cmd_benchmark_report(args) -> None:
    from .benchmark import code_version, load_set, report_set
    path = report_set(load_set(args.set), args.out, args.reports, code_version(args.commit))
    report = json.loads(path.read_text())
    print(path.with_suffix(".md").read_text())
    print(f"wrote {path} and {path.with_suffix('.md')}")
    unknown = [name for name, value in (("the image", report["environment"]["image"]),
                                        ("the lock of the image", report["environment"]["lock_sha256"]),
                                        ("whether the code differs from the commit", report["dirty"]))
               if value is None]
    if unknown or report["dirty"]:
        print("note: " + "; ".join(([f"not known: {', '.join(unknown)}"] if unknown else [])
                                   + (["the code differs from the commit"] if report["dirty"] else []))
              + ". A report to commit is written in the official image (scripts/in_docker.sh) from a clean commit.")


def _add_benchmark_commands(commands) -> None:
    benchmark = commands.add_parser(
        "benchmark", help="a set of ligands with experimental values: prepare, run, status, report")
    stages = benchmark.add_subparsers(dest="benchmark_command", required=True)

    def add(name, function, text):
        sub = stages.add_parser(name, help=text)
        sub.add_argument("set", help="directory of the set definition (set.yaml, reference.csv)")
        sub.add_argument("-o", "--out", required=True, help="work directory of the benchmark")
        sub.set_defaults(func=function)
        return sub

    sub = add("prepare", cmd_benchmark_prepare,
              "fetch and verify the inputs, write one fmeta work directory per ligand")
    sub.add_argument("--ligands", default="pilot",
                     help="pilot (default), all, or names of ligands separated by commas")
    sub.add_argument("--length-ns", type=float, help="length per replica (default: that of the set)")
    sub.add_argument("--replicas", type=int, help="replicas per ligand (default: that of the set)")

    sub = add("run", cmd_benchmark_run,
              "run or continue every stage of every prepared ligand in one scheduler call")
    sub.add_argument("--gpus", type=_gpus, help="device numbers such as 0,1, or all (default: run.gpus of the set)")
    sub.add_argument("--mode", choices=MODES, help="parallel, sequential or auto (default: run.mode of the set)")
    sub.add_argument("--until", choices=("prepare", "equilibrate", "run"), default="run",
                     help="stop after this stage of every ligand (default: run)")

    add("status", cmd_benchmark_status, "where every stage of every prepared ligand stands")

    sub = add("report", cmd_benchmark_report,
              "analyse the ligands and write the comparison with experiment and FEP")
    sub.add_argument("--reports", default="bench/reports", help="directory of the reports (default: bench/reports)")
    sub.add_argument("--commit", help="full hash of the commit of this code, where git cannot tell it "
                                      "(in a container: the output of 'git rev-parse HEAD' on the host)")

    sub = stages.add_parser("build")         # what 'benchmark run' starts for the prepare stage of one ligand
    sub.add_argument("workdir")
    sub.add_argument("--device", type=int)
    sub.set_defaults(func=cmd_benchmark_build)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="fmeta", description="Funnel metadynamics with OpenMM")
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("init", help="write a configuration with every key and its official default")
    sub.add_argument("path", nargs="?", default="fmeta.yaml")
    sub.add_argument("--preset", choices=PRESETS, default="soluble",
                     help="soluble (default): a complex in water; membrane: in a lipid bilayer")
    sub.set_defaults(func=cmd_init)

    sub = commands.add_parser("split", help="cut a complex into protein.pdb and ligand.sdf")
    sub.add_argument("complex")
    sub.add_argument("--resname", required=True, help="ligand residue name")
    sub.add_argument("--smiles", help="ligand SMILES in the protonation state to simulate")
    sub.add_argument("--chain")
    sub.add_argument("--resid")
    sub.add_argument("--drop", action="append", default=[], metavar="CHAIN[:FIRST-LAST]",
                     help="remove a chain or a residue range of it first, e.g. the fusion partner "
                          "A:1001-1106 (repeatable; insertion codes go with their number)")
    sub.add_argument("-o", "--out", default="inputs")
    sub.set_defaults(func=cmd_split)

    sub = commands.add_parser("prepare", help="build the system and place the funnel")
    _add_setup_options(sub)
    sub.set_defaults(func=cmd_prepare)

    sub = commands.add_parser("funnel", help="show the funnel report of a prepared system")
    sub.add_argument("workdir")
    sub.set_defaults(func=cmd_funnel)

    sub = commands.add_parser("equilibrate", help="minimise, heat and equilibrate")
    sub.add_argument("workdir")
    sub.add_argument("--device", type=int)
    sub.set_defaults(func=cmd_equilibrate)

    sub = commands.add_parser("run", help="run or continue production")
    sub.add_argument("workdir")
    sub.add_argument("--replica", type=int, help="run only this replica")
    sub.add_argument("--walker", type=int, help="with --replica: run only this walker of it")
    sub.add_argument("--device", type=int, help="GPU device index for --replica")
    sub.add_argument("--gpus", type=_gpus,
                     help="the GPUs for this run: device numbers such as 0,1, or all (default: run.gpus)")
    sub.add_argument("--mode", choices=MODES,
                     help="for this run: parallel (one replica per GPU, side by side), sequential (one "
                          "replica after another on one GPU) or auto (default: run.mode)")
    sub.add_argument("--length-ns", type=float, help="total length per replica")
    sub.set_defaults(func=cmd_run)

    sub = commands.add_parser("status", help="how far every replica is, and on which device")
    sub.add_argument("workdir")
    sub.set_defaults(func=cmd_status)

    sub = commands.add_parser("analyze", help="binding free energy and convergence")
    sub.add_argument("workdir")
    sub.add_argument("--plot", action="store_true", help="also write figures (needs matplotlib)")
    sub.set_defaults(func=cmd_analyze)

    sub = commands.add_parser("auto", help="prepare, equilibrate, run and analyse")
    _add_setup_options(sub)
    sub.set_defaults(func=cmd_auto)

    sub = commands.add_parser(
        "check", help="acceptance test on this machine: forces, toy free energy, full pipeline, "
                      "membrane, ligand parameters, throughput")
    sub.add_argument("-o", "--out", default="fmeta_check", help="directory for the test runs")
    sub.add_argument("--platform", help="OpenMM platform (default: the GPU platform found)")
    sub.add_argument("--precision", default="mixed")
    sub.add_argument("--device", type=int, help="GPU device index")
    sub.add_argument("--gpus", type=_gpus,
                     help="device numbers such as 0,1, or all: two or more also test replicas side by side")
    sub.add_argument("--quick", action="store_true",
                     help="minimal run lengths (a dry run of the steps, not a validation)")
    sub.set_defaults(func=cmd_check)

    sub = commands.add_parser("doctor", help="quick look at platforms and optional packages")
    sub.add_argument("--platform")
    sub.set_defaults(func=cmd_doctor)

    _add_benchmark_commands(commands)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(args)
    except (ValueError, ImportError, FileNotFoundError, RuntimeError) as error:
        if args.verbose:
            raise
        # problems with the inputs or the installation: say what is wrong, without a traceback
        sys.exit(f"fmeta: error: {error}")


if __name__ == "__main__":
    main()
