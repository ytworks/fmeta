"""Platform and device selection: prefer the GPU, fall back to the CPU."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from typing import Optional

import openmm as mm

GPU_PLATFORMS = ("CUDA", "HIP", "OpenCL")
_AUTO_ORDER = GPU_PLATFORMS + ("CPU", "Reference")
MODES = ("auto", "parallel", "sequential")


def available_platforms() -> list[str]:
    return [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]


def _usable(name: str, properties: dict) -> bool:
    """A platform can be listed yet unusable (e.g. CUDA without a device)."""
    system = mm.System()
    system.addParticle(1.0)
    try:
        mm.Context(system, mm.VerletIntegrator(0.001),
                   mm.Platform.getPlatformByName(name), properties)
    except Exception:
        return False
    return True


def select_platform(name: str = "auto", precision: str = "mixed",
                    device_index: Optional[int | str] = None) -> tuple[mm.Platform, dict]:
    """Return ``(platform, properties)``.

    ``name='auto'`` picks the first usable of CUDA, HIP, OpenCL, CPU, Reference.
    ``precision`` and ``device_index`` apply to the GPU platforms only.
    """
    # CUDA numbers the devices as nvidia-smi does only in this order. It is read when CUDA
    # starts in the process, so it is set before the first context.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    def properties_for(platform_name: str) -> dict:
        if platform_name not in GPU_PLATFORMS:
            return {}
        props = {"Precision": precision}
        if device_index is not None:
            props["DeviceIndex"] = str(device_index)
        return props

    present = available_platforms()
    if name != "auto":
        if name not in present:
            raise ValueError(f"platform {name!r} is not available (found: {', '.join(present)})")
        return mm.Platform.getPlatformByName(name), properties_for(name)
    for candidate in _AUTO_ORDER:
        if candidate in present and _usable(candidate, properties_for(candidate)):
            return mm.Platform.getPlatformByName(candidate), properties_for(candidate)
    raise RuntimeError("no usable OpenMM platform found")


@contextmanager
def platform_defaults(platform: mm.Platform, properties: dict):
    """Make ``properties`` the defaults of ``platform`` inside the block.

    For code that takes a platform but no properties (``Modeller.addMembrane``): without this
    its contexts run on the default device, whatever device the run was given."""
    previous = {key: platform.getPropertyDefaultValue(key) for key in properties}
    try:
        for key, value in properties.items():
            platform.setPropertyDefaultValue(key, value)
        yield
    finally:
        for key, value in previous.items():
            platform.setPropertyDefaultValue(key, value)


def visible_gpus() -> Optional[list[dict]]:
    """The GPUs ``nvidia-smi`` lists, in its order: ``index``, ``uuid``, ``name`` and
    ``memory_used_mib``. None when it cannot be asked (no NVIDIA driver, another vendor)."""
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"]
    try:
        text = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit():
            gpus.append(dict(index=int(fields[0]), uuid=fields[1], name=fields[2],
                             memory_used_mib=int(fields[3]) if fields[3].isdigit() else None))
    return gpus


def device_record(context: mm.Context, requested=None, before: Optional[list] = None) -> dict:
    """What ``context`` runs on, for ``device.json``: the platform, the device number that was
    asked for and the precision, device number and device name that the platform reports.

    ``before`` is :func:`visible_gpus` from before the context existed. With it the record also
    holds the memory in use on the GPU that ``nvidia-smi`` lists under the platform's device
    number, before and with the context. A context on another GPU than the one that was asked
    for shows there as memory that did not grow."""
    platform = context.getPlatform()
    record = dict(platform=platform.getName(), requested_device=requested, pid=os.getpid())
    names = platform.getPropertyNames()
    for key, name in (("precision", "Precision"), ("device_index", "DeviceIndex"),
                      ("device_name", "DeviceName")):
        if name in names:
            record[key] = platform.getPropertyValue(context, name)
    index = record.get("device_index", "")
    after = visible_gpus() if before is not None and index.isdigit() else None
    if after is not None:
        was = {gpu["index"]: gpu for gpu in before}.get(int(index))
        now = {gpu["index"]: gpu for gpu in after}.get(int(index))
        if was is not None and now is not None and was["uuid"] == now["uuid"]:
            record.update(gpu_uuid=now["uuid"], gpu_name=now["name"],
                          memory_before_mib=was["memory_used_mib"],
                          memory_after_mib=now["memory_used_mib"])
    return record


def check_devices(mode: str, gpus, per_gpu: int = 1) -> None:
    """Raise ``ValueError`` when ``run.mode``, ``run.gpus`` and ``run.per_gpu`` contradict each other."""
    if mode not in MODES:
        raise ValueError(f"run.mode is one of {', '.join(MODES)}, not {mode!r}")
    if isinstance(per_gpu, bool) or not isinstance(per_gpu, int) or per_gpu < 1:
        raise ValueError(f"run.per_gpu is the number of replicas that share a GPU, 1 or more, not {per_gpu!r}")
    if mode == "sequential" and per_gpu > 1:
        raise ValueError("run.mode sequential runs one replica at a time: leave run.per_gpu at 1")
    if gpus is None or gpus == "all":
        if mode == "sequential" and gpus == "all":
            raise ValueError("run.mode sequential runs on one GPU: give its number, not 'all'")
        return
    if (not isinstance(gpus, (list, tuple)) or not gpus
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)):
        raise ValueError(f"run.gpus is 'all' or a list of device numbers such as [0, 1], not {gpus!r}")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"run.gpus names a device twice: {list(gpus)}")
    if mode == "sequential" and len(gpus) > 1:
        raise ValueError(f"run.mode sequential runs on one GPU: give one device number, not {list(gpus)}")


def resolve_devices(mode: str = "auto", gpus=None, platform: str = "auto", visible=visible_gpus,
                    per_gpu: int = 1) -> list:
    """The places that the replicas of a run are spread over, one replica per place.

    A device is named ``per_gpu`` times, once per replica that may share it; the list goes
    through the devices ``per_gpu`` times, so the first replicas each get a device of their own.
    ``[None]`` is one device, the one the platform picks by itself.

    - ``sequential``: one device, so the replicas run one after another.
    - ``parallel``: the devices of ``gpus``, or every GPU that is visible when ``gpus`` is
      None or ``'all'``. Inside a container those are the GPUs passed to it, numbered from 0.
    - ``auto``: ``parallel`` when ``gpus`` is given, one device otherwise.

    On the CPU and Reference platforms a device number is only a place in the queue:
    ``gpus: [0, 1]`` runs two replicas side by side. ``visible`` returns the visible GPUs
    (anything with a length) or None when that is unknown.
    """
    check_devices(mode, gpus, per_gpu)
    if mode == "auto":
        mode = "sequential" if gpus is None else "parallel"
    if mode == "sequential":
        return ([gpus[0]] if gpus else [None]) * per_gpu
    listed = visible() if platform == "auto" or platform in GPU_PLATFORMS else None
    if gpus is None or gpus == "all":
        if not listed:
            raise ValueError("run.mode parallel without device numbers needs GPUs that nvidia-smi lists; "
                             "name the devices instead, such as gpus: [0, 1]")
        return list(range(len(listed))) * per_gpu
    if listed is not None and max(gpus) >= len(listed):
        raise ValueError(f"device {max(gpus)} is not there: nvidia-smi lists {len(listed)} GPU(s), numbered "
                         "from 0. Inside a container the GPUs passed to it are numbered from 0 again.")
    return list(gpus) * per_gpu
