"""Platform selection: prefer the GPU, fall back to the CPU."""

from __future__ import annotations

from typing import Optional

import openmm as mm

GPU_PLATFORMS = ("CUDA", "HIP", "OpenCL")
_AUTO_ORDER = GPU_PLATFORMS + ("CPU", "Reference")


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
