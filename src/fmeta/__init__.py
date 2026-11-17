"""fmeta: funnel metadynamics on top of OpenMM.

Units are OpenMM's native ones throughout: nm, ps, kJ/mol, K, amu.
"""

from .funnel.geometry import Funnel, compute_cvs

__version__ = "0.1.0"
__all__ = ["Funnel", "compute_cvs", "__version__"]
