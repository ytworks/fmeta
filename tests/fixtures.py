"""Test helpers. The small real system itself lives in the package
(``fmeta.testing.systems``) because ``fmeta check`` uses it too."""

from fmeta.testing.systems import (  # noqa: F401
                                   BENZENE_XML,
                                   pocket_point,
                                   write_benzene,
                                   write_complex_inputs,
                                   write_membrane_inputs,
                                   write_protein,
)
