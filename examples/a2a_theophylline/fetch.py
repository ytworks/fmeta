"""Download the OPM-oriented file of PDB entry 5MZJ (A2A receptor with theophylline).

usage (from the repository root, in the official image):
    scripts/in_docker.sh python examples/a2a_theophylline/fetch.py runs/a2a_theophylline/download

OPM has no versioned addresses. The checksum below is that of the file this example was
written with (2026-11-16); a file that differs is kept, and the script ends with status 1 so
that the numbers of README.md are not taken over without a look at what changed.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://opm-assets.storage.googleapis.com/pdb/5mzj.pdb"
SHA256 = "4be1818c64db2fa8d2f63b3b2ec74fe25111f50b1d18dbc06b8d72f42ffd3d04"
NAME = "5mzj_opm.pdb"


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    target = out / NAME
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"{target}: {len(data)} bytes, sha256 {digest}")
    if digest != SHA256:
        print(f"this is not the file the example was written with (sha256 {SHA256}): "
              "OPM has changed it. Compare before using the numbers of README.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
