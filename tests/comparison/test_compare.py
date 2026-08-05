"""Compare dstparser's grid output against the standalone reference parser.

`dstParser.py` in this directory is a standalone reimplementation that shells
out to the sdanalysis binaries directly. It hardcodes paths under
`/dicos_ui_home/anatoli/...`, i.e. a different machine, so this comparison
skips unless that install and the reference DST are both present.

Previously the body ran at import time and raised, which aborted pytest
collection for the WHOLE suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from dstparser import parse_dst_file
from dstparser.paths import sdanalysis_root

DST_FILE = (
    f"{sdanalysis_root}/sdanalysis_2018_TALE_TAx4SingleCT_DM/"
    "DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"
)


def _reference_parser():
    """Import the standalone parser, or skip if its environment is absent."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from dstParser import parse_script
    except Exception as exc:  # noqa: BLE001 - any import failure means "not here"
        pytest.skip(f"reference dstParser.py unavailable: {exc}")
    return parse_script


def test_grid_matches_reference_parser():
    if not Path(DST_FILE).exists():
        pytest.skip(f"reference DST not available: {DST_FILE}")
    parse_script = _reference_parser()

    data0 = parse_dst_file(DST_FILE)
    data1 = parse_script(DST_FILE)
    assert data0 is not None

    # dstParser.py shells out to binaries under /dicos_ui_home/...; off that
    # machine it returns an empty result instead of failing, so check for that
    # rather than comparing against nothing.
    if data1 is None or len(data1.get("energy", ())) == 0:
        pytest.skip("reference parser produced no events (its binaries are not on this host)")

    shared = [k for k in data1 if k in data0]
    assert shared, "no comparable keys"
    mismatched = []
    for key in shared:
        a, b = np.asarray(data0[key]), np.asarray(data1[key])
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-5):
            mismatched.append(key)
    assert not mismatched, f"differ from reference parser: {mismatched}"


if __name__ == "__main__":
    test_grid_matches_reference_parser()
    print("comparison PASSED")
