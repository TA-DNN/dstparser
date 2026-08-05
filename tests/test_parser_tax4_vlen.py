"""Regression test for the TAx4 vlen path (parse_dst_file_tax4_vlen).

Integration test: needs the shared TAx4 DST files. parse_dst_file_tax4_vlen is
parse_dst_file_vlen with the yyxx->xxyy detector-id swap.
"""
import numpy as np
from dstparser import parse_dst_file_tax4_vlen, parse_dst_file_vlen
from dstparser.paths import dstbank_root

# TAx4 proton files that contain triggered (nofwf>0) events -- both sub-arrays
TAX4_DST = (
    f"{dstbank_root}/tasdmc_dstbank/tax4/"
    "qgsii04proton/north/240125to240423/DAT010611_gea.rufldf.dst.gz"
)
TAX4_DST_SOUTH = (
    f"{dstbank_root}/tasdmc_dstbank/tax4/"
    "qgsii04proton/south/240125to240423/DAT000011_gea.rufldf.dst.gz"
)
# a TA-SD file, to check vlen key-set parity
TA_DST = (
    f"{dstbank_root}/tasdmc_dstbank/"
    "qgsii04proton/080417_160603/Em1_bsdinfo/XXXX03/DAT000003_gea.rufldf.dst.gz"
)


def test_tax4_vlen_structure():
    data = parse_dst_file_tax4_vlen(TAX4_DST)
    assert data is not None

    n_hits = int(data["hit_offsets"][-1])
    n_wf = int(data["tt_offsets"][-1])

    # offsets well-formed
    for off in ["hit_offsets", "tt_offsets", "hit_tt_offsets"]:
        o = data[off]
        assert o[0] == 0 and np.all(np.diff(o) >= 0)

    # per-hit arrays consistent with hit_offsets
    for k in ["arrival_times", "pulse_area", "detector_positions", "status",
              "nfold", "detector_ids"]:
        assert data[k].shape[0] == n_hits

    # waveforms consistent with nfold (nfold comes from the reader)
    assert data["time_traces"].shape[0] == n_wf
    assert np.array_equal(np.diff(data["hit_tt_offsets"]), data["nfold"])

    # only good hits kept, and every good hit has >=1 waveform (0% trace loss)
    assert np.all(data["status"] > 2)
    assert np.all(data["nfold"] > 0)
    assert np.all(np.isfinite(data["time_traces"]))


def test_tax4_detector_ids_swapped():
    """Both id components must be < 100 for the yyxx->xxyy swap to be valid,
    on BOTH sub-arrays (they occupy different id ranges)."""
    for path in (TAX4_DST, TAX4_DST_SOUTH):
        data = parse_dst_file_tax4_vlen(path)
        assert data is not None, f"no data from {path}"
        ids = data["detector_ids"]
        assert ids.min() > 0
        assert (ids // 100).max() < 100 and (ids % 100).max() < 100


def test_tax4_vlen_format_parity():
    """TAx4 vlen keys == TA-SD vlen keys, exactly."""
    tax4 = parse_dst_file_tax4_vlen(TAX4_DST)
    ta = parse_dst_file_vlen(TA_DST)
    assert set(tax4) == set(ta), (
        f"TA-only: {sorted(set(ta) - set(tax4))}, "
        f"TAx4-only: {sorted(set(tax4) - set(ta))}"
    )


if __name__ == "__main__":
    test_tax4_vlen_structure()
    test_tax4_detector_ids_swapped()
    test_tax4_vlen_format_parity()
    print("TAx4 vlen tests PASSED")
