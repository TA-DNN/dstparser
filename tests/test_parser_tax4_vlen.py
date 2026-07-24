"""Regression test for the TAx4 vlen adapter (parse_dst_file_tax4_vlen).

Integration test: needs the ceph TAx4 DST files AND the locally-built
nfold-emitting reader (see tax4_reader_build/build.sh). Mirrors
test_parser_vlen.py but with assertions on the invariants that matter for the
vlen format.
"""
import numpy as np
from dstparser import parse_dst_file_tax4_vlen, parse_dst_file_vlen

# a TAx4 proton file that contains triggered (nofwf>0) events
TAX4_DST = (
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/tax4/"
    "qgsii04proton/north/240125to240423/DAT000011_gea.rufldf.dst.gz"
)
# a TA-SD file, to check vlen key-set parity
TA_DST = (
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
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

    # waveforms consistent with nfold (nfold now comes from the reader)
    assert data["time_traces"].shape[0] == n_wf
    assert np.array_equal(np.diff(data["hit_tt_offsets"]), data["nfold"])

    # only good hits kept, and every good hit has >=1 waveform (0% trace loss)
    assert np.all(data["status"] > 2)
    assert np.all(data["nfold"] > 0)
    assert np.all(np.isfinite(data["time_traces"]))


def test_tax4_vlen_format_parity():
    """TAx4 vlen keys == TA-SD vlen keys, minus the 4 recon fields TAx4 does
    not compute; TAx4 adds none of its own."""
    tax4 = parse_dst_file_tax4_vlen(TAX4_DST)
    ta = parse_dst_file_vlen(TA_DST)
    ta_only = set(ta) - set(tax4)
    tax4_only = set(tax4) - set(ta)
    assert tax4_only == set(), f"TAx4 introduced unexpected keys: {tax4_only}"
    assert ta_only == {
        "std_recon_nsclust", "std_recon_nhits",
        "std_recon_nborder", "std_recon_qtot",
    }, f"unexpected key diff: {ta_only}"


if __name__ == "__main__":
    test_tax4_vlen_structure()
    test_tax4_vlen_format_parity()
    print("TAx4 vlen tests PASSED")
