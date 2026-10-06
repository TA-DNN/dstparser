"""Self-contained schema/alignment/storage tests; no cluster or ROOT required."""

from pathlib import Path
import struct
import sys

import h5py
import numpy as np
import pytest

from dstparser.banks import (
    Bank,
    Event,
    RAW,
    MC,
    MUON,
    RECON,
    DSTError,
    iter_events,
    write_events,
    crc,
)
from dstparser.preparation import prepare_dst, _restore
from dstparser.muon import align, grid_allocate, grid_fill
from dstparser.dst_adapter import cut_events
from dstparser.join_vlen_data import append_to_hdf5, append_to_memstore
from dstparser.h5dst.h5dst import H5DST, MultiH5DST
from dstparser.schema import promote, validate
from dstparser.campaign import add_campaign_metadata
from dstparser.cli.io import atomic_output
from dstparser.dst_adapter_vlen import detector_readings_flat


def make_event(event_id=7, n=3, muon=True, version=1, reconstructed=False):
    raw = bytearray(72 + 1280 * n)
    struct.pack_into("<iii", raw, 0, RAW, version, event_id)
    struct.pack_into("<i", raw, 68, n)
    ids = np.arange(n, dtype="<i4") + 101
    for j, a in enumerate(
        [
            np.zeros(n, dtype="<i4"),
            np.arange(n, dtype="<i4"),
            np.zeros(n, dtype="<i4"),
            ids,
            np.arange(n, dtype="<i4") + 100,
            np.full(n, 50000000, dtype="<i4"),
        ]
    ):
        raw[72 + 4 * n * j : 72 + 4 * n * (j + 1)] = a.tobytes()
    for j in range(n):
        start = 72 + 24 * n + 1256 * j + 16
        raw[start : start + 1024] = (np.arange(256, dtype="<i4") + j).tobytes()
    mc = struct.pack(
        "<6i7f", MC, 0, event_id, 14, 0, 0, 10.0, 100000.0, 0.5, 0.4, 0.0, 0.0, 0.0
    )
    banks = {MC: Bank(MC, 0, mc), RAW: Bank(RAW, version, bytes(raw))}
    if muon:
        # Deliberately signed and deliberately > total; these are legal proxies.
        arrays = [
            ids,
            np.arange(n, dtype="<i4"),
            np.full(n, -2, dtype="<f4"),
            np.full(n, 8, dtype="<f4"),
            np.full(n, 4, dtype="<f4"),
            np.full(n, 6, dtype="<f4"),
            np.zeros(n, dtype="<i4"),
            np.full(n, 3, dtype="<i4"),
        ]
        payload = struct.pack("<4i", MUON, 0, n, event_id) + b"".join(
            a.tobytes() for a in arrays
        )
        banks[MUON] = Bank(MUON, 0, payload)
    if reconstructed:
        for b in sorted(RECON):
            banks[b] = Bank(b, 0, struct.pack("<2i", b, 0))
    return Event(0, banks)


@pytest.mark.parametrize("n,version", [(0, 0), (256, 0), (257, 1), (1024, 1)])
def test_framing_roundtrip(tmp_path, n, version):
    source = make_event(n=n, version=version)
    p = tmp_path / "roundtrip.dst.gz"
    write_events(p, [source])
    events = list(iter_events(p))
    assert len(events) == 1
    assert events[0].banks == source.banks
    np.testing.assert_array_equal(
        events[0].muon["muon_signal"], np.tile([8.0, -2.0], (n, 1))
    )
    np.testing.assert_array_equal(events[0].muon["muon_nsat"], np.tile([3, 0], (n, 1)))


def test_crc_reference_and_corruption(tmp_path):
    assert crc(b"123456789") == 0x2189
    p = tmp_path / "bad.dst"
    write_events(p, [make_event()])
    b = bytearray(p.read_bytes())
    b[100] ^= 1
    p.write_bytes(b)
    with pytest.raises(DSTError, match="CRC"):
        list(iter_events(p))


def test_truncated_and_duplicate_identity(tmp_path):
    p = tmp_path / "truncated.dst"
    write_events(p, [make_event()])
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(DSTError, match="Truncated"):
        list(iter_events(p))
    e = make_event(n=2)
    payload = bytearray(e.banks[MUON].payload)
    struct.pack_into("<i", payload, 16, 999)
    e.banks[MUON] = Bank(MUON, 0, bytes(payload))
    p = tmp_path / "identity.dst"
    write_events(p, [e])
    with pytest.raises(DSTError, match="identity"):
        list(iter_events(p))


def test_unsupported_version_and_limit(tmp_path):
    for n, version in [(257, 0), (1025, 1), (1, 2)]:
        p = tmp_path / f"{n}_{version}.dst"
        write_events(p, [make_event(n=n, version=version)])
        with pytest.raises(DSTError):
            list(iter_events(p))


def test_alignment_duplicate_event_numbers_and_reordering(tmp_path):
    a = make_event(n=2)
    b = make_event(n=1)
    p = tmp_path / "two.dst"
    write_events(p, [a, b])
    waves = []
    for e in [a, b]:
        raw = e.raw
        waves.append(
            np.vstack(
                [raw["keys"][:, 0], raw["clocks"].T, raw["traces"].reshape(-1, 256).T]
            )
        )
    with pytest.warns(RuntimeWarning):
        labels, identity = align(p, waves)
    np.testing.assert_array_equal(identity["source_event_ordinal"], [0, 1])
    np.testing.assert_array_equal(identity["source_event_number"], [7, 7])
    assert labels[0]["muon_signal"][0, 1] == -2
    with pytest.raises(DSTError, match="mismatch"):
        align(p, list(reversed(waves)))


def test_grid_labels_follow_selected_first_window():
    meta = np.zeros((12, 3))
    meta[0] = [101, 102, 101]
    meta[1] = [1, 4, 4]
    meta[11] = [2, 1, 1]
    meta[4:6] = 2
    wf = np.zeros((259, 4))
    event, traces, indices = cut_events(meta, wf, return_indices=True)
    np.testing.assert_array_equal(indices, [2, 3])
    labels = {
        "muon_signal": np.arange(8, dtype=np.float32).reshape(4, 2),
        "muon_total_signal": np.ones((4, 2)),
        "muon_nsat": np.tile([1, 2], (4, 1)),
        "muon_available": np.ones(4, bool),
        "source_waveform_id": np.arange(4),
    }
    d = {}
    grid_allocate(d, (1, 3, 3), False)
    xy = np.array([[0, 1], [1, 1]])
    grid_fill(d, 0, xy, labels, indices, False)
    assert d["muon_signal_low"][0, 0, 1] == 4 and d["muon_signal_up"][0, 0, 1] == 5
    assert d["source_waveform_id"][0, 1, 1] == 3
    assert not d["muon_available"][0, 2, 2]
    assert np.isnan(d["muon_signal_low"][0, 2, 2])


def sample_vlen(muon=False):
    d = {
        "energy": np.array([10.0], dtype=np.float64),
        "std_recon_energy": np.array([9.0]),
        "arrival_times": np.ones((1, 2)),
        "pulse_area": np.ones((1, 2)),
        "detector_positions": np.ones((1, 3)),
        "status": np.array([4], np.int32),
        "nfold": np.array([1], np.int32),
        "detector_ids": np.array([101], np.int32),
        "hit_offsets": np.array([0, 1], np.int64),
        "tt_offsets": np.array([0, 1], np.int64),
        "hit_tt_offsets": np.array([0, 1], np.int64),
        "time_traces": np.full((1, 2, 128), 55, dtype=np.float32),
        "badsd": np.array([], np.int32),
        "badsd_offsets": np.array([0, 0], np.int64),
    }
    if muon:
        d = promote(d)
        d["muon_signal"][:] = [8.0, -2.0]
        d["muon_total_signal"][:] = [6.0, 4.0]
        d["muon_nsat"][:] = [3, 0]
        d["muon_available"][:] = True
    return d


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_mixed_hdf5_and_metadata_not_length_guessing(tmp_path, order):
    p = tmp_path / "mixed.h5"
    mem = {}
    with h5py.File(p, "w") as f:
        for mu in order:
            d = sample_vlen(mu)
            append_to_hdf5(f, d)
            append_to_memstore(mem, d)
        validate({k: f[k][...] for k in f})
        for k in mem:
            np.testing.assert_equal(f[k][...], mem[k])
        assert f["muon_signal"].attrs["level"] == "waveform"
        assert f["muon_signal"].attrs["layer_order"] == "lower,upper"
    h = H5DST(p)
    j = order.index(True)
    np.testing.assert_array_equal(h["muon_signal"][j][0], [[8.0, -2.0]])
    assert np.isnan(h["muon_signal"][1 - j][0]).all()
    np.testing.assert_array_equal(h["time_traces"][j][0], np.full((1, 2, 128), 55))


def test_multi_file_missing_labels(tmp_path):
    files = []
    for i, mu in enumerate([False, True]):
        p = tmp_path / f"{i}.h5"
        files.append(str(p))
        with h5py.File(p, "w") as f:
            append_to_hdf5(f, sample_vlen(mu))
    h = MultiH5DST(files)
    assert np.isnan(h["muon_signal"][0][0]).all()
    np.testing.assert_array_equal(h["muon_signal"][1][0], [[8.0, -2.0]])
    assert len(h["muon_signal"][:]) == 2
    np.testing.assert_array_equal(h["energy"][:], [10.0, 10.0])
    assert h["source_event_number"][:].shape == (2,)
    h.close()


def test_invalid_append_rejected_before_mutation(tmp_path):
    with h5py.File(tmp_path / "x.h5", "w") as f:
        append_to_hdf5(f, sample_vlen(True))
        bad = sample_vlen(True)
        bad["tt_offsets"] = np.array([0, 2])
        with pytest.raises(ValueError):
            append_to_hdf5(f, bad)
        assert len(f["energy"]) == 1 and not f.attrs["dstparser_append_in_progress"]


def test_atomic_publication_and_interrupted_read(tmp_path):
    p = tmp_path / "out.h5"
    with pytest.raises(RuntimeError):
        with atomic_output(p) as temp:
            Path(temp).write_bytes(b"partial")
            raise RuntimeError("interrupted")
    assert not p.exists()
    with h5py.File(p, "w") as f:
        append_to_hdf5(f, sample_vlen())
        f.attrs["dstparser_append_in_progress"] = True
    with pytest.raises(ValueError, match="Incomplete"):
        H5DST(p)


def test_preparation_cache_and_preservation(tmp_path):
    source = tmp_path / "source.dst.gz"
    write_events(source, [make_event()])
    tool = tmp_path / "tool.py"
    tool.write_text("""from dstparser.banks import *
import sys,struct
events=list(iter_events(sys.argv[1]))
for e in events:
 e.banks.pop(MUON,None)
 for bid in RECON:e.banks[bid]=Bank(bid,0,struct.pack('<ii',bid,0))
write_events(sys.argv[2],events)
""")
    cmd = [sys.executable, str(tool), "{input}", "{output}"]
    dep = tmp_path / "calibration"
    dep.write_text("one")
    with pytest.raises(DSTError, match="Missing reconstruction"):
        prepare_dst(source, "require")
    a = prepare_dst(
        source,
        cache_dir=tmp_path / "cache",
        commands=[cmd, cmd],
        dependencies=[tool, dep],
    )
    assert next(iter_events(a)).banks[MUON] == next(iter_events(source)).banks[MUON]
    before = a.stat().st_mtime_ns
    assert (
        prepare_dst(
            source,
            cache_dir=tmp_path / "cache",
            commands=[cmd, cmd],
            dependencies=[tool, dep],
        )
        == a
    )
    assert a.stat().st_mtime_ns == before
    dep.write_text("two")
    b = prepare_dst(
        source,
        cache_dir=tmp_path / "cache",
        commands=[cmd, cmd],
        dependencies=[tool, dep],
    )
    assert a != b
    a.write_bytes(b"broken")
    dep.write_text("one")
    with pytest.raises(DSTError, match="integrity"):
        prepare_dst(
            source,
            cache_dir=tmp_path / "cache",
            commands=[cmd, cmd],
            dependencies=[tool, dep],
        )


def test_reconstruction_reordering_fails(tmp_path):
    a = tmp_path / "a.dst"
    b = tmp_path / "b.dst"
    write_events(a, [make_event(1), make_event(2)])
    write_events(
        b, [make_event(2, reconstructed=True), make_event(1, reconstructed=True)]
    )
    with pytest.raises(DSTError, match="changed truth"):
        list(_restore(a, b))


def test_known_raw_version_transition_preserves_fields(tmp_path):
    a = tmp_path / "old.dst"
    b = tmp_path / "new.dst"
    write_events(a, [make_event(version=0)])
    write_events(b, [make_event(version=1, muon=False, reconstructed=True)])
    restored = list(_restore(a, b))
    assert restored[0].banks[RAW].version == 1
    assert MUON in restored[0].banks
    assert restored[0].fingerprint == next(iter_events(a)).fingerprint


def test_campaign_filename_does_not_define_energy(tmp_path):
    manifest = tmp_path / "grid.tsv"
    manifest.write_text(
        "tag\trunnr\tloge\ttheta_deg\tmodel\nDAT920950\t920950\t19.0\t30\tq\n"
    )
    d = add_campaign_metadata(sample_vlen(), "DAT920950.dst.gz", manifest)
    assert d["id_corsika_shower"][0] == 920950
    assert d["id_energy_bin"][0] == 190
    assert d["shower_log10_energy"][0] == 19


def test_vlen_labels_follow_all_fold_windows():
    hits = {
        "rufptn_.xxyy": np.array([101, 102]),
        "rufptn_.nfold": np.array([2, 1]),
        "rufptn_.isgood": np.array([4, 1]),
        "offsets": np.array([0, 2]),
    }
    for k in [
        "reltime[0]",
        "reltime[1]",
        "pulsa[0]",
        "pulsa[1]",
        "xyzclf[0]",
        "xyzclf[1]",
        "xyzclf[2]",
        "vem[0]",
        "vem[1]",
    ]:
        hits["rufptn_." + k] = np.ones(2)
    waves = {
        "rusdraw_.xxyy": np.array([101, 101, 102]),
        "rusdraw_.fadc": np.ones((3, 2, 128)),
        "offsets": np.array([0, 3]),
        "muon_signal": np.array([[1, -1], [2, -2], [3, -3]], np.float32),
        "source_waveform_id": np.array([10, 11, 12]),
    }
    out = detector_readings_flat("fixture", {}, hits, waves)
    np.testing.assert_array_equal(out["muon_signal"], [[1, -1], [2, -2]])
    np.testing.assert_array_equal(out["source_waveform_id"], [10, 11])
    np.testing.assert_array_equal(out["hit_tt_offsets"], [0, 2])


def test_missing_and_zero_muon_are_distinct(tmp_path):
    source = tmp_path / "mixed.dst"
    write_events(source, [make_event(n=1, muon=False), make_event(n=1)])
    waves = []
    for e in iter_events(source):
        raw = e.raw
        waves.append(
            np.vstack(
                [raw["keys"][:, 0], raw["clocks"].T, raw["traces"].reshape(-1, 256).T]
            )
        )
    with pytest.warns(RuntimeWarning):
        labels, _ = align(source, waves)
    assert not labels[0]["muon_available"][0]
    assert np.isnan(labels[0]["muon_signal"]).all()
    assert labels[1]["muon_available"][0]


def test_reconstruction_subprocess_failure_is_not_published(tmp_path):
    p = tmp_path / "input.dst"
    write_events(p, [make_event()])
    cmd = [sys.executable, "-c", "import sys; sys.exit(7)", "{input}", "{output}"]
    with pytest.raises(DSTError, match="stage 0 failed"):
        prepare_dst(p, cache_dir=tmp_path / "cache", commands=[cmd, cmd])
    assert not list((tmp_path / "cache").glob("*/complete.json"))


def test_campaign_auto_discovery(tmp_path):
    from dstparser.campaign import resolve_manifest

    manifest = tmp_path / "grid.tsv"
    manifest.write_text("tag\tloge\ttheta_deg\tmodel\nDAT920950\t19.0\t30\tq\n")
    sub = tmp_path / "outputs"
    sub.mkdir()
    p = sub / "DAT920950.dst.gz"
    p.touch()
    assert resolve_manifest(p, None) == manifest
    assert resolve_manifest(p, False) is None


def test_xmax_campaign_missing_truth_does_not_generate_random_values():
    from dstparser.xmax_reader.xmax_reader import XmaxReaderTxt

    reader = XmaxReaderTxt(None)
    row = {"tag": "DAT920950", "model": "q", "loge": "19", "theta_deg": "30"}
    with pytest.raises(ValueError, match="Measured Xmax unavailable"):
        reader.read_campaign_file("DAT920950.dst.gz", row)
    reader.read_campaign_file("DAT920950.dst.gz", {**row, "xmax_g_cm2": "750"})
    np.testing.assert_array_equal(reader(np.array([10.0]), np.array([1])), [750.0])


def test_empty_cli_group_is_explicit_and_mergeable(tmp_path):
    from dstparser.cli.worker_job import _save_grid_chunks, join_hdf5_vlen
    from dstparser.cli.io import read_h5
    from types import SimpleNamespace

    empty = tmp_path / "empty.h5"
    _save_grid_chunks([], empty)
    assert read_h5(empty) == {}
    h = MultiH5DST([str(empty)])
    assert len(h) == 0
    h.close()
    full = tmp_path / "full.h5"
    with h5py.File(full, "w") as f:
        append_to_hdf5(f, sample_vlen(True))
    final = tmp_path / "combined.h5"
    join_hdf5_vlen([empty, full], final, SimpleNamespace())
    assert len(read_h5(final)["energy"]) == 1
