"""Opt-in site tests against frozen pre-migration outputs and real campaign DSTs.

Set DSTPARSER_REGRESSION_BASELINES to the capture directory and
DSTPARSER_CAMPAIGN_ROOT to the completed campaign to enable these tests.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dstparser
from dstparser.banks import iter_events, write_events, MUON, sha256_file
from dstparser.schema import ADDITIONS
from dstparser.cli.io import read_h5
from dstparser.cli.worker_job import dst_to_hdf5, dst_to_hdf5_vlen


@pytest.mark.parametrize("kind", ["ta", "tax4"])
@pytest.mark.parametrize(
    "layout,avg", [("grid", False), ("grid", True), ("vlen", None)]
)
def test_frozen_legacy_arrays(kind, layout, avg):
    base = os.environ.get("DSTPARSER_REGRESSION_BASELINES")
    if not base:
        pytest.skip("Set DSTPARSER_REGRESSION_BASELINES for frozen array checks")
    base = Path(base)
    paths = json.loads((base / "inputs.json").read_text())
    name = f"{kind}_{layout}_{avg}"
    fn = getattr(
        dstparser,
        "parse_dst_file"
        + ("_tax4" if kind == "tax4" else "")
        + ("_vlen" if layout == "vlen" else ""),
    )
    actual = fn(paths[kind], **({"avg_traces": avg} if avg is not None else {}))
    with np.load(base / (name + ".npz")) as expected:
        assert set(actual) == set(expected.files)
        for k in expected:
            assert actual[k].dtype == expected[k].dtype
            np.testing.assert_array_equal(actual[k], expected[k], err_msg=k)


def assert_native_labels(data, source, layout):
    native = list(iter_events(source))
    for i, ordinal in enumerate(data["source_event_ordinal"]):
        mu = native[int(ordinal)].muon
        lookup = {tuple(key): j for j, key in enumerate(mu["keys"])}
        if layout == "vlen":
            h0, h1 = data["hit_offsets"][i : i + 2]
            w0, w1 = data["tt_offsets"][i : i + 2]
            ids = np.repeat(data["detector_ids"][h0:h1], data["nfold"][h0:h1])
            select = [
                lookup[(int(s), int(w))]
                for s, w in zip(ids, data["source_waveform_id"][w0:w1])
            ]
            for k in ["muon_signal", "muon_total_signal", "muon_nsat"]:
                np.testing.assert_array_equal(data[k][w0:w1], mu[k][select])
        else:
            for x, y in np.argwhere(data["muon_available"][i]):
                j = lookup[
                    (
                        int(data["detector_positions_id"][i, x, y]),
                        int(data["source_waveform_id"][i, x, y]),
                    )
                ]
                for k in ["muon_signal", "muon_total_signal", "muon_nsat"]:
                    for layer, suffix in enumerate(["low", "up"]):
                        assert data[k + "_" + suffix][i, x, y] == mu[k][j, layer]


@pytest.mark.parametrize("tag", ["DAT920460", "DAT920950", "DAT921654"])
def test_campaign_preparation_and_hdf5(tag, tmp_path):
    root = os.environ.get("DSTPARSER_CAMPAIGN_ROOT")
    if not root:
        pytest.skip("Set DSTPARSER_CAMPAIGN_ROOT for real reconstruction pilots")
    root = Path(root)
    source = next((root / "chains").glob("*/" + tag + "/outputs/" + tag + ".dst.gz"))
    selected = []
    for e in iter_events(source):
        if len(e.raw["keys"]) or len(selected) < 2:
            selected.append(e)
        if len(selected) == 18:
            break
    pilot = tmp_path / (tag + ".dst.gz")
    write_events(pilot, selected)
    original = sha256_file(pilot)
    cache = tmp_path / "cache"
    prepared = dstparser.prepare_dst(pilot, cache_dir=cache)
    new = list(iter_events(prepared))
    assert len(new) == len(selected)
    for a, b in zip(selected, new):
        assert a.fingerprint == b.fingerprint
        assert a.banks[MUON].payload == b.banks[MUON].payload
    results = {}
    for layout, fn, args in [
        ("grid", dstparser.parse_dst_file, {"avg_traces": False}),
        ("vlen", dstparser.parse_dst_file_vlen, {}),
    ]:
        with pytest.warns(RuntimeWarning, match="BSDINFO"):
            result = fn(pilot, cache_dir=cache, include_muon="require", **args)
        assert len(result["energy"]) > 0
        control = fn(prepared, include_muon=False, **args)
        assert set(result) - set(control) == set(result) & ADDITIONS
        for k in control:
            np.testing.assert_array_equal(result[k], control[k], err_msg=k)
        assert_native_labels(result, pilot, layout)
        results[layout] = result
    config = SimpleNamespace(
        xmax_dir=None,
        add_shower_params=True,
        add_standard_recon=True,
        cache_dir=cache,
        include_muon="require",
    )
    for layout, worker in [("grid", dst_to_hdf5), ("vlen", dst_to_hdf5_vlen)]:
        output = tmp_path / (layout + ".h5")
        worker([pilot], output, config)
        stored = read_h5(output)
        for k in results[layout]:
            expected = results[layout][k]
            if (
                layout == "grid"
                and k not in ADDITIONS
                and np.issubdtype(expected.dtype, np.floating)
            ):
                expected = expected.astype(np.float32)
            np.testing.assert_array_equal(stored[k], expected, err_msg=k)
    assert sha256_file(pilot) == original
