"""Native signed MUONEXTR labels, keyed and aligned before any hit selection."""

import warnings
import numpy as np
from .banks import RAW, BSD, MC, DSTError, iter_events

FIELDS = (
    "muon_signal",
    "muon_total_signal",
    "muon_nsat",
    "muon_available",
    "source_waveform_id",
)


def enabled(option, inspection):
    if option not in ("auto", "require", True, False):
        raise ValueError("include_muon must be auto, require, True or False")
    if option == "require" and inspection.muon_events != inspection.events:
        raise DSTError("Muon labels required but a MUONEXTR bank is absent")
    return (
        option is True
        or option == "require"
        or (option == "auto" and inspection.muon_events > 0)
    )


def align(path, waveforms, *, vlen=False):
    """Prove the legacy reader's event/waveform sequence against native banks.

    Event occurrences, not random event numbers alone, establish identity.
    IDs, clocks AND every FADC sample must match. Labels use lower/upper order.
    """
    result = []
    ordinals = []
    event_ids = []
    status = []
    index = 0
    for e in iter_events(path):
        if RAW not in e.banks:
            continue
        raw = e.raw
        n = len(raw["keys"])
        if not n:
            continue
        if MC not in e.banks and raw["event_code"] != 0:
            raise DSTError(
                "Muon alignment for observation/hybrid event selection is not supported"
            )
        if vlen:
            off = waveforms["offsets"]
            if index + 1 >= len(off):
                raise DSTError("Legacy reader dropped a waveform-bearing event")
            lo, hi = off[index : index + 2]
            ids = waveforms["rusdraw_.xxyy"][lo:hi]
            clocks = np.column_stack(
                (
                    waveforms["rusdraw_.clkcnt"][lo:hi],
                    waveforms["rusdraw_.mclkcnt"][lo:hi],
                )
            )
            traces = waveforms["rusdraw_.fadc"][lo:hi]
        else:
            if index >= len(waveforms):
                raise DSTError("Legacy reader dropped an event")
            w = waveforms[index]
            ids = w[0]
            clocks = w[1:3].T
            traces = w[3:].T.reshape(-1, 2, 128)
        if not (
            np.array_equal(ids, raw["keys"][:, 0])
            and np.array_equal(clocks, raw["clocks"])
            and np.array_equal(traces, raw["traces"])
        ):
            raise DSTError(
                f"Legacy/native waveform mismatch at source event {e.ordinal}"
            )
        mu = e.muon
        if mu is None:
            mu = {
                "muon_signal": np.full((n, 2), np.nan, dtype=np.float32),
                "muon_total_signal": np.full((n, 2), np.nan, dtype=np.float32),
                "muon_nsat": np.full((n, 2), -1, dtype=np.int32),
                "muon_available": np.zeros(n, dtype=bool),
            }
        result.append(
            {
                **{k: mu[k] for k in FIELDS[:-1]},
                "source_waveform_id": raw["keys"][:, 1].copy(),
            }
        )
        ordinals.append(e.ordinal)
        event_ids.append(e.event_id)
        status.append(BSD in e.banks)
        index += 1
    expected = len(waveforms["offsets"]) - 1 if vlen else len(waveforms)
    if index != expected:
        raise DSTError("Legacy/native event counts differ")
    metadata = {
        "source_event_ordinal": np.asarray(ordinals, dtype=np.int64),
        "source_event_number": np.asarray(event_ids, dtype=np.int64),
        "detector_status_available": np.asarray(status, dtype=bool),
    }
    if status and not all(status):
        warnings.warn(
            "BSDINFO absent for some events: detector health is unknown where detector_status_available=False; existing state fields are not a live-detector census.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not vlen:
        return result, metadata
    arrays = {
        k: np.concatenate([r[k] for r in result], axis=0)
        if result
        else np.empty(
            (0, 2) if k in FIELDS[:3] else (0,),
            dtype=np.float32
            if k in FIELDS[:2]
            else bool
            if k == "muon_available"
            else np.int32,
        )
        for k in FIELDS
    }
    return arrays, metadata


def grid_allocate(data, shape, avg):
    keys = (
        ["muon_signal", "muon_total_signal"]
        if avg
        else [
            "muon_signal_low",
            "muon_signal_up",
            "muon_total_signal_low",
            "muon_total_signal_up",
        ]
    )
    for k in keys:
        data[k] = np.full(shape, np.nan, dtype=np.float32)
    for k in ["muon_nsat_low", "muon_nsat_up", "source_waveform_id"]:
        data[k] = np.full(shape, -1, dtype=np.int32)
    data["muon_available"] = np.zeros(shape, dtype=bool)


def grid_fill(data, event, xy, labels, selected, avg):
    for key in FIELDS[:2]:
        values = labels[key][selected]
        if avg:
            data[key][event, xy[0], xy[1]] = values.mean(axis=1)
        else:
            for layer, suffix in enumerate(["low", "up"]):
                data[key + "_" + suffix][event, xy[0], xy[1]] = values[:, layer]
    for layer, suffix in enumerate(["low", "up"]):
        data["muon_nsat_" + suffix][event, xy[0], xy[1]] = labels["muon_nsat"][
            selected, layer
        ]
    for key in ["muon_available", "source_waveform_id"]:
        data[key][event, xy[0], xy[1]] = labels[key][selected]
