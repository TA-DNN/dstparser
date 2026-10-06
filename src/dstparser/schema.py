"""Array levels and additive missing-label representation for HDF5."""

import numpy as np

EVENT_EXTRA = (
    "source_event_ordinal",
    "source_event_number",
    "source_file_sha256",
    "detector_status_available",
)
HITS = {
    "arrival_times",
    "pulse_area",
    "detector_positions",
    "status",
    "nfold",
    "detector_ids",
}
WAVES = {
    "time_traces",
    "muon_signal",
    "muon_total_signal",
    "muon_nsat",
    "muon_available",
    "source_waveform_id",
}
GRID_MU = (
    "muon_signal",
    "muon_total_signal",
    "muon_signal_low",
    "muon_signal_up",
    "muon_total_signal_low",
    "muon_total_signal_up",
    "muon_nsat_low",
    "muon_nsat_up",
    "muon_available",
    "source_waveform_id",
)
ADDITIONS = set(GRID_MU) | set(EVENT_EXTRA) | {"muon_nsat"}


def layout(data):
    return "vlen" if "hit_offsets" in data else "grid"


def level(key, kind):
    if kind == "grid":
        return "event"
    if key in WAVES:
        return "waveform"
    if key in HITS:
        return "hit"
    if key in ("hit_offsets", "tt_offsets", "hit_tt_offsets", "badsd_offsets"):
        return "offset"
    if key == "badsd":
        return "badsd"
    return "event"


def has_muon(data):
    return any(k.startswith("muon_") for k in data)


def promote(data):
    """Return arrays with an explicit unavailable muon row for every missing label."""
    out = dict(data)
    if layout(out) == "vlen":
        n = len(out["hit_offsets"]) - 1
        w = len(out["time_traces"])
        out.setdefault("muon_signal", np.full((w, 2), np.nan, dtype=np.float32))
        out.setdefault("muon_total_signal", np.full((w, 2), np.nan, dtype=np.float32))
        out.setdefault("muon_nsat", np.full((w, 2), -1, dtype=np.int32))
        shape = (w,)
    else:
        shape = out["detector_states"].shape
        n = shape[0]
        keys = (
            ["muon_signal", "muon_total_signal"]
            if "time_traces" in out or "total_signals" in out
            else [
                "muon_signal_low",
                "muon_signal_up",
                "muon_total_signal_low",
                "muon_total_signal_up",
            ]
        )
        for k in keys:
            out.setdefault(k, np.full(shape, np.nan, dtype=np.float32))
        for k in ["muon_nsat_low", "muon_nsat_up"]:
            out.setdefault(k, np.full(shape, -1, dtype=np.int32))
    out.setdefault("muon_available", np.zeros(shape, dtype=bool))
    out.setdefault("source_waveform_id", np.full(shape, -1, dtype=np.int32))
    for k in EVENT_EXTRA[:2]:
        out.setdefault(k, np.full(n, -1, dtype=np.int64))
    out.setdefault("source_file_sha256", np.full(n, b"", dtype="S64"))
    out.setdefault("detector_status_available", np.zeros(n, dtype=bool))
    return out


def validate(data):
    if layout(data) == "vlen":
        sizes = {
            "hit_offsets": len(data["arrival_times"]),
            "tt_offsets": len(data["time_traces"]),
            "hit_tt_offsets": len(data["time_traces"]),
        }
        if "badsd_offsets" in data:
            sizes["badsd_offsets"] = len(data["badsd"])
        for k, end in sizes.items():
            o = np.asarray(data[k])
            if (
                o.ndim != 1
                or not np.issubdtype(o.dtype, np.integer)
                or not len(o)
                or o[0] != 0
                or np.any(np.diff(o) < 0)
                or o[-1] != end
            ):
                raise ValueError(f"Invalid {k}")
        n = len(data["hit_offsets"]) - 1
        nh = len(data["arrival_times"])
        nw = len(data["time_traces"])
        if (
            len(data["tt_offsets"]) != n + 1
            or len(data["hit_tt_offsets"]) != nh + 1
            or not np.array_equal(np.diff(data["hit_tt_offsets"]), data["nfold"])
        ):
            raise ValueError("Hit/waveform offset association mismatch")
        if not np.array_equal(
            data["hit_tt_offsets"][data["hit_offsets"]], data["tt_offsets"]
        ):
            raise ValueError("Cross-event hit/waveform offset mismatch")
        if "badsd_offsets" in data and len(data["badsd_offsets"]) != n + 1:
            raise ValueError("Bad-detector offsets mismatch")
        sizes = {
            "event": n,
            "hit": nh,
            "waveform": nw,
            "badsd": len(data.get("badsd", [])),
        }
        for k, a in data.items():
            lev = level(k, "vlen")
            if lev != "offset" and len(a) != sizes[lev]:
                raise ValueError(f"Wrong {lev} row count for {k}")
    if has_muon(data):
        mask = np.asarray(data["muon_available"], dtype=bool)
        if layout(data) == "grid":
            shape = data["detector_states"].shape
            if mask.shape != shape:
                raise ValueError("Grid muon availability shape mismatch")
            for key in GRID_MU:
                if key in data and np.shape(data[key]) != shape:
                    raise ValueError(f"Grid label shape mismatch for {key}")
            for key in EVENT_EXTRA:
                if key in data and len(data[key]) != shape[0]:
                    raise ValueError(f"Event metadata shape mismatch for {key}")
        for k, a in data.items():
            if k.startswith("muon_signal") or k.startswith("muon_total_signal"):
                if not np.all(np.isfinite(np.asarray(a)[mask])):
                    raise ValueError("Nonfinite available muon signal")
            if k.startswith("muon_nsat"):
                s = np.asarray(a)[mask]
                if np.any(s < 0) or np.any(s > 128):
                    raise ValueError("Invalid available saturation count")


def annotate(h5, data):
    kind = layout(data)
    h5.attrs["dstparser_schema_version"] = 2
    h5.attrs["dstparser_layout"] = kind
    for k in data:
        obj = h5[k]
        obj.attrs["level"] = level(k, kind)
        if k.startswith("muon_signal") or k.startswith("muon_total_signal"):
            obj.attrs["units"] = "VEM"
            obj.attrs["semantics"] = (
                "signed pulse-start response proxy, not waveform integral"
            )
        if kind == "vlen" and k in (
            "muon_signal",
            "muon_total_signal",
            "muon_nsat",
            "time_traces",
        ):
            obj.attrs["layer_order"] = "lower,upper"
        if k == "id_energy_bin" and "shower_log10_energy" in data:
            obj.attrs["convention"] = "round(10 * log10(shower energy / eV))"
        if k == "id_campaign_model":
            obj.attrs["codes"] = "0=qgsjetii04,1=eposlhc"


def migration_kwargs(config):
    names = (
        "include_muon",
        "reconstruction",
        "cache_dir",
        "reconstruction_commands",
        "reconstruction_dependencies",
        "reconstruction_timeout",
        "campaign_manifest",
    )
    return {k: getattr(config, k) for k in names if hasattr(config, k)}
