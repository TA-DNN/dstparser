"""
TALE MC DST → vlen HDF5 adapter.

Reads TALE MC DST files using dstio (required — dstparser's sdanalysis dst2k-ta
does not contain the TALE banks).

Banks used:
  rusdmc  — MC truth: energy, theta, phi, corexyz, parttype
  rusdraw — detector response: xxyy IDs, fadc waveforms [nofwf, 2, 128],
             mip calibration constants, clkcnt timing, fadcav pedestals

Output: a dict in the same vlen HDF5 format as parse_dst_file_vlen(), with
offset-encoded hit and time-trace levels. Suitable for direct use with
append_to_hdf5() and the srecog_dataprep pipeline.

Key differences from TASD vlen format:
  - detector_positions: set to zeros — TALE MC has no xyzclf field.
    A TaleGeometry object must supply positions for the grid step downstream.
  - status: all hits set to 4 (TALE has no pattern-recognition quality flag;
    trig_code=2 means hardware-triggered, treated as good).
  - nfold: always 1 per hit (each rusdraw hit has exactly one 128-bin window).
  - std_recon_*: all zeros — TALE reconstruction is not stored in DST.
  - xmax: not available in TALE MC rusdmc bank; set to np.nan.
  - pulse_area: computed as (fadc - pedestal) / mip_counts_per_vem summed over
    the full 128-bin window, per layer. This gives total signal in VEM, matching
    the intent of pulse_area in the TASD vlen format.
  - arrival_times: relative clock count (clkcnt - min(clkcnt)) per event,
    divided by mclkcnt (50_000_000) to give [0,1]-normalised units.
    NOTE: units differ from TASD reltime (counter-sep-dist units). The downstream
    pipeline multiplies by 4000 ns — this will need a separate calibration factor
    for TALE once the clock frequency is confirmed.

Verified fields [2026-06-03]:
  - rusdmc: energy (EeV), theta/phi (rad), corexyz (cm), parttype (CORSIKA ID)
  - rusdraw: nofwf, xxyy, fadc [nofwf,2,128], mip [nofwf,2], fadcav [nofwf,2],
             clkcnt [nofwf], mclkcnt [nofwf]

See: wiki/knowledge/tale.md, tasks/2026/06/03/01_tale_dst_exploration/
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import dstio


def _corsika_id2mass(corsika_pid: np.ndarray) -> np.ndarray:
    return np.where(corsika_pid == 14, 1, corsika_pid // 100).astype(np.int32)


def parse_tale_mc_file(dst_file: str | Path) -> dict | None:
    """Parse one TALE MC DST file into the vlen HDF5 dict format.

    Parameters
    ----------
    dst_file : path to a TALE MC DST file (.dst or .dst.gz).

    Returns
    -------
    dict with the same key set as parse_dst_file_vlen(), or None on failure.
    The dict is suitable for append_to_hdf5() from dstparser.join_vlen_data.
    """
    dst_file = Path(dst_file)
    if not dst_file.exists():
        print(f"File not found: {dst_file}")
        return None

    # Accumulate per-event lists; converted to numpy at the end.
    events_mc = []    # one dict per event from rusdmc
    events_draw = []  # one dict per event from rusdraw

    try:
        with dstio.open(str(dst_file), ["rusdmc", "rusdraw"]) as dst:
            for ev in dst:
                mc   = ev.get("rusdmc")
                draw = ev.get("rusdraw")
                if mc is None or draw is None:
                    continue
                events_mc.append(mc)
                events_draw.append(draw)
    except Exception as e:
        print(f"dstio failed on {dst_file}: {e}")
        return None

    if not events_mc:
        return None

    n_events = len(events_mc)

    # ------------------------------------------------------------------ #
    # Event-level MC truth                                                 #
    # ------------------------------------------------------------------ #
    energy      = np.array([e["energy"]   for e in events_mc], dtype=np.float64)
    theta       = np.array([e["theta"]    for e in events_mc], dtype=np.float64)
    phi         = np.array([e["phi"]      for e in events_mc], dtype=np.float64)
    parttype    = np.array([e["parttype"] for e in events_mc], dtype=np.int32)
    corexyz     = np.array([e["corexyz"]  for e in events_mc], dtype=np.float64)  # [N,3] cm

    mass_number = _corsika_id2mass(parttype)

    shower_axis = np.stack([
        np.sin(theta) * np.cos(phi + np.pi),
        np.sin(theta) * np.sin(phi + np.pi),
        np.cos(theta),
    ], axis=1).astype(np.float64)

    # xmax not available in TALE MC
    xmax = np.full(n_events, np.nan, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Event-level timing (from rusdraw)                                    #
    # ------------------------------------------------------------------ #
    yymmdd = np.array([e["yymmdd"] for e in events_draw], dtype=np.int64)
    hhmmss = np.array([e["hhmmss"] for e in events_draw], dtype=np.int64)
    usec   = np.array([e["usec"]   for e in events_draw], dtype=np.int64)
    nofwf  = np.array([e["nofwf"]  for e in events_draw], dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Hit-level arrays (flat, offset-encoded)                              #
    # ------------------------------------------------------------------ #
    hit_offsets = np.concatenate([[0], np.cumsum(nofwf)]).astype(np.int64)
    total_hits  = int(hit_offsets[-1])

    detector_ids      = np.empty(total_hits, dtype=np.int32)
    arrival_times     = np.empty((total_hits, 2), dtype=np.float32)
    pulse_area        = np.empty((total_hits, 2), dtype=np.float32)
    detector_positions = np.zeros((total_hits, 3), dtype=np.float32)  # no xyzclf in TALE
    status            = np.full(total_hits, 4, dtype=np.int32)        # hardware-triggered = good
    nfold_arr         = np.ones(total_hits, dtype=np.int32)           # one window per hit

    # Time-trace level: one 128-bin window per hit, 2 layers
    # Layout: time_traces[i] = fadc[hit_i] in VEM = (fadc - ped) / mip_count
    # hit_tt_offsets[i] = i (each hit has exactly one waveform window)
    hit_tt_offsets = np.arange(total_hits + 1, dtype=np.int64)
    time_traces    = np.empty((total_hits, 2, 128), dtype=np.float32)

    for ievt, draw in enumerate(events_draw):
        s = int(hit_offsets[ievt])
        e = int(hit_offsets[ievt + 1])
        n = e - s
        if n == 0:
            continue

        xxyy     = np.array(draw["xxyy"],    dtype=np.int32)    # [n]
        clkcnt   = np.array(draw["clkcnt"],  dtype=np.float64)  # [n]
        mclkcnt  = np.array(draw["mclkcnt"], dtype=np.float64)  # [n] max clock (~50M)
        fadcav   = np.array(draw["fadcav"],  dtype=np.float32)  # [n,2] pedestal counts
        mip_cnts = np.array(draw["pchmip"],  dtype=np.float32)  # [n,2] counts per MIP
        fadc_raw = draw["fadc"]                                  # list[n] of list[2] of tuple[128]

        detector_ids[s:e] = xxyy

        # Arrival times: relative clock count normalised by max clock.
        # Units: fraction of 50MHz clock period (~20ns steps, max ~1 second).
        clk_rel = clkcnt - clkcnt.min()
        # Divide by mclkcnt so values are in [0, 1]. Both layers share the same
        # clock, so repeat for upper layer.
        arr_t = (clk_rel / mclkcnt).astype(np.float32)
        arrival_times[s:e, 0] = arr_t   # lower layer
        arrival_times[s:e, 1] = arr_t   # upper layer (same clock)

        # Time traces: convert FADC counts to VEM.
        # VEM = (fadc_bin - pedestal) / mip_counts_per_vem, clipped to >= 0.
        for i in range(n):
            for layer in range(2):
                raw  = np.array(fadc_raw[i][layer], dtype=np.float32)
                ped  = fadcav[i, layer]
                mip  = mip_cnts[i, layer]
                if mip > 0:
                    time_traces[s + i, layer] = np.clip(
                        (raw - ped) / mip, 0.0, None
                    )
                else:
                    time_traces[s + i, layer] = 0.0

        # Pulse area: sum of the pedestal-subtracted waveform in VEM (same as
        # waveform_sum total signal used in legacy TASD pipeline).
        pulse_area[s:e, 0] = time_traces[s:e, 0].sum(axis=1)
        pulse_area[s:e, 1] = time_traces[s:e, 1].sum(axis=1)

    # Reshape time_traces to match vlen format: [total_hits, 2, 128] → stored as [total_hits, 128, 2]?
    # Check: TASD vlen stores as [ntt, 2, 128] (layer first, then bins).
    # dst_adapter_vlen.py line 469: data["time_traces"] = (waveforms["rusdraw_.fadc"] / vem)
    # shape is [ntt, 2, 128] where axis 1 = layer.
    # Our time_traces is already [total_hits, 2, 128] — correct.

    # ------------------------------------------------------------------ #
    # Assemble output dict                                                  #
    # ------------------------------------------------------------------ #
    data: dict = {}

    # MC truth
    data["energy"]      = energy
    data["xmax"]        = xmax
    data["mass_number"] = mass_number
    data["shower_axis"] = shower_axis
    data["shower_core"] = corexyz  # [N,3] cm — same convention as TASD

    # Hit-level
    data["hit_offsets"]        = hit_offsets
    data["detector_ids"]       = detector_ids
    data["arrival_times"]      = arrival_times
    data["pulse_area"]         = pulse_area
    data["detector_positions"] = detector_positions  # zeros — geometry supplied externally
    data["status"]             = status
    data["nfold"]              = nfold_arr

    # Time-trace level
    data["hit_tt_offsets"] = hit_tt_offsets
    data["time_traces"]    = time_traces  # [total_hits, 2, 128]

    # std_recon fields — all zeros/sentinel (no reconstruction in TALE DST)
    data["std_recon_yymmdd"] = yymmdd
    data["std_recon_hhmmss"] = hhmmss
    data["std_recon_usec"]   = usec
    data["std_recon_nofwf"]  = nofwf
    # Pattern-reco and LDF/geom fields: zeros (not present in TALE)
    zeros_i = np.zeros(n_events, dtype=np.int32)
    zeros_f = np.zeros(n_events, dtype=np.float64)
    data["std_recon_nsd"]           = zeros_i
    data["std_recon_nsclust"]       = zeros_i
    data["std_recon_nhits"]         = nofwf   # best available proxy
    data["std_recon_nborder"]       = zeros_i
    data["std_recon_qtot"]          = np.zeros((n_events, 2), dtype=np.float64)
    data["std_recon_energy"]        = zeros_f
    data["std_recon_ldf_scale"]     = zeros_f
    data["std_recon_ldf_scale_err"] = zeros_f
    data["std_recon_ldf_chi2"]      = zeros_f
    data["std_recon_ldf_ndof"]      = zeros_i
    data["std_recon_shower_core"]   = np.zeros((n_events, 2), dtype=np.float64)
    data["std_recon_shower_core_err"] = np.zeros((n_events, 2), dtype=np.float64)
    data["std_recon_s800"]          = zeros_f
    data["std_recon_shower_axis"]   = np.zeros((n_events, 3), dtype=np.float64)
    data["std_recon_shower_axis_err"] = zeros_f
    data["std_recon_geom_chi2"]     = zeros_f
    data["std_recon_geom_ndof"]     = zeros_i
    data["std_recon_curvature"]     = zeros_f
    data["std_recon_curvature_err"] = zeros_f
    data["std_recon_border_distance"] = zeros_f
    data["std_recon_border_distance_tshape"] = zeros_f

    return data
