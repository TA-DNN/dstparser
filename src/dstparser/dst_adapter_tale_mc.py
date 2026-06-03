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
    divided by 200 to give TASD-compatible reltime units (1 unit = 4000 ns).
    Conversion: clkcnt is a 50 MHz clock (20 ns/tick); 4000 ns / 20 ns = 200 ticks
    per TASD unit. The downstream pipeline multiplies by 4000 ns → nanoseconds.
    Verified: max spread ~63 µs across TALE array, consistent with TALE geometry.

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

    # ------------------------------------------------------------------ #
    # Fast path: single C pass via dstio.tale.fast_read_tale_mc           #
    # ------------------------------------------------------------------ #
    try:
        raw = dstio.tale.fast_read_tale_mc(str(dst_file))
    except Exception as e:
        print(f"fast_read_tale_mc failed on {dst_file}: {e}")
        return None

    if raw is None or len(raw.get("energy", [])) == 0:
        return None

    n_events = len(raw["energy"])
    nofwf    = raw["nofwf"]                            # [N] int32

    # ------------------------------------------------------------------ #
    # Event-level MC truth                                                 #
    # ------------------------------------------------------------------ #
    energy   = raw["energy"].astype(np.float64)
    theta    = raw["theta"].astype(np.float64)
    phi      = raw["phi"].astype(np.float64)
    parttype = raw["parttype"]                         # int32
    corexyz  = raw["corexyz"].astype(np.float64)       # [N,3] cm

    mass_number = _corsika_id2mass(parttype)

    shower_axis = np.stack([
        np.sin(theta) * np.cos(phi + np.pi),
        np.sin(theta) * np.sin(phi + np.pi),
        np.cos(theta),
    ], axis=1).astype(np.float64)

    xmax = np.full(n_events, np.nan, dtype=np.float64)  # not in TALE MC

    yymmdd = raw["yymmdd"].astype(np.int64)
    hhmmss = raw["hhmmss"].astype(np.int64)
    usec   = raw["usec"].astype(np.int64)

    # ------------------------------------------------------------------ #
    # Hit-level arrays (flat, offset-encoded)                              #
    # ------------------------------------------------------------------ #
    hit_offsets = np.concatenate([[0], np.cumsum(nofwf)]).astype(np.int64)
    total_hits  = int(hit_offsets[-1])

    detector_ids       = raw["xxyy"]                  # [H] int32
    detector_positions = np.zeros((total_hits, 3), dtype=np.float32)  # no xyzclf
    status             = np.full(total_hits, 4, dtype=np.int32)
    nfold_arr          = np.ones(total_hits, dtype=np.int32)

    # Arrival times — vectorised over all hits at once.
    # clkcnt is a 50 MHz DAQ clock (1 tick = 20 ns).
    # TASD reltime unit = 1 counter-separation distance = 4000 ns = 200 ticks.
    # Convert: ticks / 200 → TASD-compatible reltime units.
    # The downstream pipeline multiplies by 4000 ns to get nanoseconds,
    # so this gives physically correct timing (max ~60 µs across TALE array).
    clkcnt = raw["clkcnt"].astype(np.float64)                  # [H]
    clk_min = np.minimum.reduceat(clkcnt, hit_offsets[:-1])    # [N] per-event min
    clk_min_per_hit = np.repeat(clk_min, nofwf)                # [H]
    arr_t = ((clkcnt - clk_min_per_hit) / 200.0).astype(np.float32)
    arrival_times = np.stack([arr_t, arr_t], axis=1)           # [H, 2]

    # Time traces — FADC to VEM, fully vectorised over all hits
    # raw["fadc"]  = [H, 2, 128] int32
    # raw["fadcav"] = [H, 2]    int32  pedestal
    # raw["pchmip"] = [H, 2]    int32  MIP peak channel
    fadc_f   = raw["fadc"].astype(np.float32)                  # [H, 2, 128]
    ped      = raw["fadcav"].astype(np.float32)[:, :, np.newaxis]  # [H, 2, 1]
    mip_cnts = raw["pchmip"].astype(np.float32)[:, :, np.newaxis]  # [H, 2, 1]
    safe_mip = np.where(mip_cnts > 0, mip_cnts, 1.0)
    time_traces = np.clip((fadc_f - ped) / safe_mip, 0.0, None)
    time_traces[mip_cnts[:, :, 0] == 0] = 0.0                 # [H, 2, 128]

    # Pulse area: waveform sum in VEM
    pulse_area = time_traces.sum(axis=2).astype(np.float32)    # [H, 2]

    hit_tt_offsets = np.arange(total_hits + 1, dtype=np.int64)

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
