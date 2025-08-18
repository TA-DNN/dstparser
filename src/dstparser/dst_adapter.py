"""
This script adapts the output of the DST parser for use in machine learning models.

Overview of processing steps:
- Converts CORSIKA particle IDs to mass numbers for physics analysis.
- Calculates shower parameters (axis, core) from event data.
- Extracts standard reconstruction data (timing, geometry, energy, etc.).
- Processes detector readings into either a regular grid (for CNNs) or awkward arrays (for variable-length data).
- Normalizes coordinates so that detector positions and shower cores are centered and scaled for ML input.

The main entry point is `parse_dst_file`, which reads a DST file, parses it, and populates a dictionary with all relevant features for ML.
"""

import numpy as np
import awkward as ak
from typing import Union, Tuple, Dict, Any, List, Optional

from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
import dstparser.tasd_clf as tasd_clf

# --- Constants ---
DETECTOR_DISTANCE = 1200  # meters
CLF_ORIGIN_X = 12.2435
CLF_ORIGIN_Y = 16.4406
HEIGHT_OF_CLF = 1370  # meters
HEIGHT_EXTENT = 30  # meters
TO_METERS = 1e-2
TO_NSEC = 4 * 1000
NTIME_TRACE = 128


# --- Helper Functions ---
def corsika_id2mass(corsika_pid: np.ndarray) -> np.ndarray:
    """
    Converts CORSIKA particle ID to mass number.
    - For protons (ID 14), returns 1.
    - For nuclei, divides by 100 to get mass number.
    """
    return np.where(corsika_pid == 14, 1, corsika_pid // 100).astype(np.int32)


def rec_coreposition_to_CLF_meters(core_position_rec, option):
    """
    Converts reconstructed core position to CLF (Central Laser Facility) coordinates in meters.
    - If option is 'x' or 'y': shifts by CLF origin and scales.
    - If option is 'dx' or 'dy': returns error in meters (scales by detector distance).
    """
    detector_dist = 1200  # meters
    clf_origin_x = 12.2435
    clf_origin_y = 16.4406
    if option == "x":
        return detector_dist * (core_position_rec - clf_origin_x)
    elif option == "y":
        return detector_dist * (core_position_rec - clf_origin_y)
    elif option == "dx":
        return detector_dist * core_position_rec
    elif option == "dy":
        return detector_dist * core_position_rec


def _calculate_shower_axis(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """
    Calculates the 3D shower axis vector from zenith and azimuth angles.
    - Converts angles to radians, applies physics convention, and returns (x, y, z) direction.
    """
    zenith_rad = np.deg2rad(zenith + 0.5)
    azimuth_rad = np.deg2rad(azimuth) + np.pi
    return np.array(
        [
            np.sin(zenith_rad) * np.cos(azimuth_rad),
            np.sin(zenith_rad) * np.sin(azimuth_rad),
            np.cos(zenith_rad),
        ],
        dtype=np.float32,
    ).transpose()


def _process_time_traces(
    wform_chunk: np.ndarray,
    fadc_per_vem_low: float,
    fadc_per_vem_up: float,
    avg_traces: bool,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    De-interleaves, calibrates, and optionally averages time traces from waveform data.
    - Splits waveform into 'up' and 'low' traces, calibrates by FADC/VEM factors.
    - If avg_traces: returns average trace, else returns both traces.
    """
    ttrace_up = wform_chunk[1::2] / fadc_per_vem_low
    ttrace_low = wform_chunk[::2] / fadc_per_vem_up
    if avg_traces:
        return (ttrace_low + ttrace_up) / 2
    return ttrace_low, ttrace_up


# --- Data Population Functions ---
def shower_params(
    data: Dict[str, Any], dst_lists: List[np.ndarray], xmax_data: Optional[Any]
) -> None:
    """
    Populates shower-related parameters in the output dictionary.
    - Extracts mass number, energy, shower axis, and core position from event list.
    - Optionally computes Xmax (shower maximum) if a reader is provided.
    """
    event_list = dst_lists[0]
    data["mass_number"] = corsika_id2mass(event_list[0])
    data["energy"] = event_list[1]
    if xmax_data is not None:
        data["xmax"] = xmax_data(data["energy"], data["mass_number"])
    data["shower_axis"] = np.array(
        [
            np.sin(event_list[2]) * np.cos(event_list[3] + np.pi),
            np.sin(event_list[2]) * np.sin(event_list[3] + np.pi),
            np.cos(event_list[2]),
        ],
        dtype=np.float32,
    ).transpose()
    data["shower_core"] = np.array(
        event_list[4:7, :].transpose() * TO_METERS, dtype=np.float32
    )


def standard_recon(data: Dict[str, Any], dst_lists: List[np.ndarray]) -> None:
    """
    Populates standard reconstruction parameters in the output dictionary.
    - Extracts timing, geometry, energy, and fit quality parameters from event list.
    - Computes uncertainties for shower axis directions.
    - Handles both standard and combined reconstructions.
    """
    event_list = dst_lists[0]
    data.update(
        {
            # Basic event timing and counts
            "std_recon_yymmdd": event_list[7],
            "std_recon_hhmmss": event_list[8],
            "std_recon_usec": event_list[11],
            "std_recon_nofwf": event_list[10],
            "std_recon_nsd": event_list[9],
            # Hit and cluster counts
            "std_recon_nsclust": event_list[57],
            "std_recon_nhits": event_list[56],
            "std_recon_nborder": event_list[58],
            # Total charge (Qtot) for two reconstructions
            "std_recon_qtot": np.array([event_list[59], event_list[60]]).transpose(1, 0),
            # Energy and LDF (lateral distribution function) fit parameters
            "std_recon_energy": event_list[12],
            "std_recon_ldf_scale": event_list[13],
            "std_recon_ldf_scale_err": event_list[14],
            "std_recon_ldf_chi2": event_list[15],
            "std_recon_ldf_ndof": event_list[16],
            # Shower core positions and errors (standard and combined)
            "std_recon_shower_core": np.array(
                [
                    rec_coreposition_to_CLF_meters(event_list[17], option="x"),
                    rec_coreposition_to_CLF_meters(event_list[19], option="y"),
                ]
            ).transpose(1, 0),
            "std_recon_shower_core_err": np.array(
                [
                    rec_coreposition_to_CLF_meters(event_list[18], option="dx"),
                    rec_coreposition_to_CLF_meters(event_list[20], option="dy"),
                ]
            ).transpose(1, 0),
            "std_recon_s800": event_list[21],
            "std_recon_combined_energy": event_list[42],
            "std_recon_combined_scale": event_list[43],
            "std_recon_combined_scale_err": event_list[44],
            "std_recon_combined_chi2": event_list[45],
            "std_recon_combined_ndof": event_list[46],
            "std_recon_combined_shower_core": np.array(
                [
                    rec_coreposition_to_CLF_meters(event_list[47], option="x"),
                    rec_coreposition_to_CLF_meters(event_list[49], option="y"),
                ]
            ).transpose(1, 0),
            "std_recon_combined_shower_core_err": np.array(
                [
                    rec_coreposition_to_CLF_meters(event_list[48], option="dx"),
                    rec_coreposition_to_CLF_meters(event_list[50], option="dy"),
                ]
            ).transpose(1, 0),
            "std_recon_combined_s800": event_list[51],
            # Shower axis directions for different reconstructions
            "std_recon_shower_axis": np.array(
                [
                    np.sin(np.deg2rad(event_list[32] + 0.5))
                    * np.cos(np.deg2rad(event_list[33]) + np.pi),
                    np.sin(np.deg2rad(event_list[32] + 0.5))
                    * np.sin(np.deg2rad(event_list[33]) + np.pi),
                    np.cos(np.deg2rad(event_list[32] + 0.5)),
                ],
                dtype=np.float32,
            ).transpose(),
            "std_recon_shower_axis_fixed_curve": np.array(
                [
                    np.sin(np.deg2rad(event_list[22] + 0.5))
                    * np.cos(np.deg2rad(event_list[23]) + np.pi),
                    np.sin(np.deg2rad(event_list[22] + 0.5))
                    * np.sin(np.deg2rad(event_list[23]) + np.pi),
                    np.cos(np.deg2rad(event_list[22] + 0.5)),
                ],
                dtype=np.float32,
            ).transpose(),
            "std_recon_shower_axis_combined": np.array(
                [
                    np.sin(np.deg2rad(event_list[52] + 0.5))
                    * np.cos(np.deg2rad(event_list[53]) + np.pi),
                    np.sin(np.deg2rad(event_list[52] + 0.5))
                    * np.sin(np.deg2rad(event_list[53]) + np.pi),
                    np.cos(np.deg2rad(event_list[52] + 0.5)),
                ],
                dtype=np.float32,
            ).transpose(),
            # Fit quality and geometry
            "std_recon_geom_chi2": event_list[36],
            "std_recon_geom_ndof": event_list[37],
            "std_recon_curvature": event_list[40],
            "std_recon_curvature_err": event_list[41],
            "std_recon_geom_chi2_fixed_curve": event_list[26],
            "std_recon_geom_ndof_fixed_curve": event_list[27],
            "std_recon_border_distance": event_list[30],
            "std_recon_border_distance_tshape": event_list[31],
        }
    )
    # Uncertainties for shower axis directions (standard, fixed curve, combined)
    for key, (zenith_idx, err_idx1, err_idx2) in {
        "std_recon_shower_axis_err": (32, 34, 35),
        "std_recon_shower_axis_err_fixed_curve": (22, 24, 25),
        "std_recon_shower_axis_err_combined": (52, 54, 55),
    }.items():
        data[key] = np.sqrt(
            event_list[err_idx1] ** 2
            + (np.sin(np.deg2rad(event_list[zenith_idx])) * event_list[err_idx2]) ** 2
        )


def cut_events(
    event: np.ndarray,
    wform: np.ndarray,
    max_hits: Union[int, str] = 1,
    max_windows: Union[int, str] = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters and aligns event and waveform data for each event.

    Steps:
      1) Drop coincidence signals (event[1] <= 2)
      2) Optionally keep only the first hit per unique detector (if max_hits == 1)
      3) Align waveform columns to hits *hit-by-hit* using nfolds (robust to duplicate IDs)
      4) Optionally cap windows per hit (if max_windows != 'all'), preserving per-hit order
      5) Trim FADC rows to the number of windows actually kept
    """
    # -- Step 1: drop coincidences
    coincidence_mask = event[1] > 2
    event = event[:, coincidence_mask]
    if event.shape[1] == 0:
        return event, np.empty((wform.shape[0] - 3, 0), dtype=wform.dtype)

    # -- Step 2: optionally keep first hit per unique detector
    if max_hits == 1:
        sdid = event[0]
        _, first_indices = np.unique(sdid, return_index=True)
        event = event[:, first_indices]

    # -- Step 3: per-hit, nfold-aware alignment (handles duplicate detector IDs)
    event_ids = event[0].astype(np.int32)
    nfolds    = event[11].astype(int)

    # keep only waveforms whose det-id appears in event
    wform_ids_full = wform[0, :].astype(np.int32)
    keep_mask = np.isin(wform_ids_full, event_ids)
    wform = wform[:, keep_mask]
    wform_ids = wform[0, :].astype(np.int32)

    # queue of column indices for each det-id (in original wform order)
    id_to_queue: Dict[int, List[int]] = {}
    for j, sid in enumerate(wform_ids):
        id_to_queue.setdefault(sid, []).append(j)

    # build ordered column list hit-by-hit, consuming exactly nfold windows per hit
    ordered_cols: List[int] = []
    for sid, k in zip(event_ids, nfolds):
        q = id_to_queue.get(sid, [])
        take = q[:k]          # take this hit's windows
        ordered_cols.extend(take)
        id_to_queue[sid] = q[k:]  # consume for the next hit of the same id

    wform = wform[:, ordered_cols]

    # -- Step 4: optionally cap windows per hit *after* alignment
    if max_windows != "all":
        cap = int(max_windows)
        starts = np.concatenate(([0], np.cumsum(nfolds[:-1])))
        keep_cols = np.zeros(wform.shape[1], dtype=bool)
        for i, k in enumerate(nfolds):
            m = min(k, cap)
            keep_cols[starts[i] : starts[i] + m] = True
        wform = wform[:, keep_cols]
        max_windows_actually_kept = cap
    else:
        # when keeping all windows, rows should cover the actual max nfold
        max_windows_actually_kept = int(np.max(nfolds)) if event.shape[1] > 0 else 1

    # -- Step 5: trim FADC rows accordingly (keep 3 metadata rows)
    num_fadc_rows = 256 * max_windows_actually_kept
    wform = wform[: 3 + num_fadc_rows, :]

    # drop metadata rows from return value
    return event, wform[3:, :]



def center_tile(
    event: np.ndarray, ntile: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the center of the detector tile for each event and filters hits within the tile.
    - The tile is centered on the detector with the largest signal.
    - Returns the tile center, a mask for hits inside the tile, and their (x, y) indices.
    """
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)
    ixy = np.array([event[0] // 100, event[0] % 100]).astype(np.int32)
    ixy0 = np.copy(ixy[:, max_signal_idx]) - (ntile - 1) // 2
    ixy -= ixy0[:, np.newaxis]
    inside_tile = (ixy[0] < ntile) & (ixy[1] < ntile) & (ixy[0] >= 0) & (ixy[1] >= 0)
    return ixy0, inside_tile, ixy[:, inside_tile]


def tile_normalization(data: Dict[str, Any], ievt: int) -> None:
    """
    Normalizes detector positions and shower core data for each event.
    - Centers detector positions on the tile center and scales to [-1, 1] range.
    - Normalizes shower core and error positions similarly.
    - Ensures all spatial features are on a comparable scale for ML.
    """
    n0 = (data["detector_positions"].shape[1] - 1) // 2
    tile_extent = n0 * DETECTOR_DISTANCE
    tile_center = np.copy(data["detector_positions"][ievt, n0, n0])
    tile_center[2] = HEIGHT_OF_CLF

    dpos = data["detector_positions"][ievt]
    dpos = np.where(
        data["detector_exists"][ievt, ..., np.newaxis],
        dpos - tile_center,
        0,
    )
    dpos[..., :2] /= tile_extent
    dpos[..., 2] /= HEIGHT_EXTENT
    data["detector_positions"][ievt] = dpos

    for key in ["shower_core", "std_recon_shower_core", "std_recon_shower_core_err"]:
        if key in data:
            if "err" in key:
                data[key][ievt][:2] /= tile_extent
            else:
                data[key][ievt][:2] = (data[key][ievt][:2] - tile_center[:2]) / tile_extent
            if key == "shower_core":
                data[key][ievt][2] /= HEIGHT_EXTENT


def tile_positions(
    data: Dict[str, Any], ievt: int, ixy0: np.ndarray, tile_size: int, badsd: np.ndarray
) -> None:
    """
    Creates the centered tile for each event and populates detector positions and states.
    - Looks up detector info from tasd_clf.tasdmc_clf using tile indices.
    - Marks which detectors exist, are good, and are active in the tile.
    - Populates absolute and relative positions, IDs, and status masks.
    """
    x, y = np.mgrid[0:tile_size, 0:tile_size]
    xy_code = (x + ixy0[0]) * 100 + (y + ixy0[1])

    masks = tasd_clf.tasdmc_clf[:, 0][:, np.newaxis, np.newaxis] == xy_code
    do_exist = masks.any(axis=0)
    tasdmc_clf_indices = np.where(do_exist, np.argmax(masks, axis=0), -1)

    good = ~np.isin(tasd_clf.tasdmc_clf[tasdmc_clf_indices, 0], badsd)
    status = good & do_exist

    det_info = tasd_clf.tasdmc_clf[tasdmc_clf_indices]
    data["detector_positions"][ievt] = det_info[:, :, 1:] * TO_METERS
    data["detector_positions_abs"][ievt] = det_info[:, :, 1:] * TO_METERS
    data["detector_positions_id"][ievt] = det_info[:, :, 0]
    data["detector_states"][ievt] = status
    data["detector_exists"][ievt] = do_exist
    data["detector_good"][ievt] = good


def detector_readings_awkward(
    data: Dict[str, Any], dst_lists: List[np.ndarray], avg_traces: bool
) -> List[int]:
    """
    Processes detector readings into awkward arrays (variable-length per event).
    - For each event, filters hits and waveforms using cut_events (all hits, all windows).
    - Extracts per-hit features (ID, nfold, arrival times, signals, traces).
    - Stores results as awkward arrays for flexible ML input.
    - Returns a list of empty event indices (for later removal).
    """
    sdmeta_list, sdwaveform_list, badsdinfo_list = dst_lists[1:4]
    hits_data = {
        "det_id": [],
        "x": [],
        "y": [],
        "z": [],
        "nfold": [],
        "arrival_times": [],
        "total_signals": [],
        "time_traces": [],
        "arrival_times_low": [],
        "total_signals_low": [],
        "time_traces_low": [],
        "arrival_times_up": [],
        "total_signals_up": [],
        "time_traces_up": [],
    }
    hits_counts = []
    per_hit_ttrace_counts = []
    empty_events = []

    for ievt, (event, wform, _) in enumerate(zip(sdmeta_list, sdwaveform_list, badsdinfo_list)):
        event, wform_data = cut_events(event, wform, max_hits="all", max_windows="all")
        if event.shape[1] == 0:
            empty_events.append(ievt)
            hits_counts.append(0)
            continue

        hits_counts.append(event.shape[1])
        hits_data["det_id"].extend(event[0])
        hits_data["x"].extend(event[6])
        hits_data["y"].extend(event[7])
        hits_data["z"].extend(event[8])
        hits_data["nfold"].extend(event[11])

        fadc_per_vem_low = event[9]
        fadc_per_vem_up = event[10]
        nfolds = event[11].astype(int)
        wform_starts = np.concatenate(([0], np.cumsum(nfolds[:-1])))

        if avg_traces:
            hits_data["arrival_times"].extend((event[2] + event[3]) / 2 * TO_NSEC)
            hits_data["total_signals"].extend((event[4] + event[5]) / 2)
        else:
            hits_data["arrival_times_low"].extend(event[2] * TO_NSEC)
            hits_data["arrival_times_up"].extend(event[3] * TO_NSEC)
            hits_data["total_signals_low"].extend(event[4])
            hits_data["total_signals_up"].extend(event[5])

        for i, nfold in enumerate(nfolds):
            start_col = wform_starts[i]
            wform_chunk = wform_data[:, start_col : start_col + nfold].T.flatten()
            traces = _process_time_traces(wform_chunk, fadc_per_vem_low[i], fadc_per_vem_up[i], avg_traces)
            if avg_traces:
                hits_data["time_traces"].extend(traces)
                per_hit_ttrace_counts.append(len(traces))
            else:
                hits_data["time_traces_low"].extend(traces[0])
                hits_data["time_traces_up"].extend(traces[1])
                per_hit_ttrace_counts.append(len(traces[0]))

    # Store as awkward arrays (variable-length per event)
    data["hits_det_id"] = ak.unflatten(np.array(hits_data["det_id"]), hits_counts)
    data["hits_x"] = ak.unflatten(np.array(hits_data["x"]), hits_counts)
    data["hits_y"] = ak.unflatten(np.array(hits_data["y"]), hits_counts)
    data["hits_z"] = ak.unflatten(np.array(hits_data["z"]), hits_counts)
    data["hits_nfold"] = ak.unflatten(np.array(hits_data["nfold"]), hits_counts)

    if avg_traces:
        data["hits_arrival_times"] = ak.unflatten(np.array(hits_data["arrival_times"]), hits_counts)
        data["hits_total_signals"] = ak.unflatten(np.array(hits_data["total_signals"]), hits_counts)
        per_hit_traces = ak.unflatten(hits_data["time_traces"], per_hit_ttrace_counts)
        data["hits_time_traces"] = ak.unflatten(per_hit_traces, hits_counts)
    else:
        for suffix in ["low", "up"]:
            idx = 0 if suffix == "low" else 1
            data[f"hits_arrival_times_{suffix}"] = ak.unflatten(np.array(hits_data[f"arrival_times_{suffix}"]), hits_counts)
            data[f"hits_total_signals_{suffix}"] = ak.unflatten(np.array(hits_data[f"total_signals_{suffix}"]), hits_counts)
            per_hit_traces = ak.unflatten(hits_data[f"time_traces_{suffix}"], per_hit_ttrace_counts)
            data[f"hits_time_traces_{suffix}"] = ak.unflatten(per_hit_traces, hits_counts)

    return empty_events


def detector_readings(
    data: Dict[str, Any],
    dst_lists: List[np.ndarray],
    ntile: int,
    avg_traces: bool,
) -> List[int]:
    """
    Processes detector readings into a regular grid (tile) format for each event.
    - Initializes arrays for detector positions, states, and signals (shape: [events, ntile, ntile]).
    - For each event:
        - Filters hits and waveforms (keeps only first hit and first window per detector).
        - Centers the tile on the detector with the largest signal.
        - Populates detector positions, IDs, and status masks.
        - Normalizes positions and core coordinates.
        - Stores arrival times, signals, and time traces in the grid.
    - Returns a list of empty event indices (for later removal).
    """
    num_events = dst_lists[0][0].shape[0]
    shape = (num_events, ntile, ntile)
    data.update(
        {
            "detector_positions": np.zeros((*shape, 3), dtype=np.float32),
            "detector_positions_abs": np.zeros((*shape, 3), dtype=np.float32),
            "detector_positions_id": np.zeros(shape, dtype=np.float32),
            "detector_states": np.zeros(shape, dtype=bool),
            "detector_exists": np.zeros(shape, dtype=bool),
            "detector_good": np.zeros(shape, dtype=bool),
            "nfold": np.zeros(shape, dtype=np.float32),
        }
    )
    if avg_traces:
        data.update(
            {
                "arrival_times": np.zeros(shape, dtype=np.float32),
                "time_traces": np.zeros((*shape, NTIME_TRACE), dtype=np.float32),
                "total_signals": np.zeros(shape, dtype=np.float32),
            }
        )
    else:
        data.update(
            {
                "arrival_times_low": np.zeros(shape, dtype=np.float32),
                "arrival_times_up": np.zeros(shape, dtype=np.float32),
                "time_traces_low": np.zeros((*shape, NTIME_TRACE), dtype=np.float32),
                "time_traces_up": np.zeros((*shape, NTIME_TRACE), dtype=np.float32),
                "total_signals_low": np.zeros(shape, dtype=np.float32),
                "total_signals_up": np.zeros(shape, dtype=np.float32),
            }
        )

    empty_events = []
    sdmeta_list, sdwaveform_list, badsdinfo_list = dst_lists[1:4]

    for ievt, (event, wform, badsd) in enumerate(zip(sdmeta_list, sdwaveform_list, badsdinfo_list)):
        event, wform_data = cut_events(event, wform, max_hits=1, max_windows=1)
        if event.shape[1] == 0:
            empty_events.append(ievt)
            continue

        ixy0, inside_tile, ixy = center_tile(event, ntile)
        event_in_tile = event[:, inside_tile]
        wform_in_tile = wform_data[:, inside_tile]

        tile_positions(data, ievt, ixy0, ntile, badsd)
        tile_normalization(data, ievt)

        if event_in_tile.shape[1] > 0:
            fadc_per_vem_low = event_in_tile[9]
            fadc_per_vem_up = event_in_tile[10]
            data["nfold"][ievt, ixy[0], ixy[1]] = event_in_tile[11]

            traces = _process_time_traces(wform_in_tile, fadc_per_vem_low, fadc_per_vem_up, avg_traces)

            if avg_traces:
                data["arrival_times"][ievt, ixy[0], ixy[1]] = ((event_in_tile[2] + event_in_tile[3]) / 2 * TO_NSEC)
                data["time_traces"][ievt, ixy[0], ixy[1], :] = traces.transpose()
                data["total_signals"][ievt, ixy[0], ixy[1]] = (event_in_tile[4] + event_in_tile[5]) / 2
            else:
                data["arrival_times_low"][ievt, ixy[0], ixy[1]] = (event_in_tile[2] * TO_NSEC)
                data["arrival_times_up"][ievt, ixy[0], ixy[1]] = (event_in_tile[3] * TO_NSEC)
                data["time_traces_low"][ievt, ixy[0], ixy[1], :] = traces[0].transpose()
                data["time_traces_up"][ievt, ixy[0], ixy[1], :] = traces[1].transpose()
                data["total_signals_low"][ievt, ixy[0], ixy[1]] = event_in_tile[4]
                data["total_signals_up"][ievt, ixy[0], ixy[1]] = event_in_tile[5]

    return empty_events


def parse_dst_file(
    dst_file: str,
    ntile: int = 7,
    xmax_reader: Optional[Any] = None,
    avg_traces: bool = True,
    add_shower_params: bool = True,
    add_standard_recon: bool = True,
    config: Optional[Any] = None,
    use_grid_model: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Main entry point: parses a DST file and returns a dictionary of processed data for ML.

    Steps:
    1. Reads the DST file and parses it into lists of numpy arrays.
    2. Optionally reads Xmax data (if provided).
    3. Populates shower parameters and standard reconstruction features.
    4. Processes detector readings into either a grid (for CNNs) or awkward arrays (for variable-length models).
    5. Removes empty events from all arrays.
    6. Optionally adds event IDs using a config object.

    Returns:
        - Dictionary of processed features, ready for ML input.
    """
    dst_string = read_dst_file(dst_file)
    if not dst_string:
        return None
    dst_lists = parse_dst_string(dst_string)
    if dst_lists is None:
        return None

    if xmax_reader:
        xmax_reader.read_file(dst_file)

    data = {}
    if add_shower_params:
        shower_params(data, dst_lists, xmax_reader)
    if add_standard_recon:
        standard_recon(data, dst_lists)

    if use_grid_model:
        empty_events = detector_readings(data, dst_lists, ntile, avg_traces)
    else:
        empty_events = detector_readings_awkward(data, dst_lists, avg_traces)

    # Remove empty events from all arrays
    if empty_events:
        for key, value in data.items():
            if isinstance(value, ak.Array):
                builder = ak.ArrayBuilder()
                for i, event_data in enumerate(value):
                    if i not in empty_events:
                        builder.append(event_data)
                data[key] = builder.snapshot()
            else:
                data[key] = np.delete(value, empty_events, axis=0)

    if config and hasattr(config, "add_event_ids"):
        config.add_event_ids(data, dst_file)

    return data
