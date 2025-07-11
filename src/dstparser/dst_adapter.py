import numpy as np
import awkward as ak
from typing import Union, Tuple
from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
import dstparser.tasd_clf as tasd_clf
import re
from pathlib import Path


def corsika_id2mass(corsika_pid):
    return np.where(corsika_pid == 14, 1, corsika_pid // 100).astype(np.int32)


def rec_coreposition_to_CLF_meters(core_position_rec, option):
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


def shower_params(data, dst_lists, xmax_data):
    # Shower related
    # for details: /ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/sditerator/src/sditerator_cppanalysis.cpp
    to_meters = 1e-2
    event_list = dst_lists[0]
    data["mass_number"] = corsika_id2mass(event_list[0])
    data["energy"] = event_list[1]

    if xmax_data is not None:
        data["xmax"] = xmax_data(data["energy"])

    data["shower_axis"] = np.array(
        [
            np.sin(event_list[2]) * np.cos(event_list[3] + np.pi),
            np.sin(event_list[2]) * np.sin(event_list[3] + np.pi),
            np.cos(event_list[2]),
        ],
        dtype=np.float32,
    ).transpose()

    data["shower_core"] = np.array(
        event_list[4:7, :].transpose() * to_meters, dtype=np.float32
    )
    return data


def standard_recon(data, dst_lists):
    event_list = dst_lists[0]
    # Exempt from comments of cpp source code at:
    # /ceph/work/SATORI/projects/TA-ASIoP/benMC/sdanalysis_2019/sdmc/sdmc_spctr.c
    # // Reported by DAQ as time of the 1st signal in the triple that caused the triggger.
    # // From now on, everyhting is relative to hhmmss.  Not useful in the event reconstruction.
    # Date of event
    # rusdraw_.yymmdd = 80916; // Event date year = 08, month = 09, day = 16
    data["std_recon_yymmdd"] = event_list[7]
    # Time of event
    # rusdraw_.hhmmss = 1354;  // Event time, hour=00, minute=13, second = 54
    data["std_recon_hhmmss"] = event_list[8]
    # Microseconds for the second
    # rusdraw_.usec = 111111
    data["std_recon_usec"] = event_list[11]
    # Number of waveforms for event for all detectors
    data["std_recon_nofwf"] = event_list[10]
    # number of SDs in space-time cluster
    data["std_recon_nsd"] = event_list[9]
    # number of SDs in space cluster
    data["std_recon_nsclust"] = event_list[57]
    # number of hit SDs
    data["std_recon_nhits"] = event_list[56]
    # number of SDs in space-time cluster & lie on the border of the array
    data["std_recon_nborder"] = event_list[58]
    # total charge [VEM] of SDs in the space-time cluster, (lower & upper)
    data["std_recon_qtot"] = np.array(
        [
            event_list[59],
            event_list[60],
        ]
    ).transpose(1, 0)
    # energy reconstructed by the standard energy estimation table [EeV]
    data["std_recon_energy"] = event_list[12]
    # reconstructed scale of the Lateral Distribution Function (LDF) fit [VEM m-2]
    data["std_recon_ldf_scale"] = event_list[13]
    # uncertainty of the scale [VEM m-2]
    data["std_recon_ldf_scale_err"] = event_list[14]
    # chi-square of the LDF fit
    data["std_recon_ldf_chi2"] = event_list[15]
    # the number of degree of freedom of the LDF fit (= n - 3),
    # where "n" is the number of the SDs used for the LDF fit
    data["std_recon_ldf_ndof"] = event_list[16]
    # core position (x, y) reconstructed by the LDF fit in CLF coordinate [m]
    data["std_recon_shower_core"] = np.array(
        [
            rec_coreposition_to_CLF_meters(event_list[17], option="x"),
            rec_coreposition_to_CLF_meters(event_list[19], option="y"),
        ]
    ).transpose(1, 0)

    # uncertainty of the core position (x, y) reconstructed by the LDF fit
    data["std_recon_shower_core_err"] = np.array(
        [
            rec_coreposition_to_CLF_meters(event_list[18], option="dx"),
            rec_coreposition_to_CLF_meters(event_list[20], option="dy"),
        ]
    ).transpose(1, 0)
    # S800 (particle density at 800 m from the shower axis) [VEM m-2]
    data["std_recon_s800"] = event_list[21]

    # reconstructed values of the geometry+LDF (combined) fit
    data["std_recon_combined_energy"] = event_list[42]
    data["std_recon_combined_scale"] = event_list[43]
    data["std_recon_combined_scale_err"] = event_list[44]
    data["std_recon_combined_chi2"] = event_list[45]
    # the number of degree of freedom of the LDF fit (= 2*n - 6),
    # where "n" is the number of the SDs used for the LDF fit
    data["std_recon_combined_ndof"] = event_list[46]
    data["std_recon_combined_shower_core"] = np.array(
        [
            rec_coreposition_to_CLF_meters(event_list[47], option="x"),
            rec_coreposition_to_CLF_meters(event_list[49], option="y"),
        ]
    ).transpose(1, 0)
    data["std_recon_combined_shower_core_err"] = np.array(
        [
            rec_coreposition_to_CLF_meters(event_list[48], option="dx"),
            rec_coreposition_to_CLF_meters(event_list[50], option="dy"),
        ]
    ).transpose(1, 0)
    data["std_recon_combined_s800"] = event_list[51]

    # 3-d unit vector of the arrival direction (pointing back to the source)
    # geometry fit with a free curved parameter.
    # "+0.5" is a correction for zenith angle.
    data["std_recon_shower_axis"] = np.array(
        [
            np.sin(np.deg2rad(event_list[32] + 0.5))
            * np.cos(np.deg2rad(event_list[33]) + np.pi),
            np.sin(np.deg2rad(event_list[32] + 0.5))
            * np.sin(np.deg2rad(event_list[33]) + np.pi),
            np.cos(np.deg2rad(event_list[32] + 0.5)),
        ],
        dtype=np.float32,
    ).transpose()
    # 3-d unit vector of the arrival direction (pointing back to the source)
    # geometry fit with a fixed curved parameter
    # "+0.5" is a correction for zenith angle.
    data["std_recon_shower_axis_fixed_curve"] = np.array(
        [
            np.sin(np.deg2rad(event_list[22] + 0.5))
            * np.cos(np.deg2rad(event_list[23]) + np.pi),
            np.sin(np.deg2rad(event_list[22] + 0.5))
            * np.sin(np.deg2rad(event_list[23]) + np.pi),
            np.cos(np.deg2rad(event_list[22] + 0.5)),
        ],
        dtype=np.float32,
    ).transpose()
    # 3-d unit vector of the arrival direction (pointing back to the source)
    # geometry+LDF fit
    # "+0.5" is a correction for zenith angle.
    data["std_recon_shower_axis_combined"] = np.array(
        [
            np.sin(np.deg2rad(event_list[52] + 0.5))
            * np.cos(np.deg2rad(event_list[53]) + np.pi),
            np.sin(np.deg2rad(event_list[52] + 0.5))
            * np.sin(np.deg2rad(event_list[53]) + np.pi),
            np.cos(np.deg2rad(event_list[52] + 0.5)),
        ],
        dtype=np.float32,
    ).transpose()
    # uncertainty of the pointing direction [degree]
    # free curved parameter
    # event_list[22] is zenith angle in deg
    # event_list[24] is uncertainty zenith angle in deg
    # event_list[25] is uncertainty azimuth angle in deg
    data["std_recon_shower_axis_err"] = np.sqrt(
        event_list[34] * event_list[34]
        + np.sin(np.deg2rad(event_list[32]))
        * np.sin(np.deg2rad(event_list[32]))
        * event_list[35]
        * event_list[35]
    )
    # uncertainty of the pointing direction [degree]
    # fixed curved parameter
    data["std_recon_shower_axis_err_fixed_curve"] = np.sqrt(
        event_list[24] * event_list[24]
        + np.sin(np.deg2rad(event_list[22]))
        * np.sin(np.deg2rad(event_list[22]))
        * event_list[25]
        * event_list[25]
    )
    # uncertainty of the pointing direction [degree]
    # geometry+LDF fit
    data["std_recon_shower_axis_err_combined"] = np.sqrt(
        event_list[54] * event_list[54]
        + np.sin(np.deg2rad(event_list[52]))
        * np.sin(np.deg2rad(event_list[52]))
        * event_list[55]
        * event_list[55]
    )
    # chi-square of the geometry fit (free curvature)
    data["std_recon_geom_chi2"] = event_list[36]
    # the number of degree of freedom of the geometry fit (= n - 6),
    # where "n" is the number of the SDs used for the geometry fit
    data["std_recon_geom_ndof"] = event_list[37]
    # curvature paramter `a` of the geometry fit
    data["std_recon_curvature"] = event_list[40]
    # uncertainty of the curvature paramter `a` of the geometry fit
    data["std_recon_curvature_err"] = event_list[41]
    # chi-square of the geometry fit (fixed curvature)
    data["std_recon_geom_chi2_fixed_curve"] = event_list[26]
    # the number of degree of freedom of the geometry fit (= n - 5),
    # where "n" is the number of the SDs used for the geometry fit
    data["std_recon_geom_ndof_fixed_curve"] = event_list[27]
    # distance b/w the reconstructed core and the edge from the TA SD array [in 1,200 meter unit]
    # negative for events with the core outside of the TA SD array.
    data["std_recon_border_distance"] = event_list[30]
    # distance to the T-shape TA SD array, edge of the sub-arrays [in 1,200 meter unit]
    # this value is used as "border_distance" before implementation of the boundary trigger (on 2008/11/11)
    data["std_recon_border_distance_tshape"] = event_list[31]

    return data


def cut_events(
    event: np.ndarray,
    wform: np.ndarray,
    max_hits: Union[int, str] = 1,
    max_windows: Union[int, str] = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    event:       shape (11, n_hits)
    wform:       shape (>=3+128*n_windows, n_hits)
    max_hits:     1 (default) or 'all'
    max_windows:  1 (default), 2, 3, 4 or 'all'
    """
    # Step 1: Filter hits based on max_hits
    if max_hits == 1:
        # Keep only the first hit for each detector
        sdid = event[0]
        u, c = np.unique(sdid, return_counts=True)
        dup = u[c > 1]
        mask = np.ones(sdid.shape, dtype=bool)
        for el in dup:
            mask[np.where(sdid == el)[0][1:]] = False
        event = event[:, mask]

    # Exclude coincidence signals (where event[1] <= 2)
    event = event[:, event[1] > 2]

    # Step 2: Select corresponding waveforms
    wform_idx = []
    wform_xxyy = wform[0, :].astype(np.int32)
    for xycoord in event[0].astype(np.int32):
        # Take only the first waveform for a given detector ID
        idx = np.where(wform_xxyy == xycoord)[0]
        if len(idx) > 0:
            wform_idx.append(idx[0])
    
    wform = wform[:, wform_idx]

    # Step 3: Filter windows based on max_windows
    if max_windows != "all":
        nfolds = event[11].astype(int)
        windows_to_keep = np.minimum(nfolds, int(max_windows))

        max_windows_to_keep = np.max(windows_to_keep) if len(windows_to_keep) > 0 else 0
        num_fadc_rows = 128 * max_windows_to_keep
        
        # Keep metadata rows and FADC data
        wform = wform[: 3 + num_fadc_rows, :]

    return event, wform[3:, :]


def center_tile(event, ntile):
    # Put largest-signal SD at the center of ntile x ntile grids
    # center around detector with max signal
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)

    # ix and iy as one array [ix, iy]
    ixy = np.array([event[0] // 100, event[0] % 100]).astype(np.int32)
    # Indicies of central detector ix0, iy0
    ixy0 = np.copy(ixy[:, max_signal_idx]) - (ntile - 1) // 2
    ixy -= ixy0[:, np.newaxis]
    # cut array size to fit the tile size
    inside_tile = (ixy[0] < ntile) & (ixy[1] < ntile) & (ixy[0] >= 0) & (ixy[1] >= 0)
    ixy = ixy[:, inside_tile]
    return ixy0, inside_tile, ixy


def tile_normalization(data, ievt):
    detector_dist = 1200  # meters
    height_of_clf = 1370  # meters
    height_extent = 30  # meters, height scatter +-30 from average, z-coordinate norm

    n0 = (data["detector_positions"].shape[1] - 1) // 2
    tile_extent = (
        n0 * detector_dist
    )  # extent of tile from 0 to edge, xy-coordinates norm
    tile_center = np.copy(data["detector_positions"][ievt, n0, n0])
    # Shift to the hight of CLF (z)
    tile_center[2] = height_of_clf

    # Shift detector positions if detector exists
    dpos = data["detector_positions"][ievt, :, :, :]
    dpos = np.where(
        data["detector_exists"][ievt, :, :, np.newaxis],
        dpos - tile_center[np.newaxis, np.newaxis, :],
        0,
    )
    dpos[:, :, :2] = dpos[:, :, :2] / tile_extent
    dpos[:, :, 2] = dpos[:, :, 2] / height_extent
    data["detector_positions"][ievt, :, :, :] = dpos

    # Shift shower core array(s) if array exists
    keys = ["shower_core", "std_recon_shower_core", "std_recon_shower_core_err"]
    for key in keys:
        if data.get(key) is not None:
            if key == "std_recon_shower_core_err":
                # no need to shift, because it is an error
                data[key][ievt][:2] = data[key][ievt][:2] / tile_extent
            else:
                data[key][ievt][:2] = (
                    data[key][ievt][:2] - tile_center[:2]
                ) / tile_extent
            if key == "shower_core":
                data[key][ievt][2] = data[key][ievt][2] / height_extent

    return data


def tile_positions(ixy0, tile_size, badsd, data, ievt):
    # Create centered tile
    # n0 = (tile_size - 1) / 2
    to_meters = 1e-2
    x, y = np.mgrid[0:tile_size, 0:tile_size].astype(float)

    # Shift towards real center
    # ixy0 = [24, 10] - at the edge, uncomment for testing
    x += ixy0[0]
    y += ixy0[1]
    xy_code = x * 100 + y

    # Create mask (:, tile_size, tile_size)
    masks = tasd_clf.tasdmc_clf[:, 0][:, np.newaxis, np.newaxis] == xy_code
    tasdmc_clf_indices = np.argmax(masks, axis=0)
    do_exist = masks.any(axis=0)
    tasdmc_clf_indices = np.where(do_exist, tasdmc_clf_indices, -1)

    # Do detectors work:
    good = ~np.isin(tasd_clf.tasdmc_clf[tasdmc_clf_indices, 0], badsd)
    status = np.logical_and(good, do_exist)

    # Absolute coordinates (relative to central laser facility) in meters
    data["detector_positions"][ievt, :, :, :] = (
        tasd_clf.tasdmc_clf[tasdmc_clf_indices, 1:]
    ) * to_meters

    data["detector_positions_abs"][ievt, :, :, :] = (
        tasd_clf.tasdmc_clf[tasdmc_clf_indices, 1:]
    ) * to_meters
    data["detector_positions_id"][ievt, :, :] = tasd_clf.tasdmc_clf[
        tasdmc_clf_indices, 0
    ]

    data["detector_states"][ievt, :, :] = status
    data["detector_exists"][ievt, :, :] = do_exist
    data["detector_good"][ievt, :, :] = good
    return data


def detector_readings_awkward(data, dst_lists, avg_traces):
    to_nsec = 4 * 1000
    sdmeta_list, sdwaveform_list, badsdinfo_list = dst_lists[1:4]

    hits_counts = []
    hits_det_id = []
    hits_nfold = []

    if avg_traces:
        hits_arrival_times = []
        hits_total_signals = []
        hits_time_traces = []
        per_hit_ttrace_counts = []  # New: Counts for each hit's trace
    else:
        hits_arrival_times_low = []
        hits_arrival_times_up = []
        hits_total_signals_low = []
        hits_total_signals_up = []
        hits_time_traces_low = []
        hits_time_traces_up = []
        per_hit_ttrace_counts_low = []  # New: Counts for each hit's trace (low)
        per_hit_ttrace_counts_up = []   # New: Counts for each hit's trace (up)

    empty_events = []

    for ievt, (event, wform, badsd) in enumerate(
        zip(sdmeta_list, sdwaveform_list, badsdinfo_list)
    ):
        event, wform = cut_events(event, wform, max_hits="all", max_windows="all")

        if event.shape[1] == 0:
            empty_events.append(ievt)
            hits_counts.append(0)
            # No need to append to per_hit counts if there are no hits
            continue

        hits_counts.append(event.shape[1])
        hits_det_id.extend(event[0])
        hits_nfold.extend(event[11])

        fadc_per_vem_low = event[9]
        fadc_per_vem_up = event[10]
        nfolds = event[11].astype(int)

        if avg_traces:
            hits_arrival_times.extend((event[2] + event[3]) / 2 * to_nsec)
            hits_total_signals.extend((event[4] + event[5]) / 2)
            
            for i, nfold in enumerate(nfolds):
                ntime_trace = 128 * nfold
                wform_col = wform[:ntime_trace, i]
                
                ttrace_low = wform_col[::2] / fadc_per_vem_low[i]
                ttrace_up = wform_col[1::2] / fadc_per_vem_up[i]
                ttrace = (ttrace_low + ttrace_up) / 2
                
                hits_time_traces.extend(ttrace)
                per_hit_ttrace_counts.append(len(ttrace))
        else:
            hits_arrival_times_low.extend(event[2] * to_nsec)
            hits_arrival_times_up.extend(event[3] * to_nsec)
            hits_total_signals_low.extend(event[4])
            hits_total_signals_up.extend(event[5])

            for i, nfold in enumerate(nfolds):
                ntime_trace = 128 * nfold
                wform_col = wform[:ntime_trace, i]
                
                ttrace_low = wform_col[::2] / fadc_per_vem_low[i]
                ttrace_up = wform_col[1::2] / fadc_per_vem_up[i]
                
                hits_time_traces_low.extend(ttrace_low)
                hits_time_traces_up.extend(ttrace_up)
                per_hit_ttrace_counts_low.append(len(ttrace_low))
                per_hit_ttrace_counts_up.append(len(ttrace_up))

    # Remove empty events from per-event data
    if len(empty_events) != 0:
        for key, value in data.items():
            data[key] = np.delete(value, empty_events, axis=0)

    data["hits_det_id"] = ak.unflatten(np.array(hits_det_id), hits_counts)
    data["hits_nfold"] = ak.unflatten(np.array(hits_nfold), hits_counts)

    if avg_traces:
        data["hits_arrival_times"] = ak.unflatten(np.array(hits_arrival_times), hits_counts)
        data["hits_total_signals"] = ak.unflatten(np.array(hits_total_signals), hits_counts)
        
        # Step 1: Unflatten flat trace data into per-hit traces
        per_hit_traces = ak.unflatten(hits_time_traces, per_hit_ttrace_counts)
        # Step 2: Group per-hit traces by event
        data["hits_time_traces"] = ak.unflatten(per_hit_traces, hits_counts)
    else:
        data["hits_arrival_times_low"] = ak.unflatten(np.array(hits_arrival_times_low), hits_counts)
        data["hits_arrival_times_up"] = ak.unflatten(np.array(hits_arrival_times_up), hits_counts)
        data["hits_total_signals_low"] = ak.unflatten(np.array(hits_total_signals_low), hits_counts)
        data["hits_total_signals_up"] = ak.unflatten(np.array(hits_total_signals_up), hits_counts)
        
        # Step 1 for 'low'
        per_hit_traces_low = ak.unflatten(hits_time_traces_low, per_hit_ttrace_counts_low)
        # Step 2 for 'low'
        data["hits_time_traces_low"] = ak.unflatten(per_hit_traces_low, hits_counts)
        
        # Step 1 for 'up'
        per_hit_traces_up = ak.unflatten(hits_time_traces_up, per_hit_ttrace_counts_up)
        # Step 2 for 'up'
        data["hits_time_traces_up"] = ak.unflatten(per_hit_traces_up, hits_counts)

    return data, empty_events


def detector_readings(data, dst_lists, ntile, avg_traces):
    ntime_trace = 128  # number of time trace of waveform
    to_nsec = 4 * 1000

    num_events = dst_lists[0][0].shape[0]
    shape = num_events, ntile, ntile
    data["detector_positions"] = np.zeros((*shape, 3), dtype=np.float32)
    data["detector_positions_abs"] = np.zeros((*shape, 3), dtype=np.float32)
    data["detector_positions_id"] = np.zeros(shape, dtype=np.float32)
    data["detector_states"] = np.zeros(shape, dtype=bool)
    data["detector_exists"] = np.zeros(shape, dtype=bool)
    data["detector_good"] = np.zeros(shape, dtype=bool)
    data["nfold"] = np.zeros(shape, dtype=np.float32)

    if avg_traces:
        data["arrival_times"] = np.zeros(shape, dtype=np.float32)
        data["time_traces"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
        data["total_signals"] = np.zeros(shape, dtype=np.float32)
    else:
        data["arrival_times_low"] = np.zeros(shape, dtype=np.float32)
        data["arrival_times_up"] = np.zeros(shape, dtype=np.float32)
        data["time_traces_low"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
        data["time_traces_up"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
        data["total_signals_low"] = np.zeros(shape, dtype=np.float32)
        data["total_signals_up"] = np.zeros(shape, dtype=np.float32)

    empty_events = []

    sdmeta_list, sdwaveform_list, badsdinfo_list = dst_lists[1:4]

    for ievt, (event, wform, badsd) in enumerate(
        zip(sdmeta_list, sdwaveform_list, badsdinfo_list)
    ):
        # event.shape = (11, number of detectors)
        event, wform = cut_events(event, wform)

        if event.shape[1] == 0:
            empty_events.append(ievt)
            continue

        ixy0, inside_tile, ixy = center_tile(event, ntile)
        # Populate absolute detector positions and states
        data = tile_positions(ixy0, ntile, badsd, data, ievt)
        # Shift and normalize detector positions and shower cores
        data = tile_normalization(data, ievt)

        # Populate detector readings and arrival times
        wform = wform[:, inside_tile]
        fadc_per_vem_low = event[9][inside_tile]
        fadc_per_vem_up = event[10][inside_tile]

        # foldedness of the hit (over how many 128 fadc widnows this signal extends)
        # (e.g.) If the waveform consists of 128 * 3 = 384 time bins, `nfold` is 3.
        data["nfold"][ievt, ixy[0], ixy[1]] = event[11][inside_tile]

        if avg_traces:
            atimes = (event[2] + event[3]) / 2
            data["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

            ntime_trace = 128
            ttrace_low = wform[:ntime_trace] / fadc_per_vem_low
            if wform.shape[0] > ntime_trace:
                ttrace_up = wform[ntime_trace:] / fadc_per_vem_up
            else:
                ttrace_up = np.zeros_like(ttrace_low)
            ttrace = (ttrace_low + ttrace_up) / 2
            data["time_traces"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()

            data["total_signals"][ievt, ixy[0], ixy[1]] = (
                event[4][inside_tile] + event[5][inside_tile]
            ) / 2

        else:
            ntime_trace = 128
            ttrace_low = wform[:ntime_trace] / fadc_per_vem_low
            if wform.shape[0] > ntime_trace:
                ttrace_up = wform[ntime_trace:] / fadc_per_vem_up
            else:
                ttrace_up = np.zeros_like(ttrace_low)

            data["arrival_times_low"][ievt, ixy[0], ixy[1]] = (
                event[2][inside_tile] * to_nsec
            )
            data["arrival_times_up"][ievt, ixy[0], ixy[1]] = (
                event[3][inside_tile] * to_nsec
            )

            data["total_signals_low"][ievt, ixy[0], ixy[1]] = event[4][inside_tile]
            data["total_signals_up"][ievt, ixy[0], ixy[1]] = event[5][inside_tile]

        if avg_traces:
            data["arrival_times"][ievt, :, :] = np.where(
                data["detector_states"][ievt, :, :],
                data["arrival_times"][ievt, :, :],
                0,
            )
        else:
            for arrv_array_name in ["arrival_times_low", "arrival_times_up"]:
                data[arrv_array_name][ievt, :, :] = np.where(
                    data["detector_states"][ievt, :, :],
                    data[arrv_array_name][ievt, :, :],
                    0,
                )

    # Remove empty events
    if len(empty_events) != 0:
        for key, value in data.items():
            data[key] = np.delete(value, empty_events, axis=0)
    return data, empty_events


def parse_dst_file(
    dst_file,
    ntile=7,
    xmax_reader=None,
    avg_traces=True,
    add_shower_params=True,
    add_standard_recon=True,
    config=None,
    use_grid_model=True,
):
    #  ntile - number of SD per one side
    dst_string = read_dst_file(dst_file)
    dst_lists = parse_dst_string(dst_string)

    if dst_lists is None:
        return None

    # Load xmax info for current dst file
    if xmax_reader is not None:
        xmax_reader.read_file(dst_file)

    # Dictionary with parsed data
    data = dict()
    if add_shower_params:
        data = shower_params(data, dst_lists, xmax_reader)

    if add_standard_recon:
        data = standard_recon(data, dst_lists)

    if use_grid_model:
        data, empty_events = detector_readings(data, dst_lists, ntile, avg_traces)
    else:
        data, empty_events = detector_readings_awkward(data, dst_lists, avg_traces)

    # Remove empty events
    if len(empty_events) != 0:
        for key, value in data.items():
            if isinstance(value, ak.Array):
                # For awkward arrays, we need to rebuild the array without the empty events
                builder = ak.ArrayBuilder()
                for i, event_data in enumerate(value):
                    if i not in empty_events:
                        builder.append(event_data)
                data[key] = builder.snapshot()
            else:
                data[key] = np.delete(value, empty_events, axis=0)

    if (config is not None) and (hasattr(config, "add_event_ids")):
        data = config.add_event_ids(data, dst_file)
    return data
