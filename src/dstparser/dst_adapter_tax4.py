import numpy as np
from dstparser.dst_reader import read_dst_file_all_events
from dstparser.dst_parsers import dst_sections, parse_event, parse_sdwaveform, parse_badsdinfo
import dstparser.tasd_clf as tasd_clf
import re
from pathlib import Path


def parse_sdmeta_tax4(sdmeta_list_str):
    """
    #SD meta DATA has 11 fields/hit for BOTH TAx4 readers (printAll and the
    TALE install's add_standard_recon), NOT 12 like TA-SD's own
    add_standard_recon_v2 -- no `nfold` field is emitted for TAx4
    (sditerator_cppanalysis_printAll.cpp:38-43 and
    sditerator_cppanalysis_add_standard_recon.cpp:49-54):
        xxyy, isgood, reltime[0], reltime[1], pulsa[0], pulsa[1],
        xyzclf[0], xyzclf[1], xyzclf[2], vem[0], vem[1]
    CAVEAT: in printAll's variant specifically, the vem[1] slot is actually a
    duplicate of vem[0] (upstream printf bug -- real vem[1] is never printed
    by that variant). The add_standard_recon variant prints real vem[1].
    """
    record_size = 11
    return [
        np.fromstring(line, sep=" ").reshape(-1, record_size).transpose()
        for line in sdmeta_list_str
    ]


def parse_dst_string_tax4(dst_string):
    # Same section layout as the standard-recon dumps (dst_sections is
    # format-agnostic), but the #SD meta DATA record width differs (11 vs 12
    # fields/hit) and #EVENT DATA is truth/counts-only (11 fields, no
    # standard-recon block) -- see parse_sdmeta_tax4 and shower_params/
    # raw_event_counts below.
    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = (
        dst_sections(dst_string)
    )

    if len(event_list_str) == 0:
        return None

    event_list = parse_event(event_list_str)
    sdmeta_list = parse_sdmeta_tax4(sdmeta_list_str)
    sdwaveform_list = parse_sdwaveform(sdwaveform_list_str)
    badsdinfo_list = parse_badsdinfo(badsdinfo_list_str)

    return event_list, sdmeta_list, sdwaveform_list, badsdinfo_list


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
    # for details: /ceph/sharedfs/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/sditerator/src/sditerator_cppanalysis.cpp
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


def raw_event_counts(data, dst_lists):
    """
    Use with the printAll reader (read_dst_file_all_events / dst_lists built
    via parse_dst_string_tax4 on printAll output). printAll's #EVENT DATA line
    has only 11 fields -- truth (0-6, handled by shower_params above) + these
    raw counts, NO reconstruction fields (sditerator_cppanalysis_printAll.cpp).
    printAll lists EVERY thrown event,
    triggered or not -- use this path for full per-shower statistics.

    """
    event_list = dst_lists[0]
    data["yymmdd"] = event_list[7]
    data["hhmmss"] = event_list[8]
    # number of SDs in space-time cluster (rufptn_.nstclust) -- the closest
    # thing to a quality-cut variable available for TAx4; NOT equivalent to
    # TA-SD's std_recon_nsd (that's a std-recon-stage count; this is raw).
    data["nstclust"] = event_list[9]
    # number of waveforms for event, all detectors (rusdraw_.nofwf)
    data["nofwf"] = event_list[10]
    return data


def cut_events(event, wform):
    # ! If the signal > 128 bins it is divided on parts with 128 in each
    # ! The code below takes only first part (waveform) in case if
    # ! the signal consists of several such parts
    # Set all repeating elements to False, except first one

    # In TAx4, the SD ID is a 4-digit number, where the first two digits are
    # the y-coordinate and the last two digits are the x-coordinate.
    # We need to change from yyxx to xxyy format for further processing
    event[0, :] = ((event[0, :] % 100) * 100 + (event[0, :] // 100)).astype(np.int32)
    
    # print("event shape:", event.shape)
    
    # for i, e in enumerate(event):
    #     print(i, e)

    sdid = event[0]
    u, c = np.unique(sdid, return_counts=True)
    dup = u[c > 1]
    mask = sdid == sdid
    
    # print(f"duplicate SD IDs: {dup}")
    
    for el in dup:
        mask[np.where(sdid == el)[0][1:]] = False

    event = event[:, mask]
    # exclude coincidence signals
    # the signal is a part of the event
    event = event[:, event[1] > 2]

    # Pick corresponding waveforms. For TAx4, waveform data is present for
    # only a minority of events (~16% in a spot check) -- when
    # absent it is absent for EVERY hit in that event (all-or-nothing, matches
    # the event-level `nofwf` truth count), not a per-hit data error. Hits
    # without a matching waveform get has_wf=False; their time-trace stays
    # zero-filled but their meta-derived fields (position, signal, arrival
    # time) are unaffected -- those come from #SD meta DATA, not the waveform.
    has_wf = np.zeros(event.shape[1], dtype=bool)
    if wform.shape[1] == 0:
        # No waveform recorded for this event at all (matches nofwf==0).
        wform = np.zeros((wform.shape[0] - 3, event.shape[1]), dtype=wform.dtype)
    else:
        wform_idx = []
        for i, xycoord in enumerate(event[0].astype(np.int32)):
            match = np.where(wform[0] == xycoord)[0]
            if match.size > 0:
                wform_idx.append(match[0])
                has_wf[i] = True
            else:
                wform_idx.append(0)
        wform = wform[3:, wform_idx]
    return event, wform, has_wf


def center_tile(event, ntile):
    # Put largest-signal SD at the center of ntile x ntile grids
    # center around detector with max signal
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)

    # ix and iy as one array [ix, iy]
    ixy = np.array([event[0] // 100, event[0] % 100]).astype(np.int32)
    
    
    # print(f"ix = {ixy}")
    # print(f"event[6]= {event[6]}")
    # print(f"event[7]= {event[7]}")
    # print(f"event[8]= {event[8]}")
    
    # Indicies of central detector ix0, iy0
    ixy0 = np.copy(ixy[:, max_signal_idx]) - (ntile - 1) // 2
    ixy -= ixy0[:, np.newaxis]
    # cut array size to fit the tile size
    inside_tile = (ixy[0] < ntile) & (ixy[1] < ntile) & (ixy[0] >= 0) & (ixy[1] >= 0)
    ixy = ixy[:, inside_tile]
    return ixy0, inside_tile, ixy


def tile_normalization(data, ievt):
    # detector_dist = 1200  # meters
    detector_dist = 2080  #!!! meters, for TAx4
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


def tile_positions(ixy0, tile_size, badsd, data, ievt, hits_positions, hits_ids):
    # Create centered tile
    # n0 = (tile_size - 1) / 2
    to_meters = 1e-2
    x, y = np.mgrid[0:tile_size, 0:tile_size].astype(float)

    # Shift towards real center
    # ixy0 = [24, 10] - at the edge, uncomment for testing
    x += ixy0[0]
    y += ixy0[1]
    xy_code = x * 100 + y
    
    all_ids = hits_ids
    all_pos = hits_positions

    # masks: (n_hits, tile_size, tile_size)
    masks = all_ids[:, None, None] == xy_code[None, :, :]
    best_idx = np.argmax(masks, axis=0)           # index into your hits array
    exists   = masks.any(axis=0)
    best_idx = np.where(exists, best_idx, -1)

    # pull out the IDs & positions per cell
    cell_ids  = np.where(exists,
                         all_ids[best_idx],
                         0)
    cell_pos  = np.where(exists[...,None],
                         all_pos[best_idx],
                         0.0)

    # good/bad
    good   = ~np.isin(cell_ids, badsd)
    status = good & exists

    # write into data
    data["detector_positions"][ievt]     = cell_pos * to_meters
    data["detector_positions_abs"][ievt] = cell_pos * to_meters
    data["detector_positions_id"][ievt]  = cell_ids

    data["detector_states"][ievt] = status
    data["detector_exists"][ievt] = exists
    data["detector_good"][ievt]   = good
    return data


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
        event, wform, has_wf = cut_events(event, wform)

        if event.shape[1] == 0:
            empty_events.append(ievt)
            continue

        ixy0, inside_tile, ixy = center_tile(event, ntile)
        # Populate absolute detector positions and states
        # -------------------------------------------------------------------
        # calculate each hit’s absolute (x,y,z) in cm from your event arrays:
        # event[6], event[7], event[8] are in units of 1200 m (i.e. detector_dist),
        # so multiply back to meters, then to cm:
        detector_dist = 1200.0    # [m]
        # stack into shape (n_hits,3) in **cm**:
        hits_m = np.vstack([event[6],
                            event[7],
                            event[8]]).T * detector_dist
        hits_cm = hits_m * 100.0
        hit_ids = event[0].astype(int)
        
        
        data = tile_positions(ixy0, ntile, badsd, data, ievt, hits_cm, hit_ids)
        # Shift and normalize detector positions and shower cores
        data = tile_normalization(data, ievt)

        # Populate detector readings and arrival times
        wform = wform[:, inside_tile]
        wf_mask = has_wf[inside_tile]
        fadc_per_vem_low = event[9][inside_tile]
        fadc_per_vem_up = event[10][inside_tile]

        # `nfold` (foldedness) is NOT available for TAx4 -- printAll's #SD meta
        # DATA has only 11 fields/hit (see parse_sdmeta_tax4), no nfold slot.
        # data["nfold"] stays at its zero-init.

        # Meta-derived fields (position/signal/arrival time, from #SD meta
        # DATA) apply to every hit regardless of waveform availability.
        # Waveform-derived fields (time_traces) are only filled where
        # wf_mask is True -- for TAx4, most events have NO waveform for ANY
        # of their hits (all-or-nothing per event), and
        # those cells stay at their zero-init rather than being computed
        # from a meaningless placeholder waveform index.
        ix_wf, iy_wf = ixy[0][wf_mask], ixy[1][wf_mask]

        if avg_traces:
            atimes = (event[2] + event[3]) / 2
            data["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

            if wf_mask.any():
                ttrace = (
                    wform[:ntime_trace, wf_mask] / fadc_per_vem_low[wf_mask]
                    + wform[ntime_trace:, wf_mask] / fadc_per_vem_up[wf_mask]
                ) / 2
                data["time_traces"][ievt, ix_wf, iy_wf, :] = ttrace.transpose()

            data["total_signals"][ievt, ixy[0], ixy[1]] = (
                event[4][inside_tile] + event[5][inside_tile]
            ) / 2

        else:
            if wf_mask.any():
                ttrace = wform[:ntime_trace, wf_mask] / fadc_per_vem_low[wf_mask]
                data["time_traces_low"][ievt, ix_wf, iy_wf, :] = ttrace.transpose()

                ttrace = wform[ntime_trace:, wf_mask] / fadc_per_vem_up[wf_mask]
                data["time_traces_up"][ievt, ix_wf, iy_wf, :] = ttrace.transpose()

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
    return data


def parse_dst_file_tax4(
    dst_file,
    ntile=7,
    xmax_reader=None,
    avg_traces=True,
    add_shower_params=True,
    add_raw_counts=True,
    config=None,
):
    #  ntile - number of SD per one side
    # Reads via sditerator_printAll.run: EVERY thrown event (triggered or
    # not), truth + raw SD data, no reconstruction. For reconstructed TAx4
    # events in the vlen format, use parse_dst_file_tax4_vlen instead.
    dst_string = read_dst_file_all_events(dst_file)
    dst_lists = parse_dst_string_tax4(dst_string)

    if dst_lists is None:
        return None

    # Load xmax info for current dst file
    if xmax_reader is not None:
        xmax_reader.read_file(dst_file)

    # Dictionary with parsed data
    data = dict()
    if add_shower_params:
        data = shower_params(data, dst_lists, xmax_reader)

    if add_raw_counts:
        data = raw_event_counts(data, dst_lists)

    data = detector_readings(data, dst_lists, ntile, avg_traces)

    if (config is not None) and (hasattr(config, "add_event_ids")):
        data = config.add_event_ids(data, dst_file)
    return data
