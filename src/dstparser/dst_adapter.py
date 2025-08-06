import numpy as np
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


def shower_params(data, dst_data, xmax_data):
    # Shower related
    # for details: /ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/sditerator/src/sditerator_cppanalysis.cpp
    to_meters = 1e-2
    events = dst_data["events"]
    data["mass_number"] = corsika_id2mass(events["rusdmc_.parttype"])
    data["energy"] = events["rusdmc_.energy"]

    if xmax_data is not None:
        data["xmax"] = xmax_data(data["energy"])

    # Theta and phi are in radians (see dst_fields.md)
    data["shower_axis"] = np.array(
        [
            np.sin(events["rusdmc_.theta"]) * np.cos(events["rusdmc_.phi"] + np.pi),
            np.sin(events["rusdmc_.theta"]) * np.sin(events["rusdmc_.phi"] + np.pi),
            np.cos(events["rusdmc_.theta"]),
        ],
        dtype=np.float32,
    ).transpose()

    # shower core in cm
    data["shower_core"] = np.array(
        np.stack(
            [
                events["rusdmc_.corexyz[0]"],
                events["rusdmc_.corexyz[1]"],
                events["rusdmc_.corexyz[2]"],
            ],
            axis=1,
        ),
        dtype=np.float32,
    )

    return data


def standard_recon(
    data, dst_data, include_combined_fit=False, include_fixed_curve_fit=False
):
    events = dst_data["events"]
    # Exempt from comments of cpp source code at:
    # /ceph/work/SATORI/projects/TA-ASIoP/benMC/sdanalysis_2019/sdmc/sdmc_spctr.c
    # // Reported by DAQ as time of the 1st signal in the triple that caused the triggger.
    # // From now on, everyhting is relative to hhmmss.  Not useful in the event reconstruction.
    # Date of event
    # rusdraw_.yymmdd = 80916; // Event date year = 08, month = 09, day = 16
    data["std_recon_yymmdd"] = events["rusdraw_.yymmdd"]
    # Time of event
    # rusdraw_.hhmmss = 1354;  // Event time, hour=00, minute=13, second = 54
    data["std_recon_hhmmss"] = events["rusdraw_.hhmmss"]
    # Microseconds for the second
    # rusdraw_.usec = 111111
    data["std_recon_usec"] = events["rusdraw_.usec"]
    # Number of waveforms for event for all detectors
    data["std_recon_nofwf"] = events["rusdraw_.nofwf"]
    # number of SDs in space-time cluster
    data["std_recon_nsd"] = events["rufptn_.nstclust"]
    # number of SDs in space cluster
    data["std_recon_nsclust"] = events["rufptn_.nsclust"]
    # number of hit SDs
    data["std_recon_nhits"] = events["rufptn_.nhits"]
    # number of SDs in space-time cluster & lie on the border of the array
    data["std_recon_nborder"] = events["rufptn_.nborder"]
    # total charge [VEM] of SDs in the space-time cluster, (lower & upper)
    data["std_recon_qtot"] = np.array(
        [
            events["rufptn_.qtot[0]"],
            events["rufptn_.qtot[1]"],
        ]
    ).transpose(1, 0)
    # energy reconstructed by the standard energy estimation table [EeV]
    data["std_recon_energy"] = events["rufldf_.energy[0]"]
    # reconstructed scale of the Lateral Distribution Function (LDF) fit [VEM m-2]
    data["std_recon_ldf_scale"] = events["rufldf_.sc[0]"]
    # uncertainty of the scale [VEM m-2]
    data["std_recon_ldf_scale_err"] = events["rufldf_.dsc[0]"]
    # chi-square of the LDF fit
    data["std_recon_ldf_chi2"] = events["rufldf_.chi2[0]"]
    # the number of degree of freedom of the LDF fit (= n - 3),
    # where "n" is the number of the SDs used for the LDF fit
    data["std_recon_ldf_ndof"] = events["rufldf_.ndof[0]"]
    # core position (x, y) reconstructed by the LDF fit in CLF coordinate [m]
    data["std_recon_shower_core"] = np.array(
        [
            rec_coreposition_to_CLF_meters(events["rufldf_.xcore[0]"], option="x"),
            rec_coreposition_to_CLF_meters(events["rufldf_.ycore[0]"], option="y"),
        ]
    ).transpose(1, 0)

    # uncertainty of the core position (x, y) reconstructed by the LDF fit
    data["std_recon_shower_core_err"] = np.array(
        [
            rec_coreposition_to_CLF_meters(events["rufldf_.dxcore[0]"], option="dx"),
            rec_coreposition_to_CLF_meters(events["rufldf_.dycore[0]"], option="dy"),
        ]
    ).transpose(1, 0)
    # S800 (particle density at 800 m from the shower axis) [VEM m-2]
    data["std_recon_s800"] = events["rufldf_.s800[0]"]

    if include_combined_fit:
        # reconstructed values of the geometry+LDF (combined) fit
        data["std_recon_combined_energy"] = events["rufldf_.energy[1]"]
        data["std_recon_combined_scale"] = events["rufldf_.sc[1]"]
        data["std_recon_combined_scale_err"] = events["rufldf_.dsc[1]"]
        data["std_recon_combined_chi2"] = events["rufldf_.chi2[1]"]
        # the number of degree of freedom of the LDF fit (= 2*n - 6),
        # where "n" is the number of the SDs used for the LDF fit
        data["std_recon_combined_ndof"] = events["rufldf_.ndof[1]"]
        data["std_recon_combined_shower_core"] = np.array(
            [
                rec_coreposition_to_CLF_meters(events["rufldf_.xcore[1]"], option="x"),
                rec_coreposition_to_CLF_meters(events["rufldf_.ycore[1]"], option="y"),
            ]
        ).transpose(1, 0)
        data["std_recon_combined_shower_core_err"] = np.array(
            [
                rec_coreposition_to_CLF_meters(
                    events["rufldf_.dxcore[1]"], option="dx"
                ),
                rec_coreposition_to_CLF_meters(
                    events["rufldf_.dycore[1]"], option="dy"
                ),
            ]
        ).transpose(1, 0)
        data["std_recon_combined_s800"] = events["rufldf_.s800[1]"]

        # 3-d unit vector of the arrival direction (pointing back to the source)
        # geometry+LDF fit
        # "+0.5" is a correction for zenith angle.
        theta = np.deg2rad(events["rufldf_.theta"] + 0.5)
        phi = np.deg2rad(events["rufldf_.phi"]) + np.pi
        data["std_recon_shower_axis_combined"] = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            dtype=np.float32,
        ).transpose()

        # uncertainty of the pointing direction [degree]
        # geometry+LDF fit
        theta = np.deg2rad(events["rufldf_.theta"])
        dtheta = events["rufldf_.dtheta"]
        dphi = events["rufldf_.dphi"]

        data["std_recon_shower_axis_err_combined"] = np.sqrt(
            dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
        )

    if include_fixed_curve_fit:
        # chi-square of the geometry fit (fixed curvature)
        data["std_recon_geom_chi2_fixed_curve"] = events["rusdgeom_.chi2[1]"]
        # the number of degree of freedom of the geometry fit (= n - 5),
        # where "n" is the number of the SDs used for the geometry fit
        data["std_recon_geom_ndof_fixed_curve"] = events["rusdgeom_.ndof[1]"]

        # 3-d unit vector of the arrival direction (pointing back to the source)
        # geometry fit with a fixed curved parameter
        # "+0.5" is a correction for zenith angle.
        theta = np.deg2rad(events["rusdgeom_.theta[1]"] + 0.5)
        phi = np.deg2rad(events["rusdgeom_.phi[1]"]) + np.pi

        data["std_recon_shower_axis_fixed_curve"] = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            dtype=np.float32,
        ).transpose()

        # uncertainty of the pointing direction [degree]
        # fixed curved parameter

        theta = np.deg2rad(events["rusdgeom_.theta[1]"])
        dtheta = events["rusdgeom_.dtheta[1]"]
        dphi = events["rusdgeom_.dphi[1]"]

        data["std_recon_shower_axis_err_fixed_curve"] = np.sqrt(
            dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
        )

    # 3-d unit vector of the arrival direction (pointing back to the source)
    # geometry fit with a free curved parameter.
    # "+0.5" is a correction for zenith angle.
    theta = np.deg2rad(events["rusdgeom_.theta[2]"] + 0.5)
    phi = np.deg2rad(events["rusdgeom_.phi[2]"]) + np.pi
    data["std_recon_shower_axis"] = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float32,
    ).transpose()
    # uncertainty of the pointing direction [degree]
    # free curved parameter
    # event_list[22] is zenith angle in deg
    # event_list[24] is uncertainty zenith angle in deg
    # event_list[25] is uncertainty azimuth angle in deg

    theta = np.deg2rad(events["rusdgeom_.theta[2]"])
    dtheta = events["rusdgeom_.dtheta[2]"]
    dphi = events["rusdgeom_.dphi[2]"]

    # Uncertainty in degrees
    data["std_recon_shower_axis_err"] = np.sqrt(
        dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
    )

    # chi-square of the geometry fit (free curvature)
    data["std_recon_geom_chi2"] = events["rusdgeom_.chi2[2]"]
    # the number of degree of freedom of the geometry fit (= n - 6),
    # where "n" is the number of the SDs used for the geometry fit
    data["std_recon_geom_ndof"] = events["rusdgeom_.ndof[2]"]
    # curvature paramter `a` of the geometry fit
    data["std_recon_curvature"] = events["rusdgeom_.a"]
    # uncertainty of the curvature paramter `a` of the geometry fit
    data["std_recon_curvature_err"] = events["rusdgeom_.da"]
    # distance b/w the reconstructed core and the edge from the TA SD array [in 1,200 meter unit]
    # negative for events with the core outside of the TA SD array.
    data["std_recon_border_distance"] = events["rufldf_.bdist"]
    # distance to the T-shape TA SD array, edge of the sub-arrays [in 1,200 meter unit]
    # this value is used as "border_distance" before implementation of the boundary trigger (on 2008/11/11)
    data["std_recon_border_distance_tshape"] = events["rufldf_.tdist"]

    return data


def cut_events(event, wform):
    # ! If the signal > 128 bins it is divided on parts with 128 in each
    # ! The code below takes only first part (waveform) in case if
    # ! the signal consists of several such parts
    # Set all repeating elements to False, except first one
    sdid = event[0]
    u, c = np.unique(sdid, return_counts=True)
    dup = u[c > 1]
    mask = sdid == sdid
    for el in dup:
        mask[np.where(sdid == el)[0][1:]] = False

    event = event[:, mask]
    # exclude coincidence signals
    # the signal is a part of the event
    event = event[:, event[1] > 2]

    # Pick corresponding waveforms
    wform_idx = []
    for xycoord in event[0].astype(np.int32):
        # Take only the first waveform (second [0])
        wform_idx.append(np.where(wform[0] == xycoord)[0][0])

    wform = wform[3:, wform_idx]
    return event, wform


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

            ttrace = (
                wform[:ntime_trace] / fadc_per_vem_low
                + wform[ntime_trace:] / fadc_per_vem_up
            ) / 2
            data["time_traces"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()

            data["total_signals"][ievt, ixy[0], ixy[1]] = (
                event[4][inside_tile] + event[5][inside_tile]
            ) / 2

        else:
            ttrace = wform[:ntime_trace] / fadc_per_vem_low
            data["time_traces_low"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()

            ttrace = wform[ntime_trace:] / fadc_per_vem_up
            data["time_traces_up"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()

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


def filter_offsets(mask, offsets):
    # 1) compute counts per segment
    counts = np.add.reduceat(mask.astype(int), offsets[:-1])
    # 2) build new offsets
    return np.concatenate(([0], np.cumsum(counts)))


def detector_readings_flat(data, hits, waveforms):
    # for c = 3e10 cm/s:
    # to_nsec = 4 * 1000
    # The below is more correct for c = 2.998e10 cm/c,
    # to_nsec = 4002.7691424

    # Filter for isgood > 2:
    hit_mask = hits["rufptn_.isgood"] > 2
    # Repeat hit_mask nfold times
    wf_mask = np.repeat(hit_mask, hits["rufptn_.nfold"])

    # Filter offsets according to masks
    hits_offsets = filter_offsets(hit_mask, hits["offsets"])
    waveforms_offsets = filter_offsets(wf_mask, waveforms["offsets"])

    # Filter hits
    hits = {key: value[hit_mask] for key, value in hits.items() if key != "offsets"}
    hits["offsets"] = hits_offsets
    hits["wf_offsets"] = np.concatenate(([0], np.cumsum(hits["rufptn_.nfold"])))

    # Filter waveforms
    waveforms = {
        key: value[wf_mask] for key, value in waveforms.items() if key != "offsets"
    }
    waveforms["offsets"] = waveforms_offsets

    # Check if filtering is correct
    # Compare detectors ids for hits and corresponding waveforms
    assert np.all(
        np.repeat(hits["rufptn_.xxyy"], hits["rufptn_.nfold"])
        == waveforms["rusdraw_.xxyy"]
    ), "Hits are not in agreement with waverforms"

    # -------- Example/reminder how to work with offsets:
    # We work with flattened arrays like this:
    # Example: get hits for the event with index ievent:
    # start, end = hits["offsets"][ievent], hits["wf_offsets"][ievent + 1]
    # hits_arrv_time_lower = hits[rufptn_.reltime[0]][start:end]
    #

    # Example: get a waveform for specific hit:
    # ihit - global index in hits
    # start, end = hits["wf_offsets"][ihit], hits["wf_offsets"][ihit + 1]
    # waveforms_fadc = waveforms["rusdraw_.fadc"][start:end]
    #
    # For vectorized work use np.split and np.reduceat
    # ---------

    # Create data arrays:

    # We use counter sep. dist units, to get in nsec, multiply to to_nsec
    # We normalize it anyway ...
    data["arrival_times"] = np.stack(
        [hits["rufptn_.reltime[0]"], hits["rufptn_.reltime[1]"]], axis=1
    )

    # Pulse area in VEM (pedestals subtracted)
    data["pulse_area"] = np.stack(
        [hits["rufptn_.pulsa[0]"], hits["rufptn_.pulsa[1]"]], axis=1
    )

    # SD coordinates in CLF frame [1200m units]
    # Probably for TAx4 it is in [2080m units]
    # Looks like reasonable choice (not very large magnitude)
    data["detector_positions"] = np.stack(
        [
            hits["rufptn_.xyzclf[0]"],
            hits["rufptn_.xyzclf[1]"],
            hits["rufptn_.xyzclf[2]"],
        ],
        axis=1,
    )

    # DNN might be useful to know (3,4,5) as we filtered out (0,1,2)
    # 0: Counter not working properly
    # 1: Hit not part of any cluster
    # 2: Part of space cluster
    # 3: Passed rough time pattern recognition
    # 4: Part of the event (highest quality)
    # 5: Saturated counter
    data["status"] = hits["rufptn_.isgood"]
    # Might be useful to know how long is time trace
    data["nfold"] = hits["rufptn_.nfold"]
    # For DNN it might be easier to cut hidden space based on integer ids
    data["detector_ids"] = hits["rufptn_.xxyy"]
    # Devision of flattened array to events
    data["hit_offsets"] = hits["offsets"]
    data["hit_tt_offsets"] = hits["wf_offsets"]

    # Current suggestion is following detector_features:
    # arrival_times 2
    # pulse_area 2
    # detector_positions 3
    # status 1
    # nfold 1
    # detector_ids 1
    # --- Total 10 features

    # Combine vem values (VEM/count)
    vem = np.stack([hits["rufptn_.vem[0]"], hits["rufptn_.vem[1]"]], axis=1)
    # Reshape to the same shape as fadc
    vem = np.repeat(vem, hits["rufptn_.nfold"], axis=0)[:, :, None]
    # Convert FADC to VEMs
    data["time_traces"] = waveforms["rusdraw_.fadc"] / vem
    # Devision of flattened time_traces to events
    data["tt_offsets"] = waveforms["offsets"]


def parse_dst_file(
    dst_file,
    ntile=7,
    xmax_reader=None,
    avg_traces=True,
    add_shower_params=True,
    add_standard_recon=True,
    config=None,
):
    #  ntile - number of SD per one side
    import time

    start = time.time()
    dst_string = read_dst_file(dst_file)
    end = time.time()
    print(f"time0 = {end-start} sec")

    start = time.time()
    dst_data = parse_dst_string(dst_string)
    end = time.time()
    print(f"time1 = {end-start} sec")

    if dst_data is None:
        return None

    # Load xmax info for current dst file
    if xmax_reader is not None:
        xmax_reader.read_file(dst_file)

    # Dictionary with parsed data
    data = dict()

    start = time.time()
    if add_shower_params:
        data = shower_params(data, dst_data, xmax_reader)
    end = time.time()
    print(f"time_shp = {end-start} sec")

    start = time.time()
    if add_standard_recon:
        data = standard_recon(data, dst_data)

    end = time.time()
    print(f"time_recon = {end-start} sec")

    # for key in data:
    #     # if "std" in key:
    #     print(key, data[key].shape)

    # for key in dst_data["hits"]:
    #     print(key, dst_data["hits"][key].shape)

    # to_nsec = 4 * 1000
    to_nsec = 4002.7691424
    hits = dst_data["hits"]
    waveforms = dst_data["waveforms"]

    hit_mask = hits["rufptn_.isgood"] > 2
    hit_nfold = hits["rufptn_.nfold"]
    wf_mask = np.repeat(hit_mask, hit_nfold)

    hits_offsets = filter_offsets(hit_mask, hits["offsets"])
    waveforms_offsets = filter_offsets(wf_mask, waveforms["offsets"])

    hits = {key: value[hit_mask] for key, value in hits.items() if key != "offsets"}
    hits["offsets"] = hits_offsets
    hits["wf_offsets"] = np.concatenate(([0], np.cumsum(hits["rufptn_.nfold"])))

    # Example: get hits for the event:
    # start, end = hits["offsets"][ievent], hits["wf_offsets"][ievent + 1]
    # hits_arrv_time_lower = hits[rufptn_.reltime[0]]
    #

    # Example: get a waveform for specific hit:
    # ihit - global index in hits
    # start, end = hits["wf_offsets"][ihit], hits["wf_offsets"][ihit + 1]
    # waveforms_fadc = waveforms["rusdraw_.fadc"][start:end]

    waveforms = {
        key: value[wf_mask] for key, value in waveforms.items() if key != "offsets"
    }
    waveforms["offsets"] = waveforms_offsets

    print(waveforms["rusdraw_.xxyy"])
    # Probably, should be deleted if nothing caught in future
    assert np.all(
        np.repeat(hits["rufptn_.xxyy"], hits["rufptn_.nfold"])
        == waveforms["rusdraw_.xxyy"]
    ), "Hits are not in agreement with waverforms"

    arrival_times = (
        np.stack([hits["rufptn_.reltime[0]"], hits["rufptn_.reltime[1]"]], axis=1)
        * to_nsec
    )

    pulse_area = np.stack([hits["rufptn_.pulsa[0]"], hits["rufptn_.pulsa[1]"]], axis=1)

    detector_positions = np.stack(
        [
            hits["rufptn_.xyzclf[0]"],
            hits["rufptn_.xyzclf[1]"],
            hits["rufptn_.xyzclf[2]"],
        ],
        axis=1,
    )

    status = hits["rufptn_.isgood"]
    nfold = hits["rufptn_.nfold"]
    detector_ids = hits["rufptn_.xxyy"]

    print(detector_positions[0])

    # Combine vem
    vem = np.stack([hits["rufptn_.vem[0]"], hits["rufptn_.vem[1]"]], axis=1)
    # Reshape to the same shape as fadc
    vem = np.repeat(vem, hits["rufptn_.nfold"], axis=0)[:, :, None]
    time_traces = waveforms["rusdraw_.fadc"] / vem
    print(time_traces.dtype)

    # np.repeat(vem, hits["rufptn_.nfold"])
    # vem_repeat = np.repeat(vem, hits["rufptn_.nfold"], axis=0)
    # print(vem.shape, hits["rufptn_.nfold"].shape)
    # print(time_traces[0, 0])
    # print(waveforms["rusdraw_.fadc"][0, 0])

    # print(waveforms["offsets"][0:10].dtype)

    # print(np.diff(hits["wf_offsets"])[0:40])
    # wf = np.split(waveforms["rusdraw_.fadc"], hits["wf_offsets"][19 : 19 + 2])[1]
    # hh = np.split(hits["offsets"])

    # print(wf[:, 0, :].ravel().shape)
    # print(wf[:, 0, :])

    # for ii, w in enumerate(wf):
    #     print(ii, w.shape)

    # print(hits["wf_offsets"].shape)
    # print(hits["rufptn_.isgood"].shape)

    # nevt = len(hits["offsets"]) - 1
    # for i in nevt:
    #     st = hits["offsets"][i]
    #     end = hits["offsets"][i+1]
    #     print(hits["wf_offsets"].shape)

    # for key in waveforms:
    #     waveforms[key] = waveforms[key][wf_mask]

    # new_offsets = filter_offsets(hits["rufptn_.isgood"] > 2, hits["offsets"])
    # print("old_off", hits["offsets"])
    # print("new_off", new_offsets)
    # print("new_shape", hits["rufptn_.isgood"][hits["rufptn_.isgood"] > 2].shape)

    # arrival_times = (
    #     np.stack([hits["rufptn_.reltime[0]"], hits["rufptn_.reltime[1]"]], axis=1)
    #     * to_nsec
    # )

    # print(arrival_times.shape)
    # pulse_area = np.stack([hits["rufptn_.pulsa[0]"], hits["rufptn_.pulsa[1]"]], axis=1)
    # detector_positions = np.stack(
    #     [
    #         hits["rufptn_.xyzclf[0]"],
    #         hits["rufptn_.xyzclf[1]"],
    #         hits["rufptn_.xyzclf[2]"],
    #     ],
    #     axis=1,
    # )
    # nfold = hits["rufptn_.nfold"]
    # detector_ids = hits["rufptn_.xxyy"]
    # isgood = hits["rufptn_.isgood"]

    # # print(isgood.shape, np.sum(isgood > 1), np.sum(isgood > 0), np.sum(isgood == 0))
    # # print(hits["rufptn_.xxyy"][isgood == 0])

    # waveforms = dst_data["waveforms"]

    # print("ursdraw_.xxyy", waveforms["rusdraw_.xxyy"])
    # nevt = len(waveforms["offsets"]) - 1
    # for i in range(nevt):
    #     s = waveforms["offsets"][i]
    #     e = waveforms["offsets"][i + 1]

    #     sh = hits["offsets"][i]
    #     eh = hits["offsets"][i + 1]
    #     print(
    #         "w",
    #         i,
    #         waveforms["rusdraw_.xxyy"][s:e],
    #         waveforms["rusdraw_.xxyy"][s:e].shape,
    #     )

    #     print("h", i, hits["rufptn_.xxyy"][sh:eh], hits["rufptn_.xxyy"][sh:eh].shape)
    #     print(
    #         "f",
    #         i,
    #         hits["rufptn_.nfold"][sh:eh],
    #         np.cumsum(hits["rufptn_.nfold"][sh:eh]),
    #     )
    #     print("g", i, hits["rufptn_.isgood"][sh:eh])

    # badsd = dst_data["badsd"]
    # print(np.unique(badsd["bsdinfo_.xxyyout[x]"]))
    # print(hits["rufptn_.xxyy"])
    # # print(badsd["offsets"])
    # # print(hits["offsets"])

    # print(
    #     "npnpn= ", np.sum(np.isin(hits["rufptn_.xxyy"], badsd["bsdinfo_.xxyyout[x]"]))
    # )

    # nevents = len(hits["offsets"]) - 1
    # for ievt in range(nevents):
    #     hit_start = hits["offsets"][ievt]
    #     hit_end = hits["offsets"][ievt + 1]
    #     hit_detectors = hits["rufptn_.xxyy"][hit_start:hit_end]

    #     bsd_start = badsd["offsets"][ievt]
    #     bsd_end = badsd["offsets"][ievt + 1]
    #     bad_detectors = badsd["bsdinfo_.xxyyout[x]"][bsd_start:bsd_end]

    #     # print("hit_det", hit_detectors)
    #     # print("bad_det", bad_detectors)
    #     print(ievt, np.sum(np.isin(hit_detectors, bad_detectors)))

    # print(ievt, hit_detectors)

    # print(data["energy"].shape)
    # for key in dst_data["waveforms"]:
    #     print(key, dst_data["waveforms"][key].shape)

    # data = detector_readings(data, dst_lists, ntile, avg_traces)

    if (config is not None) and (hasattr(config, "add_event_ids")):
        data = config.add_event_ids(data, dst_file)
    return data
