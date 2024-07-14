import numpy as np
from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
import dstparser.tasd_clf as tasd_clf


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
    # number of SDs in space-time cluster
    data["std_recon_nsd"] = event_list[9]
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
    # 3-d unit vector of the arrival direction (pointing back to the source)
    data["std_recon_shower_axis"] = np.array(
        [
            np.sin(np.deg2rad(event_list[22]))
            * np.cos(np.deg2rad(event_list[23]) + np.pi),
            np.sin(np.deg2rad(event_list[22]))
            * np.sin(np.deg2rad(event_list[23]) + np.pi),
            np.cos(np.deg2rad(event_list[22])),
        ],
        dtype=np.float32,
    ).transpose()
    # uncertainty of the pointing direction [degree]
    # event_list[22] is zenith angle in deg
    # event_list[24] is uncertainty zenith angle in deg
    # event_list[25] is uncertainty azimuth angle in deg
    data["std_recon_shower_axis_err"] = np.sqrt(
        event_list[24] * event_list[24]
        + np.sin(np.deg2rad(event_list[22]))
        * np.sin(np.deg2rad(event_list[22]))
        * event_list[25]
        * event_list[25]
    )
    # chi-square of the geometry fit
    data["std_recon_geom_chi2"] = event_list[26]
    # the number of degree of freedom of the geometry fit (= n - 5),
    # where "n" is the number of the SDs used for the geometry fit
    data["std_recon_geom_ndof"] = event_list[27]
    # distance b/w the reconstructed core and the edge from the TA SD array [in 1,200 meter unit]
    # negative for events with the core outside of the TA SD array.
    data["std_recon_border_distance"] = event_list[30]
    # distance to the T-shape TA SD array, edge of the sub-arrays [in 1,200 meter unit]
    # this value is used as "border_distance" before implementation of the boundary trigger (on 2008/11/11)
    data["std_recon_border_distance_tshape"] = event_list[31]

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
            data[key][ievt][:2] = (data[key][ievt][:2] - tile_center[:2]) / tile_extent
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
    data["detector_states"] = np.zeros(shape, dtype=bool)
    data["detector_exists"] = np.zeros(shape, dtype=bool)
    data["detector_good"] = np.zeros(shape, dtype=bool)

    if avg_traces:
        data["arrival_times"] = np.zeros(shape, dtype=np.float32)
        data["time_traces"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
    else:
        data["arrival_times_low"] = np.zeros(shape, dtype=np.float32)
        data["arrival_times_up"] = np.zeros(shape, dtype=np.float32)
        data["time_traces_low"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
        data["time_traces_up"] = np.zeros((*shape, ntime_trace), dtype=np.float32)

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

        if avg_traces:
            atimes = (event[2] + event[3]) / 2
            data["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

            ttrace = (
                wform[:ntime_trace] / fadc_per_vem_low
                + wform[ntime_trace:] / fadc_per_vem_up
            ) / 2
            data["time_traces"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()
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


def parse_dst_file(
    dst_file,
    ntile=7,
    xmax_reader=None,
    avg_traces=True,
    add_shower_params=True,
    add_standard_recon=True,
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

    data = detector_readings(data, dst_lists, ntile, avg_traces)
    return data
