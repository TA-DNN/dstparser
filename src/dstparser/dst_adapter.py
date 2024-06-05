import json
import numpy as np
from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
import dstparser.tasd_clf as tasd_clf


def default_metadata():
    meta_data = dict()
    meta_data["interaction_model"] = "QGSJET-II-03"
    meta_data["atmosphere_model"] = None
    meta_data["emin"] = None
    meta_data["emax"] = None
    meta_data["espectrum"] = "HiRes"


def fill_metadata(data, dst_file, meta_data=None):
    if meta_data is None:
        meta_data = default_metadata()
    meta_data["dst_file_name"] = dst_file

    data["metadata"] = json.dumps(meta_data, indent=4)
    return data


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


def shower_params(data, dst_lists, xmax_data, add_standard_recon):
    # Shower related
    # for details: /ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/sditerator/src/sditerator_cppanalysis.cpp

    to_meters = 1e-2
    event_list = dst_lists[0]
    data["mass_number"] = corsika_id2mass(event_list[0])
    data["energy"] = event_list[1]
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

    if add_standard_recon:
        # dictionary for standard-reconstructed values
        data["std_recon"] = dict()
        # number of SDs in sapce-time cluster
        data["std_recon"]["nstclust"] = event_list[9]
        # energy reconstructed by the standard energy estimation table [EeV]
        data["std_recon"]["energy_rec"] = event_list[12]
        # reconstructed scale of the Lateral Distribution Function (LDF) fit [VEM m-2]
        data["std_recon"]["LDF_scale_rec"] = event_list[13]
        # uncertainty of the scale [VEM m-2]
        data["std_recon"]["d_LDF_scale_rec"] = event_list[14]
        # chi-square of the LDF fit
        data["std_recon"]["chi2_LDF"] = event_list[15]
        # the number of degree of freedom of the LDF fit (= n - 3),
        # where "n" is the number of the SDs used for the LDF fit
        data["std_recon"]["ndof_LDF"] = event_list[16]
        # core position (x, y) reconstructed by the LDF fit in CLF coordinate [m]
        data["std_recon"]["shower_core_rec"] = np.array(
            [
                rec_coreposition_to_CLF_meters(event_list[17], option="x"),
                rec_coreposition_to_CLF_meters(event_list[19], option="y"),
            ]
        )
        # uncertainty of the core position (x, y) reconstructed by the LDF fit
        # data["std_recon"]["d_shower_core_rec"] = np.array(
        #     [
        #         rec_coreposition_to_CLF_meters(event_list[18],
        #                                       option = "dx"),
        #         rec_coreposition_to_CLF_meters(event_list[20],
        #                                       option = "dy")
        #     ]
        # )
        # S800 (particle density at 800 m from the shower axis) [VEM m-2]
        data["std_recon"]["s800_rec"] = event_list[21]
        # 3-d unit vector of the arrival direction (pointing back to the source)
        data["std_recon"]["shower_axis_rec"] = np.array(
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
        data["std_recon"]["point_dir_err"] = np.sqrt(
            event_list[24] * event_list[24]
            + np.sin(np.deg2rad(event_list[22]))
            * np.sin(np.deg2rad(event_list[22]))
            * event_list[25]
            * event_list[25]
        )
        # chi-square of the geometry fit
        data["std_recon"]["chi2_geom"] = event_list[26]
        # the number of degree of freedom of the geometry fit (= n - 5),
        # where "n" is the number of the SDs used for the geometry fit
        data["std_recon"]["ndof_geom"] = event_list[27]
        # distance b/w the reconstructed core and the edge from the TA SD array [in 1,200 meter unit]
        # negative for events with the core outside of the TA SD array.
        data["std_recon"]["border_distance"] = event_list[30]
        # distance to the T-shape TA SD array, edge of the sub-arrays [in 1,200 meter unit]
        # this value is used as "border_distance" before implementation of the boundary trigger (on 2008/11/11)
        data["std_recon"]["border_distance_Tshape"] = event_list[31]
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


def tile_normalization(abs_coord, do_exist, shower_core):
    detector_dist = 1200  # meters
    height_of_clf = 1370  # meters
    height_scatter = 30  # meters, +-30 from average

    # Normalization of a tile for DNN
    n0 = (abs_coord.shape[0] - 1) // 2
    tile_center = np.copy(abs_coord[n0, n0])
    # Shift to the hight of CLF (z)
    tile_center[2] = height_of_clf

    # Shift shower core
    shower_core[:2] = shower_core[:2] - tile_center[:2]

    tile_center = tile_center[np.newaxis, np.newaxis, :]
    rel_coord = np.where(do_exist[:, :, np.newaxis], abs_coord - tile_center, 0)

    # xy coordinate normalization
    tile_extent = n0 * detector_dist  # extent of tile
    rel_coord[:, :, 0:2] = rel_coord[:, :, 0:2] / tile_extent
    shower_core[:2] = shower_core[:2] / tile_extent
    # z coordinate normalization
    height_extent = height_scatter
    rel_coord[:, :, 2] = rel_coord[:, :, 2] / height_extent
    shower_core[2] = shower_core[2] / height_extent

    return rel_coord, shower_core


def tile_positions(ixy0, tile_size, badsd, shower_core):
    # Create centered tile
    # n0 = (tile_size - 1) / 2
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

    abs_coord = tasd_clf.tasdmc_clf[tasdmc_clf_indices, 1:] / 1e2
    rel_coord, rel_shower_core = tile_normalization(abs_coord, do_exist, shower_core)

    return rel_coord, status, do_exist, good, rel_shower_core


def detector_readings(data, dst_lists, ntile, up_low_traces):
    ntime_trace = 128  # number of time trace of waveform
    to_nsec = 4 * 1000

    num_events = data["mass_number"].shape[0]
    shape = num_events, ntile, ntile
    data["arrival_times"] = np.zeros(shape, dtype=np.float32)
    data["time_traces"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
    data["detector_positions"] = np.zeros((*shape, 3), dtype=np.float32)
    data["detector_states"] = np.zeros(shape, dtype=bool)
    data["detector_exists"] = np.zeros(shape, dtype=bool)
    data["detector_good"] = np.zeros(shape, dtype=bool)

    if up_low_traces:
        data["time_traces_low"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
        data["time_traces_up"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
        data["arrival_times_low"] = np.zeros(shape, dtype=np.float32)
        data["arrival_times_up"] = np.zeros(shape, dtype=np.float32)

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
        wform = wform[:, inside_tile]
        fadc_per_vem_low = event[9][inside_tile]
        fadc_per_vem_up = event[10][inside_tile]

        # averaged arrival times
        atimes = (event[2] + event[3]) / 2
        # # relative time of first arrived particle
        # atimes -= np.min(atimes)
        data["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

        if up_low_traces:
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

        ttrace = (
            wform[:ntime_trace] / fadc_per_vem_low
            + wform[ntime_trace:] / fadc_per_vem_up
        ) / 2
        data["time_traces"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()

        shower_core = data["shower_core"][ievt]
        # Return detector coordinates of the tile centered in ixy0
        (
            data["detector_positions"][ievt, :, :],
            data["detector_states"][ievt, :, :],
            data["detector_exists"][ievt, :, :],
            data["detector_good"][ievt, :, :],
            data["shower_core"][ievt][:],
        ) = tile_positions(ixy0, ntile, badsd, shower_core)

        data["arrival_times"][ievt, :, :] = np.where(
            data["detector_states"][ievt, :, :], data["arrival_times"][ievt, :, :], 0
        )

        if up_low_traces:
            for arrv_array_name in ["arrival_times_low", "arrival_times_up"]:
                data[arrv_array_name][ievt, :, :] = np.where(
                    data["detector_states"][ievt, :, :],
                    data[arrv_array_name][ievt, :, :],
                    0,
                )

    # Remove empty events
    if len(empty_events) != 0:
        for key, value in data.items():
            if key == "metadata":
                continue
            data[key] = np.delete(value, empty_events, axis=0)

    return data


def parse_dst_file(
    dst_file,
    xmax_reader,
    meta_data,
    ntile=7,
    up_low_traces=False,
    add_standard_recon=False,
):
    #  ntile  # number of SD per one side
    dst_string = read_dst_file(dst_file, add_standard_recon)
    dst_lists = parse_dst_string(dst_string)

    if dst_lists is None:
        return None

    # Load xmax info for current dst file
    xmax_reader.read_file(dst_file)
    # Dictionary with parsed data
    data = dict()
    data = fill_metadata(data, dst_file, meta_data)
    data = shower_params(data, dst_lists, xmax_reader, add_standard_recon)
    data = detector_readings(data, dst_lists, ntile, up_low_traces)

    return data
