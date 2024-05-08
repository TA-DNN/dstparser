import json
import numpy as np
from dstparser.dst_reader import read_dst_file, read_xmax_data
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


def shower_params(data, dst_lists, xmax_data):
    # Shower related
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


def parse_dst_file(dst_file, meta_data, ntile=7, up_low_traces=False):
    #  ntile  # number of SD per one side
    dst_string = read_dst_file(dst_file)
    dst_lists = parse_dst_string(dst_string)

    if dst_lists is None:
        return None

    xmax_data = read_xmax_data(dst_file)

    # Dictionary with parsed data
    data = dict()
    data = fill_metadata(data, dst_file, meta_data)
    data = shower_params(data, dst_lists, xmax_data)
    data = detector_readings(data, dst_lists, ntile, up_low_traces)

    return data
