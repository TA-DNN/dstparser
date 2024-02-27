import numpy as np
from utils import rufptn_xxyy2sds
from utils import tile_positions
import json


def fill_metadata(data, dst_file):
    meta_data = dict()
    meta_data["interaction_model"] = "QGSJET-II-03"
    meta_data["atmosphere_model"] = None
    meta_data["emin"] = None
    meta_data["emax"] = None
    meta_data["espectrum"] = "HiRes"
    meta_data["dst_file_name"] = dst_file

    data["metadata"] = json.dumps(meta_data, indent=4)
    return data


def dst_sections(dst_string):
    ## make lists
    event_readout = False
    sdmeta_readout = False
    sdwaveform_readout = False
    badsdinfo_readout = False
    event_list_str = []
    sdmeta_list_str = []
    sdwaveform_list_str = []
    badsdinfo_list_str = []

    for il, line in enumerate(dst_string):
        if "EVENT DATA" in line:
            event_readout = True
            sdmeta_readout = False
            sdwaveform_readout = False
            badsdinfo_readout = False
            continue
        elif "SD meta DATA" in line:
            event_readout = False
            sdmeta_readout = True
            sdwaveform_readout = False
            badsdinfo_readout = False
            continue
        elif "SD waveform DATA" in line:
            event_readout = False
            sdmeta_readout = False
            sdwaveform_readout = True
            badsdinfo_readout = False
            continue
        elif "badsdinfo" in line:
            event_readout = False
            sdmeta_readout = False
            sdwaveform_readout = False
            badsdinfo_readout = True
            continue
        if event_readout:
            event_list_str.append(line)
        elif sdmeta_readout:
            sdmeta_list_str.append(line)
        elif sdwaveform_readout:
            sdwaveform_list_str.append(line)
        elif badsdinfo_readout:
            badsdinfo_list_str.append(line)

    return event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str


def shower_params(event_list_str, data):
    # Shower related
    event_list = [[float(c) for c in l.split(" ") if c != ""] for l in event_list_str]
    """
    event_format = ["mass_number",
                    "rusdmc_energy",
                    "rusdmc_theta",
                    "rusdmc_phi",
                    "rusdmc_corexyz[0]",
                    "rusdmc_corexyz[1]",
                    "rusdmc_corexyz[2]",
                    "rusdraw_yymmdd",
                    "rusdraw_hhmmss",
                    "rufptn_nstclust",
                    "rusdraw_nofwf"]
    """

    event_list = np.array(event_list).astype(np.float32).transpose()
    data["mass_number"] = event_list[0].astype(np.int32)
    data["energy"] = event_list[1]
    data["xmax"] = np.zeros(event_list.shape[1], dtype=np.float32)
    data["shower_axis"] = np.array(
        [
            np.sin(event_list[2]) * np.cos(event_list[3]),
            np.sin(event_list[2]) * np.sin(event_list[3]),
            np.cos(event_list[2]),
        ],
        dtype=np.float32,
    ).transpose()

    # ?? should it be float32
    data["shower_core"] = np.array(event_list[4:7, :].transpose(), dtype=np.int32)

    return data


def parse_sdmeta(sdmeta_list_str):
    """
    sdmeta_format = ["rufptn_.xxyy",
                     "rufptn_.isgood",
                     "rufptn_.reltime[0]",
                     "rufptn_.reltime[1]",
                     "rufptn_.pulsa[0]",
                     "rufptn_.pulsa[1]",
                     "rufptn_.xyzclf[0]",
                     "rufptn_.xyzclf[1]",
    "rufptn_.xyzclf[2]",
    "rufptn_.vem[0]",
    "rufptn_.vem[1]"]
    sdmeta_dict = [
        [
            {
                sdmeta_format[k]: int(sdmeta_list[i][j][k]) if sdmeta_format[k] in ["rufptn_.xxyy", "rufptn_.isgood"] else float(sdmeta_list[i][j][k])
                for k in range(len(sdmeta_format))
            }
            for j in range(len(sdmeta_list[i]))
        ]
        for i in range(len(sdmeta_list))
    ]
    """

    ## Detection related

    record_size = 11
    sdmeta_list = [
        np.fromstring(line, sep=" ").reshape(-1, record_size).transpose()
        for line in sdmeta_list_str
    ]

    return sdmeta_list


def parse_sdwaveform(sdwaveform_list_str):
    ## sd waveform data
    """
    sdwaveform_format = ["rusdraw_.xxyy",
                        "rusdraw_.clkcnt",
                        "rusdraw_.mclkcnt",
                        "rusdraw_.fadc[0]",
                        "rusdraw_.fadc[1]"]
    sdwaveform_dict = [
        [
            {
                sdwaveform_format[0]: int(sdwaveform_list[i][j][0]),
                sdwaveform_format[1]: int(sdwaveform_list[i][j][1]),
                sdwaveform_format[2]: int(sdwaveform_list[i][j][2]),
                #f"{sdwaveform_format[3]}[{k}]": int(sdwaveform_list[i][j][3 + k]),
                #f"{sdwaveform_format[4]}[{k}]": int(sdwaveform_list[i][j][3 + 128 + k])
                sdwaveform_format[3]: [int(sdwaveform_list[i][j][3 + k]) for k in range(128)],
                sdwaveform_format[4]: [int(sdwaveform_list[i][j][3 + 128 + k]) for k in range(128)]
            }
            for j in range(len(sdwaveform_list[i]))
            #for k in range(128)
        ]
        for i in range(len(sdwaveform_list))
    ]
    """

    record_size = 3 + 128 * 2
    sdwaveform_list = [
        np.fromstring(line, sep=" ", dtype=np.int32)
        .reshape(-1, record_size)
        .transpose()
        for line in sdwaveform_list_str
    ]

    return sdwaveform_list


def parse_badsdinfo(badsdinfo_list_str):
    """
    Int_t nsdsout;         // number of SDs either completely out (absent in the live detector list during event)
    vector<Int_t> xxyyout; // SDs that are completely out (can't participate in event readout)
    vector<Int_t> bitfout;
    """

    badsdinfo_list = []
    for line in badsdinfo_list_str:
        mixed_array = np.fromstring(line, sep=" ", dtype=np.int32)
        # xxyyout = mixed_array[::2]
        # bitfout = mixed_array[1::2]
        badsdinfo_list.append(mixed_array[::2])

    return badsdinfo_list


def init_detector_readings(num_events, ntile, ntime_trace, data):
    # Put largest-signal SD at the center of nTile x nTile grids
    shape = num_events, ntile, ntile
    data["arrival_times"] = np.zeros(shape, dtype=np.float32)
    data["time_traces"] = np.zeros((*shape, ntime_trace), dtype=np.float32)
    data["detector_positions"] = np.zeros((*shape, 3), dtype=np.float32)
    data["detector_states"] = np.ones(shape, dtype=bool)
    return data


def cut_events(event, wform, min_ndet):
    # Events with > 2 detectors
    event = event[:, event[1] > min_ndet]

    # Pick corresponding waveforms
    wform_idx = []
    for xycoord in event[0].astype(np.int32):
        # Take only the first waveform (second [0])
        wform_idx.append(np.where(wform[0] == xycoord)[0][0])

    wform = wform[3:, wform_idx]
    return event, wform


def center_tile(event, tile_size):
    # center around detector with max signal
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)

    # ix and iy as one array [ix, iy]
    ixy = np.array([event[0] // 100, event[0] % 100]).astype(np.int32)
    # Indicies of central detector ix0, iy0
    ixy0 = np.copy(ixy[:, max_signal_idx])
    ixy -= ixy0[:, np.newaxis]
    # cut array size to fit the tile size
    inside_tile = (abs(ixy[0]) < tile_size) & (abs(ixy[1]) < tile_size)
    ixy = ixy[:, inside_tile]
    return ixy0, inside_tile, ixy


def detector_readings(
    sdmeta_list, sdwaveform_list, badsdinfo_list, ntile, ntime_trace, data
):
    to_nsec = 4 * 1000
    num_events = data["mass_number"].shape[0]
    data = init_detector_readings(num_events, ntile, ntime_trace, data)
    tile_size = (ntile - 1) / 2 + 1

    for ievt, (event, wform, badsd) in enumerate(
        zip(sdmeta_list, sdwaveform_list, badsdinfo_list)
    ):
        event, wform = cut_events(event, wform, 2)
        ixy0, inside_tile, ixy = center_tile(event, tile_size)

        # averaged arrival times
        atimes = (event[2] + event[3]) / 2
        # relative time of first arrived particle
        atimes -= np.min(atimes)
        data["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

        ttrace = (wform[:ntime_trace] / event[9] + wform[ntime_trace:] / event[10]) / 2
        data["time_traces"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()

        # Return detector coordinates of the tile centered in ixy0
        data["detector_positions"][ievt, :, :], data["detector_states"][ievt, :, :] = (
            tile_positions(ixy0, ntile, badsd)
        )

    return data


def detector_readings_orig(sdmeta_list, sdwaveform_list, detector_tile):

    nTile = detector_tile["arrival_times"].shape[1]

    for i in range(len(sdmeta_list)):
        # if i>0:
        #    continue
        signalMax_xx = 0
        signalMax_yy = 0
        signalMax_size = 0
        firstTime = 10**8
        for j in range(len(sdmeta_list[i])):
            if sdmeta_list[i][j][1] <= 2:
                continue  ## exclude coincidence signals
            xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
            yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
            signal_size = (sdmeta_list[i][j][4] + sdmeta_list[i][j][5]) / 2
            if (sdmeta_list[i][j][2] + sdmeta_list[i][j][3]) / 2 < firstTime:
                firstTime = (sdmeta_list[i][j][2] + sdmeta_list[i][j][3]) / 2
            if signal_size > signalMax_size:
                signalMax_xx = xx
                signalMax_yy = yy
                signalMax_size = signal_size
                center_j = j
            # print("##",xx,yy,signal_size,signalMax_xx,signalMax_yy,signalMax_size)
        for j in range(len(sdmeta_list[i])):
            if sdmeta_list[i][j][1] <= 2:
                continue  ## exclude coincidence signals
            xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
            yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
            xGrid = int(-signalMax_xx + xx + (nTile - 1) / 2)
            yGrid = int(-signalMax_yy + yy + (nTile - 1) / 2)
            # print(xx,yy,xGrid,yGrid)
            if xGrid >= 0 and xGrid < nTile and yGrid >= 0 and yGrid < nTile:
                detector_tile["arrival_times"][i][xGrid][yGrid] = (
                    ((sdmeta_list[i][j][2] + sdmeta_list[i][j][3]) / 2 - firstTime)
                    * 4
                    * 1000
                )  # nsec
                fadc_low = np.array(
                    next(
                        item[3 : 3 + 128]
                        for item in sdwaveform_list[i]
                        if item[0] == sdmeta_list[i][j][0]
                    )
                )
                fadc_up = np.array(
                    next(
                        item[3 + 128 :]
                        for item in sdwaveform_list[i]
                        if item[0] == sdmeta_list[i][j][0]
                    )
                )
                detector_tile["time_traces"][i][xGrid][yGrid][:] = (
                    fadc_low / sdmeta_list[i][j][9] + fadc_up / sdmeta_list[i][j][10]
                ) / 2  ## average of lower & upper FADC signal
                # detector_positions[i][xGrid][yGrid][0] = 1.2 * ((sdmeta_list[i][j][6]-12.2435) - (sdmeta_list[i][center_j][6]-12.2435)) * 1000 # meter
                # detector_positions[i][xGrid][yGrid][1] = 1.2 * ((sdmeta_list[i][j][7]-16.4406) - (sdmeta_list[i][center_j][7]-16.4406)) * 1000 # meter
                # detector_positions[i][xGrid][yGrid][2] = 1.2 * (sdmeta_list[i][j][8] - sdmeta_list[i][center_j][8]) * 1000 # meter
                sd_clf = rufptn_xxyy2sds(int(sdmeta_list[i][j][0])) / 100  # meter
                for ii in range(3):
                    detector_tile["detector_positions"][i][xGrid][yGrid][ii] = sd_clf[
                        ii + 1
                    ]
                detector_tile["detector_states"][i][xGrid][yGrid] = True

    return detector_tile
