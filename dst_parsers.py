import numpy as np
from utils import tile_positions
import json
from dst_content import dst_content


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


def CORSIKAparticleID2mass(corsikaPID):
    return np.where(corsikaPID == 14, 1, corsikaPID // 100).astype(np.int32)


def shower_params(event_list_str, data, dst_file, xmax_reader):
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


    try:
        event_list = np.array(event_list).astype(np.float32).transpose()
        data["mass_number"] = CORSIKAparticleID2mass(event_list[0])
    except Exception as ex:
        print(event_list_str)
        input()
    
    data["energy"] = event_list[1]
    data["xmax"] = xmax_reader(dst_file, data["energy"])
    data["shower_axis"] = np.array(
        [
            np.sin(event_list[2]) * np.cos(event_list[3] + np.pi),
            np.sin(event_list[2]) * np.sin(event_list[3] + np.pi),
            np.cos(event_list[2]),
        ],
        dtype=np.float32,
    ).transpose()

    # ?? should it be float32
    data["shower_core"] = np.array(
        event_list[4:7, :].transpose() / 100, dtype=np.float32
    )

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
    data["detector_states"] = np.zeros(shape, dtype=bool)
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
    # center around detector with max signal
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)

    # ix and iy as one array [ix, iy]
    ixy = np.array([event[0] // 100, event[0] % 100]).astype(np.int32)
    # Indicies of central detector ix0, iy0
    ixy0 = np.copy(ixy[:, max_signal_idx]) - (ntile - 1) // 2
    ixy -= ixy0[:, np.newaxis]
    # cut array size to fit the tile size
    inside_tile = (ixy[0] < ntile) & (ixy[1] < ntile)
    ixy = ixy[:, inside_tile]
    return ixy0, inside_tile, ixy


def detector_readings(
    sdmeta_list, sdwaveform_list, badsdinfo_list, ntile, ntime_trace, data
):
    to_nsec = 4 * 1000
    num_events = data["mass_number"].shape[0]
    data = init_detector_readings(num_events, ntile, ntime_trace, data)

    empty_events = []

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
        # relative time of first arrived particle
        atimes -= np.min(atimes)
        data["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

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
            data["shower_core"][ievt][:],
        ) = tile_positions(ixy0, ntile, badsd, shower_core)

        data["arrival_times"][ievt, :, :] = np.where(
            data["detector_states"][ievt, :, :], data["arrival_times"][ievt, :, :], 0
        )

    return data, empty_events


def parse_dst_file(dst_file, xmax_reader):
    dst_string = dst_content(dst_file)

    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = (
        dst_sections(dst_string)
    )
    
    if len(event_list_str) == 0:
        return None

    sdmeta_list = parse_sdmeta(sdmeta_list_str)
    sdwaveform_list = parse_sdwaveform(sdwaveform_list_str)
    badsdinfo_list = parse_badsdinfo(badsdinfo_list_str)

    # Dictionary with parsed data
    data = dict()
    data = fill_metadata(data, dst_file)
    data = shower_params(event_list_str, data, dst_file, xmax_reader)

    ntile = 7  # number of SD per one side
    ntime_trace = 128  # number of time trace of waveform
    data, empty_events = detector_readings(
        sdmeta_list, sdwaveform_list, badsdinfo_list, ntile, ntime_trace, data
    )

    # Norm for xy detector positions
    # norm_xy = 1200 * ((ntile - 1) // 2)

    if len(empty_events) != 0:
        # Remove empty events
        for key, value in data.items():
            if key == "metadata":
                continue
            data[key] = np.delete(value, empty_events, axis=0)

    return data
