import numpy as np


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

    for line in dst_string:
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


def parse_event(event_list_str):
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

    event_list = [np.fromstring(line, sep=" ") for line in event_list_str]
    event_list = np.array(event_list).astype(np.float32).transpose()
    return event_list


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


def parse_dst_string(dst_string):

    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = (
        dst_sections(dst_string)
    )

    if len(event_list_str) == 0:
        return None

    event_list = parse_event(event_list_str)
    sdmeta_list = parse_sdmeta(sdmeta_list_str)
    sdwaveform_list = parse_sdwaveform(sdwaveform_list_str)
    badsdinfo_list = parse_badsdinfo(badsdinfo_list_str)

    return event_list, sdmeta_list, sdwaveform_list, badsdinfo_list
