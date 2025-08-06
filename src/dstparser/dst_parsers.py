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

    if len(event_list_str) - len(badsdinfo_list_str) == 1:
        badsdinfo_list_str.append("")

    assert (
        len(event_list_str)
        == len(sdmeta_list_str)
        == len(sdwaveform_list_str)
        == len(badsdinfo_list_str)
    ), "lists are of different sizes"

    return event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str


def parse_event(event_list_str):
    """
    # In source code:
    # 0 rusdmc_.parttype,
    # 1 rusdmc_.energy,
    # 2 rusdmc_.theta,
    # 3 rusdmc_.phi,
    # 4 rusdmc_.corexyz[0],
    # 5 rusdmc_.corexyz[1],
    # 6 rusdmc_.corexyz[2],
    # 7 rusdraw_.yymmdd,
    # 8 rusdraw_.hhmmss,
    # 9 rufptn_.nstclust,
    # 10 rusdraw_.nofwf,
    # 11 rusdraw_.usec
    # 12-16 rufldf_.energy[0], rufldf_.sc[0], rufldf_.dsc[0],
    #       rufldf_.chi2[0], rufldf_.ndof[0],
    # 17-21 rufldf_.xcore[0], rufldf_.dxcore[0], rufldf_.ycore[0],
    #       rufldf_.dycore[0], rufldf_.s800[0],
    # 22-29 rusdgeom_.theta[1], rusdgeom_.phi[1], rusdgeom_.dtheta[1],
    #       rusdgeom_.dphi[1], rusdgeom_.chi2[1], rusdgeom_.ndof[1],
    #       rusdgeom_.t0[1], rusdgeom_.dt0[1]
    # 30-31 rufldf_.bdist, rufldf_.tdist,
    # 32-41 rusdgeom_.theta[2], rusdgeom_.phi[2], rusdgeom_.dtheta[2],
    #       rusdgeom_.dphi[2], rusdgeom_.chi2[2], rusdgeom_.ndof[2],
    #       rusdgeom_.t0[2], rusdgeom_.dt0[2], rusdgeom_.a,
    #       rusdgeom_.da
    # 42-46 rufldf_.energy[0], rufldf_.sc[0], rufldf_.dsc[0],
    #       rufldf_.chi2[0], rufldf_.ndof[0],
    # 47-51 rufldf_.xcore[1], rufldf_.dxcore[1], rufldf_.ycore[1], rufldf_.dycore[1], rufldf_.s800[1],
    # 52-55 rufldf_.theta, rufldf_.phi, rufldf_.dtheta, rufldf_.dphi
    # 56-60 rufptn_.nhits, rufptn_.nsclust, rufptn_.nborder, rufptn_.qtot[0], rufptn_.qtot[1]
    """

    # print(f"event_list_str {len(event_list_str)} lines")

    event_list = [
        np.fromstring(line, sep=" ", dtype=np.float64) for line in event_list_str
    ]
    evt = np.array(event_list).transpose()

    events = {
        "rusdmc_.parttype": evt[0],
        "rusdmc_.energy": evt[1],
        "rusdmc_.theta": evt[2],
        "rusdmc_.phi": evt[3],
        "rusdmc_.corexyz[0]": evt[4],
        "rusdmc_.corexyz[1]": evt[5],
        "rusdmc_.corexyz[2]": evt[6],
        "rusdraw_.yymmdd": evt[7],
        "rusdraw_.hhmmss": evt[8],
        "rufptn_.nstclust": evt[9],
        "rusdraw_.nofwf": evt[10],
        "rusdraw_.usec": evt[11],
        "rufldf_.energy[0]": evt[12],
        "rufldf_.sc[0]": evt[13],
        "rufldf_.dsc[0]": evt[14],
        "rufldf_.chi2[0]": evt[15],
        "rufldf_.ndof[0]": evt[16],
        "rufldf_.xcore[0]": evt[17],
        "rufldf_.dxcore[0]": evt[18],
        "rufldf_.ycore[0]": evt[19],
        "rufldf_.dycore[0]": evt[20],
        "rufldf_.s800[0]": evt[21],
        "rusdgeom_.theta[1]": evt[22],
        "rusdgeom_.phi[1]": evt[23],
        "rusdgeom_.dtheta[1]": evt[24],
        "rusdgeom_.dphi[1]": evt[25],
        "rusdgeom_.chi2[1]": evt[26],
        "rusdgeom_.ndof[1]": evt[27],
        "rusdgeom_.t0[1]": evt[28],
        "rusdgeom_.dt0[1]": evt[29],
        "rufldf_.bdist": evt[30],
        "rufldf_.tdist": evt[31],
        "rusdgeom_.theta[2]": evt[32],
        "rusdgeom_.phi[2]": evt[33],
        "rusdgeom_.dtheta[2]": evt[34],
        "rusdgeom_.dphi[2]": evt[35],
        "rusdgeom_.chi2[2]": evt[36],
        "rusdgeom_.ndof[2]": evt[37],
        "rusdgeom_.t0[2]": evt[38],
        "rusdgeom_.dt0[2]": evt[39],
        "rusdgeom_.a": evt[40],
        "rusdgeom_.da": evt[41],
        "rufldf_.energy[1]": evt[42],
        "rufldf_.sc[1]": evt[43],
        "rufldf_.dsc[1]": evt[44],
        "rufldf_.chi2[1]": evt[45],
        "rufldf_.ndof[1]": evt[46],
        "rufldf_.xcore[1]": evt[47],
        "rufldf_.dxcore[1]": evt[48],
        "rufldf_.ycore[1]": evt[49],
        "rufldf_.dycore[1]": evt[50],
        "rufldf_.s800[1]": evt[51],
        "rufldf_.theta": evt[52],
        "rufldf_.phi": evt[53],
        "rufldf_.dtheta": evt[54],
        "rufldf_.dphi": evt[55],
        "rufptn_.nhits": evt[56],
        "rufptn_.nsclust": evt[57],
        "rufptn_.nborder": evt[58],
        "rufptn_.qtot[0]": evt[59],
        "rufptn_.qtot[1]": evt[60],
    }

    # print(events["rusdmc_.energy"])
    # print(events["rusdmc_.energy"].shape)

    return events


def flat_arrays_with_offsets(list_of_strings, record_size, dtype):
    """
    Parse the list of strings into a flat array with offsets

    This is useful for events with variable length data
    where each event can have a different number of entries
    For example, in the case of badsdinfo, each event can have a different
    number of SDs that are out
    This function will return a flat array and offsets for each event
    The flat array will have shape (n_entries, record_size)
    """

    # Each entry corresponds to a single event
    events = [
        np.fromstring(line, sep=" ", dtype=dtype).reshape(-1, record_size).transpose()
        for line in list_of_strings
    ]

    # Calculate lengths of each event
    # This is the number of entries in each event
    lengths = np.fromiter(
        (arr.shape[1] for arr in events), dtype=np.int64, count=len(events)
    )

    # Create offsets for each event
    # Offsets will be used to slice the flat array into individual events
    # start = offsets[0], end = offsets[1] - slice of flat array for first event
    offsets = np.empty(len(events) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = lengths.cumsum()

    # Flatten the list of arrays into a single array
    flat_events = np.hstack(events)
    return flat_events, offsets


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
                     "rufptn_.vem[0]",     #vertical equivalent muon
                     "rufptn_.vem[1]",
                     "rufptn_.nfold"]
    """
    ## Detection related

    flat_events, offsets = flat_arrays_with_offsets(
        list_of_strings=sdmeta_list_str, record_size=12, dtype=np.float64
    )

    hits_info = {
        "rufptn_.xxyy": flat_events[0].astype(np.int32),
        "rufptn_.isgood": flat_events[1],
        "rufptn_.reltime[0]": flat_events[2],
        "rufptn_.reltime[1]": flat_events[3],
        "rufptn_.pulsa[0]": flat_events[4],
        "rufptn_.pulsa[1]": flat_events[5],
        "rufptn_.xyzclf[0]": flat_events[6],
        "rufptn_.xyzclf[1]": flat_events[7],
        "rufptn_.xyzclf[2]": flat_events[8],
        "rufptn_.vem[0]": flat_events[9],  # vertical equivalent muon
        "rufptn_.vem[1]": flat_events[10],
        "rufptn_.nfold": flat_events[11],
        "offsets": offsets,
    }

    return hits_info


def parse_sdwaveform(sdwaveform_list_str):
    ## sd waveform data
    """
    sdwaveform_format = ["rusdraw_.xxyy",
                        "rusdraw_.clkcnt",
                        "rusdraw_.mclkcnt",
                        "rusdraw_.fadc[0]",
                        "rusdraw_.fadc[1]"]
    """

    flat_events, offsets = flat_arrays_with_offsets(
        list_of_strings=sdwaveform_list_str, record_size=(3 + 128 * 2), dtype=np.int32
    )

    waveforms = {
        "rusdraw_.xxyy": flat_events[0],
        "rusdraw_.clkcnt": flat_events[1],
        "rusdraw_.mclkcnt": flat_events[2],
        # Make shape (n_waveforms, 2, 128)
        "rusdraw_.fadc": np.stack(
            [flat_events[3:131], flat_events[131:]], axis=0
        ).transpose(2, 0, 1),
        # Offsets for each event
        "offsets": offsets,
    }

    # print(waveforms["rusdraw_.xxyy"][offsets[5] : offsets[6]])
    # print(waveforms["rusdraw_.xxyy"][offsets[5] : offsets[6]].shape)

    return waveforms


def parse_badsdinfo(badsdinfo_list_str):
    """
    Int_t nsdsout;         // number of SDs either completely out (absent in the live detector list during event)
    vector<Int_t> xxyyout; // SDs that are completely out (can't participate in event readout)
    vector<Int_t> bitfout;
    """

    flat_events, offsets = flat_arrays_with_offsets(
        list_of_strings=badsdinfo_list_str, record_size=2, dtype=np.int32
    )

    res = {
        "bsdinfo_.xxyyout[x]": flat_events[0],
        "bsdinfo_.bitfout[x]": flat_events[1],
        "offsets": offsets,
    }

    return res


def parse_dst_string(dst_string):

    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = (
        dst_sections(dst_string)
    )

    if len(event_list_str) == 0:
        return None

    result = {
        "events": parse_event(event_list_str),
        "hits": parse_sdmeta(sdmeta_list_str),
        "waveforms": parse_sdwaveform(sdwaveform_list_str),
        "badsd": parse_badsdinfo(badsdinfo_list_str),
    }

    return result
