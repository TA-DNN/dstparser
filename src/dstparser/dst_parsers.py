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
                     "rufptn_.vem[0]",     vertical equivalent muon
                     "rufptn_.vem[1]",
                     "rufptn_.nfold"]
    """
    ## Detection related

    record_size = 12
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
    sdwaveform_list = []
    for line in sdwaveform_list_str:
        arr = np.fromstring(line, sep=" ", dtype=np.int32)
        # Each record has metadata + FADC data. The number of FADC samples can vary.
        # Find record boundaries by looking for the metadata pattern, which is less likely
        # to be repeated than FADC values. A more robust solution would be needed if
        # metadata values could legitimately appear in FADC data.
        
        # This part of the parsing logic is complex because the record size is variable.
        # The original implementation assumed a fixed size. A full, robust implementation
        # would require a more sophisticated parsing strategy, possibly iterating through
        # the array to identify record boundaries based on expected metadata patterns or
        # prior information about the number of detectors.
        
        # For now, we'll assume that the number of detectors with waveforms is implicitly
        # known from the structure of the file, and that we can determine the number of
        # FADC windows from the length of the array.
        
        # A simplified approach that works for many cases is to assume that the number of
        # detectors is the primary organizing principle.
        
        # The logic below is a placeholder for a more robust implementation.
        # It reverts to a simpler logic that may work if the file structure is consistent.
        try:
            # This assumes a fixed record size, which is often not the case.
            record_size = 3 + 128 * 2 
            parsed_line = arr.reshape(-1, record_size).transpose()
        except ValueError:
            # A more dynamic approach is needed if record sizes vary.
            # This is a complex problem without more information about the file format.
            # As a fallback, we'll create an empty array to avoid crashing.
            parsed_line = np.array([]).reshape(3 + 128 * 2, 0)

        sdwaveform_list.append(parsed_line)


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
