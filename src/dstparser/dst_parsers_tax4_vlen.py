"""TAx4 string-dump parsers for the vlen (flat-arrays-with-offsets) format.

Copy of dst_parsers_vlen.py, changed only where TAx4's DST dump differs from
TA-SD's. Reads the output of the locally-rebuilt reader
read_dst_file_tax4_std_recon_nfold (see dst_reader / paths), which restores the
rufptn_.nfold column the stock TAx4 reader omitted. With nfold present, only two
field-level differences from TA-SD remain (verified 2026-07-20 against real
TAx4 DST files):

  1. #EVENT DATA has 42 fields, not TA-SD's 61 (parse_event below).
  2. #SD meta DATA prints the detector id as yyxx, not xxyy -- swapped in
     parse_sdmeta below. (The nfold column is now present, so no fold-count
     inference is needed.)

The #SD waveform DATA and #badsdinfo blocks are byte-compatible with TA-SD, so
their parsers (and dst_sections / flat_arrays_with_offsets) are reused as-is.
"""

import numpy as np

from dstparser.dst_parsers_vlen import (
    dst_sections,
    flat_arrays_with_offsets,
    parse_sdwaveform,
    parse_badsdinfo,
)


def parse_event(event_list_str):
    """TAx4 standard-recon #EVENT DATA layout (42 fields/event).

    Fields 0-41 are byte-identical to the first 42 fields of TA-SD's own vlen
    parse_event: truth (0-6) + counts/timing (7-11) + one LDF fit (12-21) +
    fixed-curvature geometry fit (22-29) + border cuts (30-31) + free-curvature
    geometry fit and curvature (32-41). TAx4 does NOT emit TA-SD's fields 42-60
    (the combined LDF+geometry fit, rufldf theta/phi, and rufptn_.nhits/
    nsclust/nborder/qtot).
    """
    event_list = [
        np.fromstring(line, sep=" ", dtype=np.float64) for line in event_list_str
    ]
    evt = np.array(event_list).transpose()

    events = {
        # truth (rusdmc_)
        "rusdmc_.parttype": evt[0],
        "rusdmc_.energy": evt[1],
        "rusdmc_.theta": evt[2],
        "rusdmc_.phi": evt[3],
        "rusdmc_.corexyz[0]": evt[4],
        "rusdmc_.corexyz[1]": evt[5],
        "rusdmc_.corexyz[2]": evt[6],
        # event counts / timing (rusdraw_, rufptn_)
        "rusdraw_.yymmdd": evt[7],
        "rusdraw_.hhmmss": evt[8],
        "rufptn_.nstclust": evt[9],
        "rusdraw_.nofwf": evt[10],
        "rusdraw_.usec": evt[11],
        # LDF fit (rufldf_ [0])
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
        # fixed-curvature geometry fit (rusdgeom_ [1])
        "rusdgeom_.theta[1]": evt[22],
        "rusdgeom_.phi[1]": evt[23],
        "rusdgeom_.dtheta[1]": evt[24],
        "rusdgeom_.dphi[1]": evt[25],
        "rusdgeom_.chi2[1]": evt[26],
        "rusdgeom_.ndof[1]": evt[27],
        "rusdgeom_.t0[1]": evt[28],
        "rusdgeom_.dt0[1]": evt[29],
        # border cuts (rufldf_)
        "rufldf_.bdist": evt[30],
        "rufldf_.tdist": evt[31],
        # free-curvature geometry fit (rusdgeom_ [2]) + curvature
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
    }

    return events


def parse_sdmeta(sdmeta_list_str):
    """TAx4 #SD meta DATA: 12 fields/hit, INCLUDING rufptn_.nfold.

    Field order (matches the locally-rebuilt reader
    read_dst_file_tax4_std_recon_nfold, which restores the nfold column the
    stock TAx4 reader omitted):
        xxyy, isgood, reltime[0], reltime[1], pulsa[0], pulsa[1],
        xyzclf[0], xyzclf[1], xyzclf[2], vem[0], vem[1], nfold

    TAx4 prints the detector id as yyxx; swap it to xxyy so it matches the
    waveform-block ids (which are already xxyy) and the TA-SD convention. This
    is the ONLY field-level difference from TA-SD's own parse_sdmeta.
    """
    flat_events, offsets = flat_arrays_with_offsets(
        list_of_strings=sdmeta_list_str, record_size=12, dtype=np.float64
    )

    yyxx = flat_events[0].astype(np.int32)

    hits_info = {
        "rufptn_.xxyy": (yyxx % 100) * 100 + (yyxx // 100),
        "rufptn_.isgood": flat_events[1].astype(np.int32),
        "rufptn_.reltime[0]": flat_events[2],
        "rufptn_.reltime[1]": flat_events[3],
        "rufptn_.pulsa[0]": flat_events[4],
        "rufptn_.pulsa[1]": flat_events[5],
        "rufptn_.xyzclf[0]": flat_events[6],
        "rufptn_.xyzclf[1]": flat_events[7],
        "rufptn_.xyzclf[2]": flat_events[8],
        "rufptn_.vem[0]": flat_events[9],  # vertical equivalent muon
        "rufptn_.vem[1]": flat_events[10],
        "rufptn_.nfold": flat_events[11].astype(np.int32),
        "offsets": offsets,
    }

    return hits_info


def parse_dst_string(dst_string):

    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = (
        dst_sections(dst_string)
    )

    if len(event_list_str) == 0:
        return None, None, None, None

    events = parse_event(event_list_str)
    hits = parse_sdmeta(sdmeta_list_str)
    waveforms = parse_sdwaveform(sdwaveform_list_str)
    badsd = parse_badsdinfo(badsdinfo_list_str)

    return events, hits, waveforms, badsd
