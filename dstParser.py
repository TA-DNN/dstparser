from tqdm import tqdm
import numpy as np
from dst_content import dst_content
from dst_parsers import fill_metadata, dst_sections, shower_params
from dst_parsers import parse_sdmeta, parse_sdwaveform, parse_badsdinfo
from dst_parsers import detector_readings

dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"


def parse_dst_file(dst_file):
    dst_string = dst_content(dst_file)

    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = (
        dst_sections(dst_string)
    )

    sdmeta_list = parse_sdmeta(sdmeta_list_str)
    sdwaveform_list = parse_sdwaveform(sdwaveform_list_str)
    badsdinfo_list = parse_badsdinfo(badsdinfo_list_str)

    # Dictionary with parsed data
    data = dict()
    data = fill_metadata(data, dst_file)
    data = shower_params(event_list_str, data)

    ntile = 7  # number of SD per one side
    ntime_trace = 128  # number of time trace of waveform
    data = detector_readings(
        sdmeta_list, sdwaveform_list, badsdinfo_list, ntile, ntime_trace, data
    )

    return data


data = parse_dst_file(dst_file)


for key, val in data.items():
    if isinstance(val, np.ndarray):
        print(key, val.shape)
    else:
        print(key, len(val), val)

# for i in tqdm(range(10), total=10):
#     parse_dst_file(dst_file)
