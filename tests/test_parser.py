import numpy as np
from dstparser import parse_dst_file
from time import time


def test_parser(dst_file):
    start = time()
    
    # The only function that are required to parsing
    # is parse_dst_file
    # paths to the directories with data is dstparser.paths module
    data = parse_dst_file(dst_file, up_low_traces=True)
    end = time()
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(key, val.shape)
        else:
            print(key, len(val), val)

    # print(f'energy = {data["energy"]}')
    # print(f'xmax = {data["xmax"]}')
    # print(data["time_traces_low"])
    print(f"Parse time: {end - start} sec")

if __name__ == "__main__":
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"
    test_parser(dst_file)
