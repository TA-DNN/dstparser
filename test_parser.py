import numpy as np
from dst_adapter import parse_dst_file
from xmax_reader import XmaxReader


def test_parser(dst_file):
    xmax_reader = XmaxReader()
    data = parse_dst_file(dst_file, xmax_reader)

    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(key, val.shape)
        else:
            print(key, len(val), val)

    print(f'energy = {data["energy"]}')
    print(f'xmax = {data["xmax"]}')


if __name__ == "__main__":
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"
    test_parser(dst_file)
