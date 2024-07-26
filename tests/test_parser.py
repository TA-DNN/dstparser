import numpy as np
from dstparser import parse_dst_file
from time import time


def test_parser(dst_file):
    start = time()

    # The only function that are required to parsing
    # is parse_dst_file
    # paths to the directories with data is dstparser.paths module
    # data = parse_dst_file(dst_file, up_low_traces=True)

    data = parse_dst_file(
        dst_file,
        ntile=7,
        xmax_reader=None,
        avg_traces=False,
        add_shower_params=True,
        add_standard_recon=True,
    )

    end = time()
    # for key, val in data.items():
    #     if isinstance(val, np.ndarray):
    #         print(key, val.shape)
    #     else:
    #         print(key, len(val), val)

    # print(f'energy = {data["energy"]}')
    # print(f'xmax = {data["xmax"]}')
    # print(data["time_traces_low"])
    print(f"Parse time: {end - start} sec")


if __name__ == "__main__":
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdobs_dstbank/rufldf/tasdcalibev_pass2_080511.bsdinfo.rufldf.dst.gz"
    test_parser(dst_file)
