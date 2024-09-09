import numpy as np
from dstparser import parse_dst_file
from time import time
from dstparser.cli.cli import parse_config
import sys


def test_parser(dst_file, print_read_data=False):

    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None
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
        config=config,
    )

    end = time()
    print(f"Parse time: {end - start:.3f} sec")

    if print_read_data:
        print(f"\nConverted arrays:\n---")
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                print(key, val.shape)
            else:
                print(key, len(val), val)

    # print(f'energy = {data["energy"]}')
    # print(f'xmax = {data["xmax"]}')
    # print(f'time_traces_low = {data["time_traces_low"]}')
    # print(f'event id = {data["event_id"]}')
    # print(f'corsika shower id = {data["corsika_shower_id"]}')
    # print(f'energy bin id = {data["energy_bin_id"]}')
    if config is not None:
        print(f'event_id = {data["id_event"]}')
        print(f'corsika_shower_id = {data["id_corsika_shower"]}')
        print(f'energy_bin_id = {data["id_energy_bin"]}')
        print(f'data_set_id = {data["id_data_set"]}')


if __name__ == "__main__":
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04nitrogen/080417_160603/Em1_bsdinfo/DAT081325_gea.rufldf.dst.gz"
    test_parser(dst_file, print_read_data=True)
