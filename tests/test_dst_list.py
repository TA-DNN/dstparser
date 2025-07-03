import numpy as np
from time import time
import sys

from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
from dstparser import parse_dst_file
from dstparser.cli.cli import parse_config


def test_parser(dst_file, print_read_data=False):
    # Parse configuration if provided
    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

    start = time()
    # Use the standard parser to build the data dict
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
        print("\nParsed data keys and shapes:")
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                print(f"{key} {val.shape}")
            elif isinstance(val, list):
                print(f"{key} list length {len(val)}")
            else:
                print(f"{key}: type {type(val)}")

    # Example of other prints when using config
    if config is not None:
        print(f'id_event = {data.get("id_event", "N/A")}')
        print(f'id_corsika_shower = {data.get("id_corsika_shower", "N/A")}')
        print(f'id_energy_bin = {data.get("id_energy_bin", "N/A")}')
        print(f'id_data_set = {data.get("id_data_set", "N/A")}')


if __name__ == "__main__":
    dst_file = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/"
        "tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    test_parser(dst_file, print_read_data=True)
