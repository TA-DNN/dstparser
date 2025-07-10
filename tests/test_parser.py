import numpy as np
import awkward as ak
from dstparser import parse_dst_file
from time import time
from dstparser.cli.cli import parse_config
import sys


def test_parser(dst_file, print_read_data=False):

    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

    print("--- Testing grid model ---")
    start = time()
    data_grid = parse_dst_file(
        dst_file,
        ntile=7,
        xmax_reader=None,
        avg_traces=False,
        add_shower_params=True,
        add_standard_recon=True,
        config=config,
        use_grid_model=True,
    )
    end = time()
    print(f"Parse time: {end - start:.3f} sec")

    if print_read_data and data_grid is not None:
        print(f"\nConverted arrays:\n---")
        for key, val in data_grid.items():
            if isinstance(val, np.ndarray):
                print(key, val.shape)
            else:
                print(key, len(val))
    
    if data_grid is not None:
        assert "detector_positions" in data_grid
        assert isinstance(data_grid["detector_positions"], np.ndarray)
        assert data_grid["detector_positions"].shape[1] == 7
        assert data_grid["detector_positions"].shape[2] == 7
        print("\nGrid model tests passed!")
    else:
        print("\nGrid model produced no data, skipping tests.")

    print("\n--- Testing awkward model ---")
    start = time()
    data_ak = parse_dst_file(
        dst_file,
        ntile=7,
        xmax_reader=None,
        avg_traces=False,
        add_shower_params=True,
        add_standard_recon=True,
        config=config,
        use_grid_model=False,
    )
    end = time()
    print(f"Parse time: {end - start:.3f} sec")

    if print_read_data and data_ak is not None:
        print(f"\nConverted arrays:\n---")
        for key, val in data_ak.items():
            if isinstance(val, np.ndarray):
                print(key, val.shape)
            elif isinstance(val, ak.Array):
                print(key, f"ak.Array with type {val.type}")
            else:
                print(key, len(val))

    if data_ak is not None:
        assert "hits_det_id" in data_ak
        assert isinstance(data_ak["hits_det_id"], ak.Array)
        assert "hits_time_traces_low" in data_ak
        assert isinstance(data_ak["hits_time_traces_low"], ak.Array)
        assert "hits_time_traces_up" in data_ak
        assert isinstance(data_ak["hits_time_traces_up"], ak.Array)
        print("\nAwkward model tests passed!")
    else:
        print("\nAwkward model produced no data, skipping tests.")


if __name__ == "__main__":
    dst_file = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04nitrogen/080417_160603/Em1_bsdinfo/DAT081325_gea.rufldf.dst.gz"
    test_parser(dst_file, print_read_data=True)
