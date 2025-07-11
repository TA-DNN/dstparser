import numpy as np
import awkward as ak
from dstparser import parse_dst_file
from time import time
from dstparser.cli.cli import parse_config
import sys
import pytest
import os


@pytest.fixture
def dst_file():
    file_path = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    if not os.path.exists(file_path):
        pytest.skip(f"Test DST file not found: {file_path}")
    return file_path


@pytest.mark.parametrize(
    "avg_traces, add_shower_params, add_standard_recon",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_parser_grid_model(
    dst_file, avg_traces, add_shower_params, add_standard_recon, print_read_data=False
):
    """Tests the grid model with various parameter combinations."""
    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

    print(
        f"\n--- Testing grid model (avg_traces={avg_traces}, "
        f"add_shower_params={add_shower_params}, add_standard_recon={add_standard_recon}) ---"
    )
    start = time()
    data_grid = parse_dst_file(
        dst_file,
        ntile=7,
        xmax_reader=None,
        avg_traces=avg_traces,
        add_shower_params=add_shower_params,
        add_standard_recon=add_standard_recon,
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

        if add_shower_params:
            assert "energy" in data_grid
        else:
            assert "energy" not in data_grid

        if add_standard_recon:
            assert "std_recon_energy" in data_grid
        else:
            assert "std_recon_energy" not in data_grid

        if avg_traces:
            assert "arrival_times" in data_grid
            assert "time_traces" in data_grid
            assert "arrival_times_low" not in data_grid
        else:
            assert "arrival_times_low" in data_grid
            assert "time_traces_low" in data_grid
            assert "arrival_times" not in data_grid

        print("\nGrid model tests passed!")
    else:
        print("\nGrid model produced no data, skipping tests.")


@pytest.mark.parametrize(
    "avg_traces, add_shower_params, add_standard_recon",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_parser_awkward_model(
    dst_file, avg_traces, add_shower_params, add_standard_recon, print_read_data=False
):
    """Tests the awkward model with various parameter combinations."""
    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

    print(
        f"\n--- Testing awkward model (avg_traces={avg_traces}, "
        f"add_shower_params={add_shower_params}, add_standard_recon={add_standard_recon}) ---"
    )
    start = time()
    data_ak = parse_dst_file(
        dst_file,
        ntile=7,
        xmax_reader=None,
        avg_traces=avg_traces,
        add_shower_params=add_shower_params,
        add_standard_recon=add_standard_recon,
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

        if add_shower_params:
            assert "energy" in data_ak
        else:
            assert "energy" not in data_ak

        if add_standard_recon:
            assert "std_recon_energy" in data_ak
        else:
            assert "std_recon_energy" not in data_ak

        if avg_traces:
            assert "hits_time_traces" in data_ak
            assert isinstance(data_ak["hits_time_traces"], ak.Array)
            assert "hits_time_traces_low" not in data_ak
        else:
            assert "hits_time_traces_low" in data_ak
            assert isinstance(data_ak["hits_time_traces_low"], ak.Array)
            assert "hits_time_traces_up" in data_ak
            assert isinstance(data_ak["hits_time_traces_up"], ak.Array)
            assert "hits_time_traces" not in data_ak
