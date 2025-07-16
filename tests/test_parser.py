import numpy as np
<<<<<<< HEAD
import awkward as ak
from dstparser import parse_dst_file
=======
from dstparser import parse_dst_file, parse_dst_file_tax4
>>>>>>> e3c178ce573d24547b5ee385693a60b42df6d3ca
from time import time
from dstparser.cli.cli import parse_config
import sys
import pytest
import os


<<<<<<< HEAD
@pytest.fixture
def dst_file():
    file_path = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    if not os.path.exists(file_path):
        pytest.skip(f"Test DST file not found: {file_path}")
    return file_path
=======
def test_parser(dst_file, print_read_data=False, use_ta_x4 = False):
>>>>>>> e3c178ce573d24547b5ee385693a60b42df6d3ca


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
<<<<<<< HEAD
    data_grid = parse_dst_file(
=======

    # The only function that are required to parsing
    # is parse_dst_file
    # paths to the directories with data is dstparser.paths module
    # data = parse_dst_file(dst_file, up_low_traces=True)
    
    if use_ta_x4:
        convert_func = parse_dst_file_tax4
    else:
        convert_func = parse_dst_file

    data = convert_func(
>>>>>>> e3c178ce573d24547b5ee385693a60b42df6d3ca
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

<<<<<<< HEAD
    if print_read_data and data_grid is not None:
=======

    if print_read_data:
>>>>>>> e3c178ce573d24547b5ee385693a60b42df6d3ca
        print(f"\nConverted arrays:\n---")
        for key, val in data_grid.items():
            if isinstance(val, np.ndarray):
                print(key, val.shape)
            else:
                print(key, len(val))

<<<<<<< HEAD
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
=======
    # print(f'energy = {data["energy"]}')
    # print(f'xmax = {data["xmax"]}')
    # print(f'time_traces_low = {data["time_traces_low"]}')
    # print(f'event id = {data["event_id"]}')
    # print(f'corsika shower id = {data["corsika_shower_id"]}')
    # print(f'energy bin id = {data["energy_bin_id"]}')
    # print(f"detector_positions_id = {data['detector_positions_id'][50]}")
    # print(f"detector_positions_abs = {data['detector_positions_abs'][50][:, :, 0]}")
    # print(f"detector_positions = {data['detector_positions'][50][:, :, 0].astype(np.float16)}")
    # print(f"detector_positions_id = {data['detector_positions_abs'][50].shape}")
    if config is not None:
        print(f'event_id = {data["id_event"]}')
        print(f'corsika_shower_id = {data["id_corsika_shower"]}')
        print(f'energy_bin_id = {data["id_energy_bin"]}')
        print(f'data_set_id = {data["id_data_set"]}')


if __name__ == "__main__":
    # dst_file = (
    #     "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
    #     "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    # )
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo/DAT055402_gea.rufldf.dst.gz"
    
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/tax4/qgsii04proton/north/221101to240124/DAT000025_gea.rufldf.dst.gz"
    
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04nitrogen/080417_160603/Em1_bsdinfo/DAT081325_gea.rufldf.dst.gz"
    # !If you want to use TAx4 format, set use_ta_x4=True
    # !If you want to use TA format, set use_ta_x4=False
    # !By default, use_ta_x4=False
    test_parser(dst_file, print_read_data=True, use_ta_x4 = True)

>>>>>>> e3c178ce573d24547b5ee385693a60b42df6d3ca
