import numpy as np
import awkward as ak
import yaml
from dstparser import parse_dst_file
from time import time
import pytest
import os


@pytest.mark.parametrize(
    "avg_traces, add_shower_params, add_standard_recon",
    [
        (True, True, True),
        (False, True, True),
    ],
)
def test_parser_grid_model_ta(
    dst_file_ta, config_factory, avg_traces, add_shower_params, add_standard_recon
):
    """Tests the grid model with various parameter combinations for TA data."""
    config_path = config_factory(data_type="TA", use_grid_model=True)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["dst_parser"]["avg_traces"] = avg_traces
    config["dst_parser"]["add_shower_params"] = add_shower_params
    config["dst_parser"]["add_standard_recon"] = add_standard_recon

    data = parse_dst_file(
        dst_file_ta,
        **config["dst_parser"],
        config=config,
    )

    if data is not None:
        assert "detector_positions" in data
        assert isinstance(data["detector_positions"], np.ndarray)
        assert data["detector_positions"].shape[1] == 7
        assert data["detector_positions"].shape[2] == 7

        if add_shower_params:
            assert "energy" in data
        else:
            assert "energy" not in data

        if add_standard_recon:
            assert "std_recon_energy" in data
        else:
            assert "std_recon_energy" not in data

        if avg_traces:
            assert "arrival_times" in data
            assert "time_traces" in data
            assert "arrival_times_low" not in data
        else:
            assert "arrival_times_low" in data
            assert "time_traces_low" in data
            assert "arrival_times" not in data


@pytest.mark.parametrize(
    "avg_traces, add_shower_params, add_standard_recon",
    [
        (True, True, True),
        (False, True, True),
    ],
)
def test_parser_awkward_model_ta(
    dst_file_ta, config_factory, avg_traces, add_shower_params, add_standard_recon
):
    """Tests the awkward model with various parameter combinations for TA data."""
    config_path = config_factory(data_type="TA", use_grid_model=False)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["dst_parser"]["avg_traces"] = avg_traces
    config["dst_parser"]["add_shower_params"] = add_shower_params
    config["dst_parser"]["add_standard_recon"] = add_standard_recon

    data = parse_dst_file(
        dst_file_ta,
        **config["dst_parser"],
        config=config,
    )

    if data is not None:
        assert "hits_det_id" in data
        assert isinstance(data["hits_det_id"], ak.Array)

        if add_shower_params:
            assert "energy" in data
        else:
            assert "energy" not in data

        if add_standard_recon:
            assert "std_recon_energy" in data
        else:
            assert "std_recon_energy" not in data

        if avg_traces:
            assert "hits_time_traces" in data
            assert isinstance(data["hits_time_traces"], ak.Array)
            assert "hits_time_traces_low" not in data
        else:
            assert "hits_time_traces_low" in data
            assert isinstance(data["hits_time_traces_low"], ak.Array)
            assert "hits_time_traces_up" in data
            assert isinstance(data["hits_time_traces_up"], ak.Array)
            assert "hits_time_traces" not in data


@pytest.mark.parametrize(
    "use_grid_model",
    [
        (True),
        (False),
    ],
)
def test_parser_tax4(dst_file_tax4, config_factory, use_grid_model):
    """Tests the parser for TAX4 data."""
    config_path = config_factory(data_type="TAX4", use_grid_model=use_grid_model)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data = parse_dst_file(
        dst_file_tax4,
        **config["dst_parser"],
        config=config,
    )

    if data is not None:
        if use_grid_model:
            assert "detector_positions" in data
            assert "hits_det_id" not in data
        else:
            assert "hits_det_id" in data
            assert "detector_positions" not in data
