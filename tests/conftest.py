import pytest
import yaml
import os

@pytest.fixture
def dst_file_ta(tmp_path):
    file_path = "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    if not os.path.exists(file_path):
        pytest.skip(f"Test DST file not found: {file_path}")
    return file_path

@pytest.fixture
def dst_file_tax4(tmp_path):
    file_path = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/tax4/qgsii04proton/north/221101to240124/DAT000025_gea.rufldf.dst.gz"
    if not os.path.exists(file_path):
        pytest.skip(f"Test DST file not found: {file_path}")
    return file_path

@pytest.fixture
def config_factory(tmp_path):
    def _create_config(data_type, use_grid_model):
        config_data = {
            "dst_parser": {
                "ntile": 7,
                "xmax_reader": None,
                "avg_traces": False,
                "add_shower_params": True,
                "add_standard_recon": True,
                "use_grid_model": use_grid_model,
                "data_type": data_type,
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        return str(config_file)
    return _create_config
