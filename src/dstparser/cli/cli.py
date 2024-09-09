import argparse
import sys
from pathlib import Path
import importlib
from .slurm import run_slurm_job


def run_main_job(config):

    print(config.output_dir)

    data_dirs = " ".join(config.data_dirs)

    if hasattr(config, "num_temp_files"):
        num_temp_files = config.num_temp_files
    else:
        num_temp_files = 1000

    script = Path(__file__).parent / "main_job.py"
    log_dir = Path(config.output_dir) / "slurm_run"
    options = f"--log_dir {str(log_dir)} "
    # options += f"--data_dirs {data_dirs} "
    # options += f'--glob_patterns "{config.glob_patterns}" '
    # options += f"--output_dir {config.output_dir} "
    # options += f"--temp_h5_files {num_temp_files} "
    # options += f"--final_h5_files {config.num_final_files} "
    options += f"--configfile {config.__file__}"

    print(options)

    print("OKKK")
    run_slurm_job(
        config.slurm_settings,
        log_dir,
        script,
        options,
        suffix="_main_job",
        # batch_command="bash",
    )


def parse_config(configfile):

    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Config file not found: {str(configfile)}")

    sys.path.insert(0, str(configfile.resolve().parent))
    config = importlib.import_module(configfile.stem)

    config_dict = {
        key: value for key, value in vars(config).items() if not key.startswith("__")
    }
    for key, value in config_dict.items():
        print(f"{key}={value}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Converts DST files to HDF5 format for DNN"
    )
    parser.add_argument(
        "configfile", type=str, help="Configuration file in python syntax"
    )
    args = parser.parse_args()

    config = parse_config(args.configfile)
    run_main_job(config)
