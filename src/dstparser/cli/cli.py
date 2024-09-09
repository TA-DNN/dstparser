import argparse
import sys
from pathlib import Path
import importlib
from dstparser.cli.slurm import run_slurm_job, copy_files


def run_main_job(config):

    copy_files(Path(config.__file__), Path(config.output_dir))

    script = Path(__file__).parent / "main_job.py"
    log_dir = Path(config.output_dir) / "slurm_run"
    options = f"--log_dir {str(log_dir)} "
    copied_config = Path(config.output_dir) / Path(config.__file__).name
    options += f"--configfile {str(copied_config)}"

    print(options)

    run_slurm_job(
        config.slurm_settings,
        log_dir,
        script,
        options,
        suffix="_main_job",
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
