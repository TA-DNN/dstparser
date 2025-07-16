import argparse
import sys
from pathlib import Path
import yaml
import subprocess
from dstparser.cli.slurm import run_slurm_job, copy_files


def run_main_job(config, config_path, local=False):

    copy_files(config_path, Path(config["output"]["output_dir"]))

    script = Path(__file__).parent / "main_job.py"
    log_dir = Path(config["output"]["output_dir"]) / ("local_run" if local else "slurm_run")
    options = f"--log_dir {str(log_dir)} "
    copied_config = Path(config["output"]["output_dir"]) / Path(config_path).name
    options += f"--config {str(copied_config)}"

    if local:
        options += " --local"
        command = [sys.executable, str(script)] + options.split()
        print(f"Running locally: {' '.join(command)}")
        subprocess.run(command, check=True)
    else:
        print(options)
        run_slurm_job(
            config["slurm"],
            log_dir,
            script,
            options,
            suffix="_main_job",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Converts DST files to HDF5 format for DNN"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file in YAML syntax",
        default="/home/marktsai321/TA_DNN/dstparser/examples/config.yaml",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the processing locally without submitting to SLURM.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {str(config_path)}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_sections = ["dataset_mapping", "slurm", "input", "dst_parser", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file.")

    run_main_job(config, config_path, local=args.local)

if __name__ == "__main__":
    main()
