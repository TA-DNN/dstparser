import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dstparser.cli.slurm import run_slurm_job
from dstparser.cli.cli import parse_config
import re


def find_files(dirs, globs):

    def convert_to_list(value):
        if isinstance(value, list):
            return value
        elif isinstance(value, str) and "," in value:
            return [item.strip() for item in value.split(",")]
        elif isinstance(value, str):
            return [value]
        else:
            raise ValueError("Input must be a list or a string.")

    dirs = convert_to_list(dirs)
    globs = convert_to_list(globs)

    files = []
    for d in dirs:
        path = Path(d).resolve()
        if not path.exists():
            raise ValueError(f'"{path}" doesn\'t exist!')
        for glob in globs:
            matched_files = path.rglob(glob)
            files.extend(matched_files)
    return files


def distribute_files(files, output_pattern, output_dir, group_by, max_jobs=1):

    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d

    dbase = defaultdict(
        lambda: defaultdict(lambda: {"input_files": [], "output_file": ""})
    )

    print(f"max_jobs = {max_jobs}")

    for i, file in enumerate(files):
        group_id = i // group_by
        job_id = group_id % max_jobs
        dbase[job_id][group_id]["input_files"].append(str(file))
        dbase[job_id][group_id]["output_file"] = str(
            Path(output_dir) / output_pattern.format(group_id)
        ).strip()

    return default_to_regular(dbase)


def filter_files_by_date(files):
    # Filter data files to take only before 16/06/03
    filtered_files = list()
    for file in files:
        file_date = int(re.split(r"[_,.\s]", Path(file).parts[-1])[2])
        if file_date <= 160603:
            filtered_files.append(file)
    return sorted(filtered_files)


def run_dstparser_job(max_jobs, db_file, task_name, log_dir, config):
    slurm_settings = config.slurm_settings
    slurm_settings["array"] = f"0-{max_jobs - 1}"
    # slurm_settings["mem"] = "5gb"

    script = Path(__file__).parent / "worker_job.py"
    # for task_id in range(max_jobs):
    # fout = f"/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/07/07_ta_data/file_{task_id}.out"
    options = f"{str(db_file).strip()} {task_name}"
    log_dir = Path(log_dir)
    print(f"Options = {options}")
    run_slurm_job(
        slurm_settings,
        log_dir,
        script,
        options,
        suffix="_worker_job",
    )


def generate_db(config):

    output_dir = Path(config.output_dir)
    db_files = [output_dir / "jobs_pass1.json", output_dir / "jobs_pass2.json"]

    data_base = []
    for db_file in db_files:
        if db_file.exists():
            with open(db_file, "r") as f:
                data_base.append(json.load(f))

    if len(data_base) == 2:
        return data_base, db_files

    data_base = [None, None]

    files = find_files(config.data_dirs, config.glob_patterns)
    files = sorted(files, key=lambda x: x.name)

    if hasattr(config, "temp_group_by"):
        group_by = config.temp_group_by
    else:
        group_by = 26

    data_base[0] = distribute_files(
        files=files,
        output_pattern="temp_{:05}.h5",
        output_dir=output_dir / "temp_files",
        group_by=group_by,
        max_jobs=config.njobs_temp_pass,
    )

    print(f"data_base[0] = {len(data_base[0])}")

    files = []
    for job_key in data_base[0]:
        for file_key in data_base[0][job_key]:
            files.append(Path(data_base[0][job_key][file_key]["output_file"]))

    files = sorted(files, key=lambda x: x.name)

    if hasattr(config, "final_group_by"):
        group_by = config.final_group_by
    else:
        group_by = len(files) / 20

    data_base[1] = distribute_files(
        files=files,
        output_pattern=config.file_name_pattern + "_{:03}.h5",
        output_dir=output_dir / "final_files",
        group_by=group_by,
        max_jobs=config.njobs_final_pass,
    )

    print(f"data_base[1] = {len(data_base[1])}")

    for i in range(2):
        with open(db_files[i], "w") as f:
            json.dump(data_base[i], f, indent=4)

    return data_base, db_files


def wait_until_ready(temp_files, log_dir):
    max_time_to_wait = 2 * 3600  # in sec
    check_every = 10  # sec

    print("Waiting...")
    log_file = Path(log_dir) / "log_main.dat"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    for i in range(max_time_to_wait):
        ready_num = 0
        all_ready = True
        info_str = ""
        for i, temp_file in enumerate(temp_files):
            if not Path(temp_file).exists():
                all_ready = False
                ready_task = False
            else:
                ready_num += 1
                ready_task = True

            fname = str(Path(temp_file)).strip()
            info_str += f"\n{i} {fname} ready {ready_task}"

        info_str += (
            f"\nAfter {i*check_every}/{max_time_to_wait} sec ({i*check_every/max_time_to_wait*100:.1f}%)"
            f"\nReady num = {ready_num}/{len(temp_files)}"
        )

        with open(log_file, "a") as f:
            f.write(info_str)

        if all_ready:
            break

        time.sleep(check_every)

    return all_ready


def main_job(data_base, db_files, log_dir, config):

    print("Hey")
    # Find temp files
    temp_files = []
    for task in data_base[0].values():
        for group in task.values():
            temp_files.append(group["output_file"])

    # Test for temp files existance
    temp_files_ready = True
    for temp_file in temp_files:
        if not Path(temp_file).exists():
            temp_files_ready = False
            break

    print(f"Launching slurm {~temp_files_ready}")
    # Launch slurm if temp files not ready
    if not temp_files_ready:
        run_dstparser_job(
            len(data_base[0]),
            db_files[0],
            task_name="parse_dst",
            log_dir=log_dir,
            config=config,
        )

    print("Wait unit ready")
    # Wait when all temp files are ready
    all_ready = wait_until_ready(temp_files, log_dir)

    if all_ready:
        run_dstparser_job(
            len(data_base[1]),
            db_files[1],
            task_name="join_hdf5",
            log_dir=log_dir,
            config=config,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert DST files to HDF5 format for DNN."
    )

    parser.add_argument(
        "-c", "--configfile", type=str, help="Configuration file in python syntax"
    )

    parser.add_argument(
        "-l", "--log_dir", type=str, required=True, help="Directory for slurm output"
    )
    args = parser.parse_args()
    config = parse_config(args.configfile)

    data_base, db_files = generate_db(config)
    print("Before main")
    main_job(data_base, db_files, log_dir=args.log_dir, config=config)
