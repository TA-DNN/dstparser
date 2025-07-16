import sys
import json
import time
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
from dstparser.cli.slurm import run_slurm_job
import re
import subprocess
import os


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


def distribute_files(files, output_pattern, output_dir, assign_group_id, assign_job_id):

    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d

    dbase = defaultdict(
        lambda: defaultdict(lambda: {"input_files": [], "output_file": ""})
    )

    len_files = len(files)
    for file_id, file in enumerate(files):
        group_id = assign_group_id(file, file_id, len_files)
        job_id = assign_job_id(group_id)
        dbase[job_id][group_id]["input_files"].append(str(file))
        dbase[job_id][group_id]["output_file"] = str(
            Path(output_dir) / output_pattern.format(group_id)
        ).strip()

    # Sort files by name inside group
    for job_id in dbase:
        for group_id in dbase[job_id]:
            dbase[job_id][group_id]["input_files"] = sorted(
                dbase[job_id][group_id]["input_files"], key=lambda x: Path(x).name
            )

    return default_to_regular(dbase)


def filter_files_by_date(files):
    # Filter data files to take only before 16/06/03
    filtered_files = list()
    for file in files:
        file_date = int(re.split(r"[_,.\s]", Path(file).parts[-1])[2])
        if file_date <= 160603:
            filtered_files.append(file)
    return sorted(filtered_files)


def run_dstparser_job(max_jobs, db_file, task_name, log_dir, config, config_path, local=False):
    script = Path(__file__).parent / "worker_job.py"
    options = f"{str(db_file).strip()} {task_name} {str(config_path).strip()}"
    
    if local:
        for job_id in range(max_jobs):
            env = os.environ.copy()
            env["SLURM_ARRAY_TASK_ID"] = str(job_id)
            env["SLURM_ARRAY_TASK_MIN"] = "0"
            env["SLURM_ARRAY_TASK_MAX"] = str(max_jobs - 1)
            command = [sys.executable, str(script)] + options.split()
            print(f"Running worker job {job_id} locally: {' '.join(command)}")
            subprocess.run(command, check=True, env=env)
    else:
        slurm_settings = config["slurm"]
        slurm_settings["array"] = f"0-{max_jobs - 1}"
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

    output_dir = Path(config["output"]["output_dir"])
    db_files = [output_dir / "jobs_pass1.json", output_dir / "jobs_pass2.json"]

    data_base = []
    for db_file in db_files:
        if db_file.exists():
            with open(db_file, "r") as f:
                data_base.append(json.load(f))

    if len(data_base) == 2:
        return data_base, db_files

    data_base = [None, None]

    files = find_files(config["input"]["data_dirs"], config["input"]["glob_patterns"])

    # if hasattr(config, "filter_data_files"):
    #     files = config.filter_data_files(files)

    files = sorted(files, key=lambda x: x.name)

    def temp_group_id(data_file, file_id, len_files):
        pattern = r"DAT(\d{4})"
        match = re.search(pattern, Path(data_file).name)
        if match:
            group_id = int(match.group(1))
        else:
            # Fallback if pattern doesn't match
            group_id = file_id
        return group_id

    def temp_job_id(group_id):
        return group_id % config["processing"]["njobs"]

    data_base[0] = distribute_files(
        files=files,
        output_pattern="temp_{:05}.h5",
        output_dir=output_dir / "temp_files",
        assign_group_id=temp_group_id,
        assign_job_id=temp_job_id,
    )

    print(f"data_base[0] = {len(data_base[0])}")
    print(f"data_base[0] = {data_base[0]}")

    files = []
    for job_key in data_base[0]:
        for file_key in data_base[0][job_key]:
            files.append(Path(data_base[0][job_key][file_key]["output_file"]))

    files = sorted(files, key=lambda x: x.name)

    def final_group_id(data_file, file_id, len_files):
        pattern = r"temp_(\d{5})"
        match = re.search(pattern, Path(data_file).name)
        if match:
            group_id = int(match.group(1)) % config["processing"]["group_temp_files_by"]
        else:
            # Fallback if pattern doesn't match
            group_id = file_id % config["processing"]["group_temp_files_by"]
        return group_id

    def final_job_id(group_id):
        return group_id % config["processing"]["njobs"]

    data_base[1] = distribute_files(
        files=files,
        output_pattern=config["output"]["file_name_pattern"] + "_{:03}.h5",
        output_dir=output_dir / "final_files",
        assign_group_id=final_group_id,
        assign_job_id=final_job_id,
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

    for itime in range(max_time_to_wait):
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
            f"\nAfter {itime*check_every}/{max_time_to_wait} sec ({itime*check_every/max_time_to_wait*100:.1f}%)"
            f"\nReady num = {ready_num}/{len(temp_files)}, {ready_num/len(temp_files)}%"
        )

        with open(log_file, "a") as f:
            f.write(info_str)

        if all_ready:
            break

        time.sleep(check_every)

    return all_ready


def main_job(data_base, db_files, log_dir, config, config_path, local=False):
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
            config_path=config_path,
            local=local,
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
            config_path=config_path,
            local=local,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert DST files to HDF5 format for DNN."
    )

    parser.add_argument(
        "-c", "--config", type=str, help="Configuration file in YAML syntax"
    )

    parser.add_argument(
        "-l", "--log_dir", type=str, required=True, help="Directory for slurm output"
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

    data_base, db_files = generate_db(config)
    print("Before main")
    main_job(data_base, db_files, log_dir=args.log_dir, config=config, config_path=config_path, local=args.local)
