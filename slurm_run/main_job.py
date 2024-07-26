import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from run_dst_conversion import run_slurm_job, slurm_directives
import re


def normalize_input(input_value):
    if isinstance(input_value, list):
        return input_value
    elif isinstance(input_value, str) and "," in input_value:
        return [item.strip() for item in input_value.split(",")]
    elif isinstance(input_value, str):
        return [input_value]
    else:
        raise ValueError("Input must be a list or a string.")


def get_files_from_directories(directories, patterns):
    all_files = []
    for directory in directories:
        path = Path(directory).resolve()
        if not path.exists():
            raise ValueError(f'"{path}" doesn\'t exist!')
        for pattern in patterns:
            matched_files = path.rglob(pattern)
            all_files.extend(matched_files)
    return sorted(all_files)


def distribute_files(files, num_workers, output_filename_func, output_dir):
    worker_files = defaultdict(lambda: {"input_files": [], "output_file": ""})
    for idx, file in enumerate(files):
        # worker_id = idx % num_workers
        worker_id = (idx // 26) % 1000
        worker_files[worker_id]["input_files"].append(str(file))
        worker_files[worker_id]["output_file"] = output_filename_func(
            worker_id, output_dir
        )
    return dict(worker_files)


def distribute_files11(files, nfinal_files, output_filename_func, output_dir):
    worker_files = defaultdict(lambda: {"input_files": [], "output_file": ""})
    files_per_worker = len(files) // nfinal_files + 1
    for idx, file in enumerate(files):
        worker_id = idx // files_per_worker
        # worker_id = (idx // 26) % 1000
        worker_files[worker_id]["input_files"].append(str(file))
        worker_files[worker_id]["output_file"] = output_filename_func(
            worker_id, output_dir
        )
    return dict(worker_files)


def main(directories, patterns, num_workers, output_filename_func, output_dir):
    print("Searching files...")
    directories = normalize_input(directories)
    patterns = normalize_input(patterns)
    files = get_files_from_directories(directories, patterns)
    # files = filter_files_by_date(files)
    print(f"Found {len(files)} files")
    distributed_files = distribute_files(
        files, num_workers, output_filename_func, output_dir
    )
    assert np.sum(
        [len(val["input_files"]) for val in distributed_files.values()]
    ) == len(files)
    return distributed_files


def filter_files_by_date(files):
    # Filter data files to take only before 16/06/03
    filtered_files = list()
    for file in files:
        file_date = int(re.split(r"[_,.\s]", Path(file).parts[-1])[2])
        if file_date <= 160603:
            filtered_files.append(file)
    return sorted(filtered_files)


def generate_output_filename(worker_index, output_dir):
    return str(output_dir / f"temp_{worker_index:05}.h5").strip()


def generate_output_filename1(worker_index, output_dir):
    return str(output_dir / f"ta_data_{worker_index:03}.h5").strip()


def run_dstparser_job(max_jobs, db_file, task_name, log_dir):

    slurm_settings = slurm_directives()
    slurm_settings["array"] = f"0-{max_jobs - 1}"
    slurm_settings["mem"] = "5gb"

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
        # batch_command="bash",
    )


def generate_db(
    data_dirs,
    glob_patterns,
    output_dir,
    num_temp_h5_files,
    num_final_h5_files,
):

    output_dir = Path(output_dir)
    db_files = [output_dir / "jobs_pass1.json", output_dir / "jobs_pass2.json"]

    data_base = []
    for db_file in db_files:
        if db_file.exists():
            with open(db_file, "r") as f:
                data_base.append(json.load(f))

    print(f"Here is ok, {db_files}")
    if len(data_base) == 2:
        return data_base, db_files

    data_base = [None, None]

    data_base[0] = main(
        data_dirs,
        glob_patterns,
        num_temp_h5_files,
        generate_output_filename,
        output_dir / "temp_files",
    )

    h5_files = []
    for key in data_base[0]:
        h5_files.append(data_base[0][key]["output_file"])

    data_base[1] = distribute_files11(
        h5_files,
        num_final_h5_files,
        generate_output_filename1,
        output_dir / "final_files",
    )

    for i in range(2):
        with open(db_files[i], "w") as f:
            json.dump(data_base[i], f, indent=4)

    return data_base, db_files


def main_job(data_base, db_files, log_dir):

    ready = True
    for task_id, task in data_base[0].items():
        # print("OUTPUT FILES FROM DB", task["output_file"])
        if not Path(task["output_file"]).exists():
            print(f'{task["output_file"]} not exists')
            ready = False
            break
        if not ready:
            break

    print(f"READY {ready}")
    if not ready:
        print("Temp files are not ready!")
        print(db_files)
        print(db_files[0])
        run_dstparser_job(
            len(data_base[0]), db_files[0], task_name="parse_dst", log_dir=log_dir
        )

    ready_tasks = dict()
    ready = True

    log_file = Path(log_dir) / "log_main.dat"
    print("log_file", log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    max_time_to_wait = 2 * 3600  # in sec
    check_every = 10  # sec
    for i in range(max_time_to_wait):
        ready = True
        for task_id, task in data_base[1].items():
            ready_tasks[task_id] = True
            for ifile in task["input_files"]:
                if not Path(ifile).exists():
                    ready = False
                    ready_tasks[task_id] = False

        info_str = ""
        ready_num = 0
        for task_id, task in data_base[1].items():

            if ready_tasks[task_id]:
                ready_num += 1
            fname = str(Path(task["output_file"])).strip()
            info_str += f"\n{task_id} {fname} ready {ready_tasks[task_id]}"

        print(f"READY = {ready}")
        if ready:
            break
        info_str += f"\nAfter {i*check_every}/{max_time_to_wait} sec ({i*check_every/max_time_to_wait*100:.1f}%)\nReady num = {ready_num}/{len(ready_tasks)}"

        with open(log_file, "a") as f:
            f.write(info_str)

        time.sleep(check_every)

    if not ready:
        return

    run_dstparser_job(
        len(data_base[1]), db_files[1], task_name="join_hdf5", log_dir=log_dir
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert DST files to HDF5 format for DNN."
    )
    parser.add_argument(
        "-l", "--log_dir", type=str, required=True, help="Directory for slurm output"
    )
    parser.add_argument(
        "-d",
        "--data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of directories where to search data files",
    )
    parser.add_argument(
        "-g",
        "--glob_patterns",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of patterns for data files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output HDF5 files",
    )

    parser.add_argument(
        "-t",
        "--temp_h5_files",
        type=int,
        default=1000,
        help="Number of temporary HDF5 files (default: 1000)",
    )
    parser.add_argument(
        "-f",
        "--final_h5_files",
        type=int,
        default=100,
        help="Number of final HDF5 files (default: 100)",
    )

    args = parser.parse_args()

    data_base, db_files = generate_db(
        args.data_dirs,
        args.glob_patterns,
        args.output_dir,
        args.temp_h5_files,
        args.final_h5_files,
    )
    main_job(data_base, db_files, log_dir=args.log_dir)
