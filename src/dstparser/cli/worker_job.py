import sys
import json
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from dstparser import parse_dst_file
from dstparser.xmax_reader import XmaxReader
from .io import read_h5, save2hdf5
from .job_runner import task_info
from .data_filters import filter_full_tiles


def join_hdf5():

    task_id, ntasks = task_info()
    if task_id is None:
        raise ValueError("No slurm is found!")
        # task_id = sys.argv[3]

    with open(sys.argv[1], "r") as f:
        task_db = json.load(f)

    task_id = str(task_id)
    filename = task_db[task_id]["output_file"]

    print(f"Joining files for {filename}")

    filename = Path(filename)
    if filename.exists():
        print(f"{filename} is already exists!")
        return
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)

    acc_data = dict()
    ifiles = task_db[task_id]["input_files"]

    for ifile in tqdm(ifiles, total=len(ifiles), desc=f"Joining files for {filename}"):
        data = read_h5(ifile)
        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, filename)


def dst_to_hdf5():

    task_id, ntasks = task_info()
    if task_id is None:
        raise ValueError("No slurm is found!")
        # task_id = sys.argv[3]

    with open(sys.argv[1], "r") as f:
        task_db = json.load(f)

    task_id = str(task_id)
    filename = task_db[task_id]["output_file"]

    filename = Path(filename)
    print(f"Create file {str(filename)}")
    if filename.exists():
        return
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)

    acc_data = dict()
    ifiles = task_db[task_id]["input_files"]
    xmax_dir = Path(ifiles[0]).parent
    xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")
    # xmax_reader = None
    for file in tqdm(ifiles, total=len(ifiles), desc="DST conversion"):

        if xmax_reader is not None:
            # xmax_dir is the same as directory of the file
            # but it loads many files, so better to change and
            # initialize XmaxReader object only when the directory
            # changes
            cur_dir = Path(file).parent
            if xmax_dir != cur_dir:
                xmax_dir = cur_dir
                xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")

        data = parse_dst_file(
            file,
            ntile=7,
            xmax_reader=xmax_reader,
            avg_traces=False,
            add_shower_params=True,
            add_standard_recon=True,
        )

        if data is None:
            continue

        data = filter_full_tiles(data, max_events=50)

        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, filename)


def worker_job():

    if sys.argv[2] == "parse_dst":
        dst_to_hdf5()
    elif sys.argv[2] == "join_hdf5":
        join_hdf5()


if __name__ == "__main__":
    worker_job()
