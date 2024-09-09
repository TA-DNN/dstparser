import sys
import json
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from dstparser import parse_dst_file
from dstparser.xmax_reader import XmaxReader
from dstparser.cli.io import read_h5, save2hdf5
from dstparser.cli.slurm import task_info
from dstparser.cli.data_filters import filter_full_tiles
from dstparser.cli.cli import parse_config


def process_files(task_function, config):

    task_id, ntasks = task_info()
    if task_id is None:
        raise ValueError("No slurm is found!")

    with open(sys.argv[1], "r") as f:
        task_db = json.load(f)

    task = task_db[str(task_id)]
    ofile_ids = []
    for ofile_id in task:
        ofile = Path(task[ofile_id]["output_file"])
        if not ofile.exists():
            ofile_ids.append(ofile_id)
        else:
            print(f"Exist: {ofile}")

    for ofile_id in ofile_ids:
        ifiles = task[ofile_id]["input_files"]
        ofile = task[ofile_id]["output_file"]
        Path(ofile).parent.mkdir(parents=True, exist_ok=True)
        task_function(ifiles, ofile, config)


def dst_to_hdf5(ifiles, ofile, config):

    acc_data = dict()
    xmax_dir = Path(ifiles[0]).parent
    xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")
    # xmax_reader = None
    for file in tqdm(
        ifiles, total=len(ifiles), desc=f"DST conversion for {Path(ofile).name}"
    ):

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
            config=config,
        )

        if data is None:
            print(f"Data is empty for {file}")
            continue

        # Filter data
        data = filter_full_tiles(data, max_events=50)

        # Distribute by fields
        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, ofile)


def join_hdf5(ifiles, ofile, config):
    acc_data = dict()
    for ifile in tqdm(
        ifiles, total=len(ifiles), desc=f"Joining files for {Path(ofile).name}"
    ):
        data = read_h5(ifile)
        # Distribute by fields
        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, ofile)


def worker_job():

    config = parse_config(sys.argv[3])

    if sys.argv[2] == "parse_dst":
        process_files(dst_to_hdf5, config)
    elif sys.argv[2] == "join_hdf5":
        process_files(join_hdf5, config)


if __name__ == "__main__":
    worker_job()
