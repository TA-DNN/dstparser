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
from dstparser import append_to_hdf5, parse_dst_file_vlen
import h5py


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


def update_xmax_reader(xmax_reader, file):
    """
    Update the xmax_reader if the directory of the file changes.
    """

    if xmax_reader is None:
        return None

    parent_dir = Path(file).parent
    if xmax_reader.data_dir != parent_dir:
        glob_pattern = xmax_reader.glob_pattern
        model = xmax_reader.model
        xmax_reader = XmaxReader(parent_dir, glob_pattern, model)
    return xmax_reader


def init_xmax_reader(config, init_file=None):
    """
    Typically, xmax_reader reads data files for xmax from a parent directory of dst files
    (this is how it is used now).
    However, for TAx4 files this behavior should be changed, because xmax files are in a different directory.
    In this case xmax_dir should be set in the config.
    """

    # Default values, can be overridden by config
    # If init_file is provided and xmax_dir is not overriden,
    # use its parent directory as the xmax_dir (default behavior)
    xmax_dir = "parent_dir"
    xmax_glob_pattern = "**/DAT*_xmax.txt"
    xmax_model = "QGSJetII-04"

    # If xmax_dir is set to None, xmax_reader will be None
    # If xmax_dir is set, then only xmax_dir directory will be used
    if hasattr(config, "xmax_dir"):
        xmax_dir = config.xmax_dir

    if hasattr(config, "xmax_glob_pattern"):
        xmax_glob_pattern = config.xmax_glob_pattern

    if hasattr(config, "xmax_model"):
        xmax_model = config.xmax_model

    update_reader = False
    if xmax_dir == "parent_dir":
        if init_file is not None:
            xmax_dir = Path(init_file).parent
            update_reader = True  # standard behavior
        else:
            xmax_dir = None

    if xmax_dir is None:
        xmax_reader = None
    else:
        xmax_reader = XmaxReader(
            data_dir=xmax_dir, glob_pattern=xmax_glob_pattern, model=xmax_model
        )

    return xmax_reader, update_reader


def dst_to_hdf5(ifiles, ofile, config):

    acc_data = dict()
    xmax_reader, update_reader = init_xmax_reader(config, init_file=ifiles[0])

    for file in tqdm(
        ifiles, total=len(ifiles), desc=f"DST conversion for {Path(ofile).name}"
    ):

        if update_reader:
            # Update xmax_reader if the directory of the file changes
            xmax_reader = update_xmax_reader(xmax_reader, file)

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
            continue

        # Filter data
        # data = filter_full_tiles(data, max_events=50)

        # Distribute by fields
        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, ofile)


def join_hdf5(ifiles, ofile, config):
    acc_data = dict()
    for ifile in tqdm(
        ifiles, total=len(ifiles), desc=f"Joining files for {Path(ofile).name}"
    ):

        skipped_arrays = None
        if hasattr(config, "skipped_arrays"):
            skipped_arrays = config.skipped_arrays

        data = read_h5(ifile)
        # Distribute by fields
        for key, value in data.items():
            if skipped_arrays is not None:
                # Sometimes we need to skip ["time_traces_low", "time_traces_up"]
                if key in skipped_arrays:
                    continue
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, ofile)


def dst_to_hdf5_vlen(ifiles, ofile, config):
    xmax_reader, update_reader = init_xmax_reader(config, init_file=ifiles[0])

    with h5py.File(ofile, "a") as f:
        for file in tqdm(
            ifiles, total=len(ifiles), desc=f"DST conversion for {Path(ofile).name}"
        ):

            if update_reader:
                # Update xmax_reader if the directory of the file changes
                xmax_reader = update_xmax_reader(xmax_reader, file)

            data = parse_dst_file_vlen(
                file,
                xmax_reader=xmax_reader,
                add_shower_params=True,
                add_standard_recon=True,
                config=config,
            )

            if data is None:
                continue

            append_to_hdf5(f, data)


def join_hdf5_vlen(ifiles, ofile, config):
    with h5py.File(ofile, "a") as f:

        for ifile in tqdm(
            ifiles, total=len(ifiles), desc=f"Joining files for {Path(ofile).name}"
        ):
            data = read_h5(ifile)
            append_to_hdf5(f, data)


def worker_job():

    config = parse_config(sys.argv[3])

    if sys.argv[2] == "parse_dst":
        process_files(dst_to_hdf5, config)
    elif sys.argv[2] == "join_hdf5":
        process_files(join_hdf5, config)
    elif sys.argv[2] == "parse_dst_vlen":
        process_files(dst_to_hdf5_vlen, config)
    elif sys.argv[2] == "join_hdf5_vlen":
        process_files(join_hdf5_vlen, config)


if __name__ == "__main__":
    worker_job()
