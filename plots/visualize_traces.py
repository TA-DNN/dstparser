from tqdm import tqdm
import numpy as np
from dstparser import parse_dst_file
from srecog.loaders.read_data import data_files
from srecog.utils.hdf5_utils import save_dict_to_hdf5
from srecog.utils.hdf5_utils import read_hdf5_metadata
from srecog.loaders.split_data import shuffled_indices
from srecog.utils.info import info_wrapper
from tile_movie import tile_signal_movie
from pathlib import Path
import os
import sys


def slurm_parameters():

    env_vars = {
        "task_id": "SLURM_ARRAY_TASK_ID",
        "task_min": "SLURM_ARRAY_TASK_MIN",
        "task_max": "SLURM_ARRAY_TASK_MAX",
        "job_id": "SLURM_JOB_ID",
    }

    slurm_params = dict()

    for key, envar in env_vars.items():
        val = os.environ.get(envar)
        try:
            slurm_params[key] = int(val)
        except TypeError:
            slurm_params[key] = val

    return slurm_params


def normalized_task_id():
    slurm_params = slurm_parameters()
    task_id = None
    ntasks = None

    if slurm_params["task_id"] is not None:
        task_id = slurm_params["task_id"] - slurm_params["task_min"]
        ntasks = slurm_params["task_max"] - slurm_params["task_min"] + 1
    # if slurm_params["task_id"] is not None:
    #     task_id = slurm_params["task_id"]
    #     ntasks = slurm_params["task_max"] - slurm_params["task_min"] + 1

    return task_id, ntasks


def choose_events(data_file, max_events=10):
    shapes = read_hdf5_metadata(data_file)
    data_length = shapes["energy"]["shape"][0]
    # print(data_length)
    num_events = min(max_events, data_length)

    return shuffled_indices(data_length, seed=1)[:num_events]


def run_make_movies():
    """Returns settings for task with task_id among ntasks,
    where task_id in the range 0..ntasks-1
    """

    task_id, ntasks = normalized_task_id()
    print(f"task_id = {task_id}")
    print(f"ntasks = {ntasks}")

    data_dir = "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/03/01_TA_dst/02_uplow_traces9x9/results/proc"
    proc_files = data_files(data_dir=data_dir, glob_pattern="**/DAT*.h5")

    data_file = proc_files[task_id]
    event_idxs = choose_events(data_file)

    outdir = Path(sys.argv[1])

    for i, event_idx in enumerate(event_idxs):
        print(i, event_idx)

        tile_signal_movie(
            data_file=data_file,
            event_idx=event_idx,
            time_slice=slice(None),
            cmap="Blues",
            out_dir=outdir,
        )


def test():

    root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/03"
    out_dir = root_dir + "/03_ttrace_movie/01_TA_9x9"
    res_name = "proc"
    res_dir = out_dir + f"/results/{res_name}"

    data_dir = "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/03/01_TA_dst/02_uplow_traces9x9/results/proc"
    proc_files = data_files(data_dir=data_dir, glob_pattern="**/DAT*.h5")

    task_id = 0
    data_file = proc_files[task_id]
    event_idxs = choose_events(data_file)

    # outdir = Path(sys.argv[1])

    for i, event_idx in enumerate(event_idxs):
        print(i, event_idx)
        tile_signal_movie(
            data_file=data_file,
            event_idx=event_idx,
            time_slice=slice(None),
            cmap="Blues",
            out_dir=out_dir,
        )
        break


if __name__ == "__main__":
    info_wrapper(run_make_movies)
    # info_wrapper(test)
