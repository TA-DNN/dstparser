from tqdm import tqdm
import numpy as np
from dstparser import parse_dst_file
from srecog.loaders.read_data import data_files
from srecog.utils.hdf5_utils import save_dict_to_hdf5
from srecog.utils.info import info_wrapper
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


def run_dst_process():
    """Returns settings for task with task_id among ntasks,
    where task_id in the range 0..ntasks-1
    """

    task_id, ntasks = normalized_task_id()
    print(f"task_id = {task_id}")
    print(f"ntasks = {ntasks}")

    dst_directory = (
        "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
        "qgsii03proton/080511_230511/noCuts_HiResSpectrum/"
    )

    dst_files = data_files(
        data_dir=dst_directory, glob_pattern="**/DAT*noCuts.dst.gz", divide_in=ntasks
    )
    outdir = Path(sys.argv[1])
    dst_files_task = dst_files[task_id]

    for infile in dst_files_task:
        dir_fname = infile.parts[-2:]
        outfile = f"{dir_fname[0]}/{dir_fname[1]}".split(sep="_")[0] + ".h5"
        outfile = outdir / outfile
        outfile.parent.mkdir(parents=True, exist_ok=True)
        print(outfile)
        data = parse_dst_file(str(infile), ntile=7)
        if data is None:
            print("NONE!")
            continue
        else:
            nevents = data["mass_number"].shape[0]
            print(f"Save {nevents} events")
            save_dict_to_hdf5(outfile, data)


def test():

    root_dir = "/dicos_ui_home/antonpr/ml/erdmann/generated_data"
    out_dir = root_dir + "/12_dst_TA/02_run"
    res_name = "proc"
    res_dir = out_dir + f"/results/{res_name}"

    dst_directory = (
        "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
        "qgsii03proton/080511_230511/noCuts_HiResSpectrum/"
    )

    dst_files = data_files(
        data_dir=dst_directory, glob_pattern="**/DAT*noCuts.dst.gz", divide_in=1000
    )
    outdir = Path(res_dir)
    dst_files_task = dst_files[0]

    for infile in dst_files_task:
        dir_fname = infile.parts[-2:]
        outfile = f"{dir_fname[0]}/{dir_fname[1]}".split(sep="_")[0] + ".h5"
        outfile = outdir / outfile
        print(outfile)

        # outfile.parent.mkdir(parents=True, exist_ok=True)
        data = parse_dst_file(str(infile))
        if data is None:
            print("NONE!")
            continue
        else:
            nevents = data["mass_number"].shape[0]
            print(f"Saved {nevents} events")

    # for infile in dst_files_task:
    #     dir_fname = infile.parts[-2:]
    #     outfile = f"{dir_fname[0]}/{dir_fname[1]}".split(sep="_")[0] + ".h5"
    #     outfile = outdir / outfile
    #     print(outfile)
    # outfile.parent.mkdir(parents=True, exist_ok=True)
    # data = parse_dst_file(str(infile))
    # print(outfile)
    # save_dict_to_hdf5(outfile, data)


if __name__ == "__main__":
    info_wrapper(run_dst_process)
    # test()
