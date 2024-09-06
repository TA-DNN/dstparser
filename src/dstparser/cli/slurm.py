import os
import subprocess
import shutil
from pathlib import Path


def copy_files(src_paths, dest_dir):

    if not isinstance(src_paths, list):
        src_paths = [src_paths]

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src_path in src_paths:
        path = Path(src_path)
        if any(char in str(src_path) for char in "*?"):
            files = [Path(p) for p in path.parent.glob(path.name)]
        else:
            files = [path]
        for file in files:
            print(files, str(dest_dir / file.name))
            shutil.copyfile(file, str(dest_dir / file.name))


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


def task_info(zero_indexing=False):
    slurm_params = slurm_parameters()

    if slurm_params["task_id"] is None:
        return None, None

    task_id = slurm_params["task_id"]
    ntasks = slurm_params["task_max"] - slurm_params["task_min"] + 1

    if zero_indexing:
        task_id -= slurm_params["task_min"]

    return task_id, ntasks


def run_slurm_job(
    slurm_directives,
    log_dir,
    script,
    options="",
    app="python",
    suffix="",
    batch_command="sbatch",
):
    """
    slurm_directives is dictionary, e.g.:

    slurm_directives = {
        "job-name": "procsh",
        "array": "0",
        "ntasks": 1,
        "exclude": "hpa-wn[11,13]",
        "mem": "20gb",
        "cpus-per-task": 1,
        "partition": "hdr1-al9_short",
        "time": "01:00:00",
    }
    """

    # Create log dirs
    log_dir = Path(log_dir)
    out_dir = log_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    err_dir = log_dir / "error"
    err_dir.mkdir(parents=True, exist_ok=True)

    # Create slurm script
    sbatch = "\n#SBATCH --"
    slurm_script = "#!/bin/bash\n"
    for key, value in slurm_directives.items():
        slurm_script += f"{sbatch}{key}={value}"
    slurm_script += f"{sbatch}output={str(out_dir)}/%x-%A_%4a.out"
    slurm_script += f"{sbatch}error={str(err_dir)}/%x-%A_%4a.err"

    # cd to the directory of launching script
    working_dir = Path(script).parent
    command = f"{app.strip()} {str(script).strip()} {options.strip()}"
    slurm_script += f"\n\ncd {str(working_dir)}\n{command.strip()}"

    # Save slurm script
    slurm_file = log_dir / f"src/slurm_script{suffix}.sh"
    slurm_file.parent.mkdir(parents=True, exist_ok=True)
    with open(slurm_file, "w") as f:
        f.write(slurm_script)

    # Run slurm script
    proc = subprocess.run(
        [batch_command, f"{slurm_file}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Print whether job was submitted
    print(proc.stdout.decode())
    err = proc.stderr.decode()
    if err.strip():
        print(f"stderr:\n{err}")
