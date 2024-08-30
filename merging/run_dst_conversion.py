import subprocess
import shutil
from pathlib import Path


def slurm_directives():
    return {
        "job-name": "procsh",
        "array": "0",
        "ntasks": 1,
        # "exclude": "hpa-wn[11,13]",
        "mem": "20gb",
        "cpus-per-task": 1,
        "partition": "short_serial",
        "time": "01:00:00",
    }


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


def run_slurm_job(
    slurm_directives,
    log_dir,
    script,
    options="",
    app="python",
    suffix="",
    batch_command="sbatch",
):
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

    print(proc.stdout.decode())
    err = proc.stderr.decode()
    if err.strip():
        print(f"stderr:\n{err}")


def run_main_job():

    output_dir = (
        "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/08/01_mc_wdate/"
    )

    data_dirs = [
        # "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo",
        # "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04iron/080417_160603/Em1_bsdinfo",
        "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04helium/080417_160603/Em1_bsdinfo",
    ]

    # data_dirs = [
    #     "/ceph/work/SATORI/projects/TA-ASIoP/tasdobs_dstbank/rufldf",
    # ]

    data_dirs = " ".join(data_dirs)
    # patterns = "tasdcalibev*rufldf.dst.gz"
    patterns = "DAT*dst.gz"
    num_temp_files = 1000
    num_final_files = 50

    script = Path(__file__).parent / "main_job.py"
    log_dir = Path(output_dir) / "slurm_run"
    options = f"--log_dir {str(log_dir)} "
    options += f"--data_dirs {data_dirs} "
    options += f'--glob_patterns "{patterns}" '
    options += f"--output_dir {output_dir} "
    options += f"--temp_h5_files {num_temp_files} "
    options += f"--final_h5_files {num_final_files} "

    # print(options)
    run_slurm_job(
        slurm_directives(),
        log_dir,
        script,
        options,
        suffix="_main_job",
        batch_command="bash",
    )


if __name__ == "__main__":
    run_main_job()
