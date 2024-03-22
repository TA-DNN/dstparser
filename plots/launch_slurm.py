from pathlib import Path
from srecog.slurm.slurm_launcher import output_env, run_job


root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/03"
out_dir = root_dir + "/03_ttrace_movie/02_TA_9x9"
res_name = "proc"
res_dir = out_dir + f"/results/{res_name}"

output_file, error_file, script_file = output_env(out_dir)


slurm_directives = {
    "job-name": "proc_dst9x9",
    "array": "0-5",
    "ntasks": 1,
    "mem": "3gb",
    "cpus-per-task": 1,
    "partition": "edr1_short_serial",
    "time": "01:00:00",
    "output": output_file,
    "error": error_file,
}

python_script = (
    "/ceph/work/SATORI/antonpr/ml"
    "/erdmann/dstparser/tasks/03_trace_all/visualize_traces.py"
)


working_dir = str(Path(python_script).parent)

app_info = {
    "application": f"python {python_script} {res_dir}",
    "options": {},
    "working_dir": working_dir,
    "backup": {
        "src": f"{working_dir}/*.py",
    },
}


run_job(slurm_directives, app_info, script_file)
