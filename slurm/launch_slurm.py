from srecog.slurm.slurm_launcher import output_env, run_job


root_dir = "/dicos_ui_home/antonpr/ml/erdmann/generated_data"
out_dir = root_dir + "/12_dst_TA/03_xmax_7x7"
res_name = "proc"
res_dir = out_dir + f"/results/{res_name}"
out_dir = out_dir + f"/runs/{res_name}"

output_file, error_file, script_file = output_env(out_dir)


slurm_directives = {
    "job-name": "proc_dst9",
    "array": "0-999",
    "ntasks": 1,
    "mem": "3gb",
    "cpus-per-task": 1,
    "partition": "short_serial",
    "time": "01:00:00",
    "output": output_file,
    "error": error_file,
}


from pathlib import Path

python_script = (
    "/ceph/work/SATORI/projects/TA-ASIoP"
    "/sdanalysis_2018_TALE_TAx4SingleCT_DM/"
    "antonpr/dstparser/src/dstparser/process_files.py"
)

env_script = (
    "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM"
    "/antonpr/dstparser/src/dstparser/sdanalysis_env.sh"
)
working_dir = str(Path(python_script).parent)

app_info = {
    "application": f"{env_script}; python {python_script} {res_dir}",
    "options": {},
    "working_dir": working_dir,
    "backup": {
        "src": python_script,
    },
}


run_job(slurm_directives, app_info, script_file)
