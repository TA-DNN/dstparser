import os
from pathlib import Path
import subprocess
from env_vars import changed_env_paths


root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM"
dst_reader = "sditerator.run"
sd_analysis_env = "sdanalysis_env.sh"

sd_analysis_env = str(Path(root_dir) / sd_analysis_env)

for env_var, path_var in changed_env_paths(sd_analysis_env).items():
    os.environ[env_var] = path_var


def read_dst_file(dst_filename):
    try:
        process = subprocess.Popen(
            [dst_reader, dst_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ,
        )
        output, error = process.communicate()
    except subprocess.CalledProcessError as e:
        output = e.output

    return output.strip().split("\n")
