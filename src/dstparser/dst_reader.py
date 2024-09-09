import os
from pathlib import Path
import subprocess
from dstparser.env_vars import changed_env_paths
from dstparser.paths import (
    root_dir,
    dst_reader_add_standard_recon,
    sd_analysis_env,
)


# Loading environment from "sdanalysis_env.sh"
sd_analysis_env = str(Path(root_dir) / sd_analysis_env)
for env_var, path_var in changed_env_paths(sd_analysis_env).items():
    os.environ[env_var] = path_var


def read_dst_file(dst_filename):
    dst_reader_process = dst_reader_add_standard_recon
    try:
        process = subprocess.Popen(
            [dst_reader_process, str(dst_filename).strip()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ,
        )
        output, error = process.communicate()
    except subprocess.CalledProcessError as e:
        output = e.output

    # If output is empty
    if len(output) == 0:
        print(f'dst_reader error:\n"{error}"')

    return output.strip().split("\n")
