import os
from pathlib import Path
import subprocess
from env_vars import changed_env_paths
from xmax_reader import XmaxReader


root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM"
dst_reader = "sditerator.run"
sd_analysis_env = "sdanalysis_env.sh"
sd_analysis_env = str(Path(root_dir) / sd_analysis_env)

# Loading environment from "sdanalysis_env.sh"
for env_var, path_var in changed_env_paths(sd_analysis_env).items():
    os.environ[env_var] = path_var

# Loading data for xmax
xmax_data_files = "DAT*_xmax.txt"
xmax_data_dir = (
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
    "qgsii03proton/080511_230511/noCuts_HiResSpectrum/"
)

xmax_data = XmaxReader(xmax_data_dir, xmax_data_files)


def read_xmax_data(dst_filename):
    xmax_data._read_file(dst_filename)
    return xmax_data


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
