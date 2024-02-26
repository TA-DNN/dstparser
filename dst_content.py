import os
from pathlib import Path
import subprocess


root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM"
dst_reader = "bin/sditerator.run"

root_dir = Path(root_dir)
dst_reader = root_dir / dst_reader
for env_var, path_var in zip(["LD_LIBRARY_PATH", "PATH"], ["lib", "bin"]):
    os.environ[env_var] += ":" + str(root_dir / path_var)
    # print(env_var, os.environ[env_var])

def capture_output(cmd):
    try:
        # output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        output, error = process.communicate()
        # print(error)
        return output
    except subprocess.CalledProcessError as e:
        return e.output


def dst_content(filename):
    return capture_output([dst_reader, filename]).strip().split("\n")