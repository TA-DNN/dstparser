import os
from pathlib import Path
import subprocess


#  'ROOTSYS': '/ceph/work/SATORI/projects/TA-ASIoP/root',
# 'DYLD_LIBRARY_PATH': '/ceph/work/SATORI/projects/TA-ASIoP/root/lib',
#  'LIBPATH': '/ceph/work/SATORI/projects/TA-ASIoP/root/lib',
#  'PYTHONPATH': '/ceph/work/SATORI/projects/TA-ASIoP/root/lib',
#  'SHLIB_PATH': '/ceph/work/SATORI/projects/TA-ASIoP/root/lib',

rootsys = "/ceph/work/SATORI/projects/TA-ASIoP/root"
rootlib = str(Path(rootsys) / "lib")
root_vars = {
    "ROOTSYS": rootsys,
    "DYLD_LIBRARY_PATH": rootlib,
    "LIBPATH": rootlib,
    "PYTHONPATH": rootlib,
    "SHLIB_PATH": rootlib,
}

for key, val in root_vars.items():
    current_value = os.environ.get(key, "")
    os.environ[key] = current_value + ":" + val
    # os.environ[key] += ":" + val

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
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ,
        )
        output, error = process.communicate()
        return output
    except subprocess.CalledProcessError as e:
        return e.output


def dst_content(filename):
    return capture_output([dst_reader, filename]).strip().split("\n")
