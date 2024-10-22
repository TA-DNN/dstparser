from pathlib import Path
import re
import numpy as np

# -------------------------
# EVENT ID SCHEME:
# -------------------------

# Provide numerical code for data set
# data_set_root = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
data_set_root = "/ceph/sharedfs/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
data_set_base = dict()

data_set_base[(data_set_root + "qgsii04proton/080417_160603/Em1_bsdinfo").strip()] = (
    10010001
)

data_set_base[
    (data_set_root + "qgsii03proton/080511_230511/noCuts_HiResSpectrum").strip()
] = 10010002

data_set_base[(data_set_root + "qgsii04helium/080417_160603/Em1_bsdinfo").strip()] = (
    10040001
)

data_set_base[(data_set_root + "qgsii04nitrogen/080417_160603/Em1_bsdinfo").strip()] = (
    10140001
)

data_set_base[(data_set_root + "qgsii04iron/080417_160603/Em1_bsdinfo").strip()] = (
    10560001
)


data_set_base[(data_set_root + "qgsii03iron/6yrs").strip()] = 10560002


# Function that adds id fields to h5 files
def add_event_ids(data, filename):
    """Write additional information retrived from file name"""
    ifname = Path(filename).parts[-1]
    ifname_parts = re.split(r"[_,.\s]", ifname)

    keys = []
    values = []
    # TA DATA
    if ifname_parts[0].startswith("tasdcalibev"):
        key, value = "ta_obs_date", int(ifname_parts[2])
        keys.append(key)
        values.append(value)
    # TA MC
    elif ifname_parts[0].startswith("DAT"):
        # Add corsika_shower_id CCCC for the scheme DATCCCCXX
        key, value = "id_corsika_shower", int(ifname_parts[0][3:7])
        keys.append(key)
        values.append(value)

        # Add energy bin id XX for the scheme DATCCCCXX
        key, value = "id_energy_bin", int(ifname_parts[0][7:9])
        keys.append(key)
        values.append(value)

        # Add data set id
        dset_key = str(Path(filename).parent)
        if data_set_base.get(dset_key):
            keys.append("id_data_set")
            values.append(data_set_base[dset_key])

    else:
        key, value = None, None

    if len(keys) > 0:
        data_len = next(iter(data.values())).shape[0]
        data["id_event"] = np.arange(data_len)
        for key, value in zip(keys, values):
            data[key] = np.full((data_len,), value, dtype=np.int64)

    return data


# -------------------------
# SLURM SETTINGS:
# -------------------------

# Modify settings according your environment
slurm_settings = {
    "job-name": "procsh",
    "array": "0",
    "ntasks": 1,
    # "exclude": "hpa-wn[11,13]",
    "mem": "20gb",
    "cpus-per-task": 1,
    "partition": "edr1_short",
    "time": "01:00:00",
}

# -------------------------
# INPUT:
# -------------------------

# List of directories with the location of DST files
# Uncomment or modify to specify your own directories
# Example:
# data_dirs = [
#     "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo",
#     "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04iron/080417_160603/Em1_bsdinfo",
# ]

data_dirs = [
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii03iron/6yrs",
]

# Glob patterns to match DST files to be processed
# Uncomment or modify one of the examples or provide your own pattern
# Example:
# glob_patterns = "tasdcalibev*rufldf.dst.gz"
# glob_patterns = "DAT*dst.gz"

glob_patterns = "DAT*dst.gz"

# List of data files. If this is specified, it overrides 'data_dirs' and 'glob_patterns'
# Uncomment and specify to use specific files instead of directories
# data_files = [
#     "/path/to/file1.dst.gz",
#     "/path/to/file2.dst.gz",
# ]


# -------------------------
# OUTPUT:
# -------------------------

# Explanation:
# The conversion happens in 2 steps.
# First step "temp pass" converts dst files and writes them in multiple temp files.
# The original dst files grouped by "temp_group_by" files to produces one temp file.
# The task is separated between "njobs_temp_pass" slurm jobs.

# The "final pass" merge temporary files to larger files.
# The temporary files grouped by "final_group_by" files to produce one final file.
# The task is done by "njobs_final_pass" slurm jobs.

# Temp pass:

# Group original files to temp file by group
# By default temp_group_by = 26 group energy bins will produce 1000 temp files
temp_group_by = 26
# Number of jobs for the step that creates temporary files
njobs_temp_pass = 50

# Final pass:

# Group/merge temp files by group of final_group_by
# In default settings divide 1000 files by 50 files = 20 final files
final_group_by = 5
# Number of jobs for the step that merge temporary files
njobs_final_pass = 50

# Prefix for the final file names
# Example: if file_name_pattern = "prot", output files will be "prot_01.h5", "prot_02.h5", etc.
file_name_pattern = "final"

# Directory to save all logs, temporary, and final files. Created automatically if not exist
output_dir = (
    "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/10/02_iron_qgsii03/"
)
