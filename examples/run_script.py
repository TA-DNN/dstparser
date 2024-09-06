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
    "partition": "hdr1-al9_short",
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
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04iron/080417_160603/Em1_bsdinfo",
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

# Number of final output files
# Adjust based on the expected number of final files
num_final_files = 20

# Prefix for the final file names
# Example: if file_name_pattern = "prot", output files will be "prot01.h5", "prot02.h5", etc.
file_name_pattern = "ni_full"

# Directory to save all logs, temporary, and final files. Created automatically if not exist
output_dir = (
    "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2024/09/10_test_mc_nitrogen/"
)
