"""Config for converting TAx4 MC DST files to vlen HDF5 (same format as TA-SD).

Run with the dstparser CLI (SLURM):
    python -m dstparser.cli.cli config_tax4_vlen.py
or drive the two passes directly (see dstparser README "TAx4" section) for a
non-SLURM / single-node run.

No prerequisites: TAx4 is read with the same benMC exe as TA-SD. The only TAx4
difference in the whole vlen path is the yyxx->xxyy detector-id swap, which
`vlen_adapter = "tax4"` below selects (parse_dst_file_tax4_vlen).
"""
from pathlib import Path
import re
import numpy as np
from dstparser.paths import dstbank_root, training_data_root


# -------------------------
# ADAPTER
# -------------------------
# Use the TAx4 vlen parser (parse_dst_file_tax4_vlen) instead of TA-SD's.
vlen_adapter = "tax4"

add_shower_params = True
add_standard_recon = True

# -------------------------
# XMAX
# -------------------------
# TAx4 xmax files are not wired in here -> no xmax column. To add xmax, set
# xmax_dir to the directory holding DAT*_xmax.txt and (if needed) xmax_glob_pattern
# / xmax_model, and confirm the XmaxReader call signature in shower_params.
xmax_dir = None

# -------------------------
# EVENT IDS (optional)
# -------------------------
def add_event_ids(data, filename):
    """Record per-event id + the raw DAT file number. NOTE: the TA-SD
    DATCCCCXX (shower/energy-bin) split is NOT assumed for TAx4 -- only the raw
    filename number is stored (the verified DAT<->shower matching key)."""
    ifname_parts = re.split(r"[_,.\s]", Path(filename).parts[-1])
    if not ifname_parts[0].startswith("DAT"):
        return data
    dat_number = int(ifname_parts[0][3:])
    n = next(iter(data.values())).shape[0]
    data["id_event"] = np.arange(n)
    data["id_dat_file"] = np.full((n,), dat_number, dtype=np.int64)
    return data

# -------------------------
# INPUT
# -------------------------
_base = f"{dstbank_root}/tasdmc_dstbank/tax4/qgsii04proton"
data_dirs = [
    f"{_base}/north/221101to240124",
    f"{_base}/north/240125to240423",
    f"{_base}/south/221101to230731",
    f"{_base}/south/240125to240423",
]
data_globs = "DAT*_gea.rufldf.dst.gz"

# -------------------------
# OUTPUT
# -------------------------
output_dir = f"{training_data_root}/dnn_training_data/2026/07/tax4_qgsii04proton_vlen"

# pass1: DST -> temp_files/temp_NNNNN.h5 ; pass2: temp -> final_files/final_NNNNN.h5
temp_ngroups = 400      # ~how many DST files per temp file (tune to file count)
temp_njobs = 50         # concurrent SLURM array tasks for pass1
final_ngroups = 20      # number of final merged h5 files
final_njobs = 20        # concurrent SLURM array tasks for pass2

# -------------------------
# SLURM (edit for your cluster)
# -------------------------
slurm_settings = {
    "job-name": "tax4_vlen",
    "array": "0",
    "ntasks": 1,
    "mem": "20gb",
    "cpus-per-task": 1,
    "partition": "edr1_short",
    "time": "02:00:00",
}
