"""Machine-dependent paths.

Three category roots, because the things they point at can move independently:
software installs, input DST banks, and outputs. All three default to the same
TA project area (`ta_root`), so a remount is still a one-line change -- but you
can, say, build sdanalysis locally while the dstbanks stay on shared storage.

    ta_root              $DSTPARSER_TA_ROOT           the TA project area
    +-- sdanalysis_root  $DSTPARSER_SDANALYSIS_ROOT   benMC, TALE install, ROOT
    +-- dstbank_root     $DSTPARSER_DSTBANK_ROOT      tasdmc/tasdobs dstbanks
    +-- training_data_root $DSTPARSER_TRAINING_ROOT   dnn_training_data outputs

The mount point has moved more than once, which is why none of these names
mention the storage they happen to sit on.

Every reader used here is a binary under sdanalysis_root -- nothing has to be
built locally, including for TAx4.
"""

import os
from pathlib import Path

# The TA project area: the default base for all three category roots below.
ta_root = os.environ.get(
    "DSTPARSER_TA_ROOT", "/ceph/sharedfs/work/SATORI/projects/TA-ASIoP"
)

# Software installs: sdanalysis (benMC + TALE), ROOT, openssl10.
sdanalysis_root = os.environ.get("DSTPARSER_SDANALYSIS_ROOT", ta_root)
# Input data: tasdmc_dstbank, tasdobs_dstbank, INR_group.
dstbank_root = os.environ.get("DSTPARSER_DSTBANK_ROOT", ta_root)
# Outputs: dnn_training_data.
training_data_root = os.environ.get("DSTPARSER_TRAINING_ROOT", ta_root)

# Root path to the sdanalysis install providing the DST readers
root_dir = f"{sdanalysis_root}/benMC/sdanalysis_2019"
dst_reader = "sditerator_no_standard_recon.run"
dst_reader_add_standard_recon = "sditerator_add_standard_recon_v2.run"
dst_reader_all_events = "sditerator_printAll.run"

sd_analysis_env = "sdanalysis_env.sh"

# Both sdanalysis_env.sh scripts on shared storage `source` ROOT from a mount
# that no longer exists, and they are read-only, so ROOT is set up here
# explicitly after the install env. Without this every reader dies with
# "libCore.so: cannot open shared object file".
root_env_benmc = f"{sdanalysis_root}/install/root/bin/thisroot.sh"
root_env_tax4 = f"{sdanalysis_root}/root/bin/thisroot.sh"
openssl10_alma9 = f"{sdanalysis_root}/benMC/libs_alma9/openssl10"
openssl10_rocky_linux = f"{sdanalysis_root}/benMC/libs_rocky_linux/openssl10"

# Data for xmax
xmax_data_files = "DAT*_xmax.txt"
xmax_data_dir_prot = f"{dstbank_root}/tasdmc_dstbank/qgsii04proton/080417_160603/Em1/"

xmax_data_dir_fe = f"{dstbank_root}/tasdmc_dstbank/qgsii04iron/080417_160603/Em1/"

# A SEPARATE sdanalysis install built for TAx4/TALE geometry, with its own
# rufptn.run/rufldf.run pass1/pass2 chain. Used only by dst_adapter_tax4.py --
# the vlen path reads TAx4 with the benMC reader above.
root_dir_tax4_std_recon = f"{sdanalysis_root}/sdanalysis_2018_TALE_TAx4SingleCT_DM"
dst_reader_tax4_std_recon = "sditerator_add_standard_recon.run"
