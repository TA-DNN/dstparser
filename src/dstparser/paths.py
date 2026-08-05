"""Machine-dependent paths.

Three category roots, because the things they point at can move independently:
software installs, input DST banks, and outputs. All three default to the same
TA project area (`ta_root`), so a remount is still a one-line change -- but you
can, say, build sdanalysis locally while the dstbanks stay on shared storage.

    ta_root              $DSTPARSER_TA_ROOT           the TA project area
    +-- sdanalysis_root  $DSTPARSER_SDANALYSIS_ROOT   sdanalysis install, ROOT
    +-- dstbank_root     $DSTPARSER_DSTBANK_ROOT      tasdmc/tasdobs dstbanks
    +-- training_data_root $DSTPARSER_TRAINING_ROOT   dnn_training_data outputs

The mount point has moved more than once, which is why none of these names
mention the storage they happen to sit on.
"""

import os
from pathlib import Path

# The TA project area: the default base for all three category roots below.
ta_root = os.environ.get(
    "DSTPARSER_TA_ROOT", "/ceph/sharedfs/work/SATORI/projects/TA-ASIoP"
)

# Software installs: sdanalysis, ROOT, openssl10.
sdanalysis_root = os.environ.get("DSTPARSER_SDANALYSIS_ROOT", ta_root)
# Input data: tasdmc_dstbank, tasdobs_dstbank, INR_group.
dstbank_root = os.environ.get("DSTPARSER_DSTBANK_ROOT", ta_root)
# Outputs: dnn_training_data.
training_data_root = os.environ.get("DSTPARSER_TRAINING_ROOT", ta_root)

# Root path to the sdanalysis install providing the DST readers
root_dir = f"{sdanalysis_root}/benMC/sdanalysis_2019"
# The reader used for everything: TA-SD and TAx4 alike. Emits the 61-field
# #EVENT record and 12-field #SD meta (incl. rufptn_.nfold) for both.
dst_reader_add_standard_recon = "sditerator_add_standard_recon_v2.run"
# Every THROWN event (triggered or not), truth + raw SD data, no reconstruction.
# For full per-shower statistics; not used by the parsers.
dst_reader_all_events = "sditerator_printAll.run"

sd_analysis_env = "sdanalysis_env.sh"

# sdanalysis_env.sh `source`s ROOT from a mount that no longer exists, and it
# is read-only, so ROOT is set up here explicitly after the install env.
# Without this every reader dies with "libCore.so: cannot open shared object file".
root_env_benmc = f"{sdanalysis_root}/install/root/bin/thisroot.sh"
openssl10_alma9 = f"{sdanalysis_root}/benMC/libs_alma9/openssl10"
openssl10_rocky_linux = f"{sdanalysis_root}/benMC/libs_rocky_linux/openssl10"
