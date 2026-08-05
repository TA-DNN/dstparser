"""Machine-dependent paths.

Everything that lives on the shared SATORI storage is derived from ONE root
(`ceph_root`) so a remount is a single-line change. The mount point has already
moved twice: /ceph/sharedfs/work/... -> /ceph/work/... -> back to
/ceph/sharedfs/work/... during the 2026-07-24 downtime. Override per machine
with $DSTPARSER_CEPH_ROOT.

Every reader used here is a binary on that shared storage -- nothing has to be
built locally, including for TAx4 (see the note further down).
"""

import os
from pathlib import Path

# Root of the shared SATORI project area on ceph
ceph_root = os.environ.get(
    "DSTPARSER_CEPH_ROOT", "/ceph/sharedfs/work/SATORI/projects/TA-ASIoP"
)

# Root path to the directory with data
root_dir = f"{ceph_root}/benMC/sdanalysis_2019"
dst_reader = "sditerator_no_standard_recon.run"
dst_reader_add_standard_recon = "sditerator_add_standard_recon_v2.run"
dst_reader_all_events = "sditerator_printAll.run"

sd_analysis_env = "sdanalysis_env.sh"

# Both sdanalysis_env.sh scripts on ceph still `source` ROOT from the OLD
# /ceph/work/... mount, which disappeared in the 2026-07-24 downtime. They live
# on read-only shared storage and are not ours to edit, so ROOT is set up here
# explicitly, after the install env, instead. [verified 2026-08-05: without
# this every reader dies with "libCore.so: cannot open shared object file"]
root_env_benmc = f"{ceph_root}/install/root/bin/thisroot.sh"
root_env_tax4 = f"{ceph_root}/root/bin/thisroot.sh"
openssl10_alma9 = f"{ceph_root}/benMC/libs_alma9/openssl10"
openssl10_rocky_linux = f"{ceph_root}/benMC/libs_rocky_linux/openssl10"

# Data for xmax
xmax_data_files = "DAT*_xmax.txt"
xmax_data_dir_prot = f"{ceph_root}/tasdmc_dstbank/qgsii04proton/080417_160603/Em1/"

xmax_data_dir_fe = f"{ceph_root}/tasdmc_dstbank/qgsii04iron/080417_160603/Em1/"

# TAx4 standard reconstruction lives in a SEPARATE sdanalysis install, built
# for TAx4/TALE geometry -- NOT benMC/sdanalysis_2019 above (that one's
# add_standard_recon_v2 rejects TAx4 events; verified 2026-07-20). This
# install's own rufptn.run/rufldf.run pass1/pass2 chain produced a real
# (if smaller-field) reconstruction: LDF-fit energy, S800, core, geometry-fit
# direction, border distance.
root_dir_tax4_std_recon = f"{ceph_root}/sdanalysis_2018_TALE_TAx4SingleCT_DM"
dst_reader_tax4_std_recon = "sditerator_add_standard_recon.run"

# NOTE (2026-08-05): the TALE reader above is used ONLY by the old grid-tile
# adapter dst_adapter_tax4.py. The vlen path does NOT need it: TAx4 is read with
# the same benMC exe as TA-SD (dst_reader above), which emits the full 61-field
# #EVENT record and 12-field #SD meta including rufptn_.nfold. A locally-rebuilt
# "nfold" reader used to live here; it turned out to be solving a problem that
# only the TALE reader had. Removed, with its build tree, on 2026-08-05
# (ml/trash/2026_08_05_tax4_tale_nfold_reader_and_vlen_fork/).
