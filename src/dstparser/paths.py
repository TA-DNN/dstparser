# Root path to the directory with data
# root_dir = "/ceph/sharedfs/work/TAML2024/benMC/install/sdanalysis_2019"
# root_dir = "/ceph/sharedfs/work/TAML2024/benMC/install/sdanalysis_2019"
root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/benMC/sdanalysis_2019"
dst_reader = "sditerator_no_standard_recon.run"
dst_reader_add_standard_recon = "sditerator_add_standard_recon_v2.run"
dst_reader_all_events = "sditerator_printAll.run"

sd_analysis_env = "sdanalysis_env.sh"
openssl10_alma9 = "/ceph/work/SATORI/projects/TA-ASIoP/benMC/libs_alma9/openssl10"
openssl10_rocky_linux = (
    "/ceph/work/SATORI/projects/TA-ASIoP/benMC/libs_rocky_linux/openssl10"
)

# Data for xmax
xmax_data_files = "DAT*_xmax.txt"
xmax_data_dir_prot = (
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
    "qgsii04proton/080417_160603/Em1/"
)

xmax_data_dir_fe = (
    "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
    "qgsii04iron/080417_160603/Em1/"
)

# TAx4 standard reconstruction lives in a SEPARATE sdanalysis install, built
# for TAx4/TALE geometry -- NOT benMC/sdanalysis_2019 above (that one's
# add_standard_recon_v2 rejects TAx4 events; verified 2026-07-20). This
# install's own rufptn.run/rufldf.run pass1/pass2 chain produced a real
# (if smaller-field) reconstruction: LDF-fit energy, S800, core, geometry-fit
# direction, border distance.
root_dir_tax4_std_recon = (
    "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM"
)
dst_reader_tax4_std_recon = "sditerator_add_standard_recon.run"

# TAx4 std-recon reader REBUILT LOCALLY to additionally emit rufptn_.nfold in
# the #SD meta DATA block (12 fields/hit instead of 11) and the free-curvature
# geometry fit + curvature in #EVENT DATA (42 fields). This removes the
# waveform->hit assignment ambiguity that the stock reader leaves (it omits
# nfold), so the vlen adapter can match hits to waveforms exactly like TA-SD.
# Built from a single locally-modified source against the ceph install
# READ-ONLY -- nothing on ceph was changed. Rebuild script + source:
#   /home/antonpr/local_sdanalysis/tax4_reader/
dst_reader_tax4_std_recon_nfold = (
    "/home/antonpr/local_sdanalysis/tax4_reader/bin/"
    "sditerator_add_standard_recon_nfold.run"
)
