# Root path to the directory with data
# root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM"
root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/benMC/sdanalysis_2019"
dst_reader = "sditerator_no_standard_recon.run"
dst_reader_add_standard_recon = "sditerator_add_standard_recon_v2.run"
dst_reader_all_events = "sditerator_printAll.run"

sd_analysis_env = "sdanalysis_env.sh"
openssl10_alma9 = "/ceph/work/SATORI/projects/TA-ASIoP/benMC/libs_alma9/openssl10"

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
