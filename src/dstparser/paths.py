# Root path to the directory with data
root_dir = "/ceph/work/SATORI/projects/TA-ASIoP/benMC/sdanalysis_2019"

# `sditerator` executable file
dst_reader = "sditerator_add_standard_recon_v2.run"

# script with environment variables
sd_analysis_env = "sdanalysis_env.sh"

# Patch (workaround) to provide openssl10 library for old version of root on
# alma9 and rocky_linux that do not contain `openssl10`
# It should be deleted when proper way to install is found
openssl10_fix_dir = "/ceph/work/SATORI/projects/TA-ASIoP/benMC"
openssl10_alma9 = openssl10_fix_dir + "/libs_alma9/openssl10"
openssl10_rocky_linux = openssl10_fix_dir + "/libs_rocky_linux/openssl10"
