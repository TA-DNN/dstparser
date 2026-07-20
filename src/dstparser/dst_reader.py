import os
from pathlib import Path
import subprocess
from dstparser.env_vars import changed_env_paths, is_alma_linux, is_rocky_linux
from dstparser.paths import (
    root_dir,
    dst_reader_add_standard_recon,
    dst_reader_all_events,
    root_dir_tax4_std_recon,
    dst_reader_tax4_std_recon,
    sd_analysis_env,
    openssl10_alma9,
    openssl10_rocky_linux,
)


# Loading environment from "sdanalysis_env.sh"
sd_analysis_env = str(Path(root_dir) / sd_analysis_env)
for env_var, path_var in changed_env_paths(sd_analysis_env).items():
    os.environ[env_var] = path_var

# TAx4 standard recon lives in a separate sdanalysis install with its own
# bin/lib -- load its env the same way, additively (verified 2026-07-20: no
# binary-name collisions with the benMC install above).
_tax4_sd_analysis_env = str(Path(root_dir_tax4_std_recon) / sd_analysis_env)
for env_var, path_var in changed_env_paths(_tax4_sd_analysis_env).items():
    if env_var in ("PATH", "LD_LIBRARY_PATH") and env_var in os.environ:
        os.environ[env_var] = f"{path_var}:{os.environ[env_var]}"
    else:
        os.environ[env_var] = path_var

# Add path to openssl10 missing libs
if is_alma_linux():
    ld_paths = "LD_LIBRARY_PATH"
    os.environ[ld_paths] = f"{openssl10_alma9}:{os.environ[ld_paths]}"

if is_rocky_linux():
    ld_paths = "LD_LIBRARY_PATH"
    os.environ[ld_paths] = f"{openssl10_rocky_linux}:{os.environ[ld_paths]}"


def _run_dst_reader(dst_reader_process, dst_filename):
    try:
        process = subprocess.Popen(
            [dst_reader_process, str(dst_filename).strip()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ,
        )
        output, error = process.communicate()
    except subprocess.CalledProcessError as e:
        output = e.output

    # If output is empty
    if len(output) == 0:
        print(f'dst_reader error:\n"{error}"')

    return output.strip().split("\n")


def read_dst_file(dst_filename):
    return _run_dst_reader(dst_reader_add_standard_recon, dst_filename)


def read_dst_file_all_events(dst_filename):
    # sditerator_printAll.run (benMC install): dumps every THROWN event's
    # truth + raw SD/waveform data, no reconstruction fields. Use this for
    # full per-shower statistics including non-triggered throws.
    return _run_dst_reader(dst_reader_all_events, dst_filename)


def read_dst_file_tax4_std_recon(dst_filename):
    # sditerator_add_standard_recon.run from the SEPARATE
    # sdanalysis_2018_TALE_TAx4SingleCT_DM install (built for TAx4/TALE
    # geometry) -- real LDF-fit + geometry-fit reconstruction. Only emits
    # events with nofwf>0 (verified 2026-07-20 against
    # sditerator_cppanalysis_add_standard_recon.cpp:29). The benMC install's
    # add_standard_recon_v2 rejects TAx4 events entirely; this one does not.
    return _run_dst_reader(dst_reader_tax4_std_recon, dst_filename)
