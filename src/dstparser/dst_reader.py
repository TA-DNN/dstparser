import os
from pathlib import Path
import subprocess
from dstparser.env_vars import changed_env_paths, is_alma_linux, is_rocky_linux
from dstparser.paths import (
    root_dir,
    dst_reader_add_standard_recon,
    dst_reader_all_events,
    sd_analysis_env,
    openssl10_alma9,
    openssl10_rocky_linux,
    root_env_benmc,
)


def _load_env(shell_script, prepend=False):
    """Source a shell script and copy the path-valued vars it changed into
    os.environ. With prepend=True, PATH/LD_LIBRARY_PATH are prepended to the
    current value instead of replacing it (so a second install can be layered
    on top of the first).
    """
    for env_var, path_var in changed_env_paths(str(shell_script)).items():
        if prepend and env_var in ("PATH", "LD_LIBRARY_PATH") and env_var in os.environ:
            os.environ[env_var] = f"{path_var}:{os.environ[env_var]}"
        else:
            os.environ[env_var] = path_var


# Loading environment from "sdanalysis_env.sh"
sd_analysis_env = str(Path(root_dir) / sd_analysis_env)
_load_env(sd_analysis_env)
# The env script's own `source .../thisroot.sh` line points at a stale mount and
# silently does nothing, so ROOT must be loaded here explicitly.
_load_env(root_env_benmc, prepend=True)

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
    # sditerator_add_standard_recon_v2.run (benMC install): standard
    # reconstruction, and ONLY events that triggered (rusdraw_.nofwf > 0) --
    # a small fraction of what the file holds (~1% of thrown events for TA-SD,
    # ~16% for TAx4). For every thrown event use read_dst_file_all_events below.
    return _run_dst_reader(dst_reader_add_standard_recon, dst_filename)


def read_dst_file_all_events(dst_filename):
    # sditerator_printAll.run (benMC install): dumps every THROWN event's
    # truth + raw SD/waveform data, no reconstruction fields. Use this for
    # full per-shower statistics including non-triggered throws.
    return _run_dst_reader(dst_reader_all_events, dst_filename)
