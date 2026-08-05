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

Each root is resolved independently, highest priority first:

    1. environment variable      $DSTPARSER_TA_ROOT, ...   (per shell / SLURM job)
    2. paths_local.toml          in the repo root          (per checkout, git-ignored)
    3. the built-in default      below

Copy `paths_local.toml.example` to `paths_local.toml` to set your own. Nothing
here checks that the paths exist: storage is sometimes down while you work on
something that does not need it, and an import-time failure would also break
the diagnostic. To see what resolved and what is missing:

    python -m dstparser
"""

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    tomllib = None

package_root = Path(__file__).resolve().parents[2]
local_config_file = package_root / "paths_local.toml"


def _read_local_config():
    if tomllib is None or not local_config_file.exists():
        return {}
    try:
        with open(local_config_file, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:  # a broken config must not break the import
        print(f"dstparser: ignoring unreadable {local_config_file}: {exc}")
        return {}


_local_config = _read_local_config()
# name -> "env" | "paths_local.toml" | "default", for `python -m dstparser`
_root_sources = {}


def _resolve_root(name, env_var, default):
    if env_var in os.environ:
        value, source = os.environ[env_var], "env"
    elif name in _local_config:
        value, source = str(_local_config[name]), local_config_file.name
    else:
        value, source = default, "default"
    _root_sources[name] = (source, env_var)
    return value


# The TA project area: the default base for all three category roots below.
ta_root = _resolve_root(
    "ta_root", "DSTPARSER_TA_ROOT", "/ceph/sharedfs/work/SATORI/projects/TA-ASIoP"
)

# Software installs: sdanalysis, ROOT, openssl10.
sdanalysis_root = _resolve_root("sdanalysis_root", "DSTPARSER_SDANALYSIS_ROOT", ta_root)
# Input data: tasdmc_dstbank, tasdobs_dstbank, INR_group.
dstbank_root = _resolve_root("dstbank_root", "DSTPARSER_DSTBANK_ROOT", ta_root)
# Outputs: dnn_training_data.
training_data_root = _resolve_root(
    "training_data_root", "DSTPARSER_TRAINING_ROOT", ta_root
)

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


def report():
    """Print each root: value, where it came from, whether it exists."""
    roots = {
        "ta_root": ta_root,
        "sdanalysis_root": sdanalysis_root,
        "dstbank_root": dstbank_root,
        "training_data_root": training_data_root,
    }
    cfg = local_config_file
    print(f"config file : {cfg}" + ("" if cfg.exists() else "   (absent, using defaults)"))
    print()
    width = max(len(v) for v in roots.values())
    for name, value in roots.items():
        source, env_var = _root_sources[name]
        state = "ok" if Path(value).exists() else "MISSING"
        print(f"  {name:<19}{value:<{width}}  [{source}]  {state}")
    print()
    print("  override with $DSTPARSER_TA_ROOT / _SDANALYSIS_ROOT / _DSTBANK_ROOT /")
    print(f"  _TRAINING_ROOT, or with {cfg.name} (see {cfg.name}.example)")

    missing = [n for n, v in roots.items() if not Path(v).exists()]
    if missing:
        print()
        print(f"  {len(missing)} root(s) MISSING: {', '.join(missing)}")
        print("  Shared storage may be unmounted, or the mount may have moved again.")

