"""Options to add to an existing grid/vlen conversion config for the new MC."""

from pathlib import Path

campaign_root = Path("/ceph/sharedfs/work/SATORI/mark/TA_simulation_data")
campaign_manifest = campaign_root / "grid.tsv"
cache_dir = campaign_root / "analysis/dstparser_reconstruction_cache"
reconstruction = "auto"
include_muon = "require"
reconstruction_timeout = 3600
reconstruction_dependencies = ()  # Add site-specific external tables if used.
add_shower_params = True
add_standard_recon = True
xmax_dir = None  # Enable only with a real per-shower Xmax source.

# Keep your existing SLURM grouping/output settings. These options can also
# be passed directly to parse_dst_file() and parse_dst_file_vlen().
