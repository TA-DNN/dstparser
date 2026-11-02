"""Options to add to an existing grid/vlen conversion config for the new MC."""

from pathlib import Path

# Replace both paths with your own.
campaign_root = Path("/path/to/campaign")  # folder holding the DSTs
# Table with tag, loge, theta_deg and model columns (a campaign grid.tsv or
# manifest.tsv). Omit it to search the DST's parent folders for grid.tsv.
campaign_manifest = campaign_root / "manifest.tsv"
# Reconstructed copies of the DSTs, about 0.1 GB per 10^19 eV shower. Must be
# writable; keep it out of your home folder and out of the campaign inputs.
cache_dir = Path("/path/with/space/dstparser_cache")
reconstruction = "auto"
include_muon = "require"
reconstruction_timeout = 3600
reconstruction_dependencies = ()  # Add site-specific external tables if used.
add_shower_params = True
add_standard_recon = True
xmax_dir = None  # Enable only with a real per-shower Xmax source.

# Keep your existing SLURM grouping/output settings. These options can also
# be passed directly to parse_dst_file() and parse_dst_file_vlen().
