# dstparser

Reads \*dst TA files and fills a dictionary with data required for DNN training.

Installation:
`pip install -e .`

## Paths — check this first

Every reader is a binary on shared storage, so `dstparser` needs to know where
that is. See what it resolved, and whether it exists:

```bash
python -m dstparser
```

Defaults suit the analysis machine, so usually there is nothing to do. To point
somewhere else, copy `paths_local.toml.example` to `paths_local.toml` (in the
repo root, git-ignored) and set only the roots you want to change:

```toml
sdanalysis_root = "/home/me/local/sdanalysis"
```

Each root resolves independently — environment variable (`$DSTPARSER_TA_ROOT`,
`$DSTPARSER_SDANALYSIS_ROOT`, `$DSTPARSER_DSTBANK_ROOT`,
`$DSTPARSER_TRAINING_ROOT`), then `paths_local.toml`, then the built-in default.
Importing never fails on a missing path, so the package stays usable when the
storage is down; readers fail only when actually called.

## Basic usage (grid format)
```python
from dstparser import parse_dst_file

dst_file = "/path/to/dst/file.dst.gz"
data = parse_dst_file(dst_file)                  # averaged upper/lower traces
data = parse_dst_file(dst_file, avg_traces=False)  # keep them separate
# Use the data dict to dump to an hdf5 file (see dstparser/cli)
```

`parse_dst_file` returns only events that **triggered** the detector — a small
fraction of what a simulated file holds. For every thrown event use
`dstparser.dst_reader.read_dst_file_all_events`.

TAx4 in grid format: `parse_dst_file_tax4`, same arguments. It is
`parse_dst_file` with TAx4's detector-id swap, 2080 m spacing, and positions
taken from the file rather than the TA-SD survey table.

## vlen format (flat arrays + offsets, used for GNN/DNN training)
```python
from dstparser import parse_dst_file_vlen, append_to_hdf5
import h5py

with h5py.File("out.h5", "a") as f:
    for dst_file in files:
        data = parse_dst_file_vlen(dst_file)   # TA-SD
        if data is not None:
            append_to_hdf5(f, data)
```

## TAx4 → vlen HDF5 (same format as TA-SD)

TAx4 uses the **same vlen format** as TA-SD via `parse_dst_file_tax4_vlen`, and
the output dict has **exactly the same keys** as `parse_dst_file_vlen`.

TAx4 is read with the same reader as TA-SD, and
`parse_dst_file_tax4_vlen` is `parse_dst_file_vlen(..., swap_detector_ids=True)`.
TAx4's `#SD meta` block prints the detector id as `yyxx` while its waveform
block uses `xxyy`; that swap is the only TAx4-specific step in the vlen path.

**Convert (Python API):**
```python
from dstparser import parse_dst_file_tax4_vlen, append_to_hdf5
import h5py

with h5py.File("tax4.h5", "a") as f:
    for dst_file in tax4_files:           # DAT*_gea.rufldf.dst.gz
        data = parse_dst_file_tax4_vlen(dst_file)
        if data is not None:
            append_to_hdf5(f, data)
```

**Convert (production CLI, SLURM):** the vlen pipeline is
adapter-selectable. In your config set `vlen_adapter = "tax4"` (see
`examples/config_tax4_vlen.py`), then:
```bash
python -m dstparser.cli.cli examples/config_tax4_vlen.py
```
This runs the same two-pass pipeline as TA-SD (pass1 `parse_dst_vlen` →
`temp_files/`, pass2 `join_hdf5_vlen` → `final_files/`), just using the TAx4
parser. The individual workers (`dst_to_hdf5_vlen`, `join_hdf5_vlen` in
`dstparser.cli.worker_job`) can also be called directly for a non-SLURM run.

Notes:
- The reader reads `.dst.gz` natively (no manual decompression).
- TAx4 xmax is not wired in by default (`xmax_dir = None` in the config); set
  `xmax_dir` to the directory holding `DAT*_xmax.txt` to enable it.

## Tests

```bash
pytest tests
```

Integration tests: they read real DST files from shared storage and skip when
it is not mounted.
