# dstparser

Reads \*dst TA files and fills a dictionary with data required for DNN training.

Installation:
`pip install -e .`

## Basic usage (grid format)
```python
from dstparser import parse_dst_file

dst_file = "/path/to/dst/file.dst.gz"
data = parse_dst_file(dst_file, up_low_traces=True)
# Use the data dict to dump to an hdf5 file (see dstparser/cli)
```

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

TAx4 uses the **same vlen format** as TA-SD via `parse_dst_file_tax4_vlen`. The
output dict is identical to `parse_dst_file_vlen` except it lacks the four recon
fields TAx4's reconstruction does not compute
(`std_recon_nsclust/nhits/nborder/qtot`).

**Step 1 — build the TAx4 reader once** (emits `rufptn_.nfold`, which the stock
TAx4 reader omits; required so hits map to waveforms exactly):
```bash
bash tax4_reader_build/build.sh
```
This compiles one modified source against the ceph sdanalysis install
**read-only** (nothing on ceph is changed) and writes the binary to the path in
`paths.dst_reader_tax4_std_recon_nfold`. See `tax4_reader_build/README.md` for
details and prerequisites (g++, ROOT).

**Step 2a — convert (Python API):**
```python
from dstparser import parse_dst_file_tax4_vlen, append_to_hdf5
import h5py

with h5py.File("tax4.h5", "a") as f:
    for dst_file in tax4_files:           # DAT*_gea.rufldf.dst.gz
        data = parse_dst_file_tax4_vlen(dst_file)
        if data is not None:
            append_to_hdf5(f, data)
```

**Step 2b — convert (production CLI, SLURM):** the vlen pipeline is
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
  `xmax_dir` to enable it. (The shared `init_xmax_reader` xmax code has a
  separate pre-existing staleness vs the refactored `xmax_reader` package — only
  relevant when xmax is enabled.)
- Regression tests: `tests/test_parser_tax4_vlen.py`.
