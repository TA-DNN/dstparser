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

TAx4 uses the **same vlen format** as TA-SD via `parse_dst_file_tax4_vlen`, and
the output dict has **exactly the same keys** as `parse_dst_file_vlen`.

**No special reader, no build step.** TAx4 is read with the same benMC exe as
TA-SD, which emits the full 61-field `#EVENT` record and 12-field `#SD meta`
including `rufptn_.nfold`. `parse_dst_file_tax4_vlen` is simply
`parse_dst_file_vlen(..., swap_detector_ids=True)`: TAx4's `#SD meta` block
prints the detector id as `yyxx` while its waveform block uses `xxyy`, and that
swap is the only TAx4-specific step in the whole vlen path.

> Historical note: an earlier version routed TAx4 through a separate
> TALE-geometry sdanalysis install plus a locally-rebuilt "nfold" reader. That
> was unnecessary — the missing `nfold` and the reduced field set were
> properties of *that reader's* printf, not of TAx4 data. Verified 2026-08-05 by
> running both exes on 40 TAx4 files (north+south, 976 events): identical event
> counts, identical `#SD meta`, identical `#EVENT` fields 0–41. Neither exe
> reconstructs anything — they print the banks already stored in the pass2
> (`rufldf`) DST file.

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
  `xmax_dir` to enable it. (The shared `init_xmax_reader` xmax code has a
  separate pre-existing staleness vs the refactored `xmax_reader` package — only
  relevant when xmax is enabled.)
- Regression tests: `tests/test_parser_tax4_vlen.py`.
