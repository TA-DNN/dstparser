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

## New simulation DSTs and muon labels

The same grid/vlen functions accept detector-level MC files containing
`RUSDMC`, `RUSDRAW` and `MUONEXTR`. They inspect the actual banks, rather than
assuming that a `.dst.gz` file has already been reconstructed.

```python
from dstparser import parse_dst_file, parse_dst_file_vlen, prepare_dst

data = parse_dst_file("DAT920950.dst.gz", avg_traces=False,
                      cache_dir="/my/work/reconstruction-cache")
flat = parse_dst_file_vlen("DAT920950.dst.gz",
                          cache_dir="/my/work/reconstruction-cache")
```

### Reconstruction preparation

`reconstruction="auto"` (default) runs `rufptn.run` and `rufldf.run` when
waveform-bearing events lack the required reconstruction banks. Existing
reconstruction is reused. `reconstruction="require"` instead raises an error
for an unprepared file. `add_standard_recon=False` controls returned fields;
the existing hit selections still require reconstruction.

`prepare_dst(path, cache_dir=...)` can prepare the files before a batch job.
The default cache is `$DSTPARSER_CACHE`, or `~/.cache/dstparser`. Cache entries
are keyed by the input hash, commands, executable/shared-library hashes,
environment and explicitly supplied dependency hashes. A per-entry lock
prevents concurrent preparation. Outputs are published only after checking
event counts and exact preservation of the original truth and waveform banks.
Muon banks dropped by older reconstruction tools are restored from the original
event occurrence, after that verification. Original files are read-only.
The known RUSDRAW v000/v001 version-word transition is allowed when every
stored data field matches; waveform contents and truth values must be exact.

Site-specific commands and additional calibration/lookup files can be pinned:

```python
prepared = prepare_dst(
    path, cache_dir=cache,
    commands=[
        ["/my/bin/rufptn.run", "{input}", "-o1f", "{output}"],
        ["/my/bin/rufldf.run", "{input}", "-o1f", "{output}"],
    ],
    dependencies=["/my/calibration/table.dat"], timeout=3600,
)
```

The parsing functions expose these as `reconstruction_commands`,
`reconstruction_dependencies` and `reconstruction_timeout`. Include all
external configuration/calibration files used by a custom command in its
dependencies. Missing executables, failed stages, changed events, corrupt
banks and invalid cache entries raise errors rather than returning empty data.
Files with genuinely no waveform-bearing events still return `None`.

### Additive muon fields

`include_muon="auto"` (default) adds labels when the bank is present. Historical
files without that bank retain their old field set. `True` also creates explicit
unavailable labels for legacy MC; `False` returns only the standard fields;
`"require"` requires the bank in every input event.

| Field in vlen output | Shape / meaning |
|---|---|
| `muon_signal` | `(n_waveforms, 2)`, signed muon-attributed proxy in VEM |
| `muon_total_signal` | `(n_waveforms, 2)`, corresponding total proxy in VEM |
| `muon_nsat` | `(n_waveforms, 2)`, integer saturation counts |
| `muon_available` | `(n_waveforms,)`, whether the label was present |
| `source_waveform_id` | `(n_waveforms,)`, original waveform ID |
| `source_event_ordinal`, `source_event_number` | Per-event source occurrence and stored event number |
| `source_file_sha256` | Per-event input identity as fixed-width `S64` |
| `detector_status_available` | Per-event availability of `BSDINFO` |

Vlen layer order is **lower, upper**, matching `time_traces`. Labels share the
existing `tt_offsets`/`hit_tt_offsets` and follow every waveform filter.
Association checks compare station IDs, clocks and every FADC sample with the
native banks before selecting rows; repeated event numbers are not unique keys.

For separate-layer grid output, signal keys have `_low`/`_up` suffixes and
shape `(n_events, ntile, ntile)`. Averaged grid output has `muon_signal` and
`muon_total_signal`, averaged across layers. Grid saturation keys remain
`muon_nsat_low` and `muon_nsat_up`. Grid labels follow the **first retained
waveform window**, just like the existing grid trace; they are not sums over
all folded windows. Existing reconstructed total-signal fields keep their
original definitions.

Unavailable signal labels are `NaN`, saturation/ID placeholders are `-1`, and
`muon_available` is false. Grid padding uses the same unavailable representation.
A genuine zero-muon response has a **true** availability mask and a zero value.
Signed proxies and values exceeding their corresponding total are preserved.
They are pulse-start response estimates, not particle counts, deposited energy,
or muon-only waveform samples. Saturation counts are metadata, not an automatic
selection cut.

Some detector-level campaign files lack `BSDINFO`. This produces an explicit
warning and `detector_status_available=False`. Existing detector-state fields
in that case are not a census of working detectors; an empty bad-counter list
does not establish that all counters were operational.

### HDF5, merging and campaign identities

Grid/vlen CLI workers accept all migration arguments as config attributes.
New HDF5 datasets carry explicit event/hit/waveform level metadata, so coincident
array lengths cannot misclassify a muon field. `H5DST` and `MultiH5DST` expose
waveform-level muon arrays with the same indexing as traces. Appending/merging
old and new files pads absent labels with the unavailable representation and
adjusts all offsets. Incompatible ordinary schemas/dtypes are rejected before
append; interrupted appends are marked unreadable. CLI outputs publish
atomically. Legitimately empty conversion groups are marked `dstparser_empty`
and skipped when joining populated groups.

A campaign `grid.tsv` with `tag`, `loge`, `theta_deg`, and `model` columns is
automatically discovered in input ancestors. It can instead be supplied as
`campaign_manifest="/path/grid.tsv"`; `False` disables discovery. It supplies
the **full six-digit** `id_corsika_shower`, `shower_log10_energy`,
`shower_zenith_deg`, and `id_campaign_model` (0=QGSJET-II-04, 1=EPOS-LHC).
For these manifest-based inputs, `id_energy_bin` is `round(10*log10(E/eV))`,
not the last two DAT digits; its convention is recorded in HDF5 metadata.
Legacy filename callbacks retain their behavior for historical files; explicit
campaign identities take precedence over such callbacks for campaign inputs.

Optional Xmax readers use explicit campaign metadata when available. A measured
`xmax_g_cm2` manifest value (slant depth) or a matching real Xmax input is required;
missing campaign Xmax is not replaced with a random model draw. Use
`xmax_reader=None` / CLI `xmax_dir=None` when no measured Xmax is available.

For lossless raw access without reconstruction, `dstparser.banks.iter_events`
yields CRC-checked events with `.raw` and `.muon` arrays; the existing
`read_dst_file_all_events` text reader remains available.

### Migration verification

Self-contained format, alignment, reconstruction-cache and storage tests:

```bash
pytest tests/test_migration.py
```

`tests/test_migration_real.py` enables frozen historical array comparisons with
`DSTPARSER_REGRESSION_BASELINES` and small low/mid/high-energy reconstruction
pilots with `DSTPARSER_CAMPAIGN_ROOT`. Tests use temporary copies and never
alter campaign inputs or the production software installation.
