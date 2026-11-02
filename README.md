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
storage is down; readers fail only when actually called. Importing does change
the current process's `PATH` and `LD_LIBRARY_PATH`: it sources the sdanalysis
environment script so the reader binaries can run.

## Quick start: simulation DSTs with muon labels

For detector-level MC DSTs that carry the `MUONEXTR` muon bank. Details and
caveats are in [New simulation DSTs and muon labels](#new-simulation-dsts-and-muon-labels).

```python
import numpy as np
from dstparser import parse_dst_file_vlen

d = parse_dst_file_vlen(
    "/path/to/DAT920950.dst.gz",
    cache_dir="/path/with/space/dstparser_cache",  # ~0.1 GB kept per shower
    campaign_manifest="/path/to/manifest.tsv",     # table with tag, loge, theta_deg, model
)
# d is None when no event in the file triggered the array.

event_of_wf = np.repeat(np.arange(len(d["tt_offsets"]) - 1), np.diff(d["tt_offsets"]))
station_of_wf = np.repeat(d["detector_ids"], np.diff(d["hit_tt_offsets"]))
mu, tot, ok = d["muon_signal"], d["muon_total_signal"], d["muon_available"]
log10_e = np.log10(d["energy"] * 1e18)  # per-event thrown energy; use this as truth
```

Before running it on many files:

- The first read of a DST without reconstruction banks runs `rufptn` and
  `rufldf` and stores the result in `cache_dir`. For one 10^19.0 eV, 30° shower
  (59.5 MB DST) this took 8.3 minutes on one core and 1.4 GB of memory. It kept
  109 MB in the cache and briefly used over 0.3 GB there while building. A
  cached re-read took 40 s. Always pass `cache_dir`: the default is
  `~/.cache/dstparser` in your home.
- The output keeps only waveforms from hits that pass the reconstruction's
  quality cut (`rufptn` `isgood > 2`): 122,654 of 233,908 in that shower. To
  read every waveform with its labels in seconds, without reconstruction, use
  `dstparser.banks.iter_events` (see [Raw access](#raw-access-without-reconstruction)).
- `energy` (EeV) is each event's own thrown energy. `shower_log10_energy` is
  the shower's nominal value from the manifest. See
  [What the labels and energies mean](#what-the-labels-and-energies-mean).

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
| `source_waveform_id` | `(n_waveforms,)`, the waveform's window index within its station (0 = first 128-bin window) |
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

### What the labels and energies mean

- `muon_total_signal` is the deposit, in VEM, of every simulated particle whose
  pulse **starts** inside the waveform's 128-bin window (30 bins before the
  window's trigger bin to 98 bins after). A pulse that starts before the window
  is not counted, even if its tail falls inside. The value is unsaturated and
  pedestal-free, so it is not the integral of the FADC trace.
- `muon_signal` is the part of that deposit from the shower's CORSIKA muons
  (particle codes 5 and 6). The simulation smears each particle's deposit once,
  and both labels use that same smeared deposit.
- `muon_total_signal - muon_signal` is everything else: electrons, photons,
  hadrons (neutrons included) and the simulated atmospheric background. The
  background enters the total only.
- `muon_nsat` counts the window's FADC bins above saturation, per layer. The
  labels are not clipped; use `muon_nsat` to find saturated waveforms.
- In one 10^19.0 eV campaign shower, 9 of 245,308 layer values have muon above
  total and none is negative. Keep such values as they are.
- Near the core, inside the tile simulation's fill radius (503 m in the
  shower used to validate the channel), and in cells capped at 600 VEM, the
  muon/total ratio comes from an approximation. Leave those out of muon-fraction
  physics.
- `energy` (EeV) is `rusdmc_.energy`, the energy thrown for that event. The
  detector simulation draws an energy factor per event and scales that event's
  deposits, and so its labels, to it. In one 10^19.0 eV shower the per-event
  values span 10^18.95 to 10^19.05 eV. `shower_log10_energy` is the shower's
  nominal value from the campaign manifest. Use `energy` as the per-event truth.
- Station of each waveform: `np.repeat(detector_ids, np.diff(hit_tt_offsets))`.
  Event of each waveform: `np.repeat(np.arange(n_events), np.diff(tt_offsets))`.

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

### Raw access without reconstruction

`dstparser.banks.iter_events` reads the DST in pure Python and yields one
CRC-checked event per thrown event, triggered or not. It needs no reader
binaries and no reconstruction, and keeps every waveform: 233,908 waveforms of
a 10^19.0 eV shower in 15 s. It does not decode the thrown energy or geometry;
for those use `parse_dst_file_vlen`.

```python
from dstparser.banks import iter_events

for ev in iter_events("/path/to/DAT920950.dst.gz"):
    raw, mu = ev.raw, ev.muon
    if raw is None or len(raw["keys"]) == 0:
        continue                    # thrown, but no waveforms
    xxyy, window = raw["keys"].T    # station id and window index per waveform
    traces = raw["traces"]          # (n_waveforms, 2, 128) FADC counts, lower, upper
    labels = mu["muon_signal"]      # (n_waveforms, 2) VEM, lower, upper, same rows
```

The existing `read_dst_file_all_events` text reader remains available.

## Tests

```bash
pip install -e .[test] pandas    # pandas: needed by the Xmax reader and its test
pytest tests/test_migration.py   # self-contained, no shared storage needed
pytest tests                     # adds integration tests on real DSTs
```

The integration tests read real DST files from shared storage and skip when it
is not mounted. `tests/test_migration.py` covers format, alignment,
reconstruction-cache and storage behaviour on synthetic files.
`tests/test_migration_real.py` enables frozen historical array comparisons with
`DSTPARSER_REGRESSION_BASELINES` and small low/mid/high-energy reconstruction
pilots with `DSTPARSER_CAMPAIGN_ROOT`. Tests use temporary copies and never
alter campaign inputs or the production software installation.
