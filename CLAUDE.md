# dstparser: notes for coding agents

Reads Telescope Array surface-detector DST files into numpy dicts (grid or
vlen format) for DNN training. Start with README.md: "Paths", then "Quick
start: simulation DSTs with muon labels".

## Map

- `parse_dst_file` (`src/dstparser/dst_adapter.py`): grid format.
- `parse_dst_file_vlen` (`src/dstparser/dst_adapter_vlen.py`): flat arrays + offsets.
- `dstparser.banks.iter_events`: pure-Python bank reader; every waveform, no
  reconstruction, no reader binaries.
- `preparation.prepare_dst`: runs rufptn/rufldf into a hash-keyed cache.
- `muon.py`: aligns MUONEXTR labels to waveforms. `campaign.py`: manifest metadata.

## Setup and tests

```bash
python -m venv venv && venv/bin/pip install -e .[test] pandas
venv/bin/python -m dstparser          # which reader paths resolved, and whether they exist
venv/bin/pytest tests/test_migration.py   # self-contained
venv/bin/pytest tests                     # integration tests skip without shared storage
```

pandas is not in `pyproject.toml` but the Xmax reader and
`tests/test_parser_xmax.py` import it. The 9 tests in
`tests/test_migration_real.py` skip unless `DSTPARSER_REGRESSION_BASELINES` or
`DSTPARSER_CAMPAIGN_ROOT` is set.

Importing dstparser sources the sdanalysis environment script and changes
`PATH` and `LD_LIBRARY_PATH` of the current process.

## Rules

- Never write into input data folders. Inputs are read-only; reconstruction
  writes only to `cache_dir`.
- Always pass `cache_dir`. The default is `~/.cache/dstparser` in the home
  folder. A first read of a 10^19 eV shower takes about 8 minutes, 1.4 GB of
  memory and 0.1 GB of cache.
- Keep muon labels as they are: signed, unclipped, muon may exceed total. A
  missing label is `NaN` (signals) or `-1` (saturation, IDs) with
  `muon_available` false; a real zero is `0` with `muon_available` true. Never
  fill a missing label with zero.
- Layer order is lower, upper in the vlen output and in `banks`.
- `energy` (EeV) is the per-event truth. `shower_log10_energy` is the
  shower's nominal value.
- Do not invent missing data. A failed reconstruction stage, or a missing
  campaign Xmax when an Xmax reader is set, raises an error by design; do not
  replace it with zeros or model draws.
- This repository is public: no private paths, credentials or data files in
  commits.
