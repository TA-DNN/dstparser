#!/usr/bin/env python3
"""
1event_perdst.py — STRICT constants + skip empty DSTs

Parses all DST files in a directory and their corresponding xmax files,
and saves **one event per DST file** into HDF5 (rolling to new parts if needed).

- STRICT: All fields listed in CONSTANT_KEYS must exist and match CONSTANT_SHAPES.
- Empty DST files (0 events) are **skipped** silently and counted in a summary.
- Output HDF5 is opened **lazily** on first event so we don't create empty files.

If your DST adapter shape changes, update CONSTANT_SHAPES accordingly.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import awkward as ak
import h5py
from tqdm import tqdm
import re

# allow imports from your package
dir_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dir_root / "src"))
from dstparser.dst_adapter import parse_dst_file
from dstparser.xmax_reader import XmaxReader

# ───────────────────────────────────────────────────────────────────────────────
# Hard-coded parameters (edit these paths if needed)
DST_DIR         = Path("/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo")
OUT_PATH        = Path("/home/marktsai321/TA_DNN/temp/0817/completedataset.h5")
XMAX_DATA_DIR   = Path("/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo")
XMAX_DATA_FILES = "*_xmax.txt"
CPU_CORES       = 48
EVENTS_PER_FILE = 1_000_000  # max events per output HDF5 part

# ───────────────────────────────────────────────────────────────────────────────
# Configuration flags
USE_GRID_MODEL = False
AVG_TRACES     = False

# Fixed constants schema (order MUST match downstream inference expectations)
CONSTANT_KEYS = [
    "mass_number",
    "energy",
    "xmax",
    "shower_axis",  # 3
    "shower_core",  # 2 (truth core XY in DST; model output core may be 3-D)
    # Standard Reconstruction
    "std_recon_yymmdd",
    "std_recon_hhmmss",
    "std_recon_usec",
    "std_recon_nofwf",
    "std_recon_nsd",
    "std_recon_nsclust",
    "std_recon_nhits",
    "std_recon_nborder",
    "std_recon_qtot",
    "std_recon_energy",
    "std_recon_ldf_scale",
    "std_recon_ldf_scale_err",
    "std_recon_ldf_chi2",
    "std_recon_ldf_ndof",
    "std_recon_shower_core",          # 2
    "std_recon_shower_core_err",      # 2
    "std_recon_s800",
    "std_recon_combined_energy",
    "std_recon_combined_scale",
    "std_recon_combined_scale_err",
    "std_recon_combined_chi2",
    "std_recon_combined_ndof",
    "std_recon_combined_shower_core",     # 2
    "std_recon_combined_shower_core_err", # 2
    "std_recon_combined_s800",
    "std_recon_shower_axis",              # 3
    "std_recon_shower_axis_fixed_curve",  # 3
    "std_recon_shower_axis_combined",     # 3
    "std_recon_geom_chi2",
    "std_recon_geom_ndof",
    "std_recon_curvature",
    "std_recon_curvature_err",
    "std_recon_geom_chi2_fixed_curve",
    "std_recon_geom_ndof_fixed_curve",
    "std_recon_border_distance",
    "std_recon_border_distance_tshape",
    "std_recon_shower_axis_err",          # 3
    "std_recon_shower_axis_err_fixed_curve",  # 3
    "std_recon_shower_axis_err_combined",     # 3
]

# Expected element counts for each constant (STRICT)
CONSTANT_SHAPES = {
    "mass_number": 1,
    "energy": 1,
    "xmax": 1,
    "shower_axis": 3,
    "shower_core": 2,  # truth core XY
    # std_recon
    "std_recon_yymmdd": 1,
    "std_recon_hhmmss": 1,
    "std_recon_usec": 1,
    "std_recon_nofwf": 1,
    "std_recon_nsd": 1,
    "std_recon_nsclust": 1,
    "std_recon_nhits": 1,
    "std_recon_nborder": 1,
    "std_recon_qtot": 1,
    "std_recon_energy": 1,
    "std_recon_ldf_scale": 1,
    "std_recon_ldf_scale_err": 1,
    "std_recon_ldf_chi2": 1,
    "std_recon_ldf_ndof": 1,
    "std_recon_shower_core": 2,
    "std_recon_shower_core_err": 2,
    "std_recon_s800": 1,
    "std_recon_combined_energy": 1,
    "std_recon_combined_scale": 1,
    "std_recon_combined_scale_err": 1,
    "std_recon_combined_chi2": 1,
    "std_recon_combined_ndof": 1,
    "std_recon_combined_shower_core": 2,
    "std_recon_combined_shower_core_err": 2,
    "std_recon_combined_s800": 1,
    "std_recon_shower_axis": 3,
    "std_recon_shower_axis_fixed_curve": 3,
    "std_recon_shower_axis_combined": 3,
    "std_recon_geom_chi2": 1,
    "std_recon_geom_ndof": 1,
    "std_recon_curvature": 1,
    "std_recon_curvature_err": 1,
    "std_recon_geom_chi2_fixed_curve": 1,
    "std_recon_geom_ndof_fixed_curve": 1,
    "std_recon_border_distance": 1,
    "std_recon_border_distance_tshape": 1,
    "std_recon_shower_axis_err": 3,
    "std_recon_shower_axis_err_fixed_curve": 3,
    "std_recon_shower_axis_err_combined": 3,
}

# Hit- and trace-field counts
HIT_FIELDS   = 10  # [det_id, is_good, x, y, z, nfold, arrival_low, arrival_up, total_low, total_up]
TRACE_FIELDS = 2   # [low, up] per window

# ───────────────────────────────────────────────────────────────────────────────
# STRICT helpers

def _extract_constant_value_strict(data: dict, key: str, idx: int, src: Path) -> np.ndarray:
    """Return a 1D float64 array of length CONSTANT_SHAPES[key]. Raises if missing/malformed."""
    if key not in data:
        raise RuntimeError(
            f"Missing constant '{key}' for event idx={idx} in {src}. "
            f"This file does not match the required constants schema."
        )
    arr = np.asarray(data[key][idx], dtype=np.float64).ravel()
    need = int(CONSTANT_SHAPES[key])
    if arr.size != need:
        raise RuntimeError(
            f"Bad size for constant '{key}' in {src} (event idx={idx}): got {arr.size}, expected {need}."
        )
    return arr

def _build_constants_block_strict(data: dict, idx: int, src: Path) -> np.ndarray:
    """Concatenate constants in fixed order; raise if any key is missing/mis-sized."""
    parts = []
    for key in CONSTANT_KEYS:
        parts.append(_extract_constant_value_strict(data, key, idx, src))
    const_arr = np.concatenate(parts).astype(np.float64, copy=False)
    expected_len = sum(int(CONSTANT_SHAPES[k]) for k in CONSTANT_KEYS)
    if const_arr.size != expected_len:
        raise RuntimeError(
            f"Internal error assembling constants for {src} (event idx={idx}): "
            f"length {const_arr.size} != expected {expected_len}."
        )
    return const_arr

# ───────────────────────────────────────────────────────────────────────────────

def build_event_array(data, idx, src_path: Path):
    """Build flattened 1D array + attrs for event index `idx` in `data` (STRICT constants)."""
    # 1) constants (STRICT: no skipping)
    const_arr = _build_constants_block_strict(data, idx, src_path)
    C = int(const_arr.size)

    # 2) hits
    det_ids = ak.to_numpy(data['hits_det_id'][idx])
    x_coords = ak.to_numpy(data['hits_x'][idx])
    y_coords = ak.to_numpy(data['hits_y'][idx])
    z_coords = ak.to_numpy(data['hits_z'][idx])
    nfolds = ak.to_numpy(data['hits_nfold'][idx])
    lows   = ak.to_numpy(data['hits_arrival_times_low'][idx])
    ups    = ak.to_numpy(data['hits_arrival_times_up'][idx])
    sl     = ak.to_numpy(data['hits_total_signals_low'][idx])
    su     = ak.to_numpy(data['hits_total_signals_up'][idx])
    goods  = (ak.to_numpy(data.get('hits_good', np.ones_like(det_ids)))[idx]
              if 'hits_good' in data else np.ones_like(det_ids, dtype=np.float64))

    H = det_ids.size
    hits_flat = []
    for j in range(H):
        hits_flat.extend([
            float(det_ids[j]),
            float(goods[j]),
            float(x_coords[j]), float(y_coords[j]), float(z_coords[j]),
            float(nfolds[j]),
            float(lows[j]), float(ups[j]),
            float(sl[j]), float(su[j]),
        ])
    hits_arr = np.asarray(hits_flat, dtype=np.float64, order="C")

    # 3) time traces
    lows_list = ak.to_list(data['hits_time_traces_low'][idx])
    ups_list  = ak.to_list(data['hits_time_traces_up'][idx])
    windows_per_hit, traces_flat = [], []
    for lw, uw in zip(lows_list, ups_list):
        windows_per_hit.append(len(lw))
        for ll, uu in zip(lw, uw):
            traces_flat.extend([float(ll), float(uu)])
    traces_arr = np.asarray(traces_flat, dtype=np.float64, order="C")

    # 4) concatenate
    event_data = np.concatenate([const_arr, hits_arr, traces_arr]).astype(np.float64, copy=False)

    # 5) attributes
    attrs = {
        'constant_length': C,  # exact number of constants at the start of event_data
        'hit_fields':      HIT_FIELDS,
        'num_hits':        H,
        'windows_per_hit': np.asarray(windows_per_hit, dtype=np.int32),
        'trace_fields':    TRACE_FIELDS,
    }
    return event_data, attrs

def process_file(args):
    """
    Worker: parse one DST file and build exactly one event from it (STRICT constants).
    Returns (filename, events_list). For **empty** DSTs, returns (filename, []).
    """
    dst_file, xmax_reader = args
    data = parse_dst_file(
        str(dst_file),
        use_grid_model=USE_GRID_MODEL,
        avg_traces=AVG_TRACES,
        xmax_reader=xmax_reader,
    )
    if data is None:
        # treat as empty/invalid — no events generated
        return dst_file.name, []

    # If truly empty, skip
    n_avail = len(data.get('energy', []))
    if n_avail is None or n_avail == 0:
        return dst_file.name, []

    # Deterministic choice: pick the first event (index 0).
    ev_data, ev_attrs = build_event_array(data, 0, dst_file)
    return dst_file.name, [(ev_data, ev_attrs)]

def main():
    import sys

    # 1) check directory
    if not DST_DIR.is_dir():
        print(f"ERROR: DST directory not found: {DST_DIR}", file=sys.stderr)
        sys.exit(1)

    # 2) list files
    dst_files = sorted(DST_DIR.glob("*.dst*"))

    # Optional: filter by energy bin if needed (currently pass-through)
    filtered_files = []
    for f in dst_files:
        match = re.search(r"DAT\d{4}(\d{2})", f.name)
        if match:
            energy_bin = int(match.group(1))
            filtered_files.append(f)
    dst_files = filtered_files

    N = len(dst_files)
    if N == 0:
        print(f"ERROR: No DST files found in {DST_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {N} DST files to process.")

    # 3) create Xmax reader
    xmax_reader = XmaxReader(data_dir=XMAX_DATA_DIR, glob_pattern=XMAX_DATA_FILES)

    # 4) Start processing — open HDF5 lazily on first event
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _part_path(i: int) -> Path:
        return OUT_PATH.with_name(f"{OUT_PATH.stem}_part{i:03d}{OUT_PATH.suffix}")

    f = None
    evs_grp = None
    file_idx = 0
    events_in_file = 0
    global_idx = 0
    created_paths = []
    files_with_events = 0
    files_empty = 0

    try:
        args = [(dst_file, xmax_reader) for dst_file in dst_files]
        with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
            for _, events in tqdm(executor.map(process_file, args), total=N, desc="Files"):
                if not events:
                    files_empty += 1
                    continue

                # Lazily open first output part
                if f is None:
                    file_idx = 1
                    current_path = _part_path(file_idx)
                    f = h5py.File(current_path, 'w')
                    evs_grp = f.create_group('events')
                    created_paths.append(current_path)
                    events_in_file = 0

                files_with_events += 1
                for ev_data, ev_attrs in events:
                    # roll over to a new file if we've hit the per-file cap
                    if events_in_file >= EVENTS_PER_FILE:
                        f.close()
                        file_idx += 1
                        current_path = _part_path(file_idx)
                        f = h5py.File(current_path, 'w')
                        evs_grp = f.create_group('events')
                        created_paths.append(current_path)
                        events_in_file = 0

                    grp = evs_grp.create_group(f'event_{global_idx:06d}')
                    ds  = grp.create_dataset('event_data', data=ev_data, dtype='float64', chunks=(ev_data.size,))
                    for k, v in ev_attrs.items():
                        ds.attrs[k] = v

                    global_idx += 1
                    events_in_file += 1
    finally:
        if f is not None:
            f.close()

    # Summary
    print("\nSummary:")
    print(f"  Total DST files:    {N}")
    print(f"  With events:        {files_with_events}")
    print(f"  Empty / skipped:    {files_empty}")
    print(f"  Events written:     {global_idx}")

    if not created_paths:
        print("No events were written. No output files created.")
        return

    if len(created_paths) == 1:
        print(f"Wrote {global_idx:,} events to {created_paths[0]}")
    else:
        print(f"Wrote {global_idx:,} events across {len(created_paths)} files:")
        for p in created_paths:
            print(f"  - {p}")

if __name__ == "__main__":
    main()
