#!/usr/bin/env python3
"""
parse_3_dst.py

Parses 3 DST files in a directory and their corresponding xmax files,
and saves all events into a single HDF5 file.
Uses multiprocessing and shows a progress bar.
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
# Hard‑coded parameters (edit these paths if needed)
DST_DIR         = Path("/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo")
OUT_PATH        = Path("/home/marktsai321/TA_DNN/temp/test_outputs/small_event_dataset/parsed_events.h5")
XMAX_DATA_DIR   = Path("/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo")
XMAX_DATA_FILES = "*_xmax.txt"
CPU_CORES       = 48
# ───────────────────────────────────────────────────────────────────────────────

# Configuration flags
USE_GRID_MODEL = False
AVG_TRACES     = False

# Constant feature keys
CONSTANT_KEYS = [
    "mass_number",
    "energy",
    "xmax",
    "shower_axis",
    "shower_core",
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
    "std_recon_shower_core",
    "std_recon_shower_core_err",
    "std_recon_s800",
    "std_recon_combined_energy",
    "std_recon_combined_scale",
    "std_recon_combined_scale_err",
    "std_recon_combined_chi2",
    "std_recon_combined_ndof",
    "std_recon_combined_shower_core",
    "std_recon_combined_shower_core_err",
    "std_recon_combined_s800",
    "std_recon_shower_axis",
    "std_recon_shower_axis_fixed_curve",
    "std_recon_shower_axis_combined",
    "std_recon_geom_chi2",
    "std_recon_geom_ndof",
    "std_recon_curvature",
    "std_recon_curvature_err",
    "std_recon_geom_chi2_fixed_curve",
    "std_recon_geom_ndof_fixed_curve",
    "std_recon_border_distance",
    "std_recon_border_distance_tshape",
    "std_recon_shower_axis_err",
    "std_recon_shower_axis_err_fixed_curve",
    "std_recon_shower_axis_err_combined",
]


# Hit‑ and trace‑field counts
HIT_FIELDS   = 10   # [det_id, is_good, x, y, z, nfold, arrival_time_low, arrival_time_up, total_signal_low, total_signal_up]
TRACE_FIELDS = 2   # [low, up] per window


def build_event_array(data, idx):
    """Build flattened 1D array + attrs for event index `idx` in `data`."""
    # 1) constants
    const_vals = []
    for key in CONSTANT_KEYS:
        if key in data:
            const_vals.extend(np.asarray(data[key][idx], np.float64).ravel().tolist())
    const_arr = np.array(const_vals, dtype=np.float64)
    C = const_arr.size

    # 2) hits
    det_ids = ak.to_numpy(data['hits_det_id'][idx])
    x_coords = ak.to_numpy(data['hits_x'][idx])
    y_coords = ak.to_numpy(data['hits_y'][idx])
    z_coords = ak.to_numpy(data['hits_z'][idx])
    nfolds = ak.to_numpy(data['hits_nfold'][idx])
    arrival_lows   = ak.to_numpy(data['hits_arrival_times_low'][idx])
    arrival_ups    = ak.to_numpy(data['hits_arrival_times_up'][idx])
    signal_lows    = ak.to_numpy(data['hits_total_signals_low'][idx])
    signal_ups     = ak.to_numpy(data['hits_total_signals_up'][idx])
    goods  = (ak.to_numpy(data.get('hits_good', np.ones_like(det_ids)))[idx]
              if 'hits_good' in data else np.ones_like(det_ids, dtype=np.float64))

    H = det_ids.size
    hits_flat = []
    for j in range(H):
        did          = float(det_ids[j])
        x, y, z      = float(x_coords[j]), float(y_coords[j]), float(z_coords[j])
        is_good      = float(goods[j])
        nf           = float(nfolds[j])
        arrival_low  = float(arrival_lows[j])
        arrival_up   = float(arrival_ups[j])
        signal_low   = float(signal_lows[j])
        signal_up    = float(signal_ups[j])
        hits_flat.extend([did, is_good, x, y, z, nf, arrival_low, arrival_up, signal_low, signal_up])
    hits_arr = np.array(hits_flat, dtype=np.float64)

    # 3) time traces
    lows_list = ak.to_list(data['hits_time_traces_low'][idx])
    ups_list  = ak.to_list(data['hits_time_traces_up'][idx])
    windows_per_hit, traces_flat = [], []
    for lw, uw in zip(lows_list, ups_list):
        windows_per_hit.append(len(lw))
        for ll, uu in zip(lw, uw):
            traces_flat.extend([float(ll), float(uu)])
    traces_arr = np.array(traces_flat, dtype=np.float64)

    # 4) concatenate
    event_data = np.concatenate([const_arr, hits_arr, traces_arr]).astype(np.float64)

    # 5) attributes
    attrs = {
        'constant_length': C,
        'hit_fields':      HIT_FIELDS,
        'num_hits':        H,
        'windows_per_hit': np.array(windows_per_hit, dtype=np.int32),
        'trace_fields':    TRACE_FIELDS,
    }
    return event_data, attrs


def process_file(args):
    """Worker: parse one DST file and build all its events."""
    dst_file, xmax_reader = args
    data = parse_dst_file(
        str(dst_file),
        use_grid_model=USE_GRID_MODEL,
        avg_traces=AVG_TRACES,
        xmax_reader=xmax_reader,
    )
    if data is None:
        return dst_file.name, []

    n_avail = len(data['energy'])
    events = []
    for j in range(n_avail):
        ev_data, ev_attrs = build_event_array(data, j)
        events.append((ev_data, ev_attrs))
    return dst_file.name, events


def main():
    # 1) check directory
    if not DST_DIR.is_dir():
        print(f"ERROR: DST directory not found: {DST_DIR}", file=sys.stderr)
        sys.exit(1)

    # 2) list files
    dst_files = sorted(DST_DIR.glob("*.dst*"))[:3]

    N = len(dst_files)
    if N == 0:
        print(f"ERROR: No DST files found in {DST_DIR}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {N} DST files to process.")

    # 3) create Xmax reader
    xmax_reader = XmaxReader(data_dir=XMAX_DATA_DIR, glob_pattern=XMAX_DATA_FILES)

    # 4) open HDF5 and start filling
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT_PATH, 'w') as f:
        evs_grp = f.create_group('events')
        global_idx = 0

        # 5) parallel processing
        args = [(dst_file, xmax_reader) for dst_file in dst_files]
        with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
            for _, events in tqdm(executor.map(process_file, args), total=N, desc="Files"):
                for ev_data, ev_attrs in events:
                    grp = evs_grp.create_group(f'event_{global_idx:06d}')
                    ds  = grp.create_dataset(
                        'event_data',
                        data=ev_data,
                        dtype='float64',
                        chunks=(ev_data.size,)
                    )
                    for k, v in ev_attrs.items():
                        ds.attrs[k] = v
                    global_idx += 1

        print(f"Wrote {global_idx:,} events to {OUT_PATH}")


if __name__ == "__main__":
    main()
