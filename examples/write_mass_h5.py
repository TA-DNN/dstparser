#!/usr/bin/env python3
"""
Parse every event in each DST file in a directory using up to N cores.
Each DST file is processed by one core and written immediately to its own HDF5 file.
Once all per-file HDF5s are done, merge them into a single large file under /events,
preserving the same per-event HDF5 structure.
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import awkward as ak
import h5py
from tqdm import tqdm

# allow imports from your package
dir_root = Path(__file__).parent
sys.path.insert(0, str(dir_root / "src"))
from dstparser.dst_adapter import parse_dst_file

# ───────────────────────────────────────────────────────────────────────────────
# DST_DIR    = Path("/home/marktsai321/TA_DNN/sample_dst")
DST_DIR    = Path("/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
                   "qgsii04proton/080417_160603/Em1_bsdinfo/")
OUT_DIR    = Path("/home/marktsai321/TA_DNN/training_dataset/0731/bulk_h5_02/")
FINAL_OUT  = OUT_DIR / "all_events.h5"
CPU_CORES  = 48
# ───────────────────────────────────────────────────────────────────────────────

USE_GRID_MODEL = False
AVG_TRACES     = False

# Constant feature keys (must include 'energy')
CONSTANT_KEYS = [
    "mass_number", "energy", "xmax", "shower_axis", "shower_core",
    # Standard Reconstruction
    "std_recon_yymmdd","std_recon_hhmmss","std_recon_usec",
    "std_recon_nofwf","std_recon_nsd","std_recon_nsclust",
    "std_recon_nhits","std_recon_nborder","std_recon_qtot",
    "std_recon_energy","std_recon_ldf_scale","std_recon_ldf_scale_err",
    "std_recon_ldf_chi2","std_recon_ldf_ndof","std_recon_shower_core",
    "std_recon_shower_core_err","std_recon_s800","std_recon_combined_energy",
    "std_recon_combined_scale","std_recon_combined_scale_err",
    "std_recon_combined_chi2","std_recon_combined_ndof",
    "std_recon_combined_shower_core","std_recon_combined_shower_core_err",
    "std_recon_combined_s800","std_recon_shower_axis",
    "std_recon_shower_axis_fixed_curve","std_recon_shower_axis_combined",
    "std_recon_geom_chi2","std_recon_geom_ndof","std_recon_curvature",
    "std_recon_curvature_err","std_recon_geom_chi2_fixed_curve",
    "std_recon_geom_ndof_fixed_curve","std_recon_border_distance",
    "std_recon_border_distance_tshape","std_recon_shower_axis_err",
    "std_recon_shower_axis_err_fixed_curve",
    "std_recon_shower_axis_err_combined",
]
HIT_FIELDS   = 8   # [det_id, is_good, x, y, z, nfold, arrival_time, total_signal]
TRACE_FIELDS = 2   # [low, up] per window


def build_event_array(data, idx):
    """Build flattened 1D array + attrs for event index `idx` in `data`."""
    const_vals = []
    for key in CONSTANT_KEYS:
        if key in data:
            const_vals.extend(np.asarray(data[key][idx], np.float32).ravel().tolist())
    const_arr = np.array(const_vals, dtype=np.float32)
    C = const_arr.size

    det_ids = ak.to_numpy(data['hits_det_id'][idx])
    nfolds = ak.to_numpy(data['hits_nfold'][idx])
    lows   = ak.to_numpy(data['hits_arrival_times_low'][idx])
    ups    = ak.to_numpy(data['hits_arrival_times_up'][idx])
    sl     = ak.to_numpy(data['hits_total_signals_low'][idx])
    su     = ak.to_numpy(data['hits_total_signals_up'][idx])
    goods  = (ak.to_numpy(data.get('hits_good', np.ones_like(det_ids)))[idx]
              if 'hits_good' in data else np.ones_like(det_ids, dtype=np.float32))

    H = det_ids.size
    hits_flat = []
    for j in range(H):
        did     = float(det_ids[j])
        x, y    = float(did // 100), float(did % 100)
        is_good = float(goods[j])
        nf      = float(nfolds[j])
        arrival = float((lows[j] + ups[j]) / 2.0)
        total   = float((sl[j] + su[j]) / 2.0)
        hits_flat.extend([did, is_good, x, y, 0.0, nf, arrival, total])
    hits_arr = np.array(hits_flat, dtype=np.float32)

    lows_list = ak.to_list(data['hits_time_traces_low'][idx])
    ups_list  = ak.to_list(data['hits_time_traces_up'][idx])
    windows_per_hit, traces_flat = [], []
    for lw, uw in zip(lows_list, ups_list):
        windows_per_hit.append(len(lw))
        for ll, uu in zip(lw, uw): traces_flat.extend([float(ll), float(uu)])
    traces_arr = np.array(traces_flat, dtype=np.float32)

    event_data = np.concatenate([const_arr, hits_arr, traces_arr]).astype(np.float32)
    attrs = {
        'constant_length': C,
        'hit_fields':      HIT_FIELDS,
        'num_hits':        H,
        'windows_per_hit': np.array(windows_per_hit, dtype=np.int32),
        'trace_fields':    TRACE_FIELDS,
    }
    return event_data, attrs


def process_file(dst_file: Path):
    data = parse_dst_file(str(dst_file), use_grid_model=USE_GRID_MODEL, avg_traces=AVG_TRACES)
    if data is None:
        return None, 0
    n_events = len(ak.Array(data))
    if n_events == 0:
        return None, 0

    out_path = OUT_DIR / f"{dst_file.stem}.h5"
    with h5py.File(out_path, 'w') as f_out:
        evs_grp = f_out.create_group('events')
        for i in range(n_events):
            ev_data, ev_attrs = build_event_array(data, i)
            grp = evs_grp.create_group(f'event_{i:06d}')
            ds  = grp.create_dataset('event_data', data=ev_data, dtype='float32', chunks=(ev_data.size,))
            for k, v in ev_attrs.items(): ds.attrs[k] = v
    return out_path, n_events


def merge_h5_files(h5_paths, output_path, chunk_events=10000):
    """
    Merge multiple per-file HDF5s into a single file, storing all events
    in one chunked, appendable dataset "/events" with a compound dtype.
    """
    # variable-length data types
    vlen_f32 = h5py.vlen_dtype(np.dtype('float32'))
    vlen_i32 = h5py.vlen_dtype(np.dtype('int32'))

    # compound dtype: one record per event
    event_dt = np.dtype([
        ('event_data',      vlen_f32),
        ('constant_length', np.int32),
        ('hit_fields',      np.int32),
        ('num_hits',        np.int32),
        ('windows_per_hit', vlen_i32),
        ('trace_fields',    np.int32),
    ])

    with h5py.File(output_path, 'w') as f_out:
        ds = f_out.create_dataset(
            'events',
            shape=(0,),
            maxshape=(None,),
            dtype=event_dt,
            chunks=(chunk_events,)
        )

        total = 0
        for h5_path in tqdm(sorted(h5_paths), desc="Merging HDF5 files"):
            with h5py.File(h5_path, 'r') as f_in:
                grp_in = f_in.get('events')
                if not grp_in:
                    continue

                for evt_name in sorted(grp_in):
                    dsi   = grp_in[evt_name]['event_data']
                    data  = dsi[...]
                    attrs = dsi.attrs

                    record = (
                        data,
                        attrs['constant_length'],
                        attrs['hit_fields'],
                        attrs['num_hits'],
                        attrs['windows_per_hit'],
                        attrs['trace_fields']
                    )

                    ds.resize(total + 1, axis=0)
                    ds[total] = record
                    total += 1

        print(f"Merged {total} events into {output_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DST_DIR.is_dir():
        sys.exit(f"ERROR: DST directory not found: {DST_DIR}")

    dst_files = sorted(DST_DIR.glob("*.dst*"))
    if not dst_files:
        sys.exit(f"ERROR: No DST files found in {DST_DIR}")

    created_paths = []
    with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
        futures = {executor.submit(process_file, df): df for df in dst_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DSTs"):
            out_path, count = future.result()
            if out_path:
                created_paths.append(out_path)
                print(f"Wrote {count} events to {out_path}")

    if created_paths:
        merge_h5_files(created_paths, FINAL_OUT)
    else:
        print("No HDF5 files were created; nothing to merge.")


if __name__ == "__main__":
    main()
