#!/usr/bin/env python3
"""
save_dst_to_1linehdf5.py

Parse a DST file once (ragged hit-based model), then write all events into a single HDF5 file.
Each event is stored in its own group under /events, with a single 1D dataset `event_data`
containing:
  [<constants>, <flattened hits>, <flattened time_traces>]
and the dataset carries attributes necessary to reconstruct the original structure.

Input and output paths are hard-coded.
"""
import sys
from pathlib import Path
import numpy as np
import awkward as ak
import h5py

# allow imports from your package
dir_root = Path(__file__).parent
sys.path.insert(0, str(dir_root / "src"))
from dstparser.dst_adapter import parse_dst_file

# Hard-coded paths
dst_path = Path(
    "/home/marktsai321/TA_DNN/outputs/2025/07/17/"
    "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
)
out_path = Path(
    "/home/marktsai321/TA_DNN/outputs/2025/07/23/1_line_test.h5"
)

# Configuration flags
USE_GRID_MODEL = False
AVG_TRACES = False  # ensure low/up arrays present

# Define which keys are "constant" features, including standard reconstruction
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

# Hit-fields per hit in flattened block
HIT_FIELDS = 8   # [xxyy_id, is_good, x, y, z, nfold, arrival_time, total_signal]
# Trace-fields per window (low, up)
TRACE_FIELDS = 2


def main():
    if not dst_path.exists():
        print(f"ERROR: DST file not found: {dst_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {dst_path.name}…")
    data = parse_dst_file(str(dst_path), use_grid_model=USE_GRID_MODEL, avg_traces=AVG_TRACES)
    if data is None:
        print("ERROR: parse_dst_file returned None", file=sys.stderr)
        sys.exit(1)

    # Number of parsed events
    n_events = len(ak.Array(data))
    if n_events == 0:
        print("ERROR: no events parsed", file=sys.stderr)
        sys.exit(1)

    # Open HDF5 for writing
    with h5py.File(out_path, 'w') as f:
        evs_grp = f.create_group('events')

        for i in range(n_events):
            grp = evs_grp.create_group(f'event_{i:06d}')

            # 1) Build constant-length array
            const_vals = []
            for key in CONSTANT_KEYS:
                if key not in data:
                    continue
                arr = data[key][i]
                npv = np.asarray(arr, dtype=np.float32).ravel()
                const_vals.extend(npv.tolist())
            const_arr = np.array(const_vals, dtype=np.float32)
            C = const_arr.size

            # 2) Build flattened hits
            det_ids = ak.to_numpy(data['hits_det_id'][i])
            nfolds = ak.to_numpy(data['hits_nfold'][i])
            arr_lows = ak.to_numpy(data['hits_arrival_times_low'][i])
            arr_ups  = ak.to_numpy(data['hits_arrival_times_up'][i])
            sig_lows = ak.to_numpy(data['hits_total_signals_low'][i])
            sig_ups  = ak.to_numpy(data['hits_total_signals_up'][i])
            goods    = (ak.to_numpy(data.get('hits_good', np.ones_like(det_ids)))[i]
                        if 'hits_good' in data else np.ones_like(det_ids, dtype=np.float32))

            H = det_ids.size
            hits_flat = []
            for j in range(H):
                did = float(det_ids[j])
                x = float(did // 100)
                y = float(did % 100)
                z = 0.0
                is_good = float(goods[j])
                nf = float(nfolds[j])
                arrival = float((arr_lows[j] + arr_ups[j]) / 2.0)
                total   = float((sig_lows[j] + sig_ups[j]) / 2.0)
                hits_flat.extend([did, is_good, x, y, z, nf, arrival, total])
            hits_arr = np.array(hits_flat, dtype=np.float32)

            # 3) Build flattened time_traces
            lows_list = ak.to_list(data['hits_time_traces_low'][i])
            ups_list  = ak.to_list(data['hits_time_traces_up'][i])
            windows_per_hit = []
            traces_flat = []
            for lows, ups in zip(lows_list, ups_list):
                W = len(lows)
                windows_per_hit.append(W)
                for ll, uu in zip(lows, ups):
                    traces_flat.append(float(ll))
                    traces_flat.append(float(uu))
            traces_arr = np.array(traces_flat, dtype=np.float32)

            # 4) Concatenate all pieces
            event_data = np.concatenate([const_arr, hits_arr, traces_arr]).astype(np.float32)

            # 5) Write 1D dataset with attributes
            ds = grp.create_dataset(
                'event_data',
                data=event_data,
                dtype='float32',
                chunks=(event_data.size,)
            )
            # store reconstruction info
            ds.attrs['constant_length']   = C
            ds.attrs['hit_fields']        = HIT_FIELDS
            ds.attrs['num_hits']          = H
            ds.attrs['windows_per_hit']   = np.array(windows_per_hit, dtype=np.int32)
            ds.attrs['trace_fields']      = TRACE_FIELDS

    print(f"Wrote {n_events} events to {out_path}")


if __name__ == '__main__':
    main()
