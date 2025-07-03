## DST to HDF5 Conversion Pipeline

This document details how raw DST inputs are transformed into NumPy arrays for HDF5 storage. We focus on data conversion—how each array is derived, its input & output shapes, and processing steps—rather than code specifics.

---

### A. High-Level Flow

| Step                  | Function/Input      | Output                                               | Notes                              |
| --------------------- | ------------------- | ---------------------------------------------------- | ---------------------------------- |
| Ingestion             | `DST file`          | `raw_lines`                                          | Read all lines                     |
| Sectioning            | `raw_lines`         | `(event_strs, sdmeta_strs, sdwave_strs, badsd_strs)` | Split per event into four blocks   |
| Parsing: Event        | `event_strs`        | `E` of shape `(N, F_event)`                          | Numeric table                      |
| Parsing: Metadata     | `sdmeta_strs`       | `meta_*` arrays `(N,n_tiles,n_tiles,...)`                        | Station metadata                   |
| Parsing: Waveforms    | `sdwave_strs`       | `wave_{low,up}` `(N,n_tiles,n_tiles,128)`                        | Raw ADC traces                     |
| Parsing: Bad Stations | `badsd_strs`        | `badsd_mask` `(N,n_tiles,n_tiles)`                               | Bad station flags                  |
| Conversion & Masking  | `E, meta_*, wave_*` | Trimmed/calibrated arrays `(M,...)`                  | Units, masks applied               |
| Grid Alignment        | `signals`           | Centered arrays `(M,n_tiles,n_tiles,...)`                        | Peak → center, normalize positions |
| Event Filtering       | `total_signals`     | Final events `M ≤ N`                                 | Drop zero-signal events            |
| Assemble Output       | All arrays          | `data` dict                                          | Map keys → arrays for HDF5         |

---

### B. Conversion Details Table

| Input Array                 | Input Shape   | Procedure                                              | Output Array                     | Output Shape  |
| --------------------------- | ------------- | ------------------------------------------------------ | -------------------------------- | ------------- |
| `E[:,0]`                    | `(N,)`        | Particle ID → atomic mass lookup                       | `mass_number`                    | `(N,)`        |
| `E[:,1]`                    | `(N,)`        | Primary energy (EeV)                                   | `energy`                         | `(N,)`        |
| `E[:,2:5]`                  | `(N,3)`       | Extract unit direction vector                          | `shower_axis`                    | `(N,3)`       |
| `E[:,5:8]`                  | `(N,3)`       | Convert cm→m (`×0.01`)                                 | `shower_core`                    | `(N,3)`       |
| `E[:,7]`                    | `(N,)`        | Parse YYMMDD integer                                   | `std_recon_yymmdd`               | `(N,)`        |
| `E[:,8]`                    | `(N,)`        | Parse HHMMSS integer                                   | `std_recon_hhmmss`               | `(N,)`        |
| `E[:,9]`                    | `(N,)`        | Number of detectors in cluster                         | `std_recon_nsd`                  | `(N,)`        |
| `E[:,10]`                   | `(N,)`        | Number of waveforms                                    | `std_recon_nofwf`                | `(N,)`        |
| `E[:,11]`                   | `(N,)`        | Microsecond remainder                                  | `std_recon_usec`                 | `(N,)`        |
| `E[:,12]`                   | `(N,)`        | Reconstructed energy (EeV)                             | `std_recon_energy`               | `(N,)`        |
| `E[:,13]`                   | `(N,)`        | LDF scale (VEM·m⁻²)                                    | `std_recon_ldf_scale`            | `(N,)`        |
| `E[:,14]`                   | `(N,)`        | LDF scale uncertainty                                  | `std_recon_ldf_scale_err`        | `(N,)`        |
| `E[:,15]`                   | `(N,)`        | LDF fit χ²                                             | `std_recon_ldf_chi2`             | `(N,)`        |
| `E[:,16]`                   | `(N,)`        | LDF degrees of freedom                                 | `std_recon_ldf_ndof`             | `(N,)`        |
| `E[:,17:19]`                | `(N,2)`       | Core X,Y via LDF (m)                                   | `std_recon_shower_core`          | `(N,2)`       |
| `E[:,18:20]`                | `(N,2)`       | Core X,Y uncertainty                                   | `std_recon_shower_core_err`      | `(N,2)`       |
| `E[:,21]`                   | `(N,)`        | Particle density at 800 m (VEM·m⁻²)                    | `std_recon_s800`                 | `(N,)`        |
| `meta_id`                   | `(N,n_tiles,n_tiles)`     | Remove duplicates & bad stations mask                  | `detector_positions_id`          | `(M,n_tiles,n_tiles)`     |
| `meta_positions_abs`        | `(N,n_tiles,n_tiles,3)`   | Center on max signal; normalize XY by `/1200`; leave Z | `detector_positions`             | `(M,n_tiles,n_tiles,3)`   |
| `meta_state`                | `(N,n_tiles,n_tiles)`     | Flag installed (1) vs missing (0)                      | `detector_exists`                | `(M,n_tiles,n_tiles)`     |
| `badsd_mask`                | `(N,n_tiles,n_tiles)`     | Flag bad stations (1=bad)                              | Combined in `detector_good` mask | `(M,n_tiles,n_tiles)`     |
| `wave_low`, `wave_up`       | `(N,n_tiles,n_tiles,128)` | Pedestal subtraction; multiply by gain (ADC→VEM)       | `time_traces_low/up`             | `(M,n_tiles,n_tiles,128)` |
| Raw tick counts (from wave) | `(N,n_tiles,n_tiles)`     | Multiply by 4000 (µs→ns)                               | `arrival_times_low/up`           | `(M,n_tiles,n_tiles)`     |
| `time_traces_low/up`        | `(M,n_tiles,n_tiles,128)` | Sum over last axis                                     | `total_signals_low/up`           | `(M,n_tiles,n_tiles)`     |

---

### C. Final Output Assembly

All arrays above are packaged into:

```python
{ key: array for key, array in locals().items() if isinstance(array, np.ndarray) }
```

with shapes `(M,…)`. They map directly to HDF5 datasets using `h5py.File.create_dataset`.

*End of detailed conversion tables.*
