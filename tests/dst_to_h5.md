## DST to HDF5 Conversion Pipeline

This document describes how the `dstparser` library reads raw DST files and outputs processed data arrays ready to be stored in HDF5 format. It follows the flow of `test_parser.py`, `dst_parser.py`, and `dst_adapter.py`.

---

### 1. Entry Point: `test_parser.py`

```bash
python test_parser.py <dst_file> [--print_read_data]
```

* **`test_parser`** calls `parse_dst_file(dst_file, ntile=7, avg_traces=False, add_shower_params=True, add_standard_recon=True, config=None)`  
* Prints parse time and, if requested, lists all keys in the returned `data` dict along with their shapes.

---

### 2. Parsing the DST File (`parse_dst_file`)

1. **Read raw lines**

   ```python
   dst_string = read_dst_file(dst_file)
   ```
2. **Section splitting**

   ```python
   event_strs, sdmeta_strs, sdwave_strs, badsd_strs = parse_dst_string(dst_string)
   ```
3. **Convert to numeric arrays**

   ```python
   dst_lists = (
       parse_event(event_strs),
       parse_sdmeta(sdmeta_strs),
       parse_sdwaveform(sdwave_strs),
       parse_badsdinfo(badsd_strs)
   )
   ```

---

### 3. Shower Parameters (`shower_params`)

| Output Key    | Source                    | Description                                 |
| ------------- | ------------------------- | ------------------------------------------- |
| `mass_number` | `event_list[0]`           | Converted CORSIKA particle ID → mass number |
| `energy`      | `event_list[1]`           | Primary energy (EeV)                        |
| `xmax`        | `xmax_reader(energy)`     | Depth of shower maximum (optional)          |
| `shower_axis` | `event_list[2..3]`        | Unit vector of arrival direction            |
| `shower_core` | `event_list[4..6] * 1e-2` | Core position (x,y,z) in meters             |

---

### 4. Standard Reconstruction (`standard_recon`)

Parses fields from `event_list` into detailed geometry and LDF fit results.

#### Event timestamps & clustering

| Key                 | Index in `event_list` | Notes                                        |
| ------------------- | --------------------- | -------------------------------------------- |
| `std_recon_yymmdd`  | 7                     | Date code (YYMMDD)                           |
| `std_recon_hhmmss`  | 8                     | Time code (HHMMSS)                           |
| `std_recon_usec`    | 11                    | Microseconds part                            |
| `std_recon_nofwf`   | 10                    | Number of waveforms                          |
| `std_recon_nsd`     | 9                     | Number of SD detectors in space–time cluster |
| `std_recon_nsclust` | 57                    | Number in spatial cluster                    |
| `std_recon_nhits`   | 56                    | Hit SD count                                 |
| `std_recon_nborder` | 58                    | Border detectors                             |

#### LDF fit parameters

| Key                       | Index(es) in `event_list` | Description                                 |
| ------------------------- | ------------------------- | ------------------------------------------- |
| `std_recon_qtot`          | [59, 60]                  | Total charge (low & high, VEM)              |
| `std_recon_energy`        | 12                        | Reconstructed energy (EeV)                  |
| `std_recon_ldf_scale`     | 13                        | LDF scale (VEM·m⁻²)                         |
| `std_recon_ldf_scale_err` | 14                        | Uncertainty in scale                        |
| `std_recon_ldf_chi2`      | 15                        | Chi² of LDF fit                             |
| `std_recon_ldf_ndof`      | 16                        | Degrees of freedom (n–3)                    |
| `std_recon_shower_core`   | [17,19]                   | Core position via LDF (converted to meters) |
| `std_recon_shower_core_err` | [18,20]                 | Core position uncertainty                   |
| `std_recon_s800`          | 21                        | Particle density at 800 m (VEM·m⁻²)         |

#### Combined geometry & LDF

| Key                                  | Index(es) | Description                        |
| ------------------------------------ | --------- | ---------------------------------- |
| `std_recon_combined_energy`          | 42        | Combined fit energy                |
| `std_recon_combined_scale`           | 43        | Combined LDF scale                 |
| `std_recon_combined_scale_err`       | 44        | Uncertainty of combined scale      |
| `std_recon_combined_chi2`            | 45        | Chi² combined fit                  |
| `std_recon_combined_ndof`            | 46        | Degrees of freedom (2n–6)          |
| `std_recon_combined_shower_core`     | [47,49]   | Combined core position             |
| `std_recon_combined_shower_core_err` | [48,50]   | Combined core position uncertainty |
| `std_recon_combined_s800`            | 51        | Combined S800                      |

#### Shower axis fits

| Key                                 | Source indices                                                                 | Fit type                    |
| ----------------------------------- | ------------------------------------------------------------------------------ | --------------------------- |
| `std_recon_shower_axis`             | <img src="https://render.githubusercontent.com/render/math?math=%5B32%2B0.5%C2%B0%2C33%2B%CF%80%5D">  | Free-curvature geometry fit |
| `std_recon_shower_axis_fixed_curve` | <img src="https://render.githubusercontent.com/render/math?math=%5B22%2B0.5%C2%B0%2C23%2B%CF%80%5D">  | Fixed-curvature geometry    |
| `std_recon_shower_axis_combined`    | <img src="https://render.githubusercontent.com/render/math?math=%5B52%2B0.5%C2%B0%2C53%2B%CF%80%5D">  | Geometry + LDF combined     |

#### Angle uncertainties & geometry stats

| Key                                     | Formula / Index                                                                                                       | Description                                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `std_recon_shower_axis_err`             | <img src="https://render.githubusercontent.com/render/math?math=%5Csqrt%7B34%5E2%20%2B%20%5Csin%5E2%28%5Ctheta%29%5Ccdot35%5E2%7D">        | Uncertainty in direction (free)               |
| `std_recon_shower_axis_err_fixed_curve` | <img src="https://render.githubusercontent.com/render/math?math=%5Csqrt%7B24%5E2%20%2B%20%5Csin%5E2%28%5Ctheta_%7Bfixed%7D%29%5Ccdot25%5E2%7D"> | Fixed curve error                             |
| `std_recon_shower_axis_err_combined`    | <img src="https://render.githubusercontent.com/render/math?math=%5Csqrt%7B54%5E2%20%2B%20%5Csin%5E2%28%5Ctheta_%7Bcomb%7D%29%5Ccdot55%5E2%7D">  | Combined direction error                      |
| `std_recon_geom_chi2`                   | 36                                                                                                                    | Geometry χ² (free curvature)                  |
| `std_recon_geom_ndof`                   | 37                                                                                                                    | DOF of free-fit                               |
| `std_recon_curvature`                   | 40                                                                                                                    | Curvature parameter                           |
| `std_recon_curvature_err`               | 41                                                                                                                    | Curvature uncertainty                         |
| `std_recon_geom_chi2_fixed_curve`       | 26                                                                                                                    | χ² fixed-curvature                            |
| `std_recon_geom_ndof_fixed_curve`       | 27                                                                                                                    | DOF fixed-curvature                           |
| `std_recon_border_distance`             | 30                                                                                                                    | Core-to-array-edge distance (units of 1200 m) |
| `std_recon_border_distance_tshape`      | 31                                                                                                                    | Pre-2008 boundary shape distance              |

---

### 5. Detector Readings (`detector_readings`)

Initializes per-event 7×7 grid arrays:

* **Positions**: `detector_positions`, `detector_positions_abs`, `detector_positions_id`  
* **Status**: `detector_states`, `detector_exists`, `detector_good`  
* **Signals**: `nfold` (foldedness), plus low/high:  

  * `arrival_times_low` / `arrival_times_up` (ns)  
  * `time_traces_low` / `time_traces_up` (128 bins)  
  * `total_signals_low` / `total_signals_up` (FADC counts)  

#### Workflow per event:

1. **Filter**: `cut_events` removes duplicate/coincidence detectors.  
2. **Center**: `center_tile` recenters on the detector with max signal, defines `(ix, iy)` grid indices.  
3. **Map**: `tile_positions` fills absolute coords & states; marks bad or missing SDs.  
4. **Normalize**: `tile_normalization` shifts & scales positions into <img src="https://render.githubusercontent.com/render/math?math=%5B-1%2C1%5D"> range.  
5. **Populate**:

   * Compute `nfold`, arrival times (*4 µs ticks → ns*), traces (ADC→VEM), and total signals low/high.  
6. **Mask invalid**: Zero out entries for non-operational detectors.  
7. **Cleanup**: Remove any completely empty events.

---

### 6. Optional IDs

If a `config` with `add_event_ids` is provided, the following HDF5 datasets are appended:

* `id_event`  
* `id_corsika_shower`  
* `id_energy_bin`  
* `id_data_set`  

---

### 7. Final Output

The result is a dictionary of NumPy arrays with names matching the HDF5 dataset keys. Each key can be directly written to an HDF5 file, preserving shape and data type. For example, using `h5py`:

```python
with h5py.File('output.h5', 'w') as hf:
    for key, arr in data.items():
        hf.create_dataset(key, data=arr)
