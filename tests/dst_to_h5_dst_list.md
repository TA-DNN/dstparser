# SDMETA, SDWAVEFORM & EVENT\_LIST Overview

## 1. SDMETA Physical Parameters

Each entry in `sdmeta_list` is an array of shape `(12, n_detectors)`.  Each row index `i` corresponds to a parameter array of shape `(n_detectors,)`:

| Row | Parameter       | Field name       | Description                                | Units / Notes  | Row Shape        |
| --- | --------------- | ---------------- | ------------------------------------------ | -------------- | ---------------- |
| 0   | `sd_id`         | `rufptn_.xxyy`   | Detector code: grid‑x & grid‑y             | integer code   | `(n_detectors,)` |
| 1   | `sd_isgood`     | `rufptn_.isgood` | Quality flag (1 = good, 0 = bad)           | boolean        | `(n_detectors,)` |
| 2   | `sd_reltime_lo` | `reltime[0]`     | Arrival time from lower PMT                | FADC time‑bins | `(n_detectors,)` |
| 3   | `sd_reltime_up` | `reltime[1]`     | Arrival time from upper PMT                | FADC time‑bins | `(n_detectors,)` |
| 4   | `sd_pulsa_lo`   | `pulsa[0]`       | Pulse amplitude (lower PMT)                | FADC counts    | `(n_detectors,)` |
| 5   | `sd_pulsa_up`   | `pulsa[1]`       | Pulse amplitude (upper PMT)                | FADC counts    | `(n_detectors,)` |
| 6   | `sd_xclf`       | `xyzclf[0]`      | X‑coordinate in CLF system                 | cm (×1e‑2 → m) | `(n_detectors,)` |
| 7   | `sd_yclf`       | `xyzclf[1]`      | Y‑coordinate in CLF system                 | cm (×1e‑2 → m) | `(n_detectors,)` |
| 8   | `sd_zclf`       | `xyzclf[2]`      | Z‑coordinate (height) in CLF system        | cm (×1e‑2 → m) | `(n_detectors,)` |
| 9   | `sd_vem_lo`     | `vem[0]`         | FADC counts per VEM (lower channel)        | counts / VEM   | `(n_detectors,)` |
| 10  | `sd_vem_up`     | `vem[1]`         | FADC counts per VEM (upper channel)        | counts / VEM   | `(n_detectors,)` |
| 11  | `sd_nfold`      | `nfold`          | Number of 128‑bin windows the signal spans | integer        | `(n_detectors,)` |

---

## 2. SDWAVEFORM Physical Parameters

Each entry in `sdwaveform_list` is an array of shape `(259, n_detectors)`.  Each row index `i` yields a vector of shape `(n_detectors,)`:

| Row idx | Parameter | Description                           | Units       | Row Shape                   |
| ------- | --------- | ------------------------------------- | ----------- | --------------------------- |
| 0       | `xxyy`    | Detector code (grid‑x×100 + grid‑y)   | integer     | `(n_detectors,)`            |
| 1       | `clkcnt`  | Coarse clock count                    | raw ticks   | `(n_detectors,)`            |
| 2       | `mclkcnt` | Master clock count                    | raw ticks   | `(n_detectors,)`            |
| 3–130   | `fadc_lo` | Lower-channel FADC samples (128 bins) | FADC counts | `(n_detectors,)` × 128 rows |
| 131–258 | `fadc_up` | Upper-channel FADC samples (128 bins) | FADC counts | `(n_detectors,)` × 128 rows |


---

## 3. EVENT\_LIST Physical Parameters

The combined `event_list` returned by `parse_dst_string` is an array of shape `(61, N)` for `N` events.  Each row index `i` corresponds to a parameter vector of shape `(N,)`, except multi‑row blocks:

| Row   | Field                       | Description                           | Units      | Row Shape |
| ----- | --------------------------- | ------------------------------------- | ---------- | --------- |
| 0     | `parttype`                  | Primary particle ID (CORSIKA)         | code       | `(N,)`    |
| 1     | `energy`                    | Primary energy                        | eV         | `(N,)`    |
| 2     | `theta`                     | Zenith angle                          | rad        | `(N,)`    |
| 3     | `phi`                       | Azimuth angle                         | rad        | `(N,)`    |
| 4–6   | `corexyz`                   | True shower core (x,y,z)              | m          | `(3, N)`  |
| 7     | `yymmdd`                    | Event date (YYMMDD)                   | integer    | `(N,)`    |
| 8     | `hhmmss`                    | Event time (HHMMSS)                   | integer    | `(N,)`    |
| 9     | `nstclust`                  | # of SDs in space cluster             | count      | `(N,)`    |
| 10    | `nofwf`                     | # of waveforms                        | count      | `(N,)`    |
| 11    | `usec`                      | Microseconds within second            | µs         | `(N,)`    |
| 12–16 | LDF-fit energy/scale/…      | Standard LDF fit outputs              | various    | `(5, N)`  |
| 17–21 | LDF core & errors           | Core xy & uncertainty, S800           | m, VEM     | `(5, N)`  |
| 22–31 | Geometry (fixed & pre-fit)  | Angles, errors, border distances      | deg, units | `(10, N)` |
| 32–41 | Geometry (free & curvature) | Angles, errors, curvature params      | deg, m     | `(10, N)` |
| 42–51 | Combined LDF+geom outputs   | Combined fit energy, core, S800, axis | EeV, m, …  | `(10, N)` |
| 52–55 | Combined axis uncertainties | φ/θ errors                            | deg        | `(4, N)`  |
| 56–60 | Pattern outputs             | nhits, nsclust, nborder, qtot\[2]     | count, VEM | `(5, N)`  |


---

## A. High-Level Flow

| Step               | Input          | Input Shape | Procedure          | Output            | Output Shape           |
| ------------------ | -------------- | ----------- | ------------------ | ----------------- | ---------------------- |
| 1. Ingestion       | DST file path  | `str`       | `read_dst_file`    | raw DST string    | `str`                  |
| 2. Sectioning      | raw DST string | `str`       | `parse_dst_string` | `event_list`      | `(61, N)`              |
|                    |                |             |                    | `sdmeta_list`     | `[(12, n_det), …]`     |
|                    |                |             |                    | `sdwaveform_list` | `[(259, n_det), …]`    |
|                    |                |             |                    | `badsdinfo_list`  | `[(n_bad), …]`         |
| 3. Parsing & Conv. | DST file path  | `str`       | `parse_dst_file`   | `data` dict       | dict of arrays & lists |

---

## B. Conversion Details Table

### B.1 Event-level Features (from **event\_list**, shape `(61, N)`)

| Input Data        | Shape    | Procedure            | Output Field       | Output Shape |
| ----------------- | -------- | -------------------- | ------------------ | ------------ |
| `event_list[0]`   | `(N,)`   | PID → atomic mass    | `mass_number`      | `(N,)`       |
| `event_list[1]`   | `(N,)`   | Primary energy (EeV) | `energy`           | `(N,)`       |
| `event_list[2:5]` | `(3, N)` | Direction (θ, φ→xyz) | `shower_axis`      | `(N,3)`      |
| `event_list[4:7]` | `(3, N)` | Core coords (cm→m)   | `shower_core`      | `(N,3)`      |
| `event_list[7]`   | `(N,)`   | Parse YYMMDD         | `std_recon_yymmdd` | `(N,)`       |
| …                 | …        | …                    | …                  | …            |

### B.2 Station-level Features (from **sdmeta\_list**, **sdwaveform\_list**, **badsdinfo\_list**)

| Input                               | Shape               | Procedure                      | Output Field             | Output Shape                |
| ----------------------------------- | ------------------- | ------------------------------ | ------------------------ | --------------------------- |
| `sdmeta_list`                       | `[(12, n_det), …]`  | grid → CLF coords (cm→m)       | `detector_positions_abs` | `(N, n_tiles, n_tiles, 3)`  |
| `detector_positions_abs`            | `(N,…,3)`           | center & normalize XY          | `detector_positions`     | `(N, n_tiles, n_tiles, 3)`  |
| `sdmeta_list`                       | `[(12, n_det), …]`  | extract IDs                    | `detector_positions_id`  | `(N, n_tiles, n_tiles)`     |
| `sdmeta_list`                       | `[(12, n_det), …]`  | mask installed vs missing      | `detector_states`        | `(N, n_tiles, n_tiles)`     |
| `sdmeta_list`                       | `[(12, n_det), …]`  | mask bad stations              | `detector_good`          | `(N, n_tiles, n_tiles)`     |
| `detector_states` & `detector_good` | `(N,… )`            | count valid folds              | `nfold`                  | `(N, n_tiles, n_tiles)`     |
| `sdwaveform_list`                   | `[(259, n_det), …]` | ADC→VEM & pedestal subtraction | `time_traces_low/up`     | `(N, n_tiles, n_tiles,128)` |
| `time_traces_low/up`                | `(N, …,128)`        | sum over time                  | `total_signals_low/up`   | `(N, n_tiles, n_tiles)`     |
| `sdwaveform_list`                   | `[(259, n_det), …]` | clock ticks → ns               | `arrival_times_low/up`   | `(N, n_tiles, n_tiles)`     |

---

*End of combined DST processing overview.*
