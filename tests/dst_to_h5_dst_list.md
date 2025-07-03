## A. High-Level Flow

| Step                 | Input Data     | Input Shape | Procedure          | Output Data          | Output Shape                                  |
| -------------------- | -------------- | ----------- | ------------------ | -------------------- | --------------------------------------------- |
| Ingestion            | DST file path  | `str`       | `read_dst_file`    | raw DST string       | `str`                                         |
| Sectioning           | raw DST string | `str`       | `parse_dst_string` | **event\_list**      | `(61, N)`                                     |
| Sectioning           | raw DST string | `str`       | `parse_dst_string` | **sdmeta\_list**     | `list length N`, items `shape (12, varying)`  |
| Sectioning           | raw DST string | `str`       | `parse_dst_string` | **sdwaveform\_list** | `list length N`, items `shape (259, varying)` |
| Sectioning           | raw DST string | `str`       | `parse_dst_string` | **badsdinfo\_list**  | `list length N`, items `shape (varying,)`     |
| Parsing & Conversion | DST file path  | `str`       | `parse_dst_file`   | parsed `data` dict   | `dict` of arrays & lists                      |

---

## B. Conversion Details Table

### B.1 Event-level Features (from **event\_list**, shape `(61, N)`)

| Input Data            | Input Shape | Procedure                   | Output Field                                 | Output Shape          |
| --------------------- | ----------- | --------------------------- | -------------------------------------------- | --------------------- |
| `event_list[0]`       | `(N,)`      | PID → atomic mass           | `mass_number`                                | `(N,)`                |
| `event_list[1]`       | `(N,)`      | Primary energy (EeV)        | `energy`                                     | `(N,)`                |
| `event_list[2:5]`     | `(N,3)`     | Direction vector (θ, φ→xyz) | `shower_axis`                                | `(N,3)`               |
| `event_list[4:7]`     | `(N,3)`     | Core coords (cm→m)          | `shower_core`                                | `(N,3)`               |
| `event_list[7]`       | `(N,)`      | Parse YYMMDD                | `std_recon_yymmdd`                           | `(N,)`                |
| `event_list[8]`       | `(N,)`      | Parse HHMMSS                | `std_recon_hhmmss`                           | `(N,)`                |
| `event_list[11]`      | `(N,)`      | Microsecond remainder       | `std_recon_usec`                             | `(N,)`                |
| `event_list[10]`      | `(N,)`      | Count waveforms             | `std_recon_nofwf`                            | `(N,)`                |
| `event_list[9]`       | `(N,)`      | Count SD stations cluster   | `std_recon_nsd`                              | `(N,)`                |
| `event_list[57]`      | `(N,)`      | Count clusters (space)      | `std_recon_nsclust`                          | `(N,)`                |
| `event_list[56]`      | `(N,)`      | Count hits                  | `std_recon_nhits`                            | `(N,)`                |
| `event_list[58]`      | `(N,)`      | Count border stations       | `std_recon_nborder`                          | `(N,)`                |
| `event_list[59:61]`   | `(N,2)`     | Total charge (low, up)      | `std_recon_qtot`                             | `(N,2)`               |
| `event_list[12]`      | `(N,)`      | Energy reconstruction (EeV) | `std_recon_energy`                           | `(N,)`                |
| `event_list[13]`      | `(N,)`      | LDF scale (VEM·m⁻²)         | `std_recon_ldf_scale`                        | `(N,)`                |
| `event_list[14]`      | `(N,)`      | LDF scale uncertainty       | `std_recon_ldf_scale_err`                    | `(N,)`                |
| `event_list[15]`      | `(N,)`      | LDF fit χ²                  | `std_recon_ldf_chi2`                         | `(N,)`                |
| `event_list[16]`      | `(N,)`      | LDF fit ndof                | `std_recon_ldf_ndof`                         | `(N,)`                |
| `event_list[17:19]`   | `(N,2)`     | Core XY (LDF)               | `std_recon_shower_core`                      | `(N,2)`               |
| `event_list[18:20]`   | `(N,2)`     | Core XY uncertainty         | `std_recon_shower_core_err`                  | `(N,2)`               |
| `event_list[21]`      | `(N,)`      | Density @800 m (VEM·m⁻²)    | `std_recon_s800`                             | `(N,)`                |
| *Combined LDF & geom* | —           | Merge fits                  | `std_recon_combined_*` fields                | `(N,)`, `(N,2)` etc.  |
| *Geometry outputs*    | —           | Extract angles, errors      | `std_recon_shower_axis*`, `std_recon_geom_*` | `(N,3)` ×3, `(N,)` ×? |

### B.2 Station-level Features (from **sdmeta\_list**, **sdwaveform\_list**, **badsdinfo\_list**)

| Input Data                          | Input Shape                           | Procedure                      | Output Field             | Output Shape                 |
| ----------------------------------- | ------------------------------------- | ------------------------------ | ------------------------ | ---------------------------- |
| `sdmeta_list`                       | list length N; items `(12, varying)`  | list → grid & meter conversion | `detector_positions_abs` | `(N, n_tiles, n_tiles, 3)`   |
| `detector_positions_abs`            | `(N, n_tiles, n_tiles, 3)`            | center & normalize XY          | `detector_positions`     | `(N, n_tiles, n_tiles, 3)`   |
| `sdmeta_list`                       | list length N; items `(12, varying)`  | list → grid (IDs)              | `detector_positions_id`  | `(N, n_tiles, n_tiles)`      |
| `sdmeta_list`                       | list length N; items `(12, varying)`  | flag installed vs missing      | `detector_states`        | `(N, n_tiles, n_tiles)`      |
| `sdmeta_list`                       | list length N; items `(12, varying)`  | mask bad stations              | `detector_good`          | `(N, n_tiles, n_tiles)`      |
| `detector_states` & `detector_good` | `(N,n_tiles,n_tiles)`                 | count valid folds              | `nfold`                  | `(N, n_tiles, n_tiles)`      |
| `sdwaveform_list`                   | list length N; items `(259, varying)` | pedestal subtraction; ADC→VEM  | `time_traces_low/up`     | `(N, n_tiles, n_tiles, 128)` |
| `time_traces_low/up`                | `(N,n_tiles,n_tiles,128)`             | sum over time axis             | `total_signals_low/up`   | `(N, n_tiles, n_tiles)`      |
| `sdwaveform_list`                   | list length N; items `(259, varying)` | raw clock ticks → ns (×4000)   | `arrival_times_low/up`   | `(N, n_tiles, n_tiles)`      |

---

*End of DST sections & parsed data mapping.*
