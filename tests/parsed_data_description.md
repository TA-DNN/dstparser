# Parsed Data Output Description

This document outlines the structure and shape of the data returned by `dstparser.dst_adapter.parse_dst_file`. The output is a Python dictionary containing various NumPy and Awkward Arrays. The exact contents and their shapes depend on the function's parameters.

## I. Output Fields and Shapes

The following sections detail the output dictionary keys and the corresponding data shapes for each of the 16 possible parameter combinations.

### 1. `use_grid_model=True`, `avg_traces=True`, `add_shower_params=True`, `add_standard_recon=True`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `std_recon_border_distance`: `(146,)`
- `std_recon_border_distance_tshape`: `(146,)`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times`: `(146, 7, 7)`
- `time_traces`: `(146, 7, 7, 128)`
- `total_signals`: `(146, 7, 7)`

### 2. `use_grid_model=True`, `avg_traces=True`, `add_shower_params=True`, `add_standard_recon=False`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times`: `(146, 7, 7)`
- `time_traces`: `(146, 7, 7, 128)`
- `total_signals`: `(146, 7, 7)`

### 3. `use_grid_model=True`, `avg_traces=True`, `add_shower_params=False`, `add_standard_recon=True`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `std_recon_border_distance`: `(146,)`
- `std_recon_border_distance_tshape`: `(146,)`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times`: `(146, 7, 7)`
- `time_traces`: `(146, 7, 7, 128)`
- `total_signals`: `(146, 7, 7)`

### 4. `use_grid_model=True`, `avg_traces=True`, `add_shower_params=False`, `add_standard_recon=False`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times`: `(146, 7, 7)`
- `time_traces`: `(146, 7, 7, 128)`
- `total_signals`: `(146, 7, 7)`

### 5. `use_grid_model=True`, `avg_traces=False`, `add_shower_params=True`, `add_standard_recon=True`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `std_recon_border_distance`: `(146,)`
- `std_recon_border_distance_tshape`: `(146,)`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times_low`: `(146, 7, 7)`
- `arrival_times_up`: `(146, 7, 7)`
- `time_traces_low`: `(146, 7, 7, 128)`
- `time_traces_up`: `(146, 7, 7, 128)`
- `total_signals_low`: `(146, 7, 7)`
- `total_signals_up`: `(146, 7, 7)`

### 6. `use_grid_model=True`, `avg_traces=False`, `add_shower_params=True`, `add_standard_recon=False`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times_low`: `(146, 7, 7)`
- `arrival_times_up`: `(146, 7, 7)`
- `time_traces_low`: `(146, 7, 7, 128)`
- `time_traces_up`: `(146, 7, 7, 128)`
- `total_signals_low`: `(146, 7, 7)`
- `total_signals_up`: `(146, 7, 7)`

### 7. `use_grid_model=True`, `avg_traces=False`, `add_shower_params=False`, `add_standard_recon=True`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `std_recon_border_distance`: `(146,)`
- `std_recon_border_distance_tshape`: `(146,)`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times_low`: `(146, 7, 7)`
- `arrival_times_up`: `(146, 7, 7)`
- `time_traces_low`: `(146, 7, 7, 128)`
- `time_traces_up`: `(146, 7, 7, 128)`
- `total_signals_low`: `(146, 7, 7)`
- `total_signals_up`: `(146, 7, 7)`

### 8. `use_grid_model=True`, `avg_traces=False`, `add_shower_params=False`, `add_standard_recon=False`
- `detector_positions`: `(146, 7, 7, 3)`
- `detector_positions_abs`: `(146, 7, 7, 3)`
- `detector_positions_id`: `(146, 7, 7)`
- `detector_states`: `(146, 7, 7)`
- `detector_exists`: `(146, 7, 7)`
- `detector_good`: `(146, 7, 7)`
- `nfold`: `(146, 7, 7)`
- `arrival_times_low`: `(146, 7, 7)`
- `arrival_times_up`: `(146, 7, 7)`
- `time_traces_low`: `(146, 7, 7, 128)`
- `time_traces_up`: `(146, 7, 7, 128)`
- `total_signals_low`: `(146, 7, 7)`
- `total_signals_up`: `(146, 7, 7)`

### 9. `use_grid_model=False`, `avg_traces=True`, `add_shower_params=True`, `add_standard_recon=True`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 10. `use_grid_model=False`, `avg_traces=True`, `add_shower_params=True`, `add_standard_recon=False`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 11. `use_grid_model=False`, `avg_traces=True`, `add_shower_params=False`, `add_standard_recon=True`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 12. `use_grid_model=False`, `avg_traces=True`, `add_shower_params=False`, `add_standard_recon=False`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 13. `use_grid_model=False`, `avg_traces=False`, `add_shower_params=True`, `add_standard_recon=True`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_low`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_up`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_low`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_up`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces_low`: `{0: 146, 1: (3, 32), 2: (128, 512)}`
- `hits_time_traces_up`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 14. `use_grid_model=False`, `avg_traces=False`, `add_shower_params=True`, `add_standard_recon=False`
- `mass_number`: `(146,)`
- `energy`: `(146,)`
- `xmax`: `(146,)`
- `shower_axis`: `(146, 3)`
- `shower_core`: `(146, 3)`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_low`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_up`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_low`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_up`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces_low`: `{0: 146, 1: (3, 32), 2: (128, 512)}`
- `hits_time_traces_up`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 15. `use_grid_model=False`, `avg_traces=False`, `add_shower_params=False`, `add_standard_recon=True`
- `std_recon_yymmdd`: `(146,)`
- `std_recon_hhmmss`: `(146,)`
- `std_recon_usec`: `(146,)`
- `std_recon_nofwf`: `(146,)`
- `std_recon_nsd`: `(146,)`
- `std_recon_nsclust`: `(146,)`
- `std_recon_nhits`: `(146,)`
- `std_recon_nborder`: `(146,)`
- `std_recon_qtot`: `(146, 2)`
- `std_recon_energy`: `(146,)`
- `std_recon_ldf_scale`: `(146,)`
- `std_recon_ldf_scale_err`: `(146,)`
- `std_recon_ldf_chi2`: `(146,)`
- `std_recon_ldf_ndof`: `(146,)`
- `std_recon_shower_core`: `(146, 2)`
- `std_recon_shower_core_err`: `(146, 2)`
- `std_recon_s800`: `(146,)`
- `std_recon_combined_energy`: `(146,)`
- `std_recon_combined_scale`: `(146,)`
- `std_recon_combined_scale_err`: `(146,)`
- `std_recon_combined_chi2`: `(146,)`
- `std_recon_combined_ndof`: `(146,)`
- `std_recon_combined_shower_core`: `(146, 2)`
- `std_recon_combined_shower_core_err`: `(146, 2)`
- `std_recon_combined_s800`: `(146,)`
- `std_recon_shower_axis`: `(146, 3)`
- `std_recon_shower_axis_fixed_curve`: `(146, 3)`
- `std_recon_shower_axis_combined`: `(146, 3)`
- `std_recon_shower_axis_err`: `(146,)`
- `std_recon_shower_axis_err_fixed_curve`: `(146,)`
- `std_recon_shower_axis_err_combined`: `(146,)`
- `std_recon_geom_chi2`: `(146,)`
- `std_recon_geom_ndof`: `(146,)`
- `std_recon_curvature`: `(146,)`
- `std_recon_curvature_err`: `(146,)`
- `std_recon_geom_chi2_fixed_curve`: `(146,)`
- `std_recon_geom_ndof_fixed_curve`: `(146,)`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_low`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_up`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_low`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_up`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces_low`: `{0: 146, 1: (3, 32), 2: (128, 512)}`
- `hits_time_traces_up`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

### 16. `use_grid_model=False`, `avg_traces=False`, `add_shower_params=False`, `add_standard_recon=False`
- `hits_det_id`: `{0: 146, 1: (3, 32)}`
- `hits_nfold`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_low`: `{0: 146, 1: (3, 32)}`
- `hits_arrival_times_up`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_low`: `{0: 146, 1: (3, 32)}`
- `hits_total_signals_up`: `{0: 146, 1: (3, 32)}`
- `hits_time_traces_low`: `{0: 146, 1: (3, 32), 2: (128, 512)}`
- `hits_time_traces_up`: `{0: 146, 1: (3, 32), 2: (128, 512)}`

---

## II. Physical Meaning and Shape Explanation

This section provides a detailed description of the physical meaning of each parameter and explains the origin of its shape. The shapes are generally described as `(N, ...)` where `N` is the number of events in the parsed file.

### A. Event-Level Features
These features provide information about the cosmic ray shower as a whole. The first dimension of their shape is always `N`, the number of events.

| Field Name | Description | Shape Explanation |
| --- | --- | --- |
| `mass_number` | The atomic mass number of the primary particle that initiated the shower. Derived from the CORSIKA particle ID. | `(N,)` - A scalar value for each event. |
| `energy` | The energy of the primary particle. | `(N,)` - A scalar value for each event. |
| `xmax` | The atmospheric depth at which the shower reaches its maximum number of particles. | `(N,)` - A scalar value for each event. |
| `shower_axis` | A 3D unit vector representing the direction from which the cosmic ray shower arrived. | `(N, 3)` - A 3-element vector (x, y, z) for each event. |
| `shower_core` | The 3D coordinates (x, y, z) of the point where the shower axis intersects the ground. | `(N, 3)` - A 3-element vector (x, y, z) for each event. |

### B. Standard Reconstruction Features
These features are derived from the standard reconstruction algorithms. Their shapes are also based on `N`, the number of events.

| Field Name | Description | Shape Explanation |
| --- | --- | --- |
| `std_recon_yymmdd` | The date of the event in YYMMDD format. | `(N,)` - A scalar value for each event. |
| `std_recon_hhmmss` | The time of the event in HHMMSS format. | `(N,)` - A scalar value for each event. |
| `std_recon_usec` | The microsecond part of the event time. | `(N,)` - A scalar value for each event. |
| `std_recon_nofwf` | The total number of waveforms recorded for the event across all detectors. | `(N,)` - A scalar value for each event. |
| `std_recon_nsd` | The number of surface detectors (SDs) that are part of the space-time cluster used for reconstruction. | `(N,)` - A scalar value for each event. |
| `std_recon_qtot` | The total charge (signal) recorded by the SDs in the space-time cluster, for both lower and upper PMTs. | `(N, 2)` - A 2-element vector (lower, upper) for each event. |
| `std_recon_energy` | The energy of the shower as reconstructed by the standard energy estimation table. | `(N,)` - A scalar value for each event. |
| `std_recon_ldf_scale` | The scale parameter from the Lateral Distribution Function (LDF) fit. | `(N,)` - A scalar value for each event. |
| `std_recon_shower_core` | Reconstructed core position (x, y) from the LDF fit. | `(N, 2)` - A 2-element vector (x, y) for each event. |
| `std_recon_shower_axis` | Reconstructed 3D arrival direction. | `(N, 3)` - A 3-element vector (x, y, z) for each event. |
- *... and so on for all `std_recon` parameters. The logic remains the same: `(N,)` for scalar values per event, and `(N, D)` for D-dimensional vectors per event.*

### C. Detector-Level Features (Grid Model)
When `use_grid_model=True`, the data is organized into a `(N, ntile, ntile, ...)` grid.

-   `N`: Number of events.
-   `ntile`: The side length of the square grid of detectors centered on the detector with the maximum signal (default is 7).

| Field Name | Description | Shape Explanation |
| --- | --- | --- |
| `detector_positions` | The normalized (x, y, z) positions of the detectors in the grid. | `(N, ntile, ntile, 3)` - For each of `N` events, a grid of size `ntile`x`ntile`, with each cell containing a 3-element (x,y,z) position vector. |
| `detector_positions_abs` | The absolute (x, y, z) positions of the detectors in the CLF coordinate system. | `(N, ntile, ntile, 3)` - Same structure as above, but with absolute coordinates. |
| `detector_positions_id` | The ID of each detector in the grid. | `(N, ntile, ntile)` - For each event, a grid containing a scalar detector ID in each cell. |
| `detector_states` | A boolean flag indicating if a detector is active and part of the event. | `(N, ntile, ntile)` - For each event, a grid containing a boolean value in each cell. |
| `nfold` | The "foldedness" of the signal, indicating how many 128-bin windows the signal spans. | `(N, ntile, ntile)` - For each event, a grid containing the integer nfold value for each detector. |
| `arrival_times` / `_low` / `_up` | The arrival time of the signal at each detector. | `(N, ntile, ntile)` - For each event, a grid containing a scalar arrival time for each detector. |
| `time_traces` / `_low` / `_up` | The FADC time traces recorded by each detector. | `(N, ntile, ntile, 128)` - For each event, a grid where each cell contains a 128-bin time trace vector. `128` is the number of FADC samples in a single window. |
| `total_signals` / `_low` / `_up` | The total signal (integrated charge) recorded by each detector. | `(N, ntile, ntile)` - For each event, a grid containing the scalar total signal for each detector. |

### D. Detector-Level Features (Awkward Array Model)
When `use_grid_model=False`, the data is stored in jagged Awkward Arrays to save space. The shape is described as `var * ...` indicating a variable-length dimension.

| Field Name | Description | Shape Explanation |
| --- | --- | --- |
| `hits_det_id` | An array where each entry is a list of detector IDs for the hits in one event. | `(N, var * int)` - For each of `N` events, a variable-length list of integer detector IDs. The length of the list corresponds to the number of hits in that event. |
| `hits_nfold` | An array where each entry is a list of nfold values for the hits in one event. | `(N, var * int)` - For each of `N` events, a variable-length list of integer nfold values. |
| `hits_arrival_times` / `_low` / `_up` | Jagged array of signal arrival times for each hit. | `(N, var * float)` - For each of `N` events, a variable-length list of float arrival times. |
| `hits_total_signals` / `_low` / `_up` | Jagged array of total signals for each hit. | `(N, var * float)` - For each of `N` events, a variable-length list of float total signals. |
| `hits_time_traces` / `_low` / `_up` | A nested jagged array containing the FADC time traces for each window of each hit. | `(N, var * var * float)` - For each of `N` events, a variable-length list of hits. Each hit has its own variable-length list of time trace samples. The length of the trace is `nfold * 128` (for averaged traces) or `nfold * 256 / 2` (for low/up traces). |

### E. Note on Time Trace Processing (`time_traces` and `hits_time_traces`)

The time traces are derived from the raw FADC (Flash Analog-to-Digital Converter) data, which records the signal from two separate photomultiplier tubes (PMTs) for each detector. These are designed to cover different signal strengths:

-   **Lower-level PMT (`_low`)**: A more sensitive channel, designed to accurately measure smaller signals.
-   **Upper-level PMT (`_up`)**: A less sensitive channel, designed to avoid saturation from very large signals.

The processing pipeline is as follows:

1.  **De-interleaving**: The raw hardware output interleaves the samples from the upper and lower PMTs. The parser first separates these into two distinct traces: `ttrace_low` and `ttrace_up`.
2.  **Calibration**: Each trace is independently calibrated by dividing its FADC counts by a specific `fadc_per_vem` value (one for `_low`, one for `_up`). This converts the raw data into the physical unit of Vertical Equivalent Muons (VEM).
3.  **Averaging**: The final output depends on the `avg_traces` parameter:
    -   If `avg_traces=False`, the two calibrated traces (`ttrace_low` and `ttrace_up`) are kept separate and returned with the `_low` and `_up` suffixes.
    -   If `avg_traces=True`, the two calibrated traces are averaged together element-wise (`(ttrace_low + ttrace_up) / 2`) to produce a single, combined time trace.

---


