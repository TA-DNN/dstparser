# Comprehensive TA SD DST Parser Field Documentation

This document provides a complete reference for all fields in the TA Surface Detector DST (Data Summary Tape) format, including their C++ sources, data types, descriptions, and exact code references.

## Section 1: EVENT DATA (61 fields total)

### Monte Carlo Truth Parameters (Fields 0-11)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 0 | rusdmc_.parttype | scalar | int | CORSIKA particle type ID (e.g., 14=proton, 5626=iron) | dst2k-ta/inc/rusdmc_dst.h:33-34 |
| 1 | rusdmc_.energy | scalar | float | Total energy of primary particle [EeV] | dst2k-ta/inc/rusdmc_dst.h:37 |
| 2 | rusdmc_.theta | scalar | float | Zenith angle [rad] - convert to degrees in output | dst2k-ta/inc/rusdmc_dst.h:39 |
| 3 | rusdmc_.phi | scalar | float | Azimuthal angle (N of E) [rad] - convert to degrees in output | dst2k-ta/inc/rusdmc_dst.h:40 |
| 4 | rusdmc_.corexyz[0] | scalar | float | 3D core position X in CLF reference frame [cm] | dst2k-ta/inc/rusdmc_dst.h:41 |
| 5 | rusdmc_.corexyz[1] | scalar | float | 3D core position Y in CLF reference frame [cm] | dst2k-ta/inc/rusdmc_dst.h:41 |
| 6 | rusdmc_.corexyz[2] | scalar | float | 3D core position Z in CLF reference frame [cm] | dst2k-ta/inc/rusdmc_dst.h:41 |
| 7 | rusdraw_.yymmdd | scalar | int | Event date in YYMMDD format | dst2k-ta/inc/rusdraw_dst.h:52 |
| 8 | rusdraw_.hhmmss | scalar | int | Event time in HHMMSS format | dst2k-ta/inc/rusdraw_dst.h:53 |
| 9 | rufptn_.nstclust | scalar | int | Number of SDs in space-time cluster | dst2k-ta/inc/rufptn_dst.h:55 |
| 10 | rusdraw_.nofwf | scalar | int | Number of waveforms in event | dst2k-ta/inc/rusdraw_dst.h:57 |
| 11 | rusdraw_.usec | scalar | int | Event microsecond | dst2k-ta/inc/rusdraw_dst.h:54 |

### LDF Fit Results (Fields 12-21)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 12 | rufldf_.energy[0] | scalar | float | Energy using Rutgers formula [EeV] | dst2k-ta/inc/rufldf_dst.h:49 |
| 13 | rufldf_.sc[0] | scalar | float | LDF scaling factor [VEM m⁻²] | dst2k-ta/inc/rufldf_dst.h:42-43 |
| 14 | rufldf_.dsc[0] | scalar | float | Uncertainty in LDF scaling factor [VEM m⁻²] | dst2k-ta/inc/rufldf_dst.h:42-43 |
| 15 | rufldf_.chi2[0] | scalar | float | Chi-square of LDF fit | dst2k-ta/inc/rufldf_dst.h:51 |
| 16 | rufldf_.ndof[0] | scalar | int | Degrees of freedom for LDF fit | dst2k-ta/inc/rufldf_dst.h:80 |
| 17 | rufldf_.xcore[0] | scalar | float | Core X position + uncertainty [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:37-38 |
| 18 | rufldf_.dxcore[0] | scalar | float | Uncertainty in core X position [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:37-38 |
| 19 | rufldf_.ycore[0] | scalar | float | Core Y position [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:39-40 |
| 20 | rufldf_.dycore[0] | scalar | float | Uncertainty in core Y position [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:39-40 |
| 21 | rufldf_.s800[0] | scalar | float | S800 particle density at 800m from axis [VEM m⁻²] | dst2k-ta/inc/rufldf_dst.h:46-47 |

### Geometry Fit - Fixed Curvature (Fields 22-29)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 22 | rusdgeom_.theta[1] | scalar | float | Event zenith angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:87 |
| 23 | rusdgeom_.phi[1] | scalar | float | Event azimuthal angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:89 |
| 24 | rusdgeom_.dtheta[1] | scalar | float | Uncertainty in zenith angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:88 |
| 25 | rusdgeom_.dphi[1] | scalar | float | Uncertainty in azimuth angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:90 |
| 26 | rusdgeom_.chi2[1] | scalar | float | Chi-square of geometry fit | dst2k-ta/inc/rusdgeom_dst.h:91 |
| 27 | rusdgeom_.ndof[1] | scalar | int | Degrees of freedom for geometry fit | dst2k-ta/inc/rusdgeom_dst.h:138 |
| 28 | rusdgeom_.t0[1] | scalar | float | Core hit time when core hits CLF plane [1200m units] | dst2k-ta/inc/rusdgeom_dst.h:84-86 |
| 29 | rusdgeom_.dt0[1] | scalar | float | Uncertainty in core hit time [1200m units] | dst2k-ta/inc/rusdgeom_dst.h:86 |

### Border Distances (Fields 30-31)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 30 | rufldf_.bdist | scalar | float | Distance to array boundary (negative=outside) [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:58-59 |
| 31 | rufldf_.tdist | scalar | float | Distance to T-shape boundary for BR,LR,SK [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:65-68 |

### Geometry Fit - Free Curvature (Fields 32-41)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 32 | rusdgeom_.theta[2] | scalar | float | Final event zenith angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:87 |
| 33 | rusdgeom_.phi[2] | scalar | float | Final event azimuthal angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:89 |
| 34 | rusdgeom_.dtheta[2] | scalar | float | Final uncertainty in zenith angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:88 |
| 35 | rusdgeom_.dphi[2] | scalar | float | Final uncertainty in azimuth angle [degrees] | dst2k-ta/inc/rusdgeom_dst.h:90 |
| 36 | rusdgeom_.chi2[2] | scalar | float | Final chi-square of geometry fit | dst2k-ta/inc/rusdgeom_dst.h:91 |
| 37 | rusdgeom_.ndof[2] | scalar | int | Final degrees of freedom for geometry fit | dst2k-ta/inc/rusdgeom_dst.h:138 |
| 38 | rusdgeom_.t0[2] | scalar | float | Final core hit time when core hits CLF plane [1200m units] | dst2k-ta/inc/rusdgeom_dst.h:84-86 |
| 39 | rusdgeom_.dt0[2] | scalar | float | Final uncertainty in core hit time [1200m units] | dst2k-ta/inc/rusdgeom_dst.h:86 |
| 40 | rusdgeom_.a | scalar | float | Curvature parameter in Linsley's formula | dst2k-ta/inc/rusdgeom_dst.h:92 |
| 41 | rusdgeom_.da | scalar | float | Uncertainty in curvature parameter | dst2k-ta/inc/rusdgeom_dst.h:93 |

### LDF+Geometry Combined Fit (Fields 42-55)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 42 | rufldf_.energy[1] | scalar | float | Energy using Rutgers formula from LDF+geom combined fit [EeV] | dst2k-ta/inc/rufldf_dst.h:49 |
| 43 | rufldf_.sc[1] | scalar | float | LDF scaling factor from combined fit [VEM m⁻²] | dst2k-ta/inc/rufldf_dst.h:42-43 |
| 44 | rufldf_.dsc[1] | scalar | float | Uncertainty in combined LDF scaling factor [VEM m⁻²] | dst2k-ta/inc/rufldf_dst.h:42-43 |
| 45 | rufldf_.chi2[1] | scalar | float | Chi-square of LDF+geom combined fit | dst2k-ta/inc/rufldf_dst.h:51 |
| 46 | rufldf_.ndof[1] | scalar | int | Degrees of freedom for LDF+geom combined fit | dst2k-ta/inc/rufldf_dst.h:80 |
| 47 | rufldf_.xcore[1] | scalar | float | Core X from LDF+geom combined fit [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:37-38 |
| 48 | rufldf_.dxcore[1] | scalar | float | Uncertainty in combined core X [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:37-38 |
| 49 | rufldf_.ycore[1] | scalar | float | Core Y from LDF+geom combined fit [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:39-40 |
| 50 | rufldf_.dycore[1] | scalar | float | Uncertainty in combined core Y [counter units, 1200m] | dst2k-ta/inc/rufldf_dst.h:39-40 |
| 51 | rufldf_.s800[1] | scalar | float | S800 from LDF+geom combined fit [VEM m⁻²] | dst2k-ta/inc/rufldf_dst.h:46-47 |
| 52 | rufldf_.theta | scalar | float | Zenith angle from LDF+geom combined fit [degrees] | dst2k-ta/inc/rufldf_dst.h:53 |
| 53 | rufldf_.phi | scalar | float | Azimuth angle from LDF+geom combined fit [degrees] | dst2k-ta/inc/rufldf_dst.h:55 |
| 54 | rufldf_.dtheta | scalar | float | Uncertainty in combined zenith angle [degrees] | dst2k-ta/inc/rufldf_dst.h:54 |
| 55 | rufldf_.dphi | scalar | float | Uncertainty in combined azimuth angle [degrees] | dst2k-ta/inc/rufldf_dst.h:56 |

### Pattern Recognition Results (Fields 56-60)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 56 | rufptn_.nhits | scalar | int | Number of independent signals (hits) in trigger | dst2k-ta/inc/rufptn_dst.h:50 |
| 57 | rufptn_.nsclust | scalar | int | Number of hits in largest space cluster | dst2k-ta/inc/rufptn_dst.h:51 |
| 58 | rufptn_.nborder | scalar | int | SDs in space-time cluster on array border | dst2k-ta/inc/rufptn_dst.h:56-57 |
| 59 | rufptn_.qtot[0] | scalar | float | Total charge in event (sum over space-time cluster) | dst2k-ta/inc/rufptn_dst.h:95 |
| 60 | rufptn_.qtot[1] | scalar | float | Total charge in event (sum over space-time cluster) | dst2k-ta/inc/rufptn_dst.h:95 |

## Section 2: SD meta DATA (12 fields per hit)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 0 | rufptn_.xxyy[x] | (nhits,) | int | Position of the hit (XXYY format) | dst2k-ta/inc/rufptn_dst.h:65 |
| 1 | rufptn_.isgood[x] | (nhits,) | int | Hit quality flag (0-5, see header for details) | dst2k-ta/inc/rufptn_dst.h:58-65 |
| 2 | rufptn_.reltime[x][0] | (nhits,) | float | Hit time relative to earliest hit [counter sep. dist units] | dst2k-ta/inc/rufptn_dst.h:103 |
| 3 | rufptn_.reltime[x][1] | (nhits,) | float | Hit time relative to earliest hit [counter sep. dist units] | dst2k-ta/inc/rufptn_dst.h:103 |
| 4 | rufptn_.pulsa[x][0] | (nhits,) | float | Pulse area in VEM (pedestals subtracted) | dst2k-ta/inc/rufptn_dst.h:106 |
| 5 | rufptn_.pulsa[x][1] | (nhits,) | float | Pulse area in VEM (pedestals subtracted) | dst2k-ta/inc/rufptn_dst.h:106 |
| 6 | rufptn_.xyzclf[x][0] | (nhits,) | float | SD coordinates in CLF frame [1200m units] | dst2k-ta/inc/rufptn_dst.h:93 |
| 7 | rufptn_.xyzclf[x][1] | (nhits,) | float | SD coordinates in CLF frame [1200m units] | dst2k-ta/inc/rufptn_dst.h:93 |
| 8 | rufptn_.xyzclf[x][2] | (nhits,) | float | SD coordinates in CLF frame [1200m units] | dst2k-ta/inc/rufptn_dst.h:93 |
| 9 | rufptn_.vem[x][0] | (nhits,) | float | FADC counts / VEM from monitoring | dst2k-ta/inc/rufptn_dst.h:110 |
| 10 | rufptn_.vem[x][1] | (nhits,) | float | FADC counts / VEM from monitoring | dst2k-ta/inc/rufptn_dst.h:110 |
| 11 | rufptn_.nfold[x] | (nhits,) | int | Foldedness of hit (over how many 128 fadc windows signal extends) | dst2k-ta/inc/rufptn_dst.h:66 |

## Section 3: SD waveform DATA (259 fields per waveform)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 0 | rusdraw_.xxyy[x] | (nofwf,) | int | Detector that was hit (XXYY format) | dst2k-ta/inc/rusdraw_dst.h:60 |
| 1 | rusdraw_.clkcnt[x] | (nofwf,) | int | Clock count at waveform beginning | dst2k-ta/inc/rusdraw_dst.h:61 |
| 2 | rusdraw_.mclkcnt[x] | (nofwf,) | int | Max clock count for detector (~50E6) | dst2k-ta/inc/rusdraw_dst.h:62 |
| 3-130 | rusdraw_.fadc[x][0][i] | (nofwf, 128) | int | FADC trace for lower scintillator layer | dst2k-ta/inc/rusdraw_dst.h:65 |
| 131-258 | rusdraw_.fadc[x][1][i] | (nofwf, 128) | int | FADC trace for upper scintillator layer | dst2k-ta/inc/rusdraw_dst.h:65 |

## Section 4: badsdinfo DATA (2 fields per bad detector)

| Index | C++ Source | Shape | Type | Description | Reference |
|-------|------------|-------|------|-------------|-----------|
| 0 | bsdinfo_.xxyyout[x] | (nsdsout,) | int | SDs completely out (can't participate in readout) | dst2k-ta/inc/bsdinfo_dst.h:64 |
| 1 | bsdinfo_.bitfout[x] | (nsdsout,) | int | Bit flags of SDs considered completely out | dst2k-ta/inc/bsdinfo_dst.h:65 |

## Key Notes:

- **Units Verified from Python Parser**:
  - Counter units = 1200m detector spacing
  - Time in counter units = 4 μs per unit (converted to 4000 ns in Python)
  - Angles: radians throughout
  - Energy: eV for MC truth, EeV for reconstructed values
  - CLF = Central Laser Facility coordinate system
  - VEM = Vertical Equivalent Muon (calibration unit)

- **Scintillator Layers**: 
  - Lower layer (index [0]): Bottom scintillator in SD counter
  - Upper layer (index [1]): Top scintillator in SD counter

- **Critical Fields**: Fields 32-41 (free curvature geometry) are used in standard TA SD analyses

- **Data Flow**: C++ → text output → Python parser → processed data structures

- **Python Conversions**: 
  - Positions: multiplied by 1200 to convert to meters
  - Times: multiplied by 4000 to convert to nanoseconds
  - Energy values used as EeV for reconstructed quantities
