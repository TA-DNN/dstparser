# Technical Documentation: TA Surface Detector DST Format

## Overview

The Telescope Array (TA) Surface Detector (SD) DST (Data Summary Tape) format is a comprehensive data structure that contains Monte Carlo truth information, reconstructed physics parameters, and raw detector data for cosmic ray air shower events. This document provides detailed technical explanations of the data structure, field meanings, and inter-relationships between different sections.

## Data Structure Architecture

### 1. Four-Section Design

The TA SD DST format is organized into four distinct sections:

1. **Event Summary** (61 fields): Core physics parameters and reconstruction results
2. **SD Metadata** (12 fields per hit): Per-detector information for each triggered detector
3. **Waveform Data** (259 fields per waveform): Raw FADC traces and calibration data
4. **Bad Detector Info** (2 fields per bad detector): Status information for non-functioning detectors

### 2. Coordinate Systems and Units

#### CLF Coordinate System
The Central Laser Facility (CLF) coordinate system is the primary reference frame used throughout the TA SD analysis:

- **Origin**: CLF location in Utah
- **SD Array Origin Offset**: (-12.2435, -16.4406) in 1200m units relative to CLF origin
- **Reference**: `dst2k-ta/inc/rusdgeom_dst.h:40-41`, `dst2k-ta/inc/rufptn_dst.h:37-38`

```cpp
// SD origin with respect to CLF origin in CLF frame, in [1200m] units
#define RUSDGEOM_ORIGIN_X_CLF -12.2435
#define RUSDGEOM_ORIGIN_Y_CLF -16.4406
```

#### Unit Conversions
- **Distance**: Counter separation units = 1200m
- **Time**: Counter time units = 4μs, with conversion factor `RUSDGEOM_TIMDIST = 0.249827048333` for time-to-distance conversion
- **Reference**: `dst2k-ta/inc/rusdgeom_dst.h:47-49`

```cpp
/* For converting time in uS to distance in units of counter
   separation distance 1200m. Numerically, this is 
   c*(10^(-6)(S/uS))/1200m, where c is the speed of light in m/s */
#define RUSDGEOM_TIMDIST 0.249827048333
```

## Section 1: Event Summary (Fields 0-60)

### Monte Carlo Truth Information (Fields 0-6)

The first seven fields contain true Monte Carlo parameters from CORSIKA simulation:

#### Particle Type Classification
- **Field 0** (`rusdmc_.parttype`): CORSIKA particle code
  - **Common Values**: 14 (proton), 5626 (iron nucleus), 402 (helium)
  - **Reference**: `dst2k-ta/inc/rusdmc_dst.h:33-34`

#### Energy and Direction
- **Fields 1-3**: True energy (EeV), zenith angle (radians), azimuth angle (radians)
- **Important Note**: Angles are stored in radians but converted to degrees in analysis output
- **Reference**: `dst2k-ta/inc/rusdmc_dst.h:37-40`

#### Core Position
- **Fields 4-6**: 3D core position in CLF frame [cm]
- **Conversion**: To get position in SD array coordinates, use the offset definitions
- **Reference**: `dst2k-ta/inc/rusdmc_dst.h:41`

### Event Timing and Trigger Information (Fields 7-11)

#### Time Stamps
- **Fields 7-8**: Date (YYMMDD) and time (HHMMSS) formats
- **Field 11**: Microsecond precision timing
- **Reference**: `dst2k-ta/inc/rusdraw_dst.h:52-54`

#### Trigger Statistics
- **Field 9** (`rufptn_.nstclust`): Number of SDs in space-time cluster
  - **Critical for Quality**: Events typically require ≥4 SDs for analysis
  - **Reference**: `dst2k-ta/inc/rufptn_dst.h:55`
- **Field 10** (`rusdraw_.nofwf`): Total number of waveforms
  - **Reference**: `dst2k-ta/inc/rusdraw_dst.h:57`

### Reconstruction Results: Three-Tier Fit Hierarchy

The TA SD reconstruction employs a sophisticated three-tier fitting approach:

#### Tier 1: LDF-Only Fit (Fields 12-21)
Uses only the Lateral Distribution Function without timing information:

- **Energy** (Field 12): Uses Rutgers energy formula
- **Core Position** (Fields 17-20): X,Y coordinates and uncertainties in 1200m units
- **S800** (Field 21): Signal density at 800m from shower axis [VEM/m²]
- **Quality**: Chi-square (Field 15) and degrees of freedom (Field 16)
- **Reference**: `dst2k-ta/inc/rufldf_dst.h:37-51`

#### Tier 2: Geometry-Only Fit (Fields 22-29, 32-41)
Uses timing information to determine shower direction:

##### Fixed Curvature Fit (Fields 22-29)
- **Method**: Modified Linsley fit with fixed curvature parameter
- **Outputs**: Direction (θ,φ), timing (t₀), and uncertainties
- **Index [1]**: Indicates this is the second fit type
- **Reference**: `dst2k-ta/inc/rusdgeom_dst.h:79-91`

##### Free Curvature Fit (Fields 32-41)
- **Method**: Allows curvature parameter 'a' to vary freely
- **Index [2]**: Final geometry fit results used in standard analysis
- **Curvature Parameters** (Fields 40-41): Linsley formula parameters `a` and `da`
- **Reference**: `dst2k-ta/inc/rusdgeom_dst.h:92-93`

```cpp
real8 a;        /* Curvature parameter in Linsley's formula */
real8 da;
```

#### Tier 3: Combined LDF+Geometry Fit (Fields 42-55)
Simultaneous fit of energy, direction, and core position:

- **Index [1]**: Combined fit results for LDF parameters
- **Final Direction** (Fields 52-55): Zenith/azimuth from joint fit
- **Best Energy** (Field 42): Most accurate energy reconstruction
- **Reference**: `dst2k-ta/inc/rufldf_dst.h:53-56`

### Array Geometry and Efficiency (Fields 30-31)

#### Boundary Distance Calculations
- **Field 30** (`rufldf_.bdist`): Distance to array boundary
  - **Negative values**: Core outside detector array
  - **Purpose**: Event selection and efficiency calculations
- **Field 31** (`rufldf_.tdist`): Distance to T-shaped boundary
  - **Sub-arrays**: BR (Black Rock), LR (Long Ridge), SK (Middle Drum)
  - **Purpose**: Determining which sub-array detected the event
- **Reference**: `dst2k-ta/inc/rufldf_dst.h:58-68`

### Pattern Recognition Summary (Fields 56-60)

#### Cluster Statistics
- **Field 56** (`rufptn_.nhits`): Total independent signals in trigger
- **Field 57** (`rufptn_.nsclust`): Largest space cluster size
- **Field 58** (`rufptn_.nborder`): Border detectors in space-time cluster
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:50-57`

#### Charge Measurements
- **Fields 59-60** (`rufptn_.qtot[0/1]`): Total charge in lower/upper scintillator layers
- **Units**: VEM (Vertical Equivalent Muon)
- **Purpose**: Energy estimation and event characterization
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:95`

## Section 2: SD Metadata (12 Fields per Hit)

This section contains detailed information for each detector that participated in the event trigger.

### Detector Identification and Quality
- **Field 0** (`rufptn_.xxyy[x]`): Detector position ID in XXYY format
- **Field 1** (`rufptn_.isgood[x]`): Quality flag with specific meanings:
  - **0**: Counter not working properly
  - **1**: Hit not part of any cluster
  - **2**: Part of space cluster
  - **3**: Passed rough time pattern recognition
  - **4**: Part of the event (highest quality)
  - **5**: Saturated counter
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:58-65`

### Timing Information
- **Fields 2-3**: Relative timing for lower/upper scintillator layers
- **Units**: Counter separation distance units
- **Conversion**: Multiply by `RUFPTN_TIMDIST` and add `tearliest` for absolute time
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:103`

### Signal Measurements
- **Fields 4-5**: Pulse area in VEM (pedestals subtracted)
- **Fields 9-10**: VEM calibration factors (FADC counts per VEM)
- **Purpose**: Accurate signal measurement and calibration tracking
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:106-110`

### Spatial Information
- **Fields 6-8**: 3D detector coordinates in CLF frame
- **Units**: 1200m units with respect to CLF origin
- **Note**: SD origin offset already subtracted
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:93`

### Signal Characteristics
- **Field 11** (`rufptn_.nfold[x]`): Signal foldedness
- **Meaning**: Number of 128-bin FADC windows the signal extends over
- **Purpose**: Indicates signal duration and potential saturation
- **Reference**: `dst2k-ta/inc/rufptn_dst.h:66`

## Section 3: Waveform Data (259 Fields per Waveform)

### Header Information (Fields 0-2)
- **Field 0**: Detector ID (XXYY format)
- **Field 1**: Clock count at waveform start
- **Field 2**: Maximum clock count (~50M, for timing reference)
- **Reference**: `dst2k-ta/inc/rusdraw_dst.h:60-62`

### FADC Traces (Fields 3-258)
- **Fields 3-130**: Lower scintillator layer FADC trace (128 bins)
- **Fields 131-258**: Upper scintillator layer FADC trace (128 bins)
- **Units**: Raw FADC counts
- **Sampling**: Each bin represents one time sample of the analog signal
- **Reference**: `dst2k-ta/inc/rusdraw_dst.h:65`

## Section 4: Bad Detector Information

### Detector Status Tracking
- **Field 0** (`bsdinfo_.xxyyout[x]`): Completely non-functional detector IDs
- **Field 1** (`bsdinfo_.bitfout[x]`): 16-bit status flags
- **Purpose**: Track detector health and exclude bad detectors from analysis
- **Reference**: `dst2k-ta/inc/bsdinfo_dst.h:64-65`

#### Status Bit Meanings
The 16-bit status flags encode specific failure modes:
- **Bits 0-5**: ICRR calibration issues
- **Bits 6-10**: Rutgers Monte Carlo calibration issues  
- **Bits 11-15**: Rutgers reconstruction calibration issues
- **Reference**: `dst2k-ta/inc/bsdinfo_dst.h:46-60`

## Data Flow and Processing Chain

### 1. Event Reconstruction Pipeline
```
Raw Data → Pattern Recognition → Geometry Fit → LDF Fit → Combined Fit
   ↓              ↓                  ↓           ↓         ↓
rusdraw_      rufptn_           rusdgeom_   rufldf_   Final Results
```

### 2. Quality Hierarchy
Events are processed through increasingly stringent quality cuts:
1. **Trigger Level**: Basic detector activation
2. **Pattern Recognition**: Space-time clustering (`rufptn_.nstclust ≥ 4`)
3. **Geometry Fit**: Successful timing fit (`rusdgeom_.chi2`, `rusdgeom_.ndof`)
4. **Physics Analysis**: Combined fit quality and boundary cuts

### 3. Standard Analysis Selection
Most TA SD physics analyses use:
- **Geometry**: Final fit results (Fields 32-41, index [2])
- **Energy**: Combined fit energy (Field 42)
- **Direction**: Combined fit angles (Fields 52-55)
- **Quality Cuts**: Boundary distance, fit quality, cluster size

## Key Relationships Between Sections

### 1. Cross-Section Validation
- Event summary hit count (`rufptn_.nhits`) equals SD metadata array size
- Waveform count (`rusdraw_.nofwf`) equals waveform data array size
- Bad detector count provides context for missing detectors

### 2. Coordinate Consistency
- All spatial measurements use consistent CLF-based coordinate system
- Timing measurements reference the same earliest hit time
- Energy measurements use consistent VEM calibration

### 3. Quality Propagation
- Bad detector information affects pattern recognition
- Pattern recognition quality affects geometry fits
- Geometry fit quality affects final energy reconstruction

## Implementation Notes

### 1. Numerical Precision
- Core positions: 64-bit floating point for precision
- Timing: Sub-microsecond precision maintained throughout
- Energy: EeV scale with sufficient precision for physics analysis

### 2. Error Propagation
- All fit parameters include uncertainty estimates
- Systematic uncertainties tracked through calibration flags
- Statistical uncertainties from fit covariance matrices

### 3. Coordinate Transformations
Physical coordinate conversions are handled consistently:
- CLF coordinates → SD array coordinates: Apply origin offset
- Time units → Physical time: Apply `TIMDIST` conversion factor
- Counter units → Meters: Multiply by 1200

This technical documentation provides the foundation for understanding and working with TA SD DST data, ensuring proper interpretation of the complex multi-tier reconstruction results and their associated uncertainties.

## Critical Analysis Note: Zenith Angle Bias Correction

### The "+0.5 Degree" Systematic Correction

Throughout the TA SD analysis pipeline, a systematic correction of **+0.5 degrees** is applied to all reconstructed zenith angles. This correction appears consistently across multiple analysis codes and represents a systematic bias correction.

#### Evidence in Source Code

**1. Analysis Pipeline Applications:**
- **rusdhist**: `theta = rusdgeom_.theta[2]+0.5;` (`rusdhist/src/rusdhist_class.cpp:368`)
- **sdascii**: `theta = (rusdgeom_.theta[2] + 0.5) * DegToRad();` (`sdascii/src/sdascii.cpp:259`)
- **nuf.i12f**: `if (rusdgeom_.theta[2] + 0.5 > 55.0)` (`nuf.i12f/src/iterate.cpp:838`)
- **sd4radar**: `theta = (rusdgeom_.theta[2] + 0.5);` (`misc/sd4radar.cpp:114`)
- **dst2rt_sd**: `ani_theta = (theta[2]+0.5) * DegToRad();` (`dst2rt_sd/src/dst2rt_sd.cpp:949`)

**2. Python DST Parser Implementation:**
The Python parser explicitly applies this correction with documentation:
```python
# "+0.5" is a correction for zenith angle.
np.sin(np.deg2rad(event_list[32] + 0.5))  # Fields 32, 22, 52
```
**Reference**: `dstparser/src/dstparser/dst_adapter.py:139-172`

**3. Resolution Studies:**
The correction is explicitly accounted for in MC resolution histograms:
```cpp
// Histogram title: "#theta_{Rec} + 0.5 - #theta_{Thr}, [Degree]"
hThetaRes[icut]->Fill(rusdgeom_.theta[2] + 0.5 - rusdmc_.theta * RadToDeg(),w);
```
**Reference**: `rusdhist/src/rusdhist_class.cpp:162,557`

#### Nature of the Correction

**1. Systematic Bias:** 
The comment in `nuf.i12f/src/iterate.cpp:837` states: *"Zenith angles for p and nuclei are taken from pass2 reconstruction and corrected with p mean bias."* This indicates the +0.5° correction addresses a systematic bias observed in proton Monte Carlo studies.

**2. Universal Application:**
The correction is applied to **all three geometry fit results**:
- Fixed curvature fit (Field 22: `rusdgeom_.theta[1] + 0.5`)
- Free curvature fit (Field 32: `rusdgeom_.theta[2] + 0.5`) 
- Combined LDF+geometry fit (Field 52: `rufldf_.theta + 0.5`)

**3. Analysis-Wide Consistency:**
Every major TA SD analysis tool applies this correction, ensuring consistent zenith angle treatment across the entire analysis chain.

#### Physical Interpretation

**Likely Causes:**
1. **Detector Response Asymmetries**: Systematic effects in timing or signal response
2. **Reconstruction Algorithm Bias**: Inherent bias in the Modified Linsley timing fit
3. **Atmospheric Effects**: Systematic deviations in shower development modeling
4. **Array Geometry Effects**: Systematic effects from detector array configuration

**Impact on Physics:**
- **Energy Spectrum**: Since energy reconstruction depends on zenith angle through atmospheric corrections
- **Anisotropy Studies**: Direct impact on directional reconstruction accuracy
- **Composition Analysis**: Affects zenith-angle-dependent selection and systematic uncertainties

#### Implementation Consistency

**Storage vs. Analysis:**
- **Raw Values**: DST fields store the uncorrected reconstruction results
- **Analysis Values**: All physics analyses use the +0.5° corrected values
- **Python Parser**: Automatically applies correction for shower axis calculations

**Quality Assurance:**
The systematic application across all analysis codes and explicit documentation in the Python parser demonstrates this is an established, validated correction rather than an ad-hoc adjustment.

**References**: 
- Bias correction implementation: Multiple source files listed above
- Python documentation: `dstparser/src/dstparser/dst_adapter.py:139,152,165`
- Resolution histogram definition: `rusdhist/src/rusdhist_class.cpp:162`

## Detailed Fitting Procedures and Error Propagation

### 1. Fitting Framework Architecture

The TA SD reconstruction employs ROOT's TMinuit package for all parameter fitting, implementing sophisticated χ² minimization with full error propagation. The fitting hierarchy involves three distinct levels with specific parameter counts and degrees of freedom calculations.

#### Degrees of Freedom Calculations

**LDF-Only Fit (ndof = n - 3):**
- **Parameters**: 3 fit parameters (core X, core Y, scale factor)
- **Formula**: `ndof = nactpts - NFPARS` where `NFPARS = 3`
- **Reference**: `rufldf/inc/p2ldffitter.h:15`, `rufldf/src/p2ldffitter.cpp:291`

```cpp
#define NFPARS          3  // max. number of fit parameters
// ...
ndof = nactpts - NFPARS;
```

**Geometry Fit - Fixed Curvature (ndof = n - 5):**
- **Parameters**: 5 fit parameters (θ, φ, core X, core Y, t₀)
- **Formula**: `ndof = nfitpts - NFPARS_LINSLEY` where `NFPARS_LINSLEY = 5`
- **Reference**: `rufptn/inc/p1geomfitter.h:15`, `rufptn/src/p1geomfitter.cpp:379`

```cpp
#define NFPARS_LINSLEY  5
// ...
ndof = nfitpts - NFPARS_LINSLEY;
```

**Geometry Fit - Free Curvature (ndof = n - 6):**
- **Parameters**: 6 fit parameters (θ, φ, core X, core Y, t₀, curvature 'a')
- **Formula**: `ndof = nfitpts - NFPARS_LINSLEY1` where `NFPARS_LINSLEY1 = 6`
- **Reference**: `rufptn/inc/p1geomfitter.h:16`, `rufptn/src/p1geomfitter.cpp:477`

```cpp
#define NFPARS_LINSLEY1 6
// ...
ndof = nfitpts-nfpars; // # of d.o.f
```

**Combined LDF+Geometry Fit (ndof = 2n - 6):**
- **Parameters**: 6 parameters fit simultaneously to both timing and signal data
- **Data Points**: Both signal amplitudes AND timing measurements (2n total constraints)
- **Formula**: `ndof = nfitpts-nfpars` for combined data set
- **Reference**: `rufldf/src/p2geomfitter.cpp:575`

### 2. Statistical Error Propagation via Minuit

#### Parameter Uncertainty Extraction
All fit parameter uncertainties are derived from Minuit's covariance matrix via the MIGRAD algorithm:

```cpp
// Extract parameters and their uncertainties from Minuit fit
gMinuit->GetParameter(0, theta, dtheta);   // θ ± δθ
gMinuit->GetParameter(1, phi, dphi);       // φ ± δφ
gMinuit->GetParameter(2, R[0], dR[0]);     // core X ± δX
gMinuit->GetParameter(3, R[1], dR[1]);     // core Y ± δY
gMinuit->GetParameter(4, T0, dT0);         // t₀ ± δt₀
if (nfpars == 6)
  gMinuit->GetParameter(5, a, da);         // curvature ± δa
```
**Reference**: `rufptn/src/p1geomfitter.cpp:546-553`, `rufldf/src/p2geomfitter.cpp:544-551`

#### Chi-Square Calculation Methodology
The χ² calculation incorporates multiple error sources in the denominator:

**For Geometry Fits:**
```cpp
// Combined timing and signal uncertainties
denom = sqrt(lts*lts + dt[i]*dt[i]);  // Linsley timing model + measurement error
delta = t[i] - tvsx_plane(X[i][0], X[i][1], X[i][2], par);
chisq += delta*delta / denom;
```
**Reference**: `rufldf/src/p2geomfitter.cpp:160-165`

**For LDF Fits:**
```cpp
// Signal density error propagation
drho[NGSDS]; // error on charge density due to SD response fluctuation (vem/m^2)
```
**Reference**: `rufldf/src/p2ldffitter.cpp:13`

### 3. Systematic Error Sources and Calibration Tracking

#### VEM Calibration Error Propagation
The Vertical Equivalent Muon (VEM) calibration uncertainties are rigorously tracked through the analysis chain:

```cpp
// VEM error calculation including pedestal and MIP fit uncertainties
rufptn_.vemerr[j][k] = ((real8)(rusdraw_.rhpchmip[i][k] 
                       - rusdraw_.lhpchmip[i][k])) / 2.33 * Cos(DegToRad()*MEAN_MU_THETA);
rufptn_.vemerr[j][k] = sqrt(rufptn_.vemerr[j][k] * rufptn_.vemerr[j][k] 
                       + (real8)NMIPINTCH * (real8)NMIPINTCH 
                       * rufptn_.pederr[j][k] * rufptn_.pederr[j][k]);
```
**Reference**: `rufptn/src/rufptnAnalysis.cpp:534-538`

#### Calibration Quality Flags
Systematic uncertainties are tracked through calibration bit flags in the `bsdinfo` structure:
- **Bits 0-5**: ICRR calibration issues (MeV2pe, MeV2cnt, pedestals, saturation)
- **Bits 6-10**: Rutgers MC calibration issues (MIP values, pedestal fits, χ² quality)
- **Bits 11-15**: Rutgers reconstruction calibration issues (runtime calibration quality)
**Reference**: `dst2k-ta/inc/bsdinfo_dst.h:46-60`

### 4. Multi-Level Error Validation

#### Fit Quality Assessment
Multiple quality metrics ensure robust error estimates:

**Chi-Square per Degree of Freedom:**
```cpp
if (rusdgeom_.ndof[1] > 0)
  gfchi2pdof = rusdgeom_.chi2[1] / (double)rusdgeom_.ndof[1];
```
**Reference**: `sditerator/src/sditerator_cppanalysis_add_standard_recon_v2.cpp:121-123`

**Calibration Validation:**
```cpp
if (rusdraw_.mftchi2[i][k]/(real8)rusdraw_.mftndof[i][k] > BITMIP_CHI2PDOF_CUT)
  // Flag detector as having poor calibration fit quality
```
**Reference**: `rufptn/src/rufptnAnalysis.cpp:522`

#### Error Propagation Through Analysis Chain

**1. Raw Data Level:**
- FADC noise and pedestals: `rufptn_.pederr[j][k]`
- MIP calibration uncertainties: `rufptn_.vemerr[j][k]`
- Timing uncertainties: `rufptn_.timeerr[i][k]`

**2. Pattern Recognition Level:**
- Signal measurement errors: `rufptn_.pulsaerr[i][k]`
- Position uncertainties from detector survey

**3. Reconstruction Level:**
- Fit parameter covariances from Minuit
- Model systematic uncertainties (Linsley formula, atmospheric models)
- Cross-correlation between parameters

**References**: 
- VEM error calculation: `rufptn/src/rufptnAnalysis.cpp:534-538`
- Timing error propagation: `rufptn/src/p1geomfitter.cpp:364`
- Minuit parameter extraction: `rufptn/src/p1geomfitter.cpp:546-553`

### 5. Practical Implementation Notes

#### Error Matrix Properties
- **Positive Definite**: Minuit ensures mathematically valid covariance matrices
- **Parameter Correlations**: Off-diagonal elements capture θ-φ, position-time correlations  
- **Physical Constraints**: Parameter limits prevent unphysical solutions

#### Uncertainty Interpretations
- **Statistical Errors**: From fit covariance matrix (68% confidence level)
- **Systematic Errors**: From calibration variations and model dependencies
- **Combined Uncertainties**: Quadrature addition of statistical and systematic components

This multi-level error propagation framework ensures that parameter uncertainties accurately reflect both statistical measurement limitations and systematic calibration effects, providing robust uncertainty estimates for physics analyses.

# Mathematical Foundations of TA SD Reconstruction

### Chapter 1: Theoretical Framework

#### 1.1 The Cosmic Ray Air Shower Problem

When a high-energy cosmic ray enters Earth's atmosphere, it initiates an extensive air shower (EAS) through hadronic and electromagnetic cascading processes. The Surface Detector (SD) array measures the remnants of this shower at ground level, providing discrete sampling points of the shower's lateral distribution function (LDF) and timing structure.

**The Inverse Problem:**
Given $N$ detector measurements $\{S_1, S_2, ..., S_N\}$ at positions $\{\vec{r}_1, \vec{r}_2, ..., \vec{r}_N\}$ and times $\{t_1, t_2, ..., t_N\}$, determine:
- Primary energy $E_0$
- Arrival direction $(\theta, \phi)$ 
- Core position $(X_0, Y_0)$
- Shower development parameters

#### 1.2 Physical Models

**Lateral Distribution Function (LDF):**
The signal density $\rho(r)$ as a function of distance $r$ from the shower core follows the NKG (Nishimura-Kamata-Greisen) function:

$$\rho(r) = S \times \left(\frac{r}{r_0}\right)^{-\alpha} \times \left(1 + \frac{r}{r_0}\right)^{-\beta}$$

Where:
- $S$ = scale parameter (related to shower size)
- $r_0$ = Molière radius ($\approx$ 91.6 m for TA)
- $\alpha, \beta$ = age-dependent shape parameters
- **Reference**: Empirical parameterization used in `rufldf` fitting

**Timing Model - Modified Linsley Formula:**
The arrival time at distance $r$ from the core follows:

$$t(r,\theta) = t_0 + \frac{r^2}{2Rc} \sec(\theta) + a \times \left(\frac{r}{30\text{m}}\right)^{1.5} \times \rho^{-0.5}$$

Where:
- $t_0$ = core passage time
- $R$ = shower front radius of curvature  
- $\theta$ = zenith angle
- $a$ = curvature parameter (related to shower development)
- $\rho$ = atmospheric density
- **Reference**: `rufptn/src/p1geomfitter.cpp:55`

```cpp
(*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
```

### Chapter 2: Maximum Likelihood Estimation Framework

#### 2.1 Statistical Foundation

**Likelihood Function:**
For N independent measurements with Gaussian errors, the likelihood function is:

$$L(\theta) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma_i^2}} \times \exp\left[-\frac{(y_i - f(x_i;\theta))^2}{2\sigma_i^2}\right]$$

**Log-Likelihood and Chi-Square:**
Taking the negative log-likelihood and dropping constants:

$$-2\ln(L) = \chi^2 = \sum_{i=1}^{N} \frac{(y_i - f(x_i;\theta))^2}{\sigma_i^2}$$

This transforms the problem into χ² minimization, which is computationally more stable.

#### 2.2 Error Model Construction

**Measurement Error Components:**

*Signal Measurements:*
$$\sigma^2_{s_i} = \sigma^2_{stat} + \sigma^2_{calib} + \sigma^2_{syst}$$

Where:
- σ²ₛₜₐₜ = Poisson counting statistics + electronic noise
- σ²ᶜᵃˡⁱᵇ = VEM calibration uncertainty  
- σ²ˢʸˢᵗ = Systematic uncertainties (temperature, pressure, aging)

*Timing Measurements:*
$$\sigma^2_{t_i} = \sigma^2_{instr} + \sigma^2_{model}$$

Where:
- σ²ᵢₙₛₜᵣ = Instrumental timing resolution
- σ²ₘₒdₑₗ = Model uncertainty from Linsley formula

**Implementation in TA Code:**
```cpp
// Combined timing and signal uncertainties
denom = sqrt(lts*lts + dt[i]*dt[i]);  // σ²ₘₒdₑₗ + σ²ᵢₙₛₜᵣ
delta = t[i] - tvsx_plane(X[i][0], X[i][1], X[i][2], par);
chisq += delta*delta / denom;
```
**Reference**: `rufldf/src/p2geomfitter.cpp:160-165`

### Chapter 3: Numerical Optimization Theory

#### 3.1 The MIGRAD Algorithm

TA uses ROOT's TMinuit implementation of the MIGRAD algorithm, which is based on the Davidon-Fletcher-Powell (DFP) variable metric method.

**Algorithm Steps:**
1. **Gradient Calculation**: $\nabla\chi^2 = \frac{\partial\chi^2}{\partial\theta_i}$ computed numerically
2. **Hessian Approximation**: $H \approx \nabla^2\chi^2$ built iteratively  
3. **Step Direction**: $\Delta\theta = -H^{-1}\nabla\chi^2$
4. **Line Search**: Optimize step size along $\Delta\theta$ direction
5. **Convergence Check**: $|\nabla\chi^2| <$ tolerance

**Matrix Update Formula (DFP):**

$$H^{(k+1)} = H^{(k)} + \frac{\Delta\theta\Delta\theta^T}{\Delta\theta^T\delta g} - \frac{H^{(k)}\delta g\delta g^T H^{(k)}}{\delta g^T H^{(k)}\delta g}$$

Where $\delta g = \nabla\chi^2_{(k+1)} - \nabla\chi^2_{(k)}$

#### 3.2 Error Matrix Calculation

**Covariance Matrix from Hessian:**
At the minimum, the inverse Hessian provides the covariance matrix:

$$\text{Cov}(\theta_i,\theta_j) = [H^{-1}]_{ij} = \left[\frac{\partial^2\chi^2}{\partial\theta_i\partial\theta_j}\right]^{-1}$$

**Parameter Uncertainties:**
$$\sigma(\theta_i) = \sqrt{\text{Cov}(\theta_i,\theta_i)}$$

**Implementation:**
```cpp
gMinuit->GetParameter(0, theta, dtheta);   // dtheta = √[Cov(θ,θ)]
```
**Reference**: `rufptn/src/p1geomfitter.cpp:546`

### Chapter 4: Three-Tier Fitting Hierarchy

#### 4.1 LDF-Only Fit (Tier 1)

**Parameter Vector:** θ = [X₀, Y₀, S]
**Model Function:** ρ(r) = S × f(r; fixed geometry)
**Chi-Square:**
$$\chi^2 = \sum_{i=1}^{N} \frac{(S_i - S \times f(r_i))^2}{\sigma^2_{s_i}}$$

**Degrees of Freedom:** ν = N - 3
- N = number of triggered detectors
- 3 = parameters (X₀, Y₀, S)

**Physical Interpretation:**
- Assumes shower axis from external measurement (FD hybrid)
- Optimizes energy scale and core position only
- Fast convergence, robust for triggered detector counts N ≥ 4

#### 4.2 Geometry-Only Fit (Tier 2)

**Fixed Curvature Sub-tier:**
**Parameter Vector:** θ = [θ, φ, X₀, Y₀, t₀]
**Model Function:** t(r,θ,φ) = t₀ + f(r,θ; a=fixed)
**Chi-Square:**
$$\chi^2 = \sum_{i=1}^{N} \frac{(t_i - t_0 - f(r_i,\theta))^2}{\sigma^2_{t_i}}$$

**Degrees of Freedom:** ν = N - 5

**Free Curvature Sub-tier:**
**Parameter Vector:** θ = [θ, φ, X₀, Y₀, t₀, a]
**Degrees of Freedom:** ν = N - 6

**Physical Interpretation:**
- Uses timing information to reconstruct shower direction
- Fixed curvature: assumes average shower development
- Free curvature: allows event-by-event shower variations

#### 4.3 Combined LDF+Geometry Fit (Tier 3)

**Parameter Vector:** θ = [θ, φ, X₀, Y₀, t₀, S]
**Simultaneous Chi-Square:**
$$\chi^2 = \sum_{i=1}^{N} \left[\frac{(S_i - S \times f(r_i,\theta))^2}{\sigma^2_{\rho_i}} + \frac{(t_i - t_{predicted})^2}{\sigma^2_{t_i}} \right] + \text{constraint term}$$

Where the timing model includes both geometry and Linsley timing corrections:
$$t_{predicted} = t_0 + t_{geometry}(r_i, \theta, \phi) + t_{Linsley}(\rho_i, r_i, \theta)$$

**Reference**: `rufldf/src/p2gldffitter.cpp:182-240`

# Comprehensive LDF Formulas and Parameter Definitions

## The Lateral Distribution Function (LDF) - Exact Implementation

### Primary LDF Formula
The TA SD analysis uses a modified NKG (Nishimura-Kamata-Greisen) function to describe the lateral distribution of particles in cosmic ray air showers:

$$\rho(r, \theta) = S \times \left(\frac{r}{r_0}\right)^{-\alpha} \times \left(1 + \frac{r}{r_0}\right)^{-(\eta-\alpha)} \times \left(1 + \frac{r^2}{r_{sc}^2}\right)^{-\beta}$$

**Reference**: `inc/ldffun.h:14-31` and `rufldf/src/p2gldffitter.cpp:118-133`

### Parameter Definitions and Values

| Parameter | Symbol | Value | Units | Physical Meaning | Reference |
|-----------|--------|-------|-------|------------------|-----------|
| **Molière Radius** | $r_0$ | 91.6 | m | Characteristic scattering length in air | `inc/ldffun.h:19` |
| **Constant Slope** | $\alpha$ | 1.2 | dimensionless | Inner shower profile steepness | `inc/ldffun.h:20` |
| **Quadratic Term** | $\beta$ | 0.6 | dimensionless | Far-field profile steepness | `inc/ldffun.h:21` |
| **Zenith Dependence** | $\eta$ | $3.97 - 1.79(\sec\theta - 1)$ | dimensionless | Zenith-angle-dependent aging parameter | `inc/ldffun.h:22` |
| **Scale Radius** | $r_{sc}$ | 1000.0 | m | Scaling factor for quadratic suppression | `inc/ldffun.h:23` |

### Code Implementation
```cpp
// From inc/ldffun.h:14-31
static Double_t ldffun(Double_t r, Double_t theta)
{
    Double_t r0    =  91.6;    // Molière radius [m]
    Double_t alpha =  1.2;     // Constant slope parameter
    Double_t beta  =  0.6;     // Another constant slope parameter  
    Double_t eta   =  3.97-1.79*(1.0/Cos(DegToRad()*theta)-1.0); // Zenith angle dependent slope
    Double_t rsc   = 1000.0;   // Scaling factor for r in quadratic term [m]
    
    return  Power(r/r0, -alpha) * 
            Power((1.0+r/r0), -(eta-alpha)) * 
            Power((1.0+ r*r/rsc/rsc), -beta);
}
```

## Fitting Parameter Definitions

### LDF-Only Fit Parameters (3 parameters)
Used in `p2ldffitter_class` with parameter vector **θ = [X₀, Y₀, S]**:

| Index | Parameter | Symbol | Units | Description | Reference |
|-------|-----------|--------|-------|-------------|-----------|
| `par[0]` | Core X | $X_0$ | counter units (1200m) | X position of shower core | `rufldf/src/p2ldffitter.cpp:38` |
| `par[1]` | Core Y | $Y_0$ | counter units (1200m) | Y position of shower core | `rufldf/src/p2ldffitter.cpp:39` |
| `par[2]` | Scale Factor | $S$ | VEM/m² | LDF normalization constant | `rufldf/src/p2ldffitter.cpp:40` |

### Combined LDF+Geometry Fit Parameters (6 parameters)  
Used in `p2gldffitter_class` with parameter vector **θ = [θ, φ, X₀, Y₀, t₀, S]**:

| Index | Parameter | Symbol | Units | Description | Reference |
|-------|-----------|--------|-------|-------------|-----------|
| `par[0]` | Zenith Angle | $\theta$ | degrees | Shower zenith angle | `rufldf/src/p2gldffitter.cpp:159` |
| `par[1]` | Azimuth Angle | $\phi$ | degrees | Shower azimuthal angle | `rufldf/src/p2gldffitter.cpp:160` |
| `par[2]` | Core X | $X_0$ | counter units (1200m) | X position of shower core | `rufldf/src/p2gldffitter.cpp:161` |
| `par[3]` | Core Y | $Y_0$ | counter units (1200m) | Y position of shower core | `rufldf/src/p2gldffitter.cpp:162` |
| `par[4]` | Core Time | $t_0$ | counter units (4μs) | Time of core passage | `rufldf/src/p2gldffitter.cpp:163` |
| `par[5]` | Scale Factor | $S$ | VEM/m² | LDF normalization constant | `rufldf/src/p2gldffitter.cpp:187` |

## Chi-Square Formulations

### LDF-Only Chi-Square
$$\chi^2_{LDF} = \sum_{i=1}^{N} \frac{(\rho_i - S \times \text{LDF}(r_i, \theta))^2}{\sigma^2_{\rho_i}} + \frac{(X_0 - X_{COG})^2 + (Y_0 - Y_{COG})^2}{\sigma^2_{constraint}}$$

Where:
- $\rho_i$ = observed charge density at detector $i$ [VEM/m²]
- $r_i$ = distance from shower axis to detector $i$ [m]
- $\sigma_{\rho_i}$ = error on charge density measurement [VEM/m²]
- $X_{COG}, Y_{COG}$ = center-of-gravity position of triggered detectors
- $\sigma_{constraint} = 0.15$ counter units (constraint on core position)

**Reference**: `rufldf/src/p2ldffitter.cpp:44-80`, `rufldf/inc/p2ldffitter.h:17`

### Combined LDF+Geometry Chi-Square
$$\chi^2_{combined} = \sum_{i=1}^{N} \left[ \frac{(\rho_i - S \times \text{LDF}(r_i, \theta))^2}{\sigma^2_{\rho_i}} + \frac{(t_i - t_{predicted})^2}{\sigma^2_{t_i}} \right] + \text{constraint term}$$

Where the timing model includes both geometry and Linsley timing corrections:
$$t_{predicted} = t_0 + t_{geometry}(r_i, \theta, \phi) + t_{Linsley}(\rho_i, r_i, \theta)$$

**Reference**: `rufldf/src/p2gldffitter.cpp:182-240`

## Energy Calibration and S-Parameters

### Standard Reference Distances
The LDF is evaluated at specific distances to provide standardized energy estimators:

#### S600 (600m signal density)
$$S_{600} = S \times \text{LDF}(600\text{m}, \theta)$$

#### S800 (800m signal density)  
$$S_{800} = S \times \text{LDF}(800\text{m}, \theta)$$

**Reference**: `rufldf/src/p2ldffitter.cpp:379,385`

### Atmospheric Attenuation Correction
The observed signal is corrected for atmospheric attenuation using the AGASA formula:

$$S_{600,0} = \frac{S_{600}}{\text{Attenuation}(\theta)}$$

$$\text{Attenuation}(\theta) = \exp\left[-\frac{X_0}{L_1}(\sec\theta - 1) - \frac{X_0}{L_2}(\sec\theta - 1)^2\right]$$

**Atmospheric Parameters:**
- $X_0 = 920.0$ g/cm² (atmospheric depth parameter)
- $L_1 = 500.0$ g/cm² (electromagnetic attenuation length)  
- $L_2 = 594.0$ g/cm² (hadronic attenuation length)

**Reference**: `inc/ldffun.h:35-49`

### Energy Reconstruction Formulas

#### AGASA-Style Energy (aenergy)
$$E_{AGASA} = 0.203 \times S_{600,0} \text{ [EeV]}$$

**Reference**: `rufldf/src/p2ldffitter.cpp:382`

#### Rutgers Energy (energy)
Uses a sophisticated lookup table based on QGSJet-II.03 proton Monte Carlo simulations with polynomial fits as functions of $\log_{10}(S_{800})$ and $\sec\theta$:

$$\log_{10}(E/\text{eV}) = f(\log_{10}(S_{800}), \sec\theta)$$

Where $f$ is a 7th-order polynomial with zenith-angle-dependent coefficients stored in lookup tables.

**Reference**: `sdenergy/rusdenergy.cpp:1-100`, calibration comment at `inc/sdenergy.h:8-11`

## Physical Interpretation of LDF Parameters

### Scale Factor (S)
- **Physical Meaning**: Proportional to the total number of particles at shower maximum
- **Energy Relationship**: Directly related to primary cosmic ray energy
- **Units**: VEM/m² (Vertical Equivalent Muon per square meter)

### Zenith Angle Dependence (η)
- **Formula**: $\eta = 3.97 - 1.79(\sec\theta - 1)$
- **Physical Meaning**: Accounts for shower development changes with atmospheric depth
- **Range**: $\eta \approx 3.97$ at vertical incidence, decreases with increasing zenith angle

### Distance Dependencies
- **Inner Region** ($r < r_0$): Dominated by $r^{-\alpha}$ behavior
- **Middle Region** ($r_0 < r < r_{sc}$): Transition region with zenith-dependent steepening
- **Outer Region** ($r > r_{sc}$): Additional $r^{-2\beta}$ suppression from geometry

This comprehensive LDF framework provides the foundation for accurate energy reconstruction and shower parameter determination in the TA SD analysis.
