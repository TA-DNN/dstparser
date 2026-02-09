# Xmax Reader Module

This directory provides tools for reading, managing, and scaling Xmax (shower maximum depth) values for cosmic ray air shower simulations.

## Overview

The module consists of three main components:

1. **`xmax_reader.py`** - Core classes for reading and scaling Xmax values
2. **`xmax_auger.py`** - Parameterizations from Auger experiment for Xmax distributions
3. **`create_xmax_db.py`** - Database creation utilities for efficient Xmax lookups

---

## File Descriptions

### 1. `xmax_reader.py`

The main module providing classes for Xmax value handling.

#### Key Classes:

- **`DstPathParser`**: Extracts metadata (model, primary particle, period, bin_id, shower_id) from file paths
- **`XmaxScaler`**: Scales Xmax values to arbitrary energies using elongation rate
- **`XmaxReader`**: Abstract base class for Xmax readers
- **`XmaxReaderTxt`**: Reads Xmax values from text files
- **`XmaxReaderDatabase`**: Reads Xmax values from HDF5 databases (faster, recommended)

#### Usage Example:

```python
from dstparser.xmax_reader.xmax_reader import create_xmax_reader
import numpy as np

# Option 1: Read from text files
reader = create_xmax_reader("/path/to/data/directory")

# Option 2: Read from HDF5 database (faster)
reader = create_xmax_reader({
    "qgsjetii04": "/path/to/qgsii04_xmax_db.h5",
    "eposlhc": "/path/to/eposlhc_xmax_db.h5",
})

# Initialize with a specific file
filepath = "/path/to/DAT006413_gea.rufldf.dst.gz"
reader.read_file(filepath)

# Scale Xmax to different energies
energies = np.array([10, 11, 12])  # in EeV
masses = np.array([56, 56, 56])     # Iron (A=56)
xmax_values = reader(energies, masses)
```

### 2. `xmax_auger.py`

Contains parameterizations for Xmax distributions from Auger experiments.

#### Key Components:

- **`DXMAX_PARAMS`**: Dictionary of model parameters for elongation rates
- **`gumbel_parameters()`**: Calculate Gumbel distribution parameters (μ, σ, λ)
- **`rand_xmax()`**: Generate random Xmax values following the Gumbel distribution

#### Supported Models:

- QGSJet01, QGSJetII, QGSJetII-04
- EPOS1.99, EPOS-LHC
- Sibyll2.1, Sibyll2.3d

#### Usage Example:

```python
from dstparser.xmax_reader.xmax_auger import rand_xmax
import numpy as np

log10e = 19.0  # log10(E/eV) = 19 → 10 EeV
mass = 1       # Proton
model = "EPOS-LHC"

# Generate random Xmax values
xmax_values = rand_xmax(log10e, mass, size=1000, model=model)
```

### 3. `create_xmax_db.py`

Utilities for creating HDF5 databases from text files for faster lookups.

#### Key Functions:

- **`create_xmax_db_dict()`**: Parse text files and create nested dictionary
- **`save_xmax_db_hdf5()`**: Save data to compressed HDF5 format
- **`load_xmax_db_hdf5()`**: Load specific tables from HDF5
- **`create_xmax_db()`**: Complete pipeline to create database

#### Usage Example:

```python
from dstparser.xmax_reader.create_xmax_db import create_xmax_db

root = "/path/to/tasdmc_dstbank"
model_prefix = "qgsii04"  # or "eposlhc_" or "sibyll"
output_db = "/path/to/qgsii04_xmax_db.h5"

# Create database
create_xmax_db(root, model_prefix, output_db)
```

---

## Customization Points

### 1. **Adding New Models**

To support a new hadronic interaction model:

**In `xmax_auger.py`:**
- Add parameters to `DXMAX_PARAMS` dictionary
- Add Gumbel parameters to `params` dict in `gumbel_parameters()`

**In `xmax_reader.py`:**
- Update `xmax_models_map` in `XmaxScaler.__init__()`

**In `create_xmax_db.py`:**
- Add model prefix mapping in `models` dict in `create_xmax_db_dict()`

Example:
```python
# In xmax_auger.py
DXMAX_PARAMS["NewModel"] = (X0, D, xi, delta, p0, p1, p2, a0, a1, b)

# In xmax_reader.py XmaxScaler
xmax_models_map = {
    "newmodel": "NewModel",
}
```

### 2. **Custom Path Parsing**

Modify `DstPathParser` to handle different file naming conventions:

```python
class DstPathParser:
    def __init__(self):
        # Add new aliases
        self.model_aliases["newmodel"] = ["NEWMODEL", "newmdl"]
        self.primary_aliases["oxygen"] = ["o", "oxygen"]
        
        # Rebuild patterns
        self.model_pattern_str = self._build_alias_pattern(self.model_aliases)
```

### 3. **Custom Energy Grid**

Modify `std_ta_energy_grid()` for different energy binning:

```python
def std_ta_energy_grid():
    # Custom: 100 bins, range 15-22 in log10(E/eV)
    mids = np.linspace(15, 22, 100)
    res = {"mids": 10 ** (mids - 18)}
    # ... calculate edges
    return res
```

### 4. **Custom Data Sources**

Create a new `XmaxReader` subclass:

```python
class XmaxReaderCustom(XmaxReader):
    def __init__(self, custom_source):
        super().__init__()
        self.custom_source = custom_source
    
    def _get_xmax0(self, parsed_info):
        # Implement custom logic to retrieve xmax0
        return self._custom_lookup(parsed_info)
```

### 5. **Database Table Structure**

Modify `save_xmax_db_hdf5()` to change table structure or add columns:

```python
# Add additional columns
rows.append({
    "bin_id": int(bin_id),
    "shower_id": int(shower_id),
    "ngenerated": int(d["ngenerated"]),
    "zenith_angle": float(d["zenith_angle"]),
    "xmax": float(d["xmax"]),
    "custom_field": custom_value,  # New field
})
```

### 6. **Elongation Rate Calculation**

Modify `XmaxScaler.__call__()` for custom scaling:

```python
def __call__(self, energies, mass):
    # Custom scaling logic
    eff_elongation = self.custom_elongation_function(mass)
    return self.xmax0 + eff_elongation * np.log10(energies / self.en0)
```

---

## Workflow Recommendations

### For Fast Repeated Access:
1. Create HDF5 databases once using `create_xmax_db.py`
2. Use `XmaxReaderDatabase` for all subsequent reads
3. Database supports lazy loading and caching for efficiency

### For One-Time Analysis:
1. Use `XmaxReaderTxt` directly with data directory
2. No preprocessing required

### For Production Systems:
1. Pre-generate all required databases
2. Use `create_xmax_reader()` factory with database map
3. Leverage MultiIndex DataFrame for O(1) lookups

---

## Key Concepts

### Energy Binning
- Standard TA (Telescope Array) energy grid: 51 bins, 10^16 to 10^21 eV
- Energy bin centers in EeV (10^18 eV)
- TA index mapping handles non-contiguous energy ranges

### Xmax Scaling
- Uses elongation rate: `Xmax(E) = Xmax(E0) + D_eff * log10(E/E0)`
- `D_eff = D + δ * ln(A)` where A is atomic mass
- Handles both measured and randomly generated Xmax values

### Database Structure
- HDF5 with one table per (model, primary, period) combination
- MultiIndex on (bin_id, shower_id) for fast lookups
- Compressed with zlib (complevel=9) for storage efficiency

---

## Performance Notes

- **Text file reading**: ~seconds for small datasets, can be slow for large collections
- **HDF5 database**: ~milliseconds for lookups, recommended for production
- **Caching**: XmaxReaderDatabase caches loaded tables in memory
- **Lazy loading**: Tables loaded only when first accessed

---

## Dependencies

- `numpy`: Array operations
- `pandas`: DataFrame handling and HDF5 I/O
- `pathlib`: Path manipulations
- `re`: Regular expressions for path parsing

---

## Common Use Cases

### Load existing Xmax and scale to new energies:
```python
reader = create_xmax_reader({...})  # database map
reader.read_file(filepath)
new_xmax = reader(new_energies, masses)
```

### Generate random Xmax when none exists:
```python
# If xmax0 is None or 0, XmaxScaler automatically generates random values
scaler = XmaxScaler(model="EPOS-LHC", xmax0=None, en0=10.0)
xmax = scaler(energies, mass)  # Uses rand_xmax() internally
```

### Batch processing multiple files:
```python
reader = create_xmax_reader(database_map)
for filepath in file_list:
    reader.read_file(filepath)
    results[filepath] = reader(target_energies, masses)
```

### Create database from scratch:
```python
root = "/data/simulations"
for model_prefix in ["qgsii04", "eposlhc_", "sibyll"]:
    output_db = f"{model_prefix}_xmax_db.h5"
    create_xmax_db(root, model_prefix, output_db)
```
