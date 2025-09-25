# H5DST Reader

This document describes how to use `h5dst.py` to read HDF5 DST (Data Summary Tape) files, both individually and in collections.

## Introduction

`h5dst.py` provides two main classes for interacting with HDF5 DST files:

*   `H5DST`: For reading data from a single HDF5 file.
*   `MultiH5DST`: For reading data from multiple HDF5 files as a single, unified dataset.

## Dependencies

Make sure you have the following Python packages installed:

*   `h5py`
*   `numpy`
*   `tqdm`

You can install them using pip:
```bash
pip install h5py numpy tqdm
```

## Usage

### Reading a Single File with `H5DST`

To read a single DST file, use the `H5DST` class.

**1. Import the class:**
```python
from h5dst import H5DST
```

**2. Create an instance:**
Provide the path to your HDF5 file.
```python
data = H5DST('path/to/your/file.h5')
```

**3. Inspect the file structure:**
The `show()` method prints a tree-like view of the groups and datasets within the file.
```python
data.show()
```

**4. List available data keys:**
The `keys()` method returns a list of available datasets you can access.
```python
print(data.keys())
```

**5. Access data:**
You can access datasets using dictionary-style key access.
```python
# Access the 'shower_axis' dataset
shower_axis = data['shower_axis']

# Get its shape
print(shower_axis.shape)
```

### Reading Multiple Files with `MultiH5DST`

To work with a collection of DST files as if they were one large file, use the `MultiH5DST` class. This is useful for training machine learning models or running analysis on large datasets spread across many files.

**1. Import the class:**
```python
from h5dst import MultiH5DST
from pathlib import Path
```

**2. Create an instance:**
Provide a list of file paths.
```python
data_dir = "/path/to/your/data"
files = sorted(Path(data_dir).rglob("*.h5"))
multi_data = MultiH5DST(files)
```

**3. Get the total number of events:**
The `len()` function returns the total number of events across all files.
```python
total_events = len(multi_data)
print(f"Total events: {total_events}")
```

**4. Access data slices:**
You can slice data across file boundaries seamlessly. The class handles loading data from the correct files automatically.
```python
# Get a slice of the 'energy' dataset
energies = multi_data['energy'][1000:5000]

# Get a slice of the 'xmax' dataset
xmax_values = multi_data['xmax'][4:1000000]
```

**5. Trace an event back to its source file:**
The `from_which_file` accessor helps you identify which file a specific event or a slice of events comes from.
```python
# Get info for a single event
event_info = multi_data.from_which_file[4534542]
print(event_info)

# Get info for a slice of events
slice_info = multi_data.from_which_file[4:1000]
print(slice_info)
```

This will return a list of tuples, where each tuple contains the file index, filename, and the local index/slice within that file.

### Data Levels

The data within the HDF5 files is organized into three main levels:

1.  **Event Level**: One entry per event (e.g., `energy`, `shower_axis`).
2.  **Hit Level**: One or more entries per event, corresponding to detector hits (e.g., `arrival_times`).
3.  **Time Trace Level**: One or more entries per hit, containing raw data like FADC traces (e.g., `time_traces`).

The `H5DST` class is designed to handle all three data levels seamlessly within a single file. `MultiH5DST` is optimized for accessing event-level data across many files.

While `MultiH5DST` does not directly provide access to hit-level or time-trace-level data across file boundaries (as this would be inefficient), you can easily investigate a specific event in detail. Use the `from_which_file` accessor to find which file an event belongs to, and then use the `H5DST` class to open that specific file for a detailed look at its hits and time traces.

#### Accessing Hit-Level Data with `H5DST`

For datasets at the hit level, you can access the data for a specific event by providing the event index. This will return an array of all hits for that event.

```python
# Access arrival times for all hits in event 86649
arrival_times_for_event = data32['arrival_times'][86649]
print(arrival_times_for_event.shape)

# Access the second component (e.g., y-coordinate) of all hits for that event
arrival_y_coords = data32['arrival_times'][86649][:, 1]
print(arrival_y_coords)
```

#### Accessing Time-Trace-Level Data with `H5DST` and Understanding `nfold`

Time traces, such as FADC data, are stored at the most granular level. You can access them by providing an event index and a hit index when using `H5DST`.

The `nfold` dataset is at the hit level and tells you how many time traces are associated with each hit. Typically, this might be 2 (for high-gain and low-gain channels).

```python
# Get the number of time traces for each hit in event 86649
nfold_for_event = data32['nfold'][86649]
print(nfold_for_event)

# Access the time traces for the 3rd hit (index 2) of event 86649
# This returns a list of numpy arrays.
time_traces_for_hit = data32['time_traces'][86649, 2]
print(f"Number of traces for this hit: {len(time_traces_for_hit)}")
print(f"Shape of the first trace: {time_traces_for_hit[0].shape}")

# You can then plot these traces
import matplotlib.pyplot as plt

# Plot the first trace (e.g., high-gain)
plt.plot(time_traces_for_hit[0].T)
plt.title("Time Trace 1 for Event 86649, Hit 2")
plt.show()

# Plot the second trace (e.g., low-gain)
plt.plot(time_traces_for_hit[1].T)
plt.title("Time Trace 2 for Event 86649, Hit 2")
plt.show()
```
