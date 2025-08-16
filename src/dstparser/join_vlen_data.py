import numpy as np


def append_to_memstore(memstore, data_dict):
    """
    Add or append all arrays in `data_dict` into an in-memory dictionary `memstore`.

    - Initializes new keys with data from `data_dict`.
    - For keys containing "offset":
        * Expects a 1D array starting at 0.
        * Drops the leading zero, shifts by the existing last value, and appends.
    - For all other keys:
        * Appends the full array along axis 0.

    Parameters
    ----------
    memstore : dict[str, np.ndarray]
        Dictionary holding accumulated arrays.
    data_dict : dict[str, array-like]
        New chunk of arrays to append.
    """
    for key, array in data_dict.items():
        arr = np.asarray(array)

        if "offset" in key:
            assert (
                len(arr) > 0 and arr[0] == 0
            ), f"Offset array '{key}' must start with 0, got {arr[0] if len(arr) > 0 else 'empty array'}"

        if key not in memstore:
            memstore[key] = arr.copy()
        else:
            if "offset" in key:
                last_val = memstore[key][-1]
                new_offsets = arr[1:] + last_val
                memstore[key] = np.concatenate([memstore[key], new_offsets])
            else:
                memstore[key] = np.concatenate([memstore[key], arr])


def append_to_hdf5(h5file, data_dict):
    """
    Add or append all arrays in `data_dict` into the open HDF5 file `h5file`.

    - Creates a resizable dataset for each new key.
    - For keys containing "offset":
        * Expects a 1D array starting at 0.
        * Drops the leading zero, shifts by the existing last value, and appends.
    - For all other keys:
        * Appends the full array along axis 0.

    Parameters
    ----------
    h5file : h5py.File
        An open HDF5 file (mode 'a' or 'r+').
    data_dict : dict[str, array-like]
        Mapping of dataset names to new data chunks.
    """
    for key, array in data_dict.items():
        arr = np.asarray(array)
        if "offset" in key:
            assert (
                len(arr) > 0 and arr[0] == 0
            ), f"Offset array '{key}' must start with 0, got {arr[0] if len(arr) > 0 else 'empty array'}"

        if key not in h5file:
            # create a new resizable dataset
            maxshape = (None, *arr.shape[1:])
            h5file.create_dataset(
                key,
                data=arr,
                maxshape=maxshape,
                dtype=arr.dtype,
                chunks=True,
            )
        else:
            ds = h5file[key]
            if "offset" in key:
                new_off = np.asarray(arr[1:] + ds[-1], dtype=ds.dtype)
                old = ds.shape[0]
                ds.resize(old + new_off.size, axis=0)
                ds[old:] = new_off
            else:
                new_arr = arr.astype(ds.dtype)
                old = ds.shape[0]
                ds.resize(old + new_arr.shape[0], axis=0)
                ds[old : old + new_arr.shape[0]] = new_arr


if __name__ == "__main__":
    import h5py
    from tqdm import tqdm
    from pathlib import Path
    from dstparser import parse_dst_file_vlen

    # This is example shows how to append sequencially data to hdf5 file
    # or join data in memory
    data_dir = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo"
    files = sorted(Path(data_dir).rglob("DAT*dst.gz"))[0:26]

    join_in_memory = False

    if join_in_memory:
        memstore = {}
        for fname in tqdm(files):
            data = parse_dst_file_vlen(fname)
            append_to_memstore(memstore, data)
    else:
        with h5py.File("all_data.h5", "a") as f:
            for fname in tqdm(files):
                data = parse_dst_file_vlen(fname)
                append_to_hdf5(f, data)
