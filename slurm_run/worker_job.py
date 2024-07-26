import os
import sys
import json
import h5py
import re
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from dstparser import parse_dst_file
from dstparser.xmax_reader import XmaxReader


def minimal_uint_type(array):
    """Save integer array using minimal np.uint type capable
    to hold its max values"""

    if len(array) == 0:
        return array

    if np.any(array < 0):
        raise ValueError("Only positive arrays are allowed")

    arrmax = np.max(array)
    inttypes = [np.uint8, np.uint16, np.uint32, np.uint64]

    for inttype in inttypes:
        if np.iinfo(inttype).max > arrmax:
            return array.astype(inttype)


def restore_sparse_array(data):
    """Restore compressed array"""
    if not isinstance(data, dict) or (data.get("nz_data") is None):
        return data

    if data.get("nz_idxs") is None:
        # Restore nz_idxs:
        idxs = np.ones(data["ic_info"][0])
        idxs[data["ic_idxs"]] = data["ic_data"]
        idxs[0] = data["ic_info"][1]
        nz_idxs = np.cumsum(idxs, dtype=np.int64)
    else:
        nz_idxs = data["nz_idxs"]

    # Restore array
    array = np.zeros(data["shape"], dtype=data["nz_data"].dtype)
    if len(data["nz_data"]) > 0:
        np.ravel(array)[nz_idxs] = data["nz_data"]
    return array


def compress_sparse_array(array, np_dtype=np.float16):
    """Compress sparse numpy array with a lot of zeros.
    Returns compressed representation if possible to reduce the size,
    otherwise returns original array
    """
    nz_idxs = np.flatnonzero(array)
    flat_array = np.ravel(array)

    # If all elements are non-zero
    # return original array
    if len(nz_idxs) >= len(flat_array):
        return array.astype(np_dtype)

    nz_data = np.ravel(array)[nz_idxs]
    nz_data = nz_data.astype(np_dtype)

    data = dict()
    data["nz_data"] = nz_data
    # Here "100" is arbitrary, one can test to change it
    if len(nz_idxs) > 100:
        # Compress nz_idxs, because large np.int64 array
        # takes a lot of memory ~ the size of original array
        # Compression of nz_idxs takes advantage that
        # there are many nz_idxs[i] + 1 == nz_idxs[i+1]
        diff_idxs = np.diff(nz_idxs)
        # Record only nz_idxs[i+1] - nz_idxs[i] > 1 entries
        nu_idxs = np.where(diff_idxs > 1)[0]
        nu_data = diff_idxs[nu_idxs]
        data["ic_info"] = np.array([len(nz_idxs), nz_idxs[0]], np.uint64)
        data["ic_idxs"] = minimal_uint_type(nu_idxs + 1)
        data["ic_data"] = minimal_uint_type(nu_data)
    else:
        data["nz_idxs"] = nz_idxs

    compressed_size = 0
    for key in data:
        compressed_size += data[key].nbytes

    # Return original array, if compressed size is larger than original
    if compressed_size >= array.nbytes:
        return array.astype(np_dtype)

    data["shape"] = array.shape
    # Test whether original array could be restored
    # from compressed representation
    assert np.all(
        restore_sparse_array(data) == array.astype(np_dtype)
    ), "Bad compression"
    return data


def write_h5(filename, data):
    with h5py.File(filename, "w") as f:
        for key, value in data.items():
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    f.create_dataset(f"{key}/{key1}", data=value1)
            else:
                f.create_dataset(f"{key}", data=value)


def read_h5(filename):
    data = dict()
    with h5py.File(filename, "r") as f:
        for key, value in f.items():
            if isinstance(value, h5py.Group):
                data[key] = dict()
                for key1, value1 in value.items():
                    data[key][key1] = value1[:]
                data[key] = restore_sparse_array(data[key])
            else:
                data[key] = value[:]

    return data


def slurm_parameters():

    env_vars = {
        "task_id": "SLURM_ARRAY_TASK_ID",
        "task_min": "SLURM_ARRAY_TASK_MIN",
        "task_max": "SLURM_ARRAY_TASK_MAX",
        "job_id": "SLURM_JOB_ID",
    }

    slurm_params = dict()

    for key, envar in env_vars.items():
        val = os.environ.get(envar)
        try:
            slurm_params[key] = int(val)
        except TypeError:
            slurm_params[key] = val

    return slurm_params


def task_info(zero_indexing=False):
    slurm_params = slurm_parameters()

    if slurm_params["task_id"] is None:
        return None, None

    task_id = slurm_params["task_id"]
    ntasks = slurm_params["task_max"] - slurm_params["task_min"] + 1

    if zero_indexing:
        task_id -= slurm_params["task_min"]

    return task_id, ntasks


def save2hdf5(acc_data, filename, np_dtype=np.float32):
    compress_arrs = ["time", "arrival", "detector"]

    nattempts = 5
    for iattempt in range(nattempts):
        with h5py.File(filename, "w") as f:
            for key, value in acc_data.items():
                value = np.concatenate(value, axis=0)

                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype, np.floating
                ):
                    value = value.astype(np_dtype)

                if any(key.startswith(s) for s in compress_arrs):
                    value = compress_sparse_array(value, np_dtype)

                # Write to hdf5 file:
                if isinstance(value, dict):
                    for key1, value1 in value.items():
                        f.create_dataset(f"{key}/{key1}", data=value1)
                else:
                    f.create_dataset(f"{key}", data=value)

                del value

        try:
            read_data = read_h5(filename)
            break
        except Exception as ex:
            print(f"Attempt {iattempt + 1} failed with: {ex}")


def join_hdf5():

    task_id, ntasks = task_info()
    if task_id is None:
        raise ValueError("No slurm is found!")
        # task_id = sys.argv[3]

    with open(sys.argv[1], "r") as f:
        task_db = json.load(f)

    task_id = str(task_id)
    filename = task_db[task_id]["output_file"]

    print(f"Joining files for {filename}")

    filename = Path(filename)
    if filename.exists():
        print(f"{filename} is already exists!")
        return
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)

    acc_data = dict()
    ifiles = task_db[task_id]["input_files"]

    print(f"Joining files for {filename}")

    for ifile in tqdm(ifiles, total=len(ifiles), desc="Joining files"):
        data = read_h5(ifile)
        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, filename)


def info_from_filename(data, filename):
    """Write additional information retrived from file name"""
    ifname = Path(filename).parts[-1]
    ifname_parts = re.split(r"[_,.\s]", ifname)

    # TA DATA
    if ifname_parts[0].startswith("tasdcalibev"):
        key, value = "ta_obs_date", int(ifname_parts[2])
    # TA MC
    elif ifname_parts[0].startswith("DAT"):
        key, value = "cors_shower_id", int(ifname_parts[0][3:])
    else:
        key, value = None, None

    if key is not None:
        data_len = next(iter(data.values())).shape[0]
        data[key] = np.full((data_len,), value, dtype=np.int64)

    return data


def filter_data_max(data, max_events):

    np.random.seed()
    data_len = next(iter(data.values())).shape[0]
    current_len = min(data_len, max_events)
    inds = np.random.choice(np.arange(data_len), current_len, replace=False)

    for key in data:
        data[key] = data[key][inds]

    return data


def filter_full_tiles(data, max_events=None, invert2partial=False):

    np.random.seed()

    states = data["detector_states"]
    all_true_mask = np.all(states, axis=(1, 2))
    # # Invert condition, take partial tile
    if invert2partial:
        all_true_mask = np.logical_not(all_true_mask)

    cond_indices = np.where(all_true_mask)[0]

    # Choose only max_events from them
    cond_len = len(cond_indices)

    if max_events is None:
        current_len = cond_len
    else:
        current_len = min(cond_len, max_events)

    inds = np.random.choice(np.arange(cond_len), current_len, replace=False)
    inds = cond_indices[inds]

    for key in data:
        data[key] = data[key][inds]

    return data


def dst_to_hdf5():

    task_id, ntasks = task_info()
    if task_id is None:
        raise ValueError("No slurm is found!")
        # task_id = sys.argv[3]

    with open(sys.argv[1], "r") as f:
        task_db = json.load(f)

    task_id = str(task_id)
    filename = task_db[task_id]["output_file"]

    filename = Path(filename)
    print(f"Create file {str(filename)}")
    if filename.exists():
        return
    else:
        filename.parent.mkdir(parents=True, exist_ok=True)

    acc_data = dict()
    ifiles = task_db[task_id]["input_files"]
    # xmax_dir = Path(ifiles[0]).parent
    # xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")
    xmax_reader = None
    for file in tqdm(ifiles, total=len(ifiles), desc="DST conversion"):

        if xmax_reader is not None:
            # xmax_dir is the same as directory of the file
            # but it loads many files, so better to change and
            # initialize XmaxReader object only when the directory
            # changes
            cur_dir = Path(file).parent
            if xmax_dir != cur_dir:
                xmax_dir = cur_dir
                xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")

        data = parse_dst_file(
            file,
            ntile=7,
            xmax_reader=xmax_reader,
            avg_traces=False,
            add_shower_params=False,
            add_standard_recon=True,
        )

        if data is None:
            continue

        data = info_from_filename(data, file)
        # data = filter_full_tiles(data, max_events=50)

        for key, value in data.items():
            acc_data.setdefault(key, []).append(value)

    save2hdf5(acc_data, filename)


def worker_job():

    if sys.argv[2] == "parse_dst":
        dst_to_hdf5()
    elif sys.argv[2] == "join_hdf5":
        join_hdf5()


if __name__ == "__main__":
    worker_job()
