from pathlib import Path
import h5py
import numpy as np
import torch
import warnings


def divide_list(lst, n):
    # Calculate the approximate size of each sublist
    avg = len(lst) / n
    # Initialize a list to store the sublists
    result = []
    last = 0

    # Iterate through each sublist
    for i in range(1, n + 1):
        # Calculate the starting index of the sublist
        idx = int(round(avg * i))
        # Append the sublist to the result list
        result.append(lst[last:idx])
        # Update the starting index for the next sublist
        last = idx

    return result


def data_files(data_dir, glob_pattern, files_slice=None, divide_in=None):
    """
    Retrieve a list of filenames in a directory based on a glob pattern.

    Parameters:
    - data_dir (str): The path to the directory containing the data files.
    - glob_pattern (str): The glob pattern used to match filenames.
    - files_slice (slice, optional): A slice object to subset the list of filenames.

    Returns:
    - List[str]: A sorted list of filenames matching the glob pattern.

    Example:
    >>> data_files("/path/to/data", "*.txt", files_slice=slice(1, 5))
    ['/path/to/data/file2.txt', '/path/to/data/file3.txt', '/path/to/data/file4.txt', '/path/to/data/file5.txt']
    """

    filenames = sorted(list(Path(data_dir).glob(glob_pattern)))

    if len(filenames) == 0:
        warnings.warn("No file is found")

    if files_slice is not None:
        filenames = filenames[files_slice]

    if divide_in is not None:
        filenames = divide_list(filenames, divide_in)

    return filenames


def read_hdf5_metadata(file_path):
    info_dict = dict()
    # data_attributes = ["shape", "size", "ndim", "dtype", "nbytes"]
    data_attributes = ["shape"]

    def process_group(group, info_dict, prefix=""):
        for name, item in group.items():
            obj_path = f"{prefix}/{name}" if prefix else name

            info_dict[obj_path] = dict()
            # Print attribute names and values
            attribute_names = item.attrs.keys()
            for attr_name in attribute_names:
                attr_value = item.attrs[attr_name]
                info_dict[obj_path][attr_name] = attr_value

            if isinstance(item, h5py.Dataset):
                for dattr in data_attributes:
                    info_dict[obj_path][dattr] = getattr(item, dattr)

            if isinstance(item, h5py.Group):
                process_group(item, info_dict, obj_path)

    with h5py.File(file_path, "r") as file:
        info_dict["file"] = str(file_path)
        process_group(file, info_dict)

    for key, value in list(info_dict.items()):
        if isinstance(value, dict) and (not value):
            del info_dict[key]
    return info_dict


def array_size(array, unit):
    """
    Calculate the size of the array in megabytes.

    Parameters:
        array (numpy.ndarray or torch.Tensor): The input array.

    Returns:
        float: The size of the array in megabytes.
    """

    if unit == "MB":
        scale = 1024**2
    elif unit == "kB":
        scale = 1024
    else:
        raise ValueError(f"array_size: scale = {scale}")

    if isinstance(array, np.ndarray):
        return array.nbytes / scale
    elif isinstance(array, torch.Tensor):
        return array.element_size() * array.nelement() / scale
    else:
        return 0


def array_info_string(array, unit, key=""):
    """
    Generate information string about the array.

    Parameters:
        array (numpy.ndarray or torch.Tensor): The input array.
        key (str): The key to be used in the information string.

    Returns:
        str: Information string about the array.
    """

    if unit == "MB":
        scale = 1024**2
    elif unit == "kB":
        scale = 1024
    else:
        raise ValueError(f"array_info_string: scale = {scale}")

    if not isinstance(array, (torch.Tensor, np.ndarray)):
        return f"{key}.type={type(array)}\n"

    info = (
        f"{key}.shape={array.shape}, dtype={array.dtype},"
        f" size={array_size(array, unit):.3f} {unit}\n"
    )
    if isinstance(array, torch.Tensor):
        info = info[:-1] + f", device = {array.device}\n"

    return info


def data_info(data, unit, info_string="", total_size=0, add_info=True):
    """
    Generate information string about the data.

    Parameters:
        data (dict or iterable): The input data.
        add_info (bool): Flag to add detailed array information.

    Returns:
        str: Information string about the data.
    """

    if not isinstance(data, (list, dict)):
        raise ValueError(f"Unexpected data type: {type(data)}.")

    if isinstance(data, list):
        res_data = dict()
        for i, value in enumerate(data):
            res_data[f"array_{i}"] = value
        data = res_data

    root_level = False
    if len(info_string) == 0:
        root_level = True

    for key, value in data.items():

        if isinstance(value, dict):
            info_string += f"\n{key}\n"
            info_string, total_size = data_info(
                value,
                unit,
                info_string=info_string,
                total_size=total_size,
                add_info=add_info,
            )

        else:
            total_size += array_size(value, unit)
            if add_info:
                info_string += array_info_string(value, unit, key)

    if root_level:
        info_string += f"---\nTotal size = {total_size:.2f} {unit}\n"

    return info_string, total_size


def print_data_info(data, unit="MB"):
    """
    Print information about the data.

    Parameters:
        data (dict or iterable): The input data.
    """
    print(data_info(data, unit)[0])
