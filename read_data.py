from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
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


def array_size(array):
    """
    Calculate the size of an array or tensor in megabytes.

    Parameters:
    - array (Union[np.ndarray, torch.Tensor]): The input array or tensor.

    Returns:
    - float: The size of the array or tensor in megabytes.
    """
    if isinstance(array, np.ndarray):
        return array.nbytes / (1024**2)
    elif isinstance(array, torch.Tensor):
        return array.element_size() * array.nelement() / (1024**2)
    else:
        return 0


def array_info_string(array, key=""):
    if isinstance(array, np.ndarray):
        return (
            f"{key}.shape={array.shape}, dtype={array.dtype},"
            f" size={array_size(array):.3f} Mb\n"
        )
    elif isinstance(array, torch.Tensor):
        return (
            f"{key}.shape={array.shape}, dtype={array.dtype},"
            f" size={array_size(array):.3f} Mb, device = {array.device}\n"
        )
    else:
        return f"{key}.type={type(array)}\n"


def data_info(data):
    """
    Generate and return a string with information about the shapes, data types,
    and sizes of arrays in the given data.

    Parameters:
    - data (dict or list): A dictionary or list containing arrays.

    Returns:
    - str: A string with information about each array in the data.

    Raises:
    - ValueError: If data has an unsupported type (neither dict nor list).

    Example:
    >>> data_dict = {'array1': np.array([[1, 2], [3, 4]]), 'array2': np.array([[5, 6], [7, 8]])}
    >>> result_dict = data_info(data_dict)
    >>> print(result_dict)
    array1.shape=(2, 2), dtype=int64, size=0.000 Mb
    array2.shape=(2, 2), dtype=int64, size=0.000 Mb

    >>> data_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> result_list = data_info(data_list)
    >>> print(result_list)
    array_0.shape=(3,), dtype=int64, size=0.000 Mb
    array_1.shape=(3,), dtype=int64, size=0.000 Mb
    """
    info_string = ""
    total_size = 0

    
    if hasattr(data, "items"):
        for key, value in data.items():
            total_size += array_size(value)
            info_string += array_info_string(value, key)
    elif hasattr(data, "__iter__"):
        for i, value in enumerate(data):
            total_size += array_size(value)
            info_string += array_info_string(value, key=f"array_{i}")        
    else:
        raise ValueError(f"Unexpected data type: {type(data)}.")

    info_string += f"---\nTotal size = {total_size:.2f} Mb\n"
    return info_string


def print_data_info(data):
    print(data_info(data))


def data_size(data):
    """
    Calculate the total size of the data in megabytes.

    Parameters:
    - data: dict or iterable
        The data structure for which to calculate the size.

    Returns:
    - str
        A string containing information about the total size of the data in megabytes.
    """

    info_string = ""
    total_size = 0

    if hasattr(data, "items"):
        for value in data.values():
            total_size += array_size(value)
    elif hasattr(data, "__iter__"):
        for value in data:
            total_size += array_size(value)
    else:
        raise ValueError(f"Unexpected data type: {type(data)}.")

    info_string += f"---\nTotal size = {total_size:.4f} Mb\n"
    return info_string


def data_length(data):
    """
    Determine and return the length of the arrays in the given data dictionary.

    Parameters:
    - data (dict): A dictionary containing arrays.

    Returns:
    - int: The length of the arrays in the data.

    Raises:
    - RuntimeError: If arrays in the data dictionary have different lengths.

    Example:
    >>> data = {'array1': [1, 2, 3], 'array2': [4, 5, 6]}
    >>> result = data_length(data)
    >>> print(result)
    3
    """
    val = None
    for key, value in data.items():
        if val is None:
            val = len(value)
        if val != len(value):
            raise RuntimeError(f"Different sizes in data[{key}]: {val}!={len(value)}")
    return val


def display_data_info(filenames):
    """
    Display information about data files.

    Parameters:
    - filenames (List[str]): A list of filenames.

    Example:
    >>> display_data_info(["file1.npy", "file2.npy"])
    File: file1.npy
    key1.shape = (...)
    key2.shape = (...)

    File: file2.npy
    key1.shape = (...)
    key2.shape = (...)
    """
    for filename in filenames:
        print(f"File: {filename}")
        data = np.load(filename)
        print(data_info(data))


def dtype_mapping(dtype):
    """
    Create a dictionary mapping data types ('int' and 'float') to NumPy data
    types based on the specified dtype.

    Parameters:
    - dtype (str): The desired data type as a string ('64', '32', or '16').

    Returns:
    - dict: A dictionary mapping 'int' and 'float' to the corresponding NumPy data types.

    Raises:
    - ValueError: If the specified dtype is not implemented.
    """
    dtype_dict = dict()
    if dtype == "64":
        dtype_dict["int"] = np.int64
        dtype_dict["float"] = np.float64
    elif dtype == "32":
        dtype_dict["int"] = np.int32
        dtype_dict["float"] = np.float32
    elif dtype == "16":
        dtype_dict["int"] = np.int16
        dtype_dict["float"] = np.float16
    else:
        raise ValueError(f"dtype = {dtype} is not implemented")
    return dtype_dict


def convert_data(data, dtype_dict):
    """
    Convert the data to the specified data type using the provided dtype dictionary.

    Parameters:
    - data (np.ndarray): The input data to be converted.
    - dtype_dict (dict): A dictionary mapping 'int' and 'float' to NumPy data types.

    Returns:
    - np.ndarray: The converted data.
    """
    if np.issubdtype(data.dtype, np.integer):
        res = data.astype(dtype_dict["int"])
    else:
        res = data.astype(dtype_dict["float"])
    return res


def append_data(data, arrays, joined_data, dtype_dict):
    """
    Append data arrays to a dictionary of joined data, converting them to the specified data type.

    Parameters:
    - data (dict): A dictionary containing data arrays.
    - arrays (list): A list of keys corresponding to the arrays in the data dictionary.
    - joined_data (dict): A dictionary containing previously joined data arrays.
    - dtype_dict (dict): A dictionary mapping 'int' and 'float' to NumPy data types.

    Returns:
    - dict: A dictionary with appended and converted data arrays.

    Example:
    >>> data = {'array1': np.array([[1, 2], [3, 4]]), 'array2': np.array([[5, 6], [7, 8]])}
    >>> arrays = ['array1', 'array2']
    >>> joined_data = {'array1': np.array([[0, 0]]), 'array2': np.array([[9, 10]])}
    >>> dtype_dict = {'int': np.int64, 'float': np.float64}
    >>> result = append_data(data, arrays, joined_data, dtype_dict)
    >>> print(result)
    {'array1': array([[0, 0], [1, 2], [3, 4]]), 'array2': array([[9, 10], [5, 6], [7, 8]])}
    """
    for array in arrays:
        val = joined_data.get(array, None)
        conv_data = convert_data(data[array], dtype_dict)

        if val is None:
            joined_data[array] = conv_data
        else:
            joined_data[array] = np.append(val, conv_data, axis=0)

    return joined_data


def read_and_concat(filenames, arrays, dtype="32", info=("all", "datainfo")):
    """
    Read data from files, convert it to the specified data type, and concatenate it into a single dictionary.

    Parameters:
    - filenames (List[str]): A list of filenames.
    - arrays (List[str]): A list of keys corresponding to the arrays in the data.
    - dtype (str, optional): The desired data type as a string ('64', '32', or '16'). Default is '32'.

    Returns:
    - dict: A dictionary with concatenated and converted data arrays.

    Example:
    >>> filenames = ["file1.npy", "file2.npy"]
    >>> arrays = ['array1', 'array2']
    >>> result = read_and_concat(filenames, arrays, dtype="32")
    >>> print(result)
    {'array1': array([[0, 0], [1, 2], [3, 4]]), 'array2': array([[9, 10], [5, 6], [7, 8]])}
    """
    dtype_dict = dtype_mapping(dtype)
    joined_data = dict()
    for filename in tqdm(filenames, total=len(filenames)):
        if "all" in info or "filename" in info:
            print(f"File: {filename}")
        data = np.load(filename)
        if "all" in info or "file" in info:
            if "fileinfo" in info:
                print(data_info(data))
            else:
                print(data_size(data))
        if arrays is None:
            arrays = [*data]
        joined_data = append_data(data, arrays, joined_data, dtype_dict)
        if "all" in info or "data" in info:
            print(f"Joined data:")
            if "datainfo" in info:
                print(data_info(joined_data))
            else:
                print(data_size(joined_data))
    return joined_data


def convert_to_dict(arrays, array_names):
    """
    Convert a list of arrays to a dictionary with given array names.

    Parameters:
    - arrays (list): List of NumPy arrays.
    - array_names (list): List of strings representing the names for each array.

    Returns:
    dict: Dictionary with array names as keys and corresponding arrays as values.
    """
    data_dict = dict()
    for name, array in zip(array_names, arrays):
        data_dict[name] = array
    return data_dict


def allocate_joint_dict(data_template, length, exclude=None, dtype=np.float32):
    """
    Allocate a joint dictionary for storing data with specified template and length.

    Parameters:
    - data_template (dict): Dictionary with array names as keys and template arrays as values.
    - length (int): Number of elements to allocate for each array in the joint dictionary.
    - exclude (list or None): List of array names to exclude from the joint dictionary. Default is None.
    - dtype: NumPy data type for the joint dictionary arrays. Default is np.float32.

    Returns:
    tuple: A tuple containing array names, joint data dictionary, and shapes of the arrays.
    """
    joint_data = dict()
    array_names = [*data_template]
    if exclude is not None:
        array_names = [arr for arr in array_names if arr not in exclude]

    shapes = []
    for array in array_names:
        shape = data_template[array].shape
        shapes.append(shape)
        joint_data[array] = np.empty((shape[0] * length, *shape[1:]), dtype=dtype)

    return array_names, joint_data, shapes


def fill_joint_dict(array_names, joint_data, shapes, data, index):
    """
    Fill the joint dictionary with data for a specific index.

    Parameters:
    - array_names (list): List of array names in the joint dictionary.
    - joint_data (dict): Joint dictionary to be filled with data.
    - shapes (list): List of shapes corresponding to the arrays in the joint dictionary.
    - data (dict): Dictionary with array names as keys and corresponding arrays to fill in the joint dictionary.
    - index (int): Index specifying the position to fill in the joint dictionary.
    """
    for array, shape in zip(array_names, shapes):
        joint_data[array][index * shape[0] : (index + 1) * shape[0]] = data[array]


def join_hdf_files(files, array_paths):
    """
    Join multiple HDF files containing arrays specified by array_paths into a
    single joint dictionary.

    Parameters:
    - files (list): List of HDF file paths to be joined.
    - array_paths (list): List of strings representing the paths to arrays in the HDF files.

    Returns:
    tuple: joint data dictionary
    """
    alloc_params = None
    for i, file in tqdm(enumerate(files), total=len(files), desc="Joining HDF files"):
        # Extract arrays from the current file using provided array paths
        data_arrays = arrays_from_file(file, array_paths)

        # Convert arrays to a dictionary with specified array names
        data_dict = convert_to_dict(data_arrays, array_paths)

        # Allocate joint dictionary if not already done
        if alloc_params is None:
            alloc_params = allocate_joint_dict(data_dict, len(files))

        # Fill joint dictionary with data from the current file
        fill_joint_dict(*alloc_params, data_dict, i)

    return alloc_params[1]


def join_npz_files(files, exlude=None):
    """
    Join multiple npz files containing arrays specified by array_paths into a
    single joint dictionary.

    Parameters:
    - files (list): List of HDF file paths to be joined.
    - array_paths (list): List of strings representing the paths to arrays in the HDF files.

    Returns:
    tuple: joint data dictionary
    """
    alloc_params = None
    for i, file in tqdm(enumerate(files), total=len(files), desc="Joining NPZ files"):
        data_dict = np.load(file)

        # Allocate joint dictionary if not already done
        if alloc_params is None:
            alloc_params = allocate_joint_dict(data_dict, len(files), exlude)

        # Fill joint dictionary with data from the current file
        fill_joint_dict(*alloc_params, data_dict, i)

    return alloc_params[1]


def join_dicts(dicts, exlude=None):
    """
    Join multiple npz files containing arrays specified by array_paths into a
    single joint dictionary.

    Parameters:
    - files (list): List of HDF file paths to be joined.
    - array_paths (list): List of strings representing the paths to arrays in the HDF files.

    Returns:
    tuple: joint data dictionary
    """
    alloc_params = None
    for i, data_dict in tqdm(
        enumerate(dicts), total=len(dicts), desc="Joining data dicts"
    ):
        # Allocate joint dictionary if not already done
        if alloc_params is None:
            alloc_params = allocate_joint_dict(data_dict, len(dicts), exlude)

        # Fill joint dictionary with data from the current file
        fill_joint_dict(*alloc_params, data_dict, i)

    return alloc_params[1]


if __name__ == "__main__":
    data_dir = (
        "/home/antonpr/backup_dicos01/ml/erdmann/generated_data/06_A1_1M205/data/proc"
    )
    glob_pattern = "eas_proc*"

    # Read and inspect data files
    files = data_files(data_dir, glob_pattern, slice(0, 1))
    display_data_info(files)

    # Load, convert and concatenate arrays
    data = read_and_concat(
        filenames=data_files(data_dir, glob_pattern),
        arrays=["Energy", "showeraxis"],
        dtype="16",
    )
    # Print info of the resulting data
    print(data_info(data))
