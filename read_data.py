from pathlib import Path
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