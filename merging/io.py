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
    # assert np.all(
    #     restore_sparse_array(data) == array.astype(np_dtype)
    # ), "Bad compression"
    return data