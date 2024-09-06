import numpy as np


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
