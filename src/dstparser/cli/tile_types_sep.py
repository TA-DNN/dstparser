import numpy as np


def full_tiles_cond(data, mask):
    states = data["detector_states"][mask]
    return np.all(states, axis=(1, 2))


def partial_tiles_cond(data, mask):
    states = data["detector_states"][mask]
    all_true_mask = np.all(states, axis=(1, 2))
    return np.logical_not(all_true_mask)


def subtract_masks(mask, sub_mask):
    assert np.all(mask[sub_mask] == sub_mask[sub_mask])
    res_mask = np.zeros_like(mask, dtype=np.bool_)
    res_mask[:] = mask
    res_mask[sub_mask] = False
    return res_mask


def pick_random_events(data, mask, max_events=None, seed=1, condition=None):
    """Returns boolean mask for data with True for mask=True and condition=True.
    Number of true elements is max from max_events and number of picked elements
    """

    np.random.seed(seed)

    if condition is not None:
        all_true_mask = condition(data, mask)
    else:
        all_true_mask = mask[mask]

    cond_indices = np.where(all_true_mask)[0]

    # Choose only max_events from them
    cond_len = len(cond_indices)

    if max_events is None:
        current_len = cond_len
    else:
        current_len = min(cond_len, max_events)

    inds = np.random.choice(np.arange(cond_len), current_len, replace=False)
    inds = cond_indices[inds]

    # Get boolean mask for original (total/unmasked with "mask") data
    bool_mask = np.zeros_like(mask, dtype=np.bool_)

    # Get the indices in the original mask where `mask` is True
    true_indices = np.nonzero(mask)[0]

    # Set the values at those indices according to `inds`
    bool_mask[true_indices[inds]] = True
    return bool_mask


def pick_true_elements(mask, inds):
    bool_mask = np.zeros_like(mask, dtype=np.bool_)
    true_inds = np.where(mask)[0]
    pick_inds = true_inds[inds]
    bool_mask[pick_inds] = True
    return bool_mask


def train_test_separation(mask, test_frac=0.2):

    tot_events = np.sum(mask)
    test_events = int(tot_events * test_frac)
    # 25% to small data set
    small_test_events = int(max(test_events / 4, 1))

    train_events = tot_events - test_events
    # 10% to small data set
    small_train_events = int(max(train_events / 10, 1))

    res_masks = dict()
    res_masks["train"] = pick_true_elements(mask, slice(0, train_events))
    res_masks["small_train"] = pick_true_elements(mask, slice(0, small_train_events))
    res_masks["test"] = pick_true_elements(
        mask, slice(train_events, train_events + test_events)
    )
    res_masks["small_test"] = pick_true_elements(
        mask, slice(train_events, train_events + small_test_events)
    )
    return res_masks


def large_small_separation(mask, fraction):
    test_events = np.sum(mask)
    small_events = int(max(test_events * fraction, 1))
    return mask, pick_true_elements(mask, slice(0, small_events))


def any_tiles_special(data, mask, full_tiles_train, seed=1, max_train=20, max_test=4):
    """Test set: any tiles except the tiles used in full_tiles train set
    Train set: any tiles except the one used in THIS test set
    """
    # Choose events with any tiles:
    any_tiles = dict()
    # Choose for test set the events that full_tiles training haven't seen
    rest_events = subtract_masks(mask, full_tiles_train)
    # For each bin pick 4 events for large data set and 1 event for small data set
    any_tiles_test_events = pick_random_events(
        data, rest_events, max_events=max_test, seed=seed, condition=None
    )
    any_tiles["test"], any_tiles["small_test"] = large_small_separation(
        any_tiles_test_events, fraction=0.25
    )
    # Anything that is not test set could be used for training data set:
    allowed_non_test_events = subtract_masks(mask, any_tiles_test_events)
    any_tiles_train_events = pick_random_events(
        data,
        allowed_non_test_events,
        max_events=max_train,
        seed=seed,
        condition=None,
    )
    any_tiles["train"], any_tiles["small_train"] = large_small_separation(
        any_tiles_train_events, fraction=0.1
    )

    return any_tiles


def tile_types_masks(data, max_train, max_test, seed):

    # max_train = 20  # events
    # max_test = 4  # events
    # seed = 1

    data_set_ids = np.unique(data["id_data_set"])
    corsika_shower_ids = np.unique(data["id_corsika_shower"])
    energy_bin_ids = np.unique(data["id_energy_bin"])

    max_events = max_train + max_test

    tot_picked_events = dict()
    tot_picked_events["full_tiles"] = dict()
    tot_picked_events["partial_tiles"] = dict()
    tot_picked_events["any_tiles"] = dict()

    for data_set_id in data_set_ids:
        dset_events = data["id_data_set"] == data_set_id
        for corsika_shower_id in corsika_shower_ids:
            cors_shower_events = data["id_corsika_shower"] == corsika_shower_id
            for energy_bin_id in energy_bin_ids:
                enbin_events = data["id_energy_bin"] == energy_bin_id

                mask = enbin_events & cors_shower_events & dset_events

                picked_events = dict()
                full_tiles_mask = pick_random_events(
                    data,
                    mask,
                    max_events=max_events,
                    seed=seed,
                    condition=full_tiles_cond,
                )
                picked_events["full_tiles"] = train_test_separation(full_tiles_mask)

                partial_tiles_mask = pick_random_events(
                    data,
                    mask,
                    max_events=max_events,
                    seed=seed,
                    condition=partial_tiles_cond,
                )
                picked_events["partial_tiles"] = train_test_separation(
                    partial_tiles_mask
                )

                picked_events["any_tiles"] = any_tiles_special(
                    data,
                    mask,
                    picked_events["full_tiles"]["train"],
                    seed=seed,
                    max_train=max_train,
                    max_test=max_test,
                )

                for tile_categ in tot_picked_events:
                    for data_set_categ in picked_events[tile_categ]:
                        if tot_picked_events[tile_categ].get(data_set_categ) is None:
                            tot_picked_events[tile_categ][data_set_categ] = (
                                picked_events[tile_categ][data_set_categ]
                            )
                        else:
                            tot_picked_events[tile_categ][
                                data_set_categ
                            ] |= picked_events[tile_categ][data_set_categ]

    return tot_picked_events


def test_picked_events(data, tot_picked_events):
    for tile_categ in tot_picked_events:
        for data_set_categ in tot_picked_events[tile_categ]:
            print(
                f"{tile_categ}/{data_set_categ}",
                np.sum(tot_picked_events[tile_categ][data_set_categ]),
                len(tot_picked_events[tile_categ][data_set_categ]),
            )
            ddd = np.all(
                data["detector_states"][tot_picked_events[tile_categ][data_set_categ]],
                axis=(1, 2),
            )
            print("among them full tiles:", np.sum(ddd), np.sum(ddd) / len(ddd))
            print()

    # Test is any intersection between train and test
    full_any = np.sum(
        tot_picked_events["full_tiles"]["train"]
        & tot_picked_events["any_tiles"]["test"]
    )
    any_any = np.sum(
        tot_picked_events["any_tiles"]["train"] & tot_picked_events["any_tiles"]["test"]
    )
    print(f"full_any = {full_any}, any_any = {any_any}")
