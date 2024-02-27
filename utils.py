import numpy as np
import tasd_clf


def CORSIKAparticleID2mass(corsikaPID):
    if corsikaPID == 14:
        return 1
    else:
        return corsikaPID // 100


def convert_to_specific_type(columnName, value):
    if columnName == "rusdmc_.parttype":
        return CORSIKAparticleID2mass(int(value))
    elif columnName in [
        "rusdraw_.yymmdd",
        "rusdraw_.hhmmss",
        "rufptn_.nstclust",
        "rusdraw_.nofwf",
        "rufptn_.xxyy",
        "rufptn_.isgood",
    ]:
        return int(value)
    else:
        return float(value)


def rufptn_xxyy2sds(rufptn_xxyy_):

    print(f"tasdmc_clf.shape = {tasd_clf.tasdmc_clf.shape}")
    input()
    nowIndex = int(np.where(tasd_clf.tasdmc_clf[:, 0] == rufptn_xxyy_)[0])
    return tasd_clf.tasdmc_clf[nowIndex, :]


def find_pos(evt_pos):
    masks = tasd_clf.tasdmc_clf[:, 0][:, np.newaxis] == evt_pos
    first_indices = np.argmax(masks, axis=0)
    return tasd_clf.tasdmc_clf[first_indices, 1:] / 100


def tile_positions(ixy0, tile_size, badsd):
    # Create centered tile
    n0 = (tile_size - 1) / 2
    x, y = np.mgrid[0:tile_size, 0:tile_size].astype(float) - n0

    # Shift towards real center
    # ixy0 = [24, 10] - at the edge, uncomment for testing
    x += ixy0[0]
    y += ixy0[1]
    xy_code = x * 100 + y

    # Create mask (:, tile_size, tile_size)
    masks = tasd_clf.tasdmc_clf[:, 0][:, np.newaxis, np.newaxis] == xy_code
    tasdmc_clf_indices = np.argmax(masks, axis=0)
    do_exist = masks.any(axis=0)
    tasdmc_clf_indices = np.where(do_exist, tasdmc_clf_indices, -1)
    
    # Do detectors work: 
    good = ~np.isin(tasd_clf.tasdmc_clf[tasdmc_clf_indices, 0], badsd)
    status = np.logical_and(good, do_exist)

    return tasd_clf.tasdmc_clf[tasdmc_clf_indices, 1:], status
