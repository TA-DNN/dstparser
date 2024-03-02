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
    nowIndex = int(np.where(tasd_clf.tasdmc_clf[:, 0] == rufptn_xxyy_)[0])
    return tasd_clf.tasdmc_clf[nowIndex, :]


def find_pos(evt_pos):
    masks = tasd_clf.tasdmc_clf[:, 0][:, np.newaxis] == evt_pos
    first_indices = np.argmax(masks, axis=0)
    return tasd_clf.tasdmc_clf[first_indices, 1:] / 100


def tile_normalization(abs_coord, do_exist, shower_core):
    # Normalization of a tile for DNN
    n0 = (abs_coord.shape[0] - 1) // 2
    tile_center = abs_coord[n0, n0]
    # Shift to the hight of CLF (z)
    tile_center[2] = 1370

    # Shift shower core
    shower_core[:2] = shower_core[:2] - tile_center[:2]

    tile_center = tile_center[np.newaxis, np.newaxis, :]
    rel_coord = np.where(do_exist[:, :, np.newaxis], abs_coord - tile_center, 0)

    # xy coordinate normalization
    tile_extent = n0 * 1200  # extent of tile
    rel_coord[:, :, 0:2] = rel_coord[:, :, 0:2] / tile_extent
    shower_core[:2] = shower_core[:2] / tile_extent
    # z coordinate normalization
    height_extent = 30  # +- 30 meters
    rel_coord[:, :, 2] = rel_coord[:, :, 2] / height_extent
    shower_core[2] = shower_core[2] / height_extent

    return rel_coord, shower_core


def tile_positions(ixy0, tile_size, badsd, shower_core):
    # Create centered tile
    # n0 = (tile_size - 1) / 2
    x, y = np.mgrid[0:tile_size, 0:tile_size].astype(float)

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

    abs_coord = tasd_clf.tasdmc_clf[tasdmc_clf_indices, 1:] / 1e2
    rel_coord, rel_shower_core = tile_normalization(abs_coord, do_exist, shower_core)

    return rel_coord, status, rel_shower_core
