"""
TASD detector position tables, CLF frame, two dataset epochs.

Data stored in tasd_detector_coords.h5 (same directory), derived from:
  sdxyzclf_raw.h   (sdanalysis_2019/inc/)
  tacoortrans.h    (CLF reference point)

Epochs:
  DS1  yymmdd < 81111   (before 2008-11-11)  502 detectors
  DS2  yymmdd >= 81111  (from  2008-11-11)   507 detectors  (DS3 = DS2)

Arrays per epoch:
  ids  int16   (n,)     xxyy codes  (xx*100 + yy)
  xyz  float32 (n, 3)   x, y, z positions [metres, CLF frame]

CLF reference: lat=39.29693  lon=-112.90875  alt=1382.0 m
"""

from pathlib import Path
import h5py
import numpy as np

YYMMDD_DS1_2_DS2 = 81111  # 2008-11-11 (YYMMDD, no zero padding)

_h5_path = Path(__file__).with_name("tasd_detector_coords.h5")

with h5py.File(_h5_path, "r") as _f:
    detector_ids_ds1 = _f["ds1/ids"][:]
    detector_xyz_ds1 = _f["ds1/xyz"][:]
    detector_ids_ds2 = _f["ds2/ids"][:]
    detector_xyz_ds2 = _f["ds2/xyz"][:]


def get_detector_coords(yymmdd: int):
    """Return (ids, xyz) for the given date (YYMMDD, no zero padding).

    ids : int16   (n,)      xxyy detector codes
    xyz : float32 (n, 3)    x, y, z positions [metres, CLF frame]
    """
    if yymmdd < YYMMDD_DS1_2_DS2:
        return detector_ids_ds1, detector_xyz_ds1
    return detector_ids_ds2, detector_xyz_ds2
