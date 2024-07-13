import numpy as np
import re
from pathlib import Path
from dstparser.read_data import data_files


# Taken from https://astro.pages.rwth-aachen.de/astrotools/_modules/auger.html#mean_xmax
# Values for <Xmax>, sigma(Xmax) parameterization from [3,4,5]
# DXMAX_PARAMS[model] = (X0, D, xi, delta, p0, p1, p2, a0, a1, b)
DXMAX_PARAMS = {
    # from [3], pre-LHC
    "QGSJet01": (774.2, 49.7, -0.30, 1.92, 3852, -274, 169, -0.451, -0.0020, 0.057),
    # from [3], pre-LHC
    "QGSJetII": (781.8, 45.8, -1.13, 1.71, 3163, -237, 60, -0.386, -0.0006, 0.043),
    # from [3], pre-LHC
    "EPOS1.99": (809.7, 62.2, 0.78, 0.08, 3279, -47, 228, -0.461, -0.0041, 0.059),
    # from [3]
    "Sibyll2.1": (795.1, 57.7, -0.04, -0.04, 2785, -364, 152, -0.368, -0.0049, 0.039),
    # from [5], fit range lgE = 17 - 20
    "Sibyll2.1*": (795.1, 57.9, 0.06, 0.08, 2792, -394, 101, -0.360, -0.0019, 0.037),
    # from [4]
    "EPOS-LHC": (806.1, 55.6, 0.15, 0.83, 3284, -260, 132, -0.462, -0.0008, 0.059),
    # from [5], fit range lgE = 17 - 20
    "EPOS-LHC*": (806.1, 56.3, 0.47, 1.15, 3270, -261, 149, -0.459, -0.0005, 0.058),
    # from [4]
    "QGSJetII-04": (790.4, 54.4, -0.31, 0.24, 3738, -375, -21, -0.397, 0.0008, 0.046),
    # from [5], fit range lgE = 17 - 20
    "QGSJetII-04*": (790.4, 54.4, -0.33, 0.69, 3702, -369, 83, -0.396, 0.0010, 0.045),
}


class XmaxReader:
    def __init__(self, xmax_data_dir, xmax_data_files, model="QGSJetII-04"):

        self.empty = False
        if xmax_data_dir is None:
            self.empty = True
            return

        self.elongation_rate = DXMAX_PARAMS[model][1]
        xmax_files = data_files(data_dir=xmax_data_dir, glob_pattern=xmax_data_files)

        file_idx = []
        all_xmax = []
        for dst_file in xmax_files:
            try:
                nfile, nevents, zenith_angle, xmax = np.loadtxt(
                    dst_file, dtype=str, unpack=True
                )
            except Exception as ex:
                print(f"file: {dst_file}")
                print(ex)
                raise

            zenith_angle = np.array(zenith_angle, dtype=np.float32)
            cost = np.cos(zenith_angle * (np.pi / 180))
            xmax = np.array(xmax, dtype=np.float32) / cost
            file_idx.append(nfile)
            all_xmax.append(xmax)

        self.file_idxs = np.concatenate(file_idx)
        self.all_xmax = np.concatenate(all_xmax)

        # Energy bins in EeV
        self.en_bins = np.geomspace(1, 1000, 31)

    def _extract_idx(self, file_name):
        pattern = r"DAT(\d+)_"
        match = re.search(pattern, file_name.name)
        if match:
            return match.group(1)
        else:
            return ""

    def read_file(self, file_name):
        if self.empty:
            self._xmax0 = None
            self._en0 = None
            return

        file_name = Path(file_name)
        file_idx = self._extract_idx(file_name)
        en0 = self.en_bins[int(file_idx[-2:])]
        try:
            xmax0 = self.all_xmax[np.where(self.file_idxs == file_idx)[0]][0]
        except Exception:
            xmax0 = None

        self._xmax0 = xmax0
        self._en0 = en0

    def __call__(self, energies):
        # Getting xmax0 and scaling with energy according <Xmax>
        if (self._xmax0 == 0) or (self._xmax0 is None):
            return np.zeros(energies.shape[0], dtype=np.float32)
        else:
            return self._xmax0 + self.elongation_rate * np.log10(energies / self._en0)
