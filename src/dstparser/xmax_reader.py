import numpy as np
import re
from pathlib import Path
from dstparser.read_data import data_files


class XmaxReader:
    def __init__(self, xmax_data_dir, xmax_data_files):

        xmax_files = data_files(
            data_dir=xmax_data_dir, glob_pattern=f"**/{xmax_data_files}"
        )

        file_idx = []
        all_xmax = []
        for dst_file in xmax_files:
            nfile, nevents, zenith_angle, xmax = np.loadtxt(
                dst_file, dtype=str, unpack=True
            )

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

    def _read_file(self, file_name):
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
        # Getting xmax0 and scaling with energy
        # according <Xmax>
        if (self._xmax0 == 0) or (self._xmax0 is None):
            return np.zeros(energies.shape[0], dtype=np.float32)
        else:
            elong_rate = 45.8  # elongation rate for QGSJetII
            return self._xmax0 + elong_rate * np.log10(energies / self._en0)


class XmaxReaderEmpty(XmaxReader):
    def _read_file(self, file_name):
        self._xmax0 = None
        self._en0 = None
