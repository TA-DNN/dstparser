import numpy as np
from pathlib import Path
import re
import warnings
from abc import ABC, abstractmethod


from dstparser.xmax_reader.xmax_auger import DXMAX_PARAMS
from dstparser.paths import dstbank_root


class DstPathParser:
    def __init__(self):

        model_aliases = {
            "qgsjetii04": ["QGSII04", "QGSJETII04", "QGSJET04"],
            "eposlhc": ["EPOS", "EPOSLHC"],
            "sibyll21": ["SIBYLL21"],
            "sibyll23": [
                "SIBYLL",
                "SIBYLL23",
            ],
        }

        primary_aliases = {
            "proton": ["p", "proton"],
            "helium": ["he", "helium"],
            "nitrogen": ["n", "nitrogen"],
            "iron": ["fe", "iron"],
        }

        full_primary = {
            "proton": ["proton"],
            "helium": ["helium"],
            "nitrogen": ["nitrogen"],
            "iron": ["iron"],
        }

        self.model_aliases = model_aliases
        self.primary_aliases = primary_aliases
        self.full_primary = full_primary

        self.model_pattern_str = self._build_alias_pattern(self.model_aliases)
        self.primary_pattern_str = self._build_alias_pattern(self.primary_aliases)
        self.full_primary_pattern_str = self._build_alias_pattern(self.full_primary)

        self.model_primary_pattern = re.compile(
            rf"(?P<model>{self.model_pattern_str})_*(?P<primary>{self.primary_pattern_str})",
            re.IGNORECASE,
        )

        self.model_pattern = re.compile(
            rf"(?P<model>{self.model_pattern_str})",
            re.IGNORECASE,
        )

        self.primary_pattern = re.compile(
            rf"(?P<primary>{self.full_primary_pattern_str})",
            re.IGNORECASE,
        )

        self.period_pattern = re.compile(r"\d{6}_\d{6}")
        self.dat_pattern = re.compile(r"DAT(\d{4})(\d{2})")

    def _build_alias_pattern(self, alias_map):
        return "|".join(re.escape(a) for aliases in alias_map.values() for a in aliases)

    def _normalize(self, value, alias_map):
        value = value.lower()
        for canonical, aliases in alias_map.items():
            if value in (a.lower() for a in aliases):
                return canonical
        return None

    def parse(self, filepath):
        filepath = str(filepath)

        result = {
            "model": None,
            "primary": None,
            "period": None,
            "bin_id": None,
            "shower_id": None,
            "file_idx": None,
        }

        # "model_primary" in the same path component
        for part in Path(filepath).parts:
            match = self.model_primary_pattern.search(part)
            if match:
                result["model"] = self._normalize(
                    match.group("model"), self.model_aliases
                )
                result["primary"] = self._normalize(
                    match.group("primary"), self.primary_aliases
                )
                break

        # standalone model
        if result["model"] is None:
            match = self.model_pattern.search(filepath)
            if match:
                result["model"] = self._normalize(
                    match.group("model"), self.model_aliases
                )

        # standalone primary
        if result["primary"] is None:
            match = self.primary_pattern.search(filepath)
            if match:
                result["primary"] = self._normalize(
                    match.group("primary"), self.primary_aliases
                )

        # period
        match = self.period_pattern.search(filepath)
        if match:
            result["period"] = match.group(0)

        # DAT identifiers
        match = self.dat_pattern.search(filepath)
        if match:
            result["shower_id"] = int(match.group(1))
            result["bin_id"] = int(match.group(2))
            result["file_idx"] = match.group(1) + match.group(2)

        return result


class XmaxScaler:
    """
    Scale Xmax values to arbitrary energies using elongation rate.
    Given an Xmax value at a reference energy, scales it to target energies.
    If xmax0 is None or 0, generates random Xmax instead.
    """

    def __init__(self, model, xmax0, en0):
        """
        Args:
            model: Model name (e.g., 'QGSJetII-04'). Must be in DXMAX_PARAMS.
            xmax0: Xmax value at the reference energy (g/cm^2). Can be None or 0.
            en0: Reference energy (EeV)
        """

        # Map to existing tables
        xmax_models_map = {
            "eposlhc": "EPOS-LHC",
            "qgsjetii04": "QGSJetII-04",
            "sibyll21": "Sibyll2.1",
            "sibyll23": "Sibyll2.1",
        }
        # Model is not mapped if not in the xmax_models_map
        model = xmax_models_map[model]

        if model not in DXMAX_PARAMS:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(DXMAX_PARAMS.keys())}"
            )

        self.model = model
        params = DXMAX_PARAMS[model]
        self.D = params[1]  # Base elongation rate (per decade)
        self.delta = params[3]  # Mass-dependence correction
        self.xmax0 = xmax0
        self.en0 = en0

    def __call__(self, energies, mass):
        """
        Scale Xmax to arbitrary energies.

        Args:
            energies: Target energy in EeV (can be scalar or array)
            mass: Atomic mass number

        Returns:
            Xmax in g/cm^2 (same shape as energies)
        """
        if (self.xmax0 is None) or (self.xmax0 == 0):
            # Generate random Xmax for model and energy
            from dstparser.xmax_reader.xmax_auger import rand_xmax

            # Convert EeV to log10(eV)
            log10e = np.log10(energies) + 18
            return rand_xmax(log10e, mass, model=self.model)
        else:
            # Elongation rate with mass dependent correction
            eff_elongation = self.D + self.delta * np.log(np.maximum(mass, 1))

            # Scale from reference energy to target energies
            return self.xmax0 + eff_elongation * np.log10(energies / self.en0)


def std_ta_energy_grid():
    """Return standard TA energy grid (mids and edges) in EeV."""
    # Define 51 mid point in log10 space with 0.1 step
    mids = np.linspace(16, 21, 51)
    res = {"mids": 10 ** (mids - 18)}

    widths = mids[1:] - mids[:-1]
    left_edge = mids[0:1] - widths[0:1] / 2
    right_edge = mids[-1:] + widths[-1:] / 2
    # Edges of the bins
    edges = np.concatenate([left_edge, mids[:-1] + widths / 2, right_edge])
    calc_mids = (edges[1:] + edges[:-1]) / 2
    assert np.allclose(
        calc_mids, mids
    ), "Calculated midpoints do not match expected midpoints"
    res["edges"] = 10 ** (edges - 18)
    # ta_indx is mapping from TA indexing (including 25) [0, 25] for (1, 20.5)
    # [26, 39] (including 39) for (16.6, 17.9)
    res["ta_indx"] = np.concatenate([np.arange(20, 46), np.arange(6, 21)])

    return res


class XmaxReader(ABC):
    """
    Base class for Xmax readers.
    Reads file, determines xmax0 and en0, and creates an XmaxScaler.
    """

    def __init__(self):
        self.scaler = None
        self.path_parser = DstPathParser()

        # Standard TA energy grid
        std_grid = std_ta_energy_grid()
        self.energy_bin_centers = std_grid["mids"]
        self.ta_indx = std_grid["ta_indx"]

    def _get_energy_bin_center(self, bin_id):
        """
        Get energy at bin center from bin_id.

        Args:
            bin_id: Bin ID (0-based index)

        Returns:
            Energy at bin center in EeV
        """
        if bin_id is None:
            return None
        # Map to original index
        orig_indx = self.ta_indx[bin_id]
        return self.energy_bin_centers[orig_indx]

    @abstractmethod
    def _get_xmax0(self, parsed_info):
        """
        Get xmax0 for the given file.
        Must be implemented by subclasses.

        Args:
            parsed_info: Dict returned by path_parser.parse()

        Returns:
            xmax0 value (can be None or 0 if not available)
        """
        pass

    def read_file(self, filepath):
        """
        Read file and initialize XmaxScaler.

        Args:
            filepath: Path to the file
        """
        filepath = Path(filepath)

        # Parse filepath to extract metadata
        parsed_info = self.path_parser.parse(filepath)

        # Get model from parsed info
        model = parsed_info["model"]
        if model is None:
            raise ValueError(f"Could not extract model from filepath: {filepath}")

        # Get xmax0 from subclass implementation
        xmax0 = self._get_xmax0(parsed_info)

        # Get energy at bin center
        en0 = self._get_energy_bin_center(parsed_info["bin_id"])
        if en0 is None:
            raise ValueError(
                f"Could not determine energy bin center from filepath: {filepath}"
            )

        # Create scaler
        self.scaler = XmaxScaler(model=model, xmax0=xmax0, en0=en0)

    def __call__(self, energies, mass):
        """
        Scale Xmax to arbitrary energies.

        Args:
            energies: Energy in EeV (can be scalar or array)
            mass: Atomic mass number

        Returns:
            Xmax in g/cm^2
        """
        if self.scaler is None:
            raise RuntimeError("Must call read_file() before scaling Xmax")

        return self.scaler(energies, mass)


class XmaxReaderTxt(XmaxReader):
    """Xmax reader that loads info from text files."""

    def __init__(self, data_dir, glob_pattern="**/DAT*_xmax.txt"):
        """
        Args:
            data_dir: Directory containing xmax text files
            glob_pattern: Pattern to match xmax files
        """
        super().__init__()

        self.empty = False
        if data_dir is None:
            self.empty = True
            return

        xmax_files = sorted(Path(data_dir).glob(glob_pattern))

        file_idx = []
        all_xmax = []

        # Ignore warning from numpy.loadtxt
        warnings.filterwarnings(
            "ignore", "Input line 1 contained no data and will not be counted"
        )

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

    def _get_xmax0(self, parsed_info):
        """Get xmax0 from loaded text files."""
        if self.empty:
            return None

        file_idx = parsed_info["file_idx"]

        try:
            xmax0 = self.all_xmax[np.where(self.file_idxs == file_idx)[0]][0]
            return xmax0
        except Exception:
            return None


class XmaxReaderDatabase(XmaxReader):
    """Xmax reader that uses the HDF5 database."""

    def __init__(self, model_db_map):
        """
        Args:
            model_db_map: dict mapping model names to HDF5 file paths
                         e.g., {'qgsii04': 'xmax_qgsii04.h5', ...}
        """
        super().__init__()

        self.model_db_map = {
            model: Path(db_file) for model, db_file in model_db_map.items()
        }
        self._cache = {}  # (model, primary, period) -> DataFrame

        # Store current parsed info for _get_xmax0
        self._current_parsed_info = None

    def _load_table(self, model, primary, period):
        """
        Lazy load a table from HDF5 and cache it.

        Args:
            model: Model name (e.g., 'qgsjetii04')
            primary: Primary particle (e.g., 'proton')
            period: Time period (e.g., '080417_160603')

        Returns:
            DataFrame with MultiIndex (bin_id, shower_id)
        """
        import pandas as pd

        cache_key = (model, primary, period)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if model not in self.model_db_map:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(self.model_db_map.keys())}"
            )

        db_file = self.model_db_map[model]
        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")

        key = f"{model}_{primary}_{period}"
        try:
            df = pd.read_hdf(db_file, key=key)
            self._cache[cache_key] = df
            return df
        except KeyError:
            raise KeyError(f"Table not found in {db_file}: {key}")

    def _get_xmax0(self, parsed_info):
        """Get xmax0 from HDF5 database."""
        model = parsed_info.get("model")
        primary = parsed_info.get("primary")
        period = parsed_info.get("period")
        bin_id = parsed_info.get("bin_id")
        shower_id = parsed_info.get("shower_id")

        if None in (model, primary, period, bin_id, shower_id):
            print(f"Warning: Missing required info in parsed_info: {parsed_info}")
            return None

        try:
            # Load the appropriate table (lazy)
            df = self._load_table(model, primary, period)
            # Look up the specific shower
            return float(df.loc[(bin_id, shower_id), "xmax"])
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not get Xmax from database: {e}")
            return None


def create_xmax_reader(source, **kwargs):
    """
    Factory for XmaxReader.

    Args:
        source:
            - str or Path → directory with text files (XmaxReaderTxt)
            - dict        → model_db_map for HDF5 databases (XmaxReaderDatabase)
        **kwargs:
            Passed through to the underlying reader constructor.
            Example: glob_pattern for XmaxReaderTxt.

    Returns:
        XmaxReader instance
    """
    if isinstance(source, (str, Path)):
        return XmaxReaderTxt(data_dir=source, **kwargs)

    if isinstance(source, dict):
        return XmaxReaderDatabase(model_db_map=source)

    raise TypeError(
        "create_xmax_reader expects str/Path (text files) or dict (database map)"
    )


if __name__ == "__main__":

    # filepath = f"{dstbank_root}/INR_group/cluster82/grisha/tasdmc_EPOS_p/p2/DAT000623.corsika77420.EPOS.tar.gz.spctr1.1745.noCuts.dst.gz"
    # filepath = f"{dstbank_root}/tasdmc_dstbank/qgsii04nitrogen/160604_240422/Em1_bsdinfo/XXXX22/DAT003022_gea.rufldf.dst.gz"
    filepath = f"{dstbank_root}/tasdmc_dstbank/qgsii04iron/160604_240422/Em1_bsdinfo/XXXX13/DAT006413_gea.rufldf.dst.gz"

    rr = XmaxReaderTxt(
        f"{dstbank_root}/tasdmc_dstbank/qgsii04iron/160604_240422"
    )
    rr.read_file(filepath)

    print(rr(np.array([10, 11, 12]), np.array([56, 56, 56])))

    rr = XmaxReaderDatabase(
        {
            "qgsjetii04": "/home/antonpr/ml/tasks/2026/02/04/qgsii04_xmax_db.h5",
            "eposlhc": "/home/antonpr/ml/tasks/2026/02/04/eposlhc_xmax_db.h5",
        }
    )

    rr.read_file(filepath)

    print(rr(np.array([10, 11, 12]), np.array([56, 56, 56])))
