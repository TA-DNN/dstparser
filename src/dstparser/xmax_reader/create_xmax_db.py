from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from dstparser.paths import dstbank_root, training_data_root


model_prefix_map_global = {
    "eposlhc_": "eposlhc",
    "qgsii04": "qgsjetii04",
    "sibyll": "sibyll23",
}


def nested_dict():
    return defaultdict(nested_dict)


def to_sorted_dict(d):
    if isinstance(d, defaultdict):
        # first convert children
        d = {k: to_sorted_dict(v) for k, v in d.items()}
    # if keys are integers, sort them
    if all(isinstance(k, int) for k in d.keys()):
        d = dict(sorted(d.items()))
    return d


def create_xmax_db_dict(root, model_prefix):

    root = Path(root)
    res = nested_dict()

    for model_dir in root.glob(f"{model_prefix}*"):
        if not model_dir.is_dir():
            continue

        model = model_prefix_map_global[model_prefix]
        primary = model_dir.name.replace(model_prefix, "")

        for period_dir in model_dir.iterdir():
            if not period_dir.is_dir() or period_dir.name == "long":
                continue

            period = period_dir.name
            emdir = period_dir / "Em1_bsdinfo"
            if not emdir.is_dir():
                continue

            for f in emdir.glob("DAT*_xmax.txt"):
                try:
                    data = np.loadtxt(
                        f,
                        comments="#",
                        dtype=[
                            ("id", "i4"),  # xxxxyy
                            ("ngenerated", "i4"),
                            ("zenith", "f4"),
                            ("xmax_vert", "f4"),
                        ],
                    )
                except ValueError:
                    continue

                if data.size == 0:
                    continue

                zenith = data["zenith"]
                cost = np.cos(np.deg2rad(zenith))
                xmax = data["xmax_vert"] / cost

                for i in range(data.shape[0]):
                    raw_id = int(data["id"][i])

                    shower_id = raw_id // 100
                    bin_id = raw_id % 100

                    res[model][primary][period][bin_id][shower_id] = {
                        "ngenerated": int(data["ngenerated"][i]),
                        "zenith_angle": float(zenith[i]),
                        "xmax": float(xmax[i]),
                    }

    return to_sorted_dict(res)


def save_xmax_db_hdf5(res, h5_filename):
    """
    Save nested dict: res[model][primary][period][bin_id][shower_id] -> {ngenerated, zenith_angle, xmax}
    into a **single compressed HDF5 file** with one table per (model, primary, period).
    MultiIndex: (bin_id, shower_id) for fast lookups.
    """
    h5_filename = Path(h5_filename)

    with pd.HDFStore(h5_filename, mode="w", complevel=9, complib="zlib") as store:
        for model, primaries in res.items():
            for primary, periods in primaries.items():
                for period, bins in periods.items():
                    rows = []
                    for bin_id, showers in bins.items():
                        for shower_id, d in showers.items():
                            rows.append(
                                {
                                    "bin_id": int(bin_id),
                                    "shower_id": int(shower_id),
                                    "ngenerated": int(d["ngenerated"]),
                                    "zenith_angle": float(d["zenith_angle"]),
                                    "xmax": float(d["xmax"]),
                                }
                            )
                    df = pd.DataFrame(rows)
                    df = df.astype(
                        {
                            "bin_id": "int32",
                            "shower_id": "int32",
                            "ngenerated": "int32",
                            "zenith_angle": "float32",
                            "xmax": "float32",
                        }
                    )
                    # Set MultiIndex for fast lookups
                    df = df.set_index(["bin_id", "shower_id"])
                    # Compose a unique key for this table
                    key = f"{model}_{primary}_{period}"
                    store.put(key, df, format="table", data_columns=True)


def load_xmax_db_hdf5(h5_filename, model, primary, period):
    """
    Load a single (model, primary, period) table from HDF5 into a MultiIndex DataFrame
    """
    key = f"{model}_{primary}_{period}"
    df = pd.read_hdf(h5_filename, key=key)
    return df


def create_xmax_db(root, model_prefix, xmax_db):
    res = create_xmax_db_dict(root, model_prefix)
    Path(xmax_db).parent.mkdir(parents=True, exist_ok=True)
    save_xmax_db_hdf5(res, xmax_db)


def task_create_db():
    root = f"{dstbank_root}/tasdmc_dstbank"

    model_prefixes = ["qgsii04", "eposlhc_"]
    output_dir = f"{training_data_root}/dnn_training_data/2026/02/xmax_db"
    output_dir = Path(output_dir)

    for model_prefix in model_prefixes:

        model_name = model_prefix_map_global[model_prefix]
        output_db = output_dir / f"{model_name}_xmax_db.h5"
        create_xmax_db(root, model_prefix, output_db)


def task_test_read():

    root = f"{dstbank_root}/tasdmc_dstbank"
    model_prefix = "eposlhc_"
    xmax_db = Path(__file__).parent / "eposlhc_xmax_db.h5"

    df = load_xmax_db_hdf5(xmax_db, "qgsii04", "proton", "080417_160603")

    bin_id = 1
    shower_id = 879
    xmax = df.loc[(bin_id, shower_id), "xmax"]
    zenith = df.loc[(bin_id, shower_id), "zenith_angle"]
    ngen = df.loc[(bin_id, shower_id), "ngenerated"]

    print(f"xmax={xmax}, zenith={zenith}, ngen={ngen}")


if __name__ == "__main__":

    task_create_db()
