"""Explicit metadata for campaigns whose DAT suffix is not an energy code."""

import csv
from functools import lru_cache
from pathlib import Path
import re
import numpy as np


@lru_cache(maxsize=16)
def _read_manifest(path, mtime, size):
    with open(path, newline="") as f:
        first = f.readline()
        f.seek(0)
        rows = list(csv.DictReader(f, delimiter="\t" if "\t" in first else ","))
    result = {}
    for r in rows:
        if r["tag"] in result:
            raise ValueError("Duplicate campaign tag")
        float(r["loge"])
        float(r["theta_deg"])
        result[r["tag"]] = r
    return result


def campaign_row(filename, manifest):
    p = Path(manifest).resolve(strict=True)
    s = p.stat()
    m = re.search(r"(DAT\d{6})(?:[_.]|$)", Path(filename).name)
    if not m:
        raise ValueError("Expected full six-digit DAT identity")
    try:
        return _read_manifest(str(p), s.st_mtime_ns, s.st_size)[m[1]]
    except KeyError as e:
        raise ValueError(f"{m[1]} absent from campaign manifest") from e


def resolve_manifest(filename, manifest):
    """Recognize the campaign grid by its schema, not by a DAT suffix guess."""
    if manifest is False:
        return None
    if manifest is not None:
        return Path(manifest)
    for parent in Path(filename).resolve().parents:
        candidate = parent / "grid.tsv"
        if not candidate.is_file():
            continue
        with candidate.open() as f:
            fields = set(f.readline().strip().split("\t"))
        if {"tag", "loge", "theta_deg", "model"}.issubset(fields):
            campaign_row(filename, candidate)
            return candidate
    return None


def add_campaign_metadata(data, filename, manifest):
    r = campaign_row(filename, manifest)
    n = (
        len(data["hit_offsets"]) - 1
        if "hit_offsets" in data
        else data["detector_states"].shape[0]
    )
    data["id_corsika_shower"] = np.full(
        n, int(r.get("runnr", r["tag"][3:])), dtype=np.int64
    )
    # Explicit logE*10 convention is independent of historical DAT suffix codes.
    data["id_energy_bin"] = np.full(n, round(float(r["loge"]) * 10), dtype=np.int32)
    data["shower_log10_energy"] = np.full(n, float(r["loge"]), dtype=np.float32)
    data["shower_zenith_deg"] = np.full(n, float(r["theta_deg"]), dtype=np.float32)
    models = {"q": 0, "e": 1, "qgsjetii04": 0, "eposlhc": 1}
    if r["model"].lower() not in models:
        raise ValueError("Unknown campaign model code")
    data["id_campaign_model"] = np.full(n, models[r["model"].lower()], dtype=np.int16)
    return data


def init_xmax(reader, filename, manifest):
    if manifest is None:
        reader.read_file(filename)
    else:
        if not hasattr(reader, "read_campaign_file"):
            raise ValueError(
                "Xmax reader must support read_campaign_file for explicit campaign metadata"
            )
        reader.read_campaign_file(filename, campaign_row(filename, manifest))
