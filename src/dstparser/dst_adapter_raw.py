"""Read a TA SD DST directly with dstio (C, no ASCII dump) into the vlen layout plus raw readout fields.

Event selection = sditerator_cppanalysis_add_standard_recon_v2.cpp (the ASCII path of dst_adapter_vlen):
``rusdraw.nofwf > 0 and (tasdcalibev.trgPos >= 101 or rusdraw.event_code == 0)`` -- MC DSTs also hold the
non-triggered thrown events (nofwf == 0, no rufptn/rufldf banks), which this skips.
``id_event`` = index of the event among the selected events of the DST (same as the ASCII path).

Fields with the same name as in dst_adapter_vlen have the same definition (std_recon_*, energy, shower_axis,
shower_core, mass_number, hits of isgood > 2, time_traces = fadc / vem, badsd). Added:
  per hit (isgood > 2):    vem [nhit, 2] (FADC counts per VEM), ped [nhit, 2] (rufptn pedestal per bin)
  NOT stored: time_traces (VEM) -- recompute as fadc / np.repeat(vem, nfold, axis=0)[:, :, None] (time_traces())
  per waveform of a hit:   fadc [nwf, 2, 128] int16 (raw counts, pedestal NOT subtracted), clkcnt, mclkcnt,
                           wf_pchped, wf_mip [nwf, 2] (rusdraw calibration of the waveform's monitoring cycle)
  isolated hits (isgood <= 2), no waveforms: iso_detector_ids, iso_status, iso_arrival_times, iso_pulse_area,
                           iso_detector_positions, iso_nfold, iso_clkcnt (first window), iso_offsets
  waveforms of no hit:     orphan_detector_ids, orphan_clkcnt, orphan_offsets
Layer index 0 = lower, 1 = upper everywhere (as rufptn / rusdraw).
"""
import dstio
import h5py
import numpy as np

from dstparser.join_vlen_data import append_to_hdf5

# SD origin with respect to the CLF origin in the CLF frame, 1200 m units (as dst_adapter_vlen.standard_recon)
CLF_ORIGIN_X, CLF_ORIGIN_Y = -12.2435, -16.4406


def is_selected(ev):
    """The ASCII program's event condition."""
    r = ev["rusdraw"]
    trg = ev["tasdcalibev"]["trgPos"] >= 101 if "tasdcalibev" in ev else False
    return r["nofwf"] > 0 and (trg or r["event_code"] == 0)


def _axis(theta_deg, phi_deg):
    th, ph = np.deg2rad(theta_deg + 0.5), np.deg2rad(phi_deg) + np.pi
    return [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]


def _event_fields(ev):
    r, p, g, l = ev["rusdraw"], ev["rufptn"], ev["rusdgeom"], ev["rufldf"]
    th = np.deg2rad(g["theta"][2])
    out = {
        "std_recon_yymmdd": r["yymmdd"], "std_recon_hhmmss": r["hhmmss"], "std_recon_usec": r["usec"],
        "std_recon_nofwf": r["nofwf"], "std_recon_nsd": p["nstclust"], "std_recon_nsclust": p["nsclust"],
        "std_recon_nhits": p["nhits"], "std_recon_nborder": p["nborder"], "std_recon_qtot": p["qtot"],
        "std_recon_energy": l["energy"][0], "std_recon_ldf_scale": l["sc"][0], "std_recon_ldf_scale_err": l["dsc"][0],
        "std_recon_ldf_chi2": l["chi2"][0], "std_recon_ldf_ndof": l["ndof"][0],
        "std_recon_shower_core": [l["xcore"][0] + CLF_ORIGIN_X, l["ycore"][0] + CLF_ORIGIN_Y],
        "std_recon_shower_core_err": [l["dxcore"][0], l["dycore"][0]], "std_recon_s800": l["s800"][0],
        "std_recon_shower_axis": _axis(g["theta"][2], g["phi"][2]),
        "std_recon_shower_axis_err": np.sqrt(g["dtheta"][2] ** 2 + np.sin(th) ** 2 * g["dphi"][2] ** 2),
        "std_recon_geom_chi2": g["chi2"][2], "std_recon_geom_ndof": g["ndof"][2],
        "std_recon_curvature": g["a"], "std_recon_curvature_err": g["da"],
        "std_recon_border_distance": l["bdist"], "std_recon_border_distance_tshape": l["tdist"],
    }
    if "rusdmc" in ev:
        m = ev["rusdmc"]
        out["energy"] = m["energy"]                                              # EeV
        out["mass_number"] = 1 if m["parttype"] == 14 else m["parttype"] // 100
        out["shower_axis"] = [np.sin(m["theta"]) * np.cos(m["phi"] + np.pi),
                              np.sin(m["theta"]) * np.sin(m["phi"] + np.pi), np.cos(m["theta"])]
        out["shower_core"] = list(m["corexyz"])                                  # cm
    return out


def read_dst_raw(dst_file, keep=None):
    """Selected events of one DST as a dict of flat arrays with offsets (see module docstring).

    keep: optional callable(id_event, ev) -> bool applied after the selection, before any array is built
          (e.g. sampling and loose cuts); ev is the dstio event dict.
    Returns None if no event is kept.
    """
    ev_rows, ids = [], []
    hit_cols = {k: [] for k in ("arrival_times", "pulse_area", "detector_positions", "status", "nfold",
                                "detector_ids", "vem", "ped")}
    wf_cols = {k: [] for k in ("fadc", "clkcnt", "mclkcnt", "wf_pchped", "wf_mip")}
    iso_cols = {k: [] for k in ("iso_detector_ids", "iso_status", "iso_arrival_times", "iso_pulse_area",
                                "iso_detector_positions", "iso_nfold", "iso_clkcnt")}
    orph_cols = {"orphan_detector_ids": [], "orphan_clkcnt": []}
    n_hit, n_wf, n_iso, n_orph, hit_wf_counts, badsd, n_bad = [0], [0], [0], [0], [], [], [0]

    with dstio.open(str(dst_file)) as dst:
        i_sel = -1
        for ev in dst:
            if not is_selected(ev):
                continue
            i_sel += 1
            if keep is not None and not keep(i_sel, ev):
                continue
            r, p = ev["rusdraw"], ev["rufptn"]
            ev_rows.append(_event_fields(ev))
            ids.append(i_sel)

            isgood = np.asarray(p["isgood"])
            nfold = np.asarray(p["nfold"])
            wfi = np.asarray(p["wfindex"])
            clk = np.asarray(r["clkcnt"])
            good = isgood > 2
            owned = np.zeros(r["nofwf"], bool)
            for a, n in zip(wfi, nfold):
                owned[a:a + n] = True
            wf_sel = np.concatenate([np.arange(a, a + n) for a, n in zip(wfi[good], nfold[good])]) if good.any() \
                else np.zeros(0, int)

            hit_cols["arrival_times"].append(np.asarray(p["reltime"])[good])
            hit_cols["pulse_area"].append(np.asarray(p["pulsa"])[good])
            hit_cols["detector_positions"].append(np.asarray(p["xyzclf"])[good])
            hit_cols["status"].append(isgood[good])
            hit_cols["nfold"].append(nfold[good])
            hit_cols["detector_ids"].append(np.asarray(p["xxyy"])[good])
            hit_cols["vem"].append(np.asarray(p["vem"])[good])
            hit_cols["ped"].append(np.asarray(p["ped"])[good])
            hit_wf_counts.append(nfold[good])

            wf_cols["fadc"].append(np.asarray(r["fadc"])[wf_sel])
            wf_cols["clkcnt"].append(clk[wf_sel])
            wf_cols["mclkcnt"].append(np.asarray(r["mclkcnt"])[wf_sel])
            wf_cols["wf_pchped"].append(np.asarray(r["pchped"])[wf_sel])
            wf_cols["wf_mip"].append(np.asarray(r["mip"])[wf_sel])

            iso = ~good
            iso_cols["iso_detector_ids"].append(np.asarray(p["xxyy"])[iso])
            iso_cols["iso_status"].append(isgood[iso])
            iso_cols["iso_arrival_times"].append(np.asarray(p["reltime"])[iso])
            iso_cols["iso_pulse_area"].append(np.asarray(p["pulsa"])[iso])
            iso_cols["iso_detector_positions"].append(np.asarray(p["xyzclf"])[iso])
            iso_cols["iso_nfold"].append(nfold[iso])
            iso_cols["iso_clkcnt"].append(clk[wfi[iso]])

            orph_cols["orphan_detector_ids"].append(np.asarray(r["xxyy"])[~owned])
            orph_cols["orphan_clkcnt"].append(clk[~owned])

            bad = ev["bsdinfo"]["xxyyout"] if "bsdinfo" in ev else ()
            badsd.append(np.asarray(bad, dtype=np.int64))

            n_hit.append(n_hit[-1] + int(good.sum()))
            n_wf.append(n_wf[-1] + wf_sel.size)
            n_iso.append(n_iso[-1] + int(iso.sum()))
            n_orph.append(n_orph[-1] + int((~owned).sum()))
            n_bad.append(n_bad[-1] + len(bad))

    if not ev_rows:
        return None

    data = {k: np.asarray([row[k] for row in ev_rows]) for k in ev_rows[0]}
    data["id_event"] = np.asarray(ids, dtype=np.int32)
    for cols in (hit_cols, wf_cols, iso_cols, orph_cols):
        for k, v in cols.items():
            data[k] = np.concatenate(v)
    # per-hit / per-waveform columns in compact types (event-level columns stay float64)
    data["fadc"] = data["fadc"].astype(np.int16)
    for k in ("clkcnt", "mclkcnt", "iso_clkcnt", "orphan_clkcnt", "detector_ids", "iso_detector_ids",
              "orphan_detector_ids", "wf_pchped"):
        data[k] = data[k].astype(np.int32)
    for k in ("status", "nfold", "iso_status", "iso_nfold"):
        data[k] = data[k].astype(np.int16)
    for k in ("arrival_times", "pulse_area", "detector_positions", "vem", "ped", "wf_mip", "iso_arrival_times",
              "iso_pulse_area", "iso_detector_positions"):
        data[k] = data[k].astype(np.float32)
    data["hit_offsets"] = np.asarray(n_hit, dtype=np.int64)
    data["tt_offsets"] = np.asarray(n_wf, dtype=np.int64)
    data["hit_tt_offsets"] = np.concatenate(([0], np.cumsum(np.concatenate(hit_wf_counts)))).astype(np.int64)
    data["iso_offsets"] = np.asarray(n_iso, dtype=np.int64)
    data["orphan_offsets"] = np.asarray(n_orph, dtype=np.int64)
    data["badsd"] = np.concatenate(badsd)
    data["badsd_offsets"] = np.asarray(n_bad, dtype=np.int64)
    return data


def loose_cuts(ev):
    """Loose superset of the TA spectrum cuts, on the bank values of one selected event:
    nstclust >= 4, std zenith (rusdgeom theta[2] + 0.5, as std_recon_shower_axis) < 55 deg, finite std energy > 0.
    NO ndof requirement: the TA cuts use chi2 itself when ndof == 0 (40 % of TA-cut data events have geometry
    ndof[2] <= 0) -- requiring ndof > 0 would not be a superset."""
    p, g, l = ev["rufptn"], ev["rusdgeom"], ev["rufldf"]
    e = l["energy"][0]
    return p["nstclust"] >= 4 and g["theta"][2] + 0.5 < 55.0 and np.isfinite(e) and e > 0


class FractionSelector:
    """keep() for read_dst_raw: a fixed random fraction of the selected events of one DST, then optional cuts.

    The uniform number of every selected event is drawn in DST order from a generator seeded by `seed_key`
    (e.g. (sample, period, bin, shower)), BEFORE the cuts -> the fraction-f subsample is reproducible and does
    not depend on the cuts. Picklable (for multiprocessing).
    """

    def __init__(self, fraction, seed_key, cuts=loose_cuts):
        self.fraction, self.seed_key, self.cuts = fraction, tuple(seed_key), cuts
        self._rng = None

    def __call__(self, id_event, ev):
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.SeedSequence(list(self.seed_key)))
        if self._rng.random() >= self.fraction:
            return False
        return self.cuts is None or bool(self.cuts(ev))


def dsts_to_hdf5_raw(jobs, ofile):
    """Convert DSTs into one HDF5 file (appending, offsets shifted by append_to_hdf5).

    jobs: list of (dst_path, keep, event_constants) -- keep as in read_dst_raw (or None), event_constants a dict
    of scalars written as per-event columns (e.g. id_energy_bin, id_corsika_shower, period, sampling fraction).
    Returns (number of DSTs read, number of events written).
    """
    n_ev = 0
    with h5py.File(ofile, "a") as f:
        for path, keep, consts in jobs:
            data = read_dst_raw(path, keep=keep)
            if data is None:
                continue
            n = len(data["id_event"])
            for k, v in consts.items():
                data[k] = np.full(n, v)
            append_to_hdf5(f, data, compression="lzf")
            n_ev += n
    return len(jobs), n_ev


def time_traces(fadc, vem, nfold):
    """VEM traces as in dst_adapter_vlen (pedestal not subtracted): fadc / vem of the waveform's hit."""
    return (fadc / np.repeat(vem, nfold, axis=0)[:, :, None]).astype(np.float32)


# Raw-only fields (readout, isolated hits, orphan waveforms): not part of the dst_adapter_vlen layout
RAW_ONLY = ("fadc", "clkcnt", "mclkcnt", "wf_mip", "wf_pchped", "ped", "vem")


def to_vlen(src, dst, xmax=None):
    """Write a raw-layout file as a dst_adapter_vlen-layout file (what srecog_dataprep recipes read): time_traces()
    added, raw-only / iso_* / orphan_* fields dropped. `xmax` [n_events] (g/cm2) is added when given (the raw adapter
    does not know the CORSIKA Xmax; production files take it from the MC shower DB) (2026-10-19)."""
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        for k in f:
            if k in RAW_ONLY or k.startswith(("iso_", "orphan_")):
                continue
            g.create_dataset(k, data=f[k][:])
        g.create_dataset("time_traces", data=time_traces(f["fadc"][:], f["vem"][:], f["nfold"][:]), compression="lzf")
        if xmax is not None:
            g.create_dataset("xmax", data=np.asarray(xmax, np.float64))
