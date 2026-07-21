"""TAx4 adapter for the vlen (flat-arrays-with-offsets) format.

Copy of dst_adapter_vlen.py, changed only where TAx4 differs. shower_params
(truth) and detector_readings_flat (the DNN inputs) are reused verbatim from
dst_adapter_vlen -- once the TAx4 parser supplies a reconstructed rufptn_.nfold
and xxyy-swapped hits (see dst_parsers_tax4_vlen), the hits/waveforms dicts have
exactly the shape those functions expect. Only standard_recon needs a TAx4
variant, because TAx4 emits a smaller reconstruction field set.
"""

import numpy as np
from pathlib import Path

from dstparser.dst_reader import read_dst_file_tax4_std_recon_nfold
from dstparser.dst_parsers_tax4_vlen import parse_dst_string
from dstparser.dst_adapter_vlen import shower_params, detector_readings_flat


def standard_recon(data, events, include_fixed_curve_fit=False):
    """TAx4 standard reconstruction.

    Same formulas as dst_adapter_vlen.standard_recon, scoped to the fields TAx4
    actually emits (see dst_parsers_tax4_vlen.parse_event): one LDF fit, the
    free-curvature geometry fit (the primary reco direction), and the border
    cuts. The fixed-curvature geometry fit is available too and, as in TA-SD,
    is gated behind include_fixed_curve_fit (default off, for key-set parity
    with TA-SD's default vlen output). TAx4 has NO combined LDF+geometry fit and
    none of rufptn_.nsclust/nhits/nborder/qtot, so those TA-SD outputs are
    omitted (not defaulted -- the source fields do not exist).

    [established, verified 2026-07-20, method: field indices confirmed against
     real TAx4 DST dumps; formulas copied unchanged from
     dst_adapter_vlen.standard_recon]

    UNVERIFIED for TAx4 (copied from the TA-SD convention, flagged not assumed):
    clf_origin_x/y and the "+0.5 deg" zenith correction were derived for
    TA-SD's array; whether they apply unchanged to TAx4's array geometry is not
    checked. They affect ONLY std_recon_shower_core and the reco shower-axis
    vectors -- not truth (shower_params) nor the DNN inputs
    (detector_readings_flat).
    """
    # SD origin with respect to CLF origin in CLF frame, in [1200m] units
    clf_origin_x = -12.2435
    clf_origin_y = -16.4406

    data["std_recon_yymmdd"] = events["rusdraw_.yymmdd"].astype(np.int64)
    data["std_recon_hhmmss"] = events["rusdraw_.hhmmss"].astype(np.int64)
    data["std_recon_usec"] = events["rusdraw_.usec"].astype(np.int64)
    # Number of waveforms for event for all detectors
    data["std_recon_nofwf"] = events["rusdraw_.nofwf"].astype(np.int32)
    # number of SDs in space-time cluster
    data["std_recon_nsd"] = events["rufptn_.nstclust"].astype(np.int32)

    # -- LDF fit --
    # energy reconstructed by the standard energy estimation table [EeV]
    data["std_recon_energy"] = events["rufldf_.energy[0]"]
    # reconstructed scale of the Lateral Distribution Function (LDF) fit [VEM m-2]
    data["std_recon_ldf_scale"] = events["rufldf_.sc[0]"]
    # uncertainty of the scale [VEM m-2]
    data["std_recon_ldf_scale_err"] = events["rufldf_.dsc[0]"]
    # chi-square of the LDF fit
    data["std_recon_ldf_chi2"] = events["rufldf_.chi2[0]"]
    # number of degrees of freedom of the LDF fit (= n - 3)
    data["std_recon_ldf_ndof"] = events["rufldf_.ndof[0]"]
    # core position (x, y) from the LDF fit in CLF coordinate, in 1200m units
    data["std_recon_shower_core"] = np.array(
        [
            events["rufldf_.xcore[0]"] + clf_origin_x,
            events["rufldf_.ycore[0]"] + clf_origin_y,
        ]
    ).transpose(1, 0)
    # uncertainty of the core position (x, y), in 1200m units
    data["std_recon_shower_core_err"] = np.array(
        [
            events["rufldf_.dxcore[0]"],
            events["rufldf_.dycore[0]"],
        ]
    ).transpose(1, 0)
    # S800 (particle density at 800 m from the shower axis) [VEM m-2]
    data["std_recon_s800"] = events["rufldf_.s800[0]"]

    # -- Fixed-curvature geometry fit (off by default, as in TA-SD) --
    if include_fixed_curve_fit:
        # chi-square of the geometry fit (fixed curvature)
        data["std_recon_geom_chi2_fixed_curve"] = events["rusdgeom_.chi2[1]"]
        # number of degrees of freedom of the geometry fit (= n - 5)
        data["std_recon_geom_ndof_fixed_curve"] = events["rusdgeom_.ndof[1]"]
        # 3-d unit vector of the arrival direction (pointing back to the source)
        # "+0.5" is a correction for zenith angle.
        theta = np.deg2rad(events["rusdgeom_.theta[1]"] + 0.5)
        phi = np.deg2rad(events["rusdgeom_.phi[1]"]) + np.pi
        data["std_recon_shower_axis_fixed_curve"] = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
        ).transpose()
        # uncertainty of the pointing direction [degree]
        theta = np.deg2rad(events["rusdgeom_.theta[1]"])
        dtheta = events["rusdgeom_.dtheta[1]"]
        dphi = events["rusdgeom_.dphi[1]"]
        data["std_recon_shower_axis_err_fixed_curve"] = np.sqrt(
            dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
        )

    # -- Free-curvature geometry fit (the primary reco direction) --
    # "+0.5" is a correction for zenith angle.
    theta = np.deg2rad(events["rusdgeom_.theta[2]"] + 0.5)
    phi = np.deg2rad(events["rusdgeom_.phi[2]"]) + np.pi
    data["std_recon_shower_axis"] = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
    ).transpose()
    # uncertainty of the pointing direction [degree]
    theta = np.deg2rad(events["rusdgeom_.theta[2]"])
    dtheta = events["rusdgeom_.dtheta[2]"]
    dphi = events["rusdgeom_.dphi[2]"]
    data["std_recon_shower_axis_err"] = np.sqrt(
        dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
    )
    # chi-square of the geometry fit (free curvature)
    data["std_recon_geom_chi2"] = events["rusdgeom_.chi2[2]"]
    # number of degrees of freedom of the geometry fit (= n - 6)
    data["std_recon_geom_ndof"] = events["rusdgeom_.ndof[2]"]
    # curvature parameter `a` of the geometry fit + its uncertainty
    data["std_recon_curvature"] = events["rusdgeom_.a"]
    data["std_recon_curvature_err"] = events["rusdgeom_.da"]

    # -- Border cuts --
    # distance b/w the reconstructed core and the array edge [1200m units]
    data["std_recon_border_distance"] = events["rufldf_.bdist"]
    # distance to the T-shape sub-array edge [1200m units]
    data["std_recon_border_distance_tshape"] = events["rufldf_.tdist"]

    return data


def parse_dst_file_tax4_vlen(
    dst_file,
    xmax_reader=None,
    add_shower_params=True,
    add_standard_recon=True,
    add_badsd=True,
    config=None,
):
    """TAx4 analogue of parse_dst_file_vlen: build the same flat-with-offsets
    vlen data dict from a TAx4 MC DST file.

    Reads via the locally-rebuilt reader (read_dst_file_tax4_std_recon_nfold),
    which emits only triggered events (rusdraw_.nofwf>0) with waveforms AND the
    rufptn_.nfold column -- matching TA-SD vlen's scope and letting hits map to
    waveforms exactly. gzipped DST input is read natively; no decompression
    step is needed.
    """
    if not Path(dst_file).exists():
        print(f"File: {dst_file} doesn't exists")
        return None

    dst_string = read_dst_file_tax4_std_recon_nfold(dst_file)
    events, hits, waveforms, badsd = parse_dst_string(dst_string)

    if events is None:
        return None

    # Load xmax info for current dst file
    if xmax_reader is not None:
        xmax_reader.read_file(dst_file)

    # Dictionary with parsed data
    data = dict()

    if add_shower_params:
        data = shower_params(data, events, xmax_reader)

    if add_standard_recon:
        data = standard_recon(data, events)

    data = detector_readings_flat(
        dst_file, data, hits, waveforms, badsd=badsd, add_badsd=add_badsd
    )

    if (config is not None) and (hasattr(config, "add_event_ids")):
        data = config.add_event_ids(data, dst_file)
    return data
