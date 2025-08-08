import numpy as np
from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers_vlen import parse_dst_string


def corsika_id2mass(corsika_pid):
    return np.where(corsika_pid == 14, 1, corsika_pid // 100).astype(np.int32)


def rec_coreposition_to_CLF_meters(core_position_rec, option):
    detector_dist = 1200  # meters
    clf_origin_x = 12.2435
    clf_origin_y = 16.4406
    if option == "x":
        return detector_dist * (core_position_rec - clf_origin_x)
    elif option == "y":
        return detector_dist * (core_position_rec - clf_origin_y)
    elif option == "dx":
        return detector_dist * core_position_rec
    elif option == "dy":
        return detector_dist * core_position_rec


def shower_params(data, events, xmax_data):
    # Shower related
    # for details: /ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/sditerator/src/sditerator_cppanalysis.cpp
    to_meters = 1e-2
    # events = dst_data["events"]
    data["mass_number"] = corsika_id2mass(events["rusdmc_.parttype"])
    data["energy"] = events["rusdmc_.energy"]

    if xmax_data is not None:
        data["xmax"] = xmax_data(data["energy"])

    # Theta and phi are in radians (see dst_fields.md)
    data["shower_axis"] = np.array(
        [
            np.sin(events["rusdmc_.theta"]) * np.cos(events["rusdmc_.phi"] + np.pi),
            np.sin(events["rusdmc_.theta"]) * np.sin(events["rusdmc_.phi"] + np.pi),
            np.cos(events["rusdmc_.theta"]),
        ],
        dtype=np.float32,
    ).transpose()

    # shower core in cm
    data["shower_core"] = np.array(
        np.stack(
            [
                events["rusdmc_.corexyz[0]"],
                events["rusdmc_.corexyz[1]"],
                events["rusdmc_.corexyz[2]"],
            ],
            axis=1,
        ),
        dtype=np.float32,
    )

    return data


def standard_recon(
    data, events, include_combined_fit=False, include_fixed_curve_fit=False
):
    # events = dst_data["events"]
    # Exempt from comments of cpp source code at:
    # /ceph/work/SATORI/projects/TA-ASIoP/benMC/sdanalysis_2019/sdmc/sdmc_spctr.c
    # // Reported by DAQ as time of the 1st signal in the triple that caused the triggger.
    # // From now on, everyhting is relative to hhmmss.  Not useful in the event reconstruction.
    # Date of event
    # rusdraw_.yymmdd = 80916; // Event date year = 08, month = 09, day = 16
    data["std_recon_yymmdd"] = events["rusdraw_.yymmdd"]
    # Time of event
    # rusdraw_.hhmmss = 1354;  // Event time, hour=00, minute=13, second = 54
    data["std_recon_hhmmss"] = events["rusdraw_.hhmmss"]
    # Microseconds for the second
    # rusdraw_.usec = 111111
    data["std_recon_usec"] = events["rusdraw_.usec"]
    # Number of waveforms for event for all detectors
    data["std_recon_nofwf"] = events["rusdraw_.nofwf"]
    # number of SDs in space-time cluster
    data["std_recon_nsd"] = events["rufptn_.nstclust"]
    # number of SDs in space cluster
    data["std_recon_nsclust"] = events["rufptn_.nsclust"]
    # number of hit SDs
    data["std_recon_nhits"] = events["rufptn_.nhits"]
    # number of SDs in space-time cluster & lie on the border of the array
    data["std_recon_nborder"] = events["rufptn_.nborder"]
    # total charge [VEM] of SDs in the space-time cluster, (lower & upper)
    data["std_recon_qtot"] = np.array(
        [
            events["rufptn_.qtot[0]"],
            events["rufptn_.qtot[1]"],
        ]
    ).transpose(1, 0)
    # energy reconstructed by the standard energy estimation table [EeV]
    data["std_recon_energy"] = events["rufldf_.energy[0]"]
    # reconstructed scale of the Lateral Distribution Function (LDF) fit [VEM m-2]
    data["std_recon_ldf_scale"] = events["rufldf_.sc[0]"]
    # uncertainty of the scale [VEM m-2]
    data["std_recon_ldf_scale_err"] = events["rufldf_.dsc[0]"]
    # chi-square of the LDF fit
    data["std_recon_ldf_chi2"] = events["rufldf_.chi2[0]"]
    # the number of degree of freedom of the LDF fit (= n - 3),
    # where "n" is the number of the SDs used for the LDF fit
    data["std_recon_ldf_ndof"] = events["rufldf_.ndof[0]"]
    # core position (x, y) reconstructed by the LDF fit in CLF coordinate [m]
    data["std_recon_shower_core"] = np.array(
        [
            rec_coreposition_to_CLF_meters(events["rufldf_.xcore[0]"], option="x"),
            rec_coreposition_to_CLF_meters(events["rufldf_.ycore[0]"], option="y"),
        ]
    ).transpose(1, 0)

    # uncertainty of the core position (x, y) reconstructed by the LDF fit
    data["std_recon_shower_core_err"] = np.array(
        [
            rec_coreposition_to_CLF_meters(events["rufldf_.dxcore[0]"], option="dx"),
            rec_coreposition_to_CLF_meters(events["rufldf_.dycore[0]"], option="dy"),
        ]
    ).transpose(1, 0)
    # S800 (particle density at 800 m from the shower axis) [VEM m-2]
    data["std_recon_s800"] = events["rufldf_.s800[0]"]

    if include_combined_fit:
        # reconstructed values of the geometry+LDF (combined) fit
        data["std_recon_combined_energy"] = events["rufldf_.energy[1]"]
        data["std_recon_combined_scale"] = events["rufldf_.sc[1]"]
        data["std_recon_combined_scale_err"] = events["rufldf_.dsc[1]"]
        data["std_recon_combined_chi2"] = events["rufldf_.chi2[1]"]
        # the number of degree of freedom of the LDF fit (= 2*n - 6),
        # where "n" is the number of the SDs used for the LDF fit
        data["std_recon_combined_ndof"] = events["rufldf_.ndof[1]"]
        data["std_recon_combined_shower_core"] = np.array(
            [
                rec_coreposition_to_CLF_meters(events["rufldf_.xcore[1]"], option="x"),
                rec_coreposition_to_CLF_meters(events["rufldf_.ycore[1]"], option="y"),
            ]
        ).transpose(1, 0)
        data["std_recon_combined_shower_core_err"] = np.array(
            [
                rec_coreposition_to_CLF_meters(
                    events["rufldf_.dxcore[1]"], option="dx"
                ),
                rec_coreposition_to_CLF_meters(
                    events["rufldf_.dycore[1]"], option="dy"
                ),
            ]
        ).transpose(1, 0)
        data["std_recon_combined_s800"] = events["rufldf_.s800[1]"]

        # 3-d unit vector of the arrival direction (pointing back to the source)
        # geometry+LDF fit
        # "+0.5" is a correction for zenith angle.
        theta = np.deg2rad(events["rufldf_.theta"] + 0.5)
        phi = np.deg2rad(events["rufldf_.phi"]) + np.pi
        data["std_recon_shower_axis_combined"] = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            dtype=np.float32,
        ).transpose()

        # uncertainty of the pointing direction [degree]
        # geometry+LDF fit
        theta = np.deg2rad(events["rufldf_.theta"])
        dtheta = events["rufldf_.dtheta"]
        dphi = events["rufldf_.dphi"]

        data["std_recon_shower_axis_err_combined"] = np.sqrt(
            dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
        )

    if include_fixed_curve_fit:
        # chi-square of the geometry fit (fixed curvature)
        data["std_recon_geom_chi2_fixed_curve"] = events["rusdgeom_.chi2[1]"]
        # the number of degree of freedom of the geometry fit (= n - 5),
        # where "n" is the number of the SDs used for the geometry fit
        data["std_recon_geom_ndof_fixed_curve"] = events["rusdgeom_.ndof[1]"]

        # 3-d unit vector of the arrival direction (pointing back to the source)
        # geometry fit with a fixed curved parameter
        # "+0.5" is a correction for zenith angle.
        theta = np.deg2rad(events["rusdgeom_.theta[1]"] + 0.5)
        phi = np.deg2rad(events["rusdgeom_.phi[1]"]) + np.pi

        data["std_recon_shower_axis_fixed_curve"] = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            dtype=np.float32,
        ).transpose()

        # uncertainty of the pointing direction [degree]
        # fixed curved parameter

        theta = np.deg2rad(events["rusdgeom_.theta[1]"])
        dtheta = events["rusdgeom_.dtheta[1]"]
        dphi = events["rusdgeom_.dphi[1]"]

        data["std_recon_shower_axis_err_fixed_curve"] = np.sqrt(
            dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
        )

    # 3-d unit vector of the arrival direction (pointing back to the source)
    # geometry fit with a free curved parameter.
    # "+0.5" is a correction for zenith angle.
    theta = np.deg2rad(events["rusdgeom_.theta[2]"] + 0.5)
    phi = np.deg2rad(events["rusdgeom_.phi[2]"]) + np.pi
    data["std_recon_shower_axis"] = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float32,
    ).transpose()
    # uncertainty of the pointing direction [degree]
    # free curved parameter
    # event_list[22] is zenith angle in deg
    # event_list[24] is uncertainty zenith angle in deg
    # event_list[25] is uncertainty azimuth angle in deg

    theta = np.deg2rad(events["rusdgeom_.theta[2]"])
    dtheta = events["rusdgeom_.dtheta[2]"]
    dphi = events["rusdgeom_.dphi[2]"]

    # Uncertainty in degrees
    data["std_recon_shower_axis_err"] = np.sqrt(
        dtheta * dtheta + np.sin(theta) * np.sin(theta) * dphi * dphi
    )

    # chi-square of the geometry fit (free curvature)
    data["std_recon_geom_chi2"] = events["rusdgeom_.chi2[2]"]
    # the number of degree of freedom of the geometry fit (= n - 6),
    # where "n" is the number of the SDs used for the geometry fit
    data["std_recon_geom_ndof"] = events["rusdgeom_.ndof[2]"]
    # curvature paramter `a` of the geometry fit
    data["std_recon_curvature"] = events["rusdgeom_.a"]
    # uncertainty of the curvature paramter `a` of the geometry fit
    data["std_recon_curvature_err"] = events["rusdgeom_.da"]
    # distance b/w the reconstructed core and the edge from the TA SD array [in 1,200 meter unit]
    # negative for events with the core outside of the TA SD array.
    data["std_recon_border_distance"] = events["rufldf_.bdist"]
    # distance to the T-shape TA SD array, edge of the sub-arrays [in 1,200 meter unit]
    # this value is used as "border_distance" before implementation of the boundary trigger (on 2008/11/11)
    data["std_recon_border_distance_tshape"] = events["rufldf_.tdist"]

    return data


def filter_offsets(mask, offsets):
    # 1) compute counts per segment
    counts = np.add.reduceat(mask.astype(int), offsets[:-1])
    # 2) build new offsets
    return np.concatenate(([0], np.cumsum(counts)))


def remove_mismatch(expected, actual):
    """
    Return a boolean mask such that applying it to `actual` will give `expected`.
    The order of elements in `expected` must appear in `actual` in the same order.
    """
    mask = np.zeros(len(actual), dtype=bool)
    i = 0  # index in expected

    for j in range(len(actual)):
        if i >= len(expected):
            break
        if actual[j] == expected[i]:
            mask[j] = True
            i += 1

    if i != len(expected):
        raise ValueError("Could not match all elements of `expected` in `actual`")

    if not np.array_equal(actual[mask], expected):
        raise ValueError("Mismatch after masking: actual[mask] != expected")

    return mask


def detector_readings_flat(data, hits, waveforms):
    # for c = 3e10 cm/s:
    # to_nsec = 4 * 1000
    # The below is more correct for c = 2.998e10 cm/c,
    # to_nsec = 4002.7691424

    # check if number folds from hits equal to number of waveforms
    if np.sum(hits["rufptn_.nfold"]) != waveforms["rusdraw_.xxyy"].shape:
        print("Missmatch between hits and waveforms!")

        # Compute expected detector IDs from hits
        expected_xxyy = np.repeat(hits["rufptn_.xxyy"], hits["rufptn_.nfold"])

        # Get actual waveform detector IDs
        actual_xxyy = waveforms["rusdraw_.xxyy"]

        mask_matched = remove_mismatch(expected_xxyy, actual_xxyy)

        waveforms_offsets = filter_offsets(mask_matched, waveforms["offsets"])

        waveforms = {
            key: value[mask_matched]
            for key, value in waveforms.items()
            if key != "offsets"
        }

        waveforms["offsets"] = waveforms_offsets

    # Filter for isgood > 2:
    hit_mask = hits["rufptn_.isgood"] > 2
    # Repeat hit_mask nfold times
    wf_mask = np.repeat(hit_mask, hits["rufptn_.nfold"])

    # Filter offsets according to masks
    hits_offsets = filter_offsets(hit_mask, hits["offsets"])
    waveforms_offsets = filter_offsets(wf_mask, waveforms["offsets"])

    # Filter hits
    hits = {key: value[hit_mask] for key, value in hits.items() if key != "offsets"}
    hits["offsets"] = hits_offsets
    hits["wf_offsets"] = np.concatenate(([0], np.cumsum(hits["rufptn_.nfold"])))

    # Filter waveforms
    waveforms = {
        key: value[wf_mask] for key, value in waveforms.items() if key != "offsets"
    }
    waveforms["offsets"] = waveforms_offsets

    # Check if filtering is correct
    # Compare detectors ids for hits and corresponding waveforms
    assert np.all(
        np.repeat(hits["rufptn_.xxyy"], hits["rufptn_.nfold"])
        == waveforms["rusdraw_.xxyy"]
    ), "Hits are not in agreement with waverforms"

    # -------- Example/reminder how to work with offsets:
    # We work with flattened arrays like this:
    # Example: get hits for the event with index ievent:
    # start, end = hits["offsets"][ievent], hits["wf_offsets"][ievent + 1]
    # hits_arrv_time_lower = hits[rufptn_.reltime[0]][start:end]
    #

    # Example: get a waveform for specific hit:
    # ihit - global index in hits
    # start, end = hits["wf_offsets"][ihit], hits["wf_offsets"][ihit + 1]
    # waveforms_fadc = waveforms["rusdraw_.fadc"][start:end]
    #
    # For vectorized work use np.split and np.reduceat
    # ---------

    # Create data arrays:

    # We use counter sep. dist units, to get in nsec, multiply to to_nsec
    # We normalize it anyway ...
    data["arrival_times"] = np.stack(
        [hits["rufptn_.reltime[0]"], hits["rufptn_.reltime[1]"]], axis=1
    )

    # Pulse area in VEM (pedestals subtracted)
    data["pulse_area"] = np.stack(
        [hits["rufptn_.pulsa[0]"], hits["rufptn_.pulsa[1]"]], axis=1
    )

    # SD coordinates in CLF frame [1200m units]
    # Probably for TAx4 it is in [2080m units]
    # Looks like reasonable choice (not very large magnitude)
    data["detector_positions"] = np.stack(
        [
            hits["rufptn_.xyzclf[0]"],
            hits["rufptn_.xyzclf[1]"],
            hits["rufptn_.xyzclf[2]"],
        ],
        axis=1,
    )

    # DNN might be useful to know (3,4,5) as we filtered out (0,1,2)
    # 0: Counter not working properly
    # 1: Hit not part of any cluster
    # 2: Part of space cluster
    # 3: Passed rough time pattern recognition
    # 4: Part of the event (highest quality)
    # 5: Saturated counter
    data["status"] = hits["rufptn_.isgood"]
    # Might be useful to know how long is time trace
    data["nfold"] = hits["rufptn_.nfold"]
    # For DNN it might be easier to cut hidden space based on integer ids
    data["detector_ids"] = hits["rufptn_.xxyy"]
    # Devision of flattened array to events
    data["hit_offsets"] = hits["offsets"]
    data["hit_tt_offsets"] = hits["wf_offsets"]

    # Current suggestion is following detector_features:
    # arrival_times 2
    # pulse_area 2
    # detector_positions 3
    # status 1
    # nfold 1
    # detector_ids 1
    # --- Total 10 features

    # Combine vem values (VEM/count)
    vem = np.stack([hits["rufptn_.vem[0]"], hits["rufptn_.vem[1]"]], axis=1)
    # Reshape to the same shape as fadc
    vem = np.repeat(vem, hits["rufptn_.nfold"], axis=0)[:, :, None]
    # Convert FADC to VEMs
    data["time_traces"] = waveforms["rusdraw_.fadc"] / vem
    # Devision of flattened time_traces to events
    data["tt_offsets"] = waveforms["offsets"]

    return data


def parse_dst_file_vlen(
    dst_file,
    xmax_reader=None,
    add_shower_params=True,
    add_standard_recon=True,
    config=None,
):
    dst_string = read_dst_file(dst_file)
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

    data = detector_readings_flat(data, hits, waveforms)

    if (config is not None) and (hasattr(config, "add_event_ids")):
        data = config.add_event_ids(data, dst_file)
    return data
