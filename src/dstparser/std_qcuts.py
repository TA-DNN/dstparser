import numpy as np


def angular_dist(dirs0, dirs1=None):
    if dirs1 is None:
        dirs1 = np.expand_dims(np.array([0, 0, 1]), axis=0)

    cos = np.sum(dirs0 * dirs1, axis=-1)
    cos = cos / np.linalg.norm(dirs0, axis=-1) / np.linalg.norm(dirs1, axis=-1)
    cos = np.clip(cos, -1, 1)
    return np.arccos(cos) * 180 / np.pi


def compute_chi2_per_ndof(chi2, ndof):
    """Compute chi2/ndof safely."""
    chi2 = np.asarray(chi2[:], dtype=np.float32)
    ndof = np.asarray(ndof[:], dtype=np.float32)

    # Remove division by zero warning
    result = np.empty_like(chi2, dtype=float)
    mask = ndof > 0
    result[mask] = chi2[mask] / ndof[mask]
    result[~mask] = chi2[~mask]
    return result


def define_std_cuts(
    data,
    nsd_min=None,
    zenith_max=None,
    dborder_min=None,
    point_dir_max=None,
    chi2_geom_max=None,
    chi2_ldf_max=None,
    chi2_combined_max=None,
    s800_relerr_max=None,
    use_tshape_cut=False,
):
    """Define standard quality cuts individually, only computing required arrays.
    If defaults are provided, they will be used to fill any missing values.

    Args:
        use_tshape_cut: Apply T-shape border cut for early data (< 2008/11/11)
                               to match iterate.cpp behavior
    """

    cuts = dict()

    if nsd_min is not None:
        nsd = np.asarray(data["std_recon_nsd"][:], dtype=np.int32)
        cuts["nsd"] = nsd >= nsd_min

    if zenith_max is not None:
        zenith = angular_dist(
            np.asarray(data["std_recon_shower_axis"][:], dtype=np.float32)
        )
        cuts["zenith"] = zenith < zenith_max

    if dborder_min is not None:
        border_distance = np.asarray(
            data["std_recon_border_distance"][:], dtype=np.float32
        )
        cuts["dborder"] = border_distance >= dborder_min

        # Add T-shape border cut for early data if requested
        if use_tshape_cut:
            event_date = np.asarray(data["std_recon_yymmdd"][:], dtype=np.int32)
            tshape_border_distance = np.asarray(
                data["std_recon_border_distance_tshape"][:], dtype=np.float32
            )

            # DS1_TO_DS2_DATE = 81111 (2008/11/11)
            DS1_TO_DS2_DATE = 81111
            early_data_mask = event_date < DS1_TO_DS2_DATE
            tshape_cut = (tshape_border_distance >= dborder_min) | ~early_data_mask

            cuts["tshape_border"] = tshape_cut

    if point_dir_max is not None:
        axis_err = np.asarray(data["std_recon_shower_axis_err"][:], dtype=np.float32)
        cuts["point_dir"] = axis_err < point_dir_max

    if chi2_geom_max is not None:
        chi2_geom = compute_chi2_per_ndof(
            data["std_recon_geom_chi2"], data["std_recon_geom_ndof"]
        )
        cuts["chi2_geom"] = chi2_geom < chi2_geom_max

    if chi2_ldf_max is not None:
        chi2_ldf = compute_chi2_per_ndof(
            data["std_recon_ldf_chi2"], data["std_recon_ldf_ndof"]
        )
        cuts["chi2_ldf"] = chi2_ldf < chi2_ldf_max

    if chi2_combined_max is not None:
        chi2_combined = compute_chi2_per_ndof(
            data["std_recon_combined_chi2"], data["std_recon_combined_ndof"]
        )
        cuts["chi2_combined"] = chi2_combined < chi2_combined_max

    if s800_relerr_max is not None:
        s800_scale = np.asarray(data["std_recon_ldf_scale"][:], dtype=np.float32)
        s800_scale_err = np.asarray(
            data["std_recon_ldf_scale_err"][:], dtype=np.float32
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            cuts["s800_err"] = (s800_scale_err / s800_scale) < s800_relerr_max

    return cuts


def print_cut_efficiencies(cuts):
    """Print efficiencies of each cut and cumulative efficiency."""
    n_total = len(next(iter(cuts.values())))  # Length of first cut array
    print(f"Total = {n_total}")
    header = f"{'cut':20}: {'efficiency':>9}, {'combined':>9}, {'after cut':>14}"
    print(header)
    print("-" * len(header))
    combined = np.ones(n_total, dtype=bool)
    for key, mask in cuts.items():
        eff = 100 * np.sum(mask) / n_total  # Individual cut efficiency
        combined &= mask  # Combine with previous cuts
        comb_eff = 100 * np.sum(combined) / n_total  # Cumulative efficiency
        print(f"{key:20}: {eff:8.2f}%, {comb_eff:8.2f}%, {np.sum(combined):14d}")


def combine_cuts(cuts_dict, verbose=True):
    """Combine all cuts into a single mask using AND logic, with optional verbosity to print efficiencies."""
    n_total = len(next(iter(cuts_dict.values())))  # Length of first cut array
    combined_cut = np.ones(n_total, dtype=bool)

    if verbose:
        print_cut_efficiencies(cuts_dict)  # Print efficiencies if verbose is True

    # Combine all cuts (the printing logic is already handled in print_cut_efficiencies)
    for cut in cuts_dict.values():
        combined_cut &= cut  # Combine with previous cuts

    return combined_cut


def std_spectrum_quality_cuts(data, zenith_max=45, verbose=True, use_tshape_cut=True):
    cuts = define_std_cuts(
        data,
        nsd_min=5,
        zenith_max=zenith_max,
        dborder_min=1,
        point_dir_max=5,
        chi2_geom_max=4,
        chi2_ldf_max=4,
        chi2_combined_max=None,
        s800_relerr_max=0.25,
        use_tshape_cut=use_tshape_cut,
    )
    return combine_cuts(cuts, verbose=verbose)


def std_composit_quality_cuts(data, zenith_max=45, verbose=True, use_tshape_cut=True):
    """
    The same as std_spectrum_quality_cuts, but uses tighter
    quality cuts used in https://arxiv.org/pdf/1808.03680

    nsd_min=7
    chi2_combined_max=None (disabled - iterate.cpp uses custom P chi2)
    use_tshape_cut=True (apply T-shape border cut for early data)

    NOTE: The chi2_combined_max=5 cut is disabled because iterate.cpp
    uses a custom combined chi2 calculation (P) that includes timing,
    mixed Gaussian/Poisson likelihood, and different parameter counting.
    This cannot be replicated with the standard reconstruction chi2.

    The T-shape border cut is applied for data before 2008/11/11 to match
    iterate.cpp behavior exactly.
    """

    cuts = define_std_cuts(
        data,
        nsd_min=7,
        zenith_max=zenith_max,
        dborder_min=1,
        point_dir_max=5,
        chi2_geom_max=4,
        chi2_ldf_max=4,
        chi2_combined_max=None,  # Disabled - iterate.cpp uses custom P chi2
        s800_relerr_max=0.25,
        use_tshape_cut=use_tshape_cut,
    )
    return combine_cuts(cuts, verbose=verbose)


def std_custom_quality_cuts(
    data,
    nsd_min=7,
    zenith_max=45,
    verbose=True,
    chi2_combined_max=None,
    s800_relerr_max=0.25,
    use_tshape_cut=True,
):
    """
    Custom quality cuts with full control over parameters.

    This function is designed to exactly match iterate.cpp behavior including:
    - T-shape border cut for early data (< 2008/11/11)
    - Disabled combined chi2 cut
    - All other parameters matching iterate.cpp

    Args:
        data: Event data dictionary
        nsd_min: Minimum number of SDs
        zenith_max: Maximum zenith angle
        verbose: Print cut efficiencies
        chi2_combined_max: Combined chi2 cut (None to disable)
        s800_relerr_max: Maximum S800 relative error
        use_tshape_cut: Apply T-shape border cut for early data

    Returns:
        Combined cut mask
    """

    cuts = define_std_cuts(
        data,
        nsd_min=nsd_min,
        zenith_max=zenith_max,
        dborder_min=1,
        point_dir_max=5,
        chi2_geom_max=4,
        chi2_ldf_max=4,
        chi2_combined_max=chi2_combined_max,
        s800_relerr_max=s800_relerr_max,
        use_tshape_cut=use_tshape_cut,
    )
    return combine_cuts(cuts, verbose=verbose)
