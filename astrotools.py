import numpy as np

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


def mean_xmax(log10e, mass, model="EPOS-LHC"):
    """
    <Xmax> values for given energies log10e(E / eV), mass numbers A
    and hadronic interaction model, according to [3,4].

    :param log10e: energy log10(E/eV)
    :type log10e: array_like
    :param mass: mass number
    :type mass: array_like
    :param model: hadronic interaction model
    :type model: string
    :return: mean Xmax value in [g/cm^2]
    """
    x0, d, xi, delta = DXMAX_PARAMS[model][:4]
    l_e = log10e - 19
    return x0 + d * l_e + (xi - d / np.log(10) + delta * l_e) * np.log(mass)


def xmax_scaling(en0, xmax0, en):
    d = 45.8  # elongation rate for QGSJetII (see above)
    return xmax0 + d * np.log10(en / en0)
