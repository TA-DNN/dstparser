# This is excerpt from https://astro.pages.rwth-aachen.de/astrotools/_modules/auger.html#mean_xmax

import numpy as np


def gumbel_parameters(log10e, mass, model="EPOS-LHC"):
    """
    Location, scale and shape parameter of the Gumbel Xmax distribution from [1], equations 3.1 - 3.6.
    parameters from [10]

    :param log10e: energy log10(E/eV)
    :type log10e: array_like
    :param mass: mass number
    :type mass: array_like
    :param model: hadronic interaction model
    :type model: string
    :return: mu (array_like, location paramater [g/cm^2]), sigma (array_like, scale parameter [g/cm^2]),
             lamda (array_like, shape parameter)
    :rtype: tuple
    """
    l_e = log10e - 19  # log10(E/10 EeV)
    ln_mass = np.log(mass)
    d = np.array([np.ones_like(mass), ln_mass, ln_mass**2])

    # Parameters for mu, sigma and lambda of the Gumble Xmax distribution from [1], table 1.
    #   'model' : {
    #       'mu'     : ((a0, a1, a2), (b0, b1, b2), (c0, c1, c2))
    #       'sigma'  : ((a0, a1, a2), (b0, b1, b2))
    #       'lambda' : ((a0, a1, a2), (b0, b1, b2))}
    params = {
        "QGSJetII": {
            "mu": (
                (758.444, -10.692, -1.253),
                (48.892, 0.02, 0.179),
                (-2.346, 0.348, -0.086),
            ),
            "sigma": ((39.033, 7.452, -2.176), (4.390, -1.688, 0.170)),
            "lambda": ((0.857, 0.686, -0.040), (0.179, 0.076, -0.0130)),
        },
        "QGSJetII-04": {
            "mu": (
                (758.65, -12.3571, -1.24539),
                (56.5943, -1.01244, 0.228689),
                (-0.534683, -0.17284, -0.019159),
            ),
            "sigma": ((35.4234, 6.75921, -1.46182), (-0.796042, 0.201762, -0.0142452)),
            "lambda": (
                (0.671545, 0.373902, 0.075325),
                (0.0304335, 0.0473985, -0.000564531),
            ),
        },
        "Sibyll2.1": {
            "mu": (
                (770.104, -15.873, -0.960),
                (58.668, -0.124, -0.023),
                (-1.423, 0.977, -0.191),
            ),
            "sigma": ((31.717, 1.335, -0.601), (-1.912, 0.007, 0.086)),
            "lambda": ((0.683, 0.278, 0.012), (0.008, 0.051, 0.003)),
        },
        "Sibyll2.3d": {
            "mu": (
                (785.852, -15.5994, -1.06906),
                (60.5929, -0.786014, 0.200728),
                (-0.689462, -0.294794, 0.0399432),
            ),
            "sigma": ((41.0345, -2.17329, -0.306202), (-0.309466, -1.16496, 0.225445)),
            "lambda": (
                (0.799493, 0.235235, 0.00856884),
                (0.0632135, -0.0012847, 0.000330525),
            ),
        },
        "EPOS1.99": {
            "mu": (
                (780.013, -11.488, -1.906),
                (61.911, -0.098, 0.038),
                (-0.405, 0.163, -0.095),
            ),
            "sigma": ((28.853, 8.104, -1.924), (-0.083, -0.961, 0.215)),
            "lambda": ((0.538, 0.524, 0.047), (0.009, 0.023, 0.010)),
        },
        "EPOS-LHC": {
            "mu": (
                (775.457, -10.3991, -1.75261),
                (58.5306, -0.827668, 0.231144),
                (-1.40781, 0.225624, -0.10008),
            ),
            "sigma": ((32.2632, 3.94252, -0.864421), (1.27601, -1.81337, 0.231914)),
            "lambda": (
                (0.641093, 0.219762, 0.171124),
                (0.0726131, 0.0353188, -0.0131158),
            ),
        },
    }
    par = params[model]

    p0, p1, p2 = np.dot(par["mu"], d)
    mu = p0 + p1 * l_e + p2 * l_e**2
    p0, p1 = np.dot(par["sigma"], d)
    sigma = p0 + p1 * l_e
    p0, p1 = np.dot(par["lambda"], d)
    lambd = p0 + p1 * l_e

    return mu, sigma, lambd


def rand_xmax(log10e, mass, size=None, model="QGSJetII-04"):
    """
    Random Xmax values for given energy E [EeV] and mass number A, cf. [1].

    :param log10e: energy log10(E/eV)
    :type log10e: array_like
    :param mass: mass number
    :type mass: array_like
    :param model: hadronic interaction model
    :type model: string
    :param size: number of xmax values to create
    :type size: int or None
    :return: random Xmax values in [g/cm^2]
    :rtype: array_like
    """
    mu, sigma, lambd = gumbel_parameters(log10e, mass, model)

    # From [2], theorem 3.1:
    # Y = -ln X is generalized Gumbel distributed for Erlang distributed X
    # Erlang is a special case of the gamma distribution
    return mu - sigma * np.log(np.random.gamma(lambd, 1.0 / lambd, size=size))
