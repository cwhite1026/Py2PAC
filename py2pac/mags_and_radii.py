#External code
import numpy as np
from astropy.cosmology import default_cosmology

#===============================================================================
#===============================================================================
#===============================================================================

def get_mags_and_radii(size, radii=True, min_mag = 20, max_mag = 28, z = 1.7):
    """Class method to generate a completeness function dependent only
    on magnitude.

    **Syntax**

    completeness_function = CompletenessFunction.from_1D_array(completeness_array, mag_range)

    Parameters
    ----------

    completeness_array : 1D array-like
        A 1D array describing completeness over a set of magnitudes

    mag_range : 1D array-like
        A 1D array containing the magnitude range over which
        completeness is known. Can either be of length 2, containing
        the minimum and maximum magnitude, or one greater than the
        length of the completeness array, containing completeness
        histogram bin edges.

    Returns
    -------
    completeness_function_1D : CompletenessFunction instance
        An object that describes completeness over the specified
        magnitude range.

    Notes
    -----
    Assumes equal bin widths.
    """
    df = default_cosmology.get_cosmology_from_string('Planck13')
    distmod = df.distmod(z).value
    mags = get_schechter_mags(size, distmod, min_mag, max_mag)
    radii = get_radii(mags)
    return mags + distmod, radii

def get_radii(m, r0 = 0.21 / 0.06, m0 = -21., beta = 0.3, sigma = 0.7):
    exp = 10**(0.4*(m0 - m))
    mean = r0 * exp**beta
    rand = np.random.lognormal(mean, sigma)
    log_rand = np.log10(rand)
    if type(log_rand).__name__ != 'ndarray':
        log_rand = np.array([log_rand])
    log_rand[log_rand > 2.5] = 2.5
    log_rand[log_rand < -1.] = -1.
    return log_rand

def schechter(m, phi_star=0.00681, m_star=-19.61, alpha=-1.33):
    fudge_factor = np.log(10)*0.4 * phi_star
    base_exp = 10**(0.4*(m_star - m))
    function = fudge_factor * (base_exp**(1 + alpha)) * np.exp(-base_exp)
    return function

def get_schechter_mags(size, distmod, min_mag, max_mag):
    nmax = schechter(max_mag - distmod)
    mags = []
    while len(mags) < size:
        x = np.random.uniform(min_mag, max_mag, size=size) - distmod
        y = np.random.uniform(0, high=nmax, size=size)
        mags += list(x[y<=schechter(x)])   
    mags = np.asarray(mags[:size])
    return mags
