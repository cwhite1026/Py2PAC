#External code
import numpy as np
from astropy.cosmology import default_cosmology
from scipy import stats

#===============================================================================
#===============================================================================
#===============================================================================

def get_mags_and_radii(size, radii=True, min_mag = 20, max_mag = 28, z = 1.7):
    """
    This is a function that generates fake magnitudes and optionally radii for
    randoms.

    Parameters
    ----------
    size : int
        Number of magnitudes + radii to generate
        
    radii : Boolean, option
        Whether to generate radii as well as magnitudes or not.
        Default is true

    min_mag : scalar, optional
        Numerically lowest (brightest) allowable magnitude

    max_mag : scalar, optional
        Numerically highest (dimmest) allowable magnitude

    z : scalar, optional
        Redshift of fake galaxies

    Returns
    ----------
        mags : array
            Array of apparent magnitudes for randoms
        radii : array
            Array of log of radii in pixels for randoms
    """
    df = default_cosmology.get_cosmology_from_string('Planck13')
    shape, loc, scale = 0.519904595968, -0.398557693623, 1.74601245963
    z_array = stats.lognorm.rvs(shape, loc, scale, size=size)
    z_array[z_array < 0.3] = z
    z_array[z_array > z] = z
    dmod_array = df.distmod(z_array).value
    distmod = df.distmod(z).value
    mags = get_schechter_mags(size, distmod, min_mag, max_mag)
    radii = get_radii(mags)
    apparent_mags = mags + dmod_array
    apparent_mags[apparent_mags < min_mag] = min_mag
    apparent_mags[apparent_mags > max_mag] = max_mag
    return apparent_mags, radii

def get_radii(m, r0 = 0.21 / 0.06, m0 = -21., beta = 0.3, sigma = 0.7):
    """
    Generates artificial galaxy radii given magnitudes. Parameters from here:
    http://iopscience.iop.org/article/10.1088/0004-637X/765/1/68/pdf

    Parameters
    ----------
    m : float or array
        Input magnitudes
        
    r0 : scalar, optional
        Default is 0.21 / 0.06 (arcsec / GOODS pixel scale)

    m0 : scalar, optional
        Default is -21

    beta : scalar, optional
        Default is 0.3

    sigma : scalar, optional
        Default is 0.7

    Returns
    ----------
        log_rand : array
            Array of log of radii in pixels
    """
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
    """
    Returns number densities of magnitudes according to schechter function
    Default parameters for z=1.7 from here:
    http://mnras.oxfordjournals.org/content/456/3/3194.full.pdf

    Parameters
    ----------
    m : float or array
        Input magnitudes
        
    phi_star : scalar, optional
        Default is 0.21 / 0.06 (arcsec / GOODS pixel scale)

    m_star : scalar, optional
        Default is -19.61

    alpha : scalar, optional
        Faint-end slope of Schechter function

    Returns
    ----------
        n : float or array
            Number density of galaxies by magnitude
    """
    fudge_factor = np.log(10)*0.4 * phi_star
    base_exp = 10**(0.4*(m_star - m))
    n = fudge_factor * (base_exp**(1 + alpha)) * np.exp(-base_exp)
    return n

def get_schechter_mags(size, distmod, min_mag, max_mag):
    """
    Cuts off sample of random magnitudes according to schechter function

    Parameters
    ----------
    size : int
        Number of magnitudes to generate
        
    distmod : scalar
        Distance modulus for conversion between absolute and apparent mags

    min_mag : scalar
        Numerically lowest (brightest) allowable magnitude

    max_mag : scalar
        Numerically highest (dimmest) allowable magnitude

    Returns
    ----------
        mags : array
            Array of absolute magnitudes for randoms
            sampled according to Schechter function
    """
    nmax = schechter(max_mag - distmod)
    mags = []
    while len(mags) < size:
        m = np.random.uniform(min_mag, max_mag, size=size) - distmod
        n = np.random.uniform(0, high=nmax, size=size)
        mags += list(m[n <= schechter(m)])   
    mags = np.asarray(mags[:size])
    return mags
