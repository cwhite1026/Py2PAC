#External code
import numpy as np
import numpy.ma as ma
from astropy.cosmology import default_cosmology

#==========================================================================
#==========================================================================
#==========================================================================

'''
The lookup table is a fairly hacky way to get the fit parameters for 
Gaussians we fit to the magnitude distribution at certain bins of redshift 
and  stellar mass. The first column is the average of the redshift bin 
edges, the second is the same for log stellar mass, the third is the center 
of the Gaussian fit, and the fourth is the standard deviation.
'''

lookup_table = np.array([[1.25,8.5,26.34667761,0.67892732],
                        [1.25,9.5,24.07570187,0.50435506],
                        [1.25,11.,22.33910589,0.56583984],
                        [2.,8.5,27.00955795,0.67639655],
                        [2.,9.5,24.98014983,0.53994574],
                        [2.,11.,23.7487221,1.05912046],
                        [3.5,8.5,27.99870314,0.76291203],
                        [3.5,9.5,26.46102337,1.00514572],
                        [3.5,11.,25.79922306,1.83962567],
                        [5.25,8.5,28.39936568,0.55549091],
                        [5.25,9.5,27.48977546,1.13281005],
                        [5.25,11.,27.38154824,1.75470488]])

def get_mags_and_radii(size, min_mag = 20, max_mag = 28, z = 1.25, mstar = 9.5):
    """
    This is a function that generates fake magnitudes and optionally radii 
    for randoms.

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
        Allowed values: 1.25, 2, 3.5, 5.25

    mstar : scalar, optional
        Log of stellar mass of fake galaxies
        Allowed values: 8.5, 9.5, 11
    mstar : scalar, optional
        Log of stellar mass of fake galaxies. Default is 9.5
        Allowed values are 8.5, 9.5, and 11

    Returns
    ----------
        mags : array
            Array of apparent magnitudes for randoms
        radii : array
            Array of log of radii in pixels for randoms
    """
    
    df = default_cosmology.get_cosmology_from_string('Planck13')
    dmod = df.distmod(z).value
    mags = get_mags(size, z, mstar, min_mag, max_mag)
    absolute_mags = mags - dmod
    radii = get_radii(absolute_mags)
    return mags, radii

def get_mags(size, z, mstar, min_mag, max_mag, lookup_table = lookup_table):
    '''
    Queries the lookup table for appropriate Gaussian parameters for 
    the magnitude distribution and generates a random sample with those

    Parameters
    ----------
    size : int
        Number of magnitudes + radii to generate
        
    z : scalar
        Redshift of fake galaxies. Default is 1.25
        Allowed values are 1.25, 2, 4.5, and 5.25

    mstar : scalar
        Log of stellar mass of fake galaxies. Default is 9.5
        Allowed values are 8.5, 9.5, and 11

    lookup_table : array of shape N x 4
        Table of parameters for Gaussian magnitude distribution at
        z and mstar; described more fully above

    Returns
    ----------
        mags : array
            Array of apparent magnitudes for randoms
    '''
    row = lookup_table[(lookup_table[:,0] == z) & (lookup_table[:,1] == mstar)][0]
    loc, scale = row[2], row[3]
    n_needed = size
    mags = np.array([])
    while n_needed > 0:
        new_mags = np.random.normal(loc=loc,scale=scale,size=n_needed)
        masked_mags = ma.masked_inside(new_mags, min_mag, max_mag)
        n_valid = n_needed - len(masked_mags.compressed())
        if n_valid > 0:
            new_mags = new_mags[masked_mags.mask]
            mags = np.concatenate((mags, new_mags))
        n_needed = size - len(mags)
    
    return mags

def get_radii(m, r0 = 0.21 / 0.06, m0 = -21., beta = 0.3, sigma = 0.7):
    """
    Generates artificial galaxy radii given magnitudes. Parameters from 
    here:
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
    exp = 10**(-0.4*(m - m0))
    mean = r0 * exp**beta
    rand = np.random.lognormal(np.log(mean), sigma)
    log_rand = np.log10(rand)
    if type(log_rand).__name__ != 'ndarray':
        log_rand = np.array([log_rand])
    log_rand[log_rand > 2.5] = 2.5
    log_rand[log_rand < -1.] = -1.
    return log_rand

