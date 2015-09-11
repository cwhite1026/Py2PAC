#This is a copy of the astroML.correlation package with some modifications made by Cathy White.  The main body of work is not hers!

import numpy as np
import numpy.ma as ma
from sklearn.neighbors import BallTree
from astroML.utils import check_random_state

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def uniform_sphere(RAlim, DEClim, size=1):
    """Draw a uniform sample on a sphere

    Parameters
    ----------
    RAlim : tuple
        select Right Ascension between RAlim[0] and RAlim[1]
        units are degrees
    DEClim : tuple
        select Declination between DEClim[0] and DEClim[1]
    size : int (optional)
        the size of the random arrays to return (default = 1)

    Returns
    -------
    RA, DEC : ndarray
        the random sample on the sphere within the given limits.
        arrays have shape equal to size.
    """
    zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
    DEC = (180. / np.pi) * np.arcsin(z)
    RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)

    return RA, DEC

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def ra_dec_to_xyz(ra, dec):
    """Convert ra & dec to Euclidean points

    Parameters
    ----------
    ra, dec : ndarrays

    Returns
    x, y, z : ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return (cos_ra * sin_dec,
            sin_ra * sin_dec,
            cos_dec)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def xyz_to_ra_dec(x, y, z):
    #Written by CW
    """Convert Euclidean points to ra and dec

    Parameters
    ----------
    x, y, z : ndarrays (dimension irrelevant as long as all the same)

    Returns
    ra, dec : ndarrays (degrees)
    """

    r = np.sqrt(x**2. + y**2. + z**2.)
    ra = np.arctan(-y/x) * 180./np.pi
    dec = np.arcsin(z/r) * 180./np.pi

    return (ra, dec)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def euclidian_dist_to_angular_dist(dist, r=1):
    """convert euclidian distances to angular distances
    Cathy White added 9/26/14- inverts angular_dist_to_euclidian_dist
       -argument is euclidian distance in the same units as r
       -returns angular sep in degrees"""
    
    return 180./np.pi * 2. * np.asin(dist/(2.*r))


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def angular_dist_to_euclidean_dist(D, r=1):
    """convert angular distances to euclidean distances
       -argument is angle in degrees
       -returns in whatever units r is in"""
    
    return 2 * r * np.sin(0.5 * D * np.pi / 180.)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def two_point(data, bins, BT_D=None, BT_R=None, method='standard',
              data_R=None, random_state=None, return_trees=False, 
              verbose=False, RR=None, return_RR=False, return_DD=False):
    #Edited by CW to allow user to supply the BallTrees or ask
    #for them to be returned
    """Two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    BT_D : BallTree (optional)
        ball tree created with the data positions
    BT_R : BallTree (optional)
        ball tree created with the random positions
    method : string
        "standard" or "landy-szalay".
    data_R : array_like (optional if no BT_R)
        if specified, use this as the random comparison sample
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background
    return_trees : True | False
        if True, the returns will be corr, data_tree,
        random_tree.  Default is False.
    RR : if this exact set of randoms and theta bins has been
        run, you can supply the RR counts and not calculate them again.
        You also need the data in case you're running landy-szalay.
    return_RR : If you know you'll be running a CF with this
        exact same random sample and binning (like with a bootstrap),
        you can get the RR counts returned and feed them back in the
        next time
    return_DD : In case you want to fit to the pair counts rather
        than the w(theta) estimator, you can get this back too
 
    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    data_tree : BallTree (optional)
        the ball tree used to calculate distances between objects
        quickly in the data.  only returned if return_trees == True
    random_tree : BallTree (optional)
        the ball tree used to calculate distances between objects
        quickly in the randomly generated set.  only returned if
        return_trees == True
    RR : ndarray (optional)
        the RR counts may be returned (if return_RR==True) and used
        again without recomputing if the theta bins and the random
        sample is exactly the same
    DD : ndarray (optional)
        the DD pair counts, returned if return_DD==True   
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    n_samples, n_features = data.shape
    Nbins = len(bins) - 1

    # shuffle all but one axis to get background distribution
    if data_R is None:
        print "two_point says: generating random sample"
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])
    else:
        data_R = np.asarray(data_R)
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')

    factor = len(data_R) * 1. / len(data)

    if BT_D is None:
        if verbose:
            print "two_point says: computing BallTree for data"
        BT_D = BallTree(data)
    if BT_R is None:
        if verbose:
            print "two_point says: computing BallTree for random sample"
        BT_R = BallTree(data_R)

    counts_DD = np.zeros(Nbins + 1)
    counts_RR = np.zeros(Nbins + 1)

    if verbose:
        print "two_point says: working through the CF calc.  This could take a while"
    for i in range(Nbins + 1):
        counts_DD[i] = np.sum(BT_D.query_radius(data, bins[i],
                                                count_only=True))
        if RR is None:
            counts_RR[i] = np.sum(BT_R.query_radius(data_R, bins[i],
                                                    count_only=True))

    if verbose:
        print "two_point says: binning done!"
    DD = np.diff(counts_DD)
    if RR is None:
        RR = np.diff(counts_RR)

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    if method == 'standard':
        corr = factor**2 * DD / RR - 1
    elif method == 'landy-szalay':
        counts_DR = np.zeros(Nbins + 1)
        for i in range(Nbins + 1):
            counts_DR[i] = np.sum(BT_R.query_radius(data, bins[i],
                                                    count_only=True))
        DR = np.diff(counts_DR)

        corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    corr[RR_zero] = np.nan

    to_return=corr
    if return_trees:
        to_return=[to_return]
        to_return.append(BT_D)
        to_return.append(BT_R)
    if return_RR:
        if not return_trees:
            to_return=[to_return]
        to_return.append(RR)
    if return_DD:
        if (not return_trees) and (not return_RR):
            to_return=[to_return]
        to_return.append(DD)        
    
    return to_return

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def bootstrap_two_point(data, bins, Nbootstrap=10, BT_D=None, BT_R=None,
                        method='standard', return_bootstraps=False,
                        return_trees=False, random_state=None):
    """Bootstrapped two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    Nbootstrap : integer
        number of bootstrap resamples to perform (default = 10)
    BT_D : BallTree
        ball tree created with the data positions (optional)
    BT_R : BallTree
        ball tree created with the random positions (optional)
    method : string
        "standard" or "landy-szalay".
    return_bootstraps: bool
        if True, return full bootstrapped samples
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background
    return_trees : True | False
        if True, the returns will include the ball
        trees for the data and random sets

    Returns
    -------
    This has several options for the returns.  The order will be
    corr, corr_err, bootstraps, BallTree_Data, BallTree_Rand

    Whatever subset of this you choose to have returned will
    retain this ordering.
    
    corr, corr_err : ndarrays
        the estimate of the correlation function and the bootstrap
        error within each bin. shape = Nbins
    bootstraps: thingy of some sort (optional)
        only returned when return_bootstraps == True
    data_tree : BallTree (optional)
        the ball tree used to calculate distances between objects
        quickly in the data.  only returned if return_trees == True
    random_tree : BallTree (optional)
        the ball tree used to calculate distances between objects
        quickly in the randomly generated set.  only returned if
        return_trees == True
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    if Nbootstrap < 2:
        raise ValueError("Nbootstrap must be greater than 1")

    n_samples, n_features = data.shape

    # get the baseline estimate
    if return_trees:
        corr, BT_D, BT_R = two_point(data, bins, BT_D=BT_D, BT_R=BT_R, method=method, random_state=rng, return_trees=True)
    else:
        corr = two_point(data, bins, BT_D=BT_D, BT_R=BT_R, method=method, random_state=rng)

    bootstraps = np.zeros((Nbootstrap, len(corr)))

    for i in range(Nbootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        bootstraps[i] = two_point(data[indices, :], bins, method=method,
                                  random_state=rng)

    # use masked std dev in case of NaNs
    corr_err = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))

    if return_bootstraps and return_trees:
        return corr, corr_err, bootstraps, BT_D, BT_R
    if return_bootstraps and not return_trees:
        return corr, corr_err, bootstraps
    if not return_bootstraps and return_trees:
        return corr, corr_err, BT_D, BT_R
    if not return_boostraps and not return_trees:
        return corr, corr_err

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def two_point_angular(ra, dec, bins, BT_D=None, BT_R=None, method='standard',
                      ra_R=None, dec_R=None, random_state=None, return_trees=False, 
                      verbose=False, RR=None, return_RR=False, return_DD=False):
    """Angular two-point correlation function

    A separate function is needed because angular distances are not
    euclidean, and random sampling needs to take into account the
    spherical volume element.

    Parameters
    ----------
    ra : array_like
        input right ascention, shape = (n_samples,)
    dec : array_like
        input declination
    completeness : array_like (optional)
        completenesses for each object if performing weighted CF
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    BT_D : BallTree (optional)
        ball tree created with the data positions
    BT_R : BallTree (optional)
        ball tree created with the random positions
    method : string
        "standard" or "landy-szalay".
    ra_R, dec_R : array_like (optional if no BT_R)
        the random sample to be used.  If you pass BT_R
        as an argument, you must also pass the random sample
        it was made from with this
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background
    return_trees : True | False
        if True, the returns will include the ball
        trees for the data and random sets
    RR : if this exact set of randoms and theta bins has been
        run, you can supply the RR counts and not calculate them again.
        You also need the data in case you're running landy-szalay.
    return_RR : If you know you'll be running a CF with this
        exact same random sample and binning (like with a bootstrap),
        you can get the RR counts returned and feed them back in the
        next time
    return_DD : In case you want to fit to the pair counts rather
        than the w(theta) estimator, you can get this back too
        
    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    data_tree : BallTree (optional)
        the ball tree used to calculate distances between objects
        quickly in the data.  only returned if return_trees == True
    random_tree : BallTree (optional)
        the ball tree used to calculate distances between objects
        quickly in the randomly generated set.  only returned if
        return_trees == True
    RR : ndarray (optional)
        the RR counts may be returned (if return_RR==True) and used
        again without recomputing if the theta bins and the random
        sample is exactly the same
    DD : ndarray (optional)
        the DD pair counts, returned if return_DD==True
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if (ra.ndim != 1) or (dec.ndim != 1) or (ra.shape != dec.shape):
        raise ValueError('ra and dec must be 1-dimensional '
                         'arrays of the same length')

    n_features = len(ra)
    Nbins = len(bins) - 1

    # draw a random sample with N points
    if (ra_R is None) | (dec_R is None):
        if verbose:
            print "two_point_angular says: generating random sample"
        ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                     (min(dec), max(dec)),
                                     2 * len(ra))

    data = np.asarray(ra_dec_to_xyz(ra, dec), order='F').T
    data_R = np.asarray(ra_dec_to_xyz(ra_R, dec_R), order='F').T

    # convert spherical bins to cartesian bins
    bins_transform = angular_dist_to_euclidean_dist(bins)
    if verbose:
        print "two_point_angular says: transform complete, calling two_point"
    return two_point(data, bins_transform, method=method, BT_D=BT_D,
                     BT_R=BT_R, data_R=data_R, random_state=rng,
                     return_trees=return_trees, RR=RR, return_RR=return_RR,
                     return_DD=return_DD)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def bootstrap_two_point_angular(ra, dec, bins, method='standard', BT_D=None, BT_R=None,
                                ra_R=None, dec_R=None, Nbootstraps=10, random_state=None, oversample_factor=1):
    """Angular two-point correlation function

    A separate function is needed because angular distances are not
    euclidean, and random sampling needs to take into account the
    spherical volume element.

    Parameters
    ----------
    ra : array_like
        input right ascention, shape = (n_samples,)
    dec : array_like
        input declination
    ra_R, dec_R : array_like (taken but ignored)
        the random sample to be used.  If you pass BT_R
        as an argument, you must also pass the random sample
        it was made from with this
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    BT_D : BallTree
        ball tree created with the data positions (taken but ignored)
    BT_R : BallTree
        ball tree created with the random positions (taken but ignored)
    method : string
        "standard" or "landy-szalay".
    Nbootstraps : int
        number of bootstrap resamples
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background
        
    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    dcorr : ndarray
        error estimate on dcorr (sample standard deviation of
        bootstrap resamples)
    bootstraps : ndarray
        The full sample of bootstraps used to compute corr and dcorr
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if (ra.ndim != 1) or (dec.ndim != 1) or (ra.shape != dec.shape):
        raise ValueError('ra and dec must be 1-dimensional '
                         'arrays of the same length')

    n_features = len(ra)
    Nbins = len(bins) - 1
    data = np.asarray(ra_dec_to_xyz(ra, dec), order='F').T

    # convert spherical bins to cartesian bins
    bins_transform = angular_dist_to_euclidean_dist(bins)

    bootstraps = []

    for i in range(Nbootstraps):
        # draw a random sample with N points if we don't have a random set already
        if (ra_R is None) or (dec_R is None):
            ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                         (min(dec), max(dec)),
                                         2 * len(ra))

        data_R = np.asarray(ra_dec_to_xyz(ra_R, dec_R), order='F').T

        if i > 0:
            # random sample of the data
            ind = np.random.randint(0, data.shape[0], oversample_factor*data.shape[0])
            data_b = data[ind]
        else:
            data_b = data

        bootstraps.append(two_point(data_b, bins_transform, method=method,
                                    data_R=data_R, random_state=rng))

    bootstraps = np.asarray(bootstraps)
    corr = np.mean(bootstraps, 0)
    corr_err = np.std(bootstraps, 0, ddof=1)

    return corr, corr_err, bootstraps
