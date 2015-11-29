#This is a copy of the astroML.correlation package with some
#modifications made by Cathy White.  The main body of work was done
#by Jake VanderPlas.

import numpy as np
import numpy.ma as ma
from sklearn.neighbors import BallTree
from astroML.utils import check_random_state

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def uniform_sphere(RAlim, DEClim, size=1):
    """
    Draw a uniform sample on a sphere

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
    """
    Convert ra & dec to Euclidean points

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
    """
    Convert Euclidean points to ra and dec

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
    """
    The inverse of angular_dist_to_euclidian_dist.

    Parameters
    ----------
    dist: array-like or scalar
        Euclidian distance in the same units as r
    r: scalar (optional)
        The radius of the sphere that you're projecting onto.  Default
        is 1.

    Returns
    -------
    angular_separation: numpy array or scalar (matches input)
       Angular separation in degrees that corresponds to the Euclidean
       distance(s)
    """

    #If it's an array, make sure that it's a numpy array so that nothing
    #weird happens in the vector operations
    try:
        n_dists = len(dist)
    except TypeError:
        n_dists = 1
    else:
        dist = np.array(dist)

    #Return the distance
    return 180./np.pi * 2. * np.asin(dist/(2.*r))


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def angular_dist_to_euclidean_dist(D, r=1):
    """
    Convert angular distances to euclidean distances

    Parameters
    ----------
    D: array-like or scalar
       Angle in degrees
    r: scalar (optional)
       The radius of the sphere you're projecting onto

    Returns
    -------
    dist: array-like or scalar
        Euclidean distance in whatever units r is in
    """

    #If it's an array, make sure that it's a numpy array so that nothing
    #weird happens in the vector operations
    try:
        n_Ds = len(D)
    except TypeError:
        n_Ds = 1
    else:
        D = np.array(D)

    #Return the distance(s)
    return 2 * r * np.sin(0.5 * D * np.pi / 180.)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def two_point(data, bins, BT_D=None, BT_R=None, method='standard',
              data_R=None, random_state=None, return_trees=False, 
              verbose=False, RR=None, return_RR=False, return_DD=False):
    #Edited by CW to allow user to supply more things and have more things
    #returned.
    """
    Two-point correlation function in Euclidean space.  Options to return
    a number of things.  What gets returned is up to the user but the order
    will always be correlation_function, data_balltree, random_balltree,
    random_random, data_data.  If the user asks for a subset of those, the
    list will be shorter but the order will be maintained.

    Parameters
    ----------
    data : array_like
        Input data, shape = [n_samples, n_features]
    bins : array_like
        Bins within which to compute the 2-point correlation.
        Shape = Nbins + 1
    BT_D : BallTree (optional)
        Ball tree created with the data positions
    BT_R : BallTree (optional)
        Ball tree created with the random positions
    method : string (optional)
        "standard" or "landy-szalay".  Default is 'standard'.
    data_R : array_like (optional if no BT_R)
        If specified, use this as the random comparison sample.  This must
        be included if you wish to use a pre-computed random ball tree
    random_state : integer, np.random.RandomState, or None (optional)
        Specify the random state to use for generating background.  Not
        used if the randoms are provided by the user.  Default is None
    RR : 1D array-like, shape = Nbins
        If this exact set of randoms and theta bins has been
        run, you can supply the RR counts and not calculate them again.
        You also need the data if you're running with method='landy-szalay'
    return_trees : boolean (optional)
        If True, the returns will include the data and random ball trees.
        Default is False.
    return_RR : boolean (optional)
        If you know you'll be running a CF with this
        exact same random sample and binning (like with a bootstrap),
        you can get the RR counts returned and feed them back in the
        next time
    return_DD : boolean (optional)
        In case you want to fit to the pair counts rather
        than the w(theta) estimator, you can get this back too
    verbose: boolean (optional)
        Determines whether or not the function narrates what it's doing.
        Default is False.
 
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
def two_point_angular(ra, dec, bins, BT_D=None, BT_R=None,
                      method='standard', ra_R=None, dec_R=None,
                      random_state=None, return_trees=False, verbose=False,
                      RR=None, return_RR=False, return_DD=False):
    """
    Angular two-point correlation function

    A separate function is needed because angular distances are not
    euclidean, and random sampling needs to take into account the
    spherical volume element.

    There are a number of options for what gets returned.  The order
    will always be correlation_function, data_balltree, random_balltree,
    random_random, data_data.  If the user asks for a subset of those, the
    list will be shorter but the order will be maintained.

    Parameters
    ----------
    ra : array_like
        input right ascention, shape = (n_samples,)
    dec : array_like
        input declination, shape = (n_samples,)
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    BT_D : BallTree (optional)
        ball tree created with the data positions.  The positions given to
        the BallTree should be Euclidean and not angular
    BT_R : BallTree (optional)
        ball tree created with the random positions. The positions given to
        the BallTree should be Euclidean and not angular
    method : string (optional)
        "standard" or "landy-szalay".  Default is 'standard'
    ra_R, dec_R : array_like (optional if no BT_R)
        the random sample to be used.  If you pass BT_R
        as an argument, you must also pass the random sample
        it was made from with this
    random_state : integer, np.random.RandomState, or None (optional)
        specify the random state to use for generating background.
        Default is None
    RR : array-like, shape = Nbins (optional)
        If this exact set of randoms and theta bins has been
        run, you can supply the RR counts and not calculate them again.
        You also need the data if you're running landy-szalay.
    return_RR : boolean (optional)
        If you know you'll be running a CF with this
        exact same random sample and binning (like with a bootstrap),
        you can get the RR counts returned and feed them back in the
        next time.  Default is False.
    return_DD : boolean (optional)
        In case you want to fit to the pair counts rather than the w(theta)
        estimator, you can get this back too.  Default is False
    return_trees : boolean (optional)
        if True, the returns will include the ball trees for the data and
        random sets.  Default is False
        
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

