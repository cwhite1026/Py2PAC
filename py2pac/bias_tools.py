#For things that might not want to be built into the angular catalog

import numpy as np
import scipy.special as sp
import scipy.integrate as ig
import scipy.optimize as opt
from scipy.interpolate import interp1d
import cosmology as cos
import hmf
from copy import *
import useful_things as u
#Speed of light
c=3.e5  #km/s

options={}
options['fname']="planck_for_hmf_transfer_out.dat"
mf=hmf.MassFunction(transfer_fit="FromFile", transfer_options=options, z=0)
delta_c = mf.delta_c

#==========================================================================
#==========================================================================
def apply_zbinned_linear_bias_correction(biases, z, fit_x='derived', 
                                         fit_to_set = 'fine', 
                                         fixed_slope=False, 
                                         outliers = False):
    """
    Takes a derived bias and a redshift.  Returns the actual bias according
    to whatever fit you choose.
    """

    if fit_to_set not in ['fine', 'coarse']:
        raise ValueError("You must choose a fit_to_set of either 'fine' or"
                         " 'coarse'.")

    #Read in the fits
    if fit_x == 'actual':
        if fixed_slope:
            raise ValueError("Cathy hasn't run the fits for x=actual "
                            "with a fixed slope.")
        if outliers:
            fn = '/Users/caviglia/Dropbox/plots/plots_for_notes/bias_correction/linear_fits.PICKLE'         
        else:
            fn = '/Users/caviglia/Dropbox/plots/plots_for_notes/bias_correction/linear_fits_no_outliers.PICKLE'   
    elif fit_x == "derived":
        if fixed_slope:
            if outliers:
                fn = "/Users/caviglia/Dropbox/plots/plots_for_notes/bias_correction/fixed_slope_0.73_linear_fits_x_derived.PICKLE"
            else:
                fn = "/Users/caviglia/Dropbox/plots/plots_for_notes/bias_correction/fixed_slope_0.73_linear_fits_x_derived_no_outliers.PICKLE"            
        else:
            if not outliers:
                fn = "/Users/caviglia/Dropbox/plots/plots_for_notes/bias_correction/linear_fits_x_derived_no_outliers.PICKLE"
            else:
                fn =  '/Users/caviglia/Dropbox/plots/plots_for_notes/bias_correction/linear_fits_x_derived.PICKLE'
    else:
        raise ValueError("Your options for fit_x are 'derived' and "
                         "'actual'.  You have not chosen either")
    fits = u.load_pickle(fn)
    
    #Figure out the fit to use
    z_centers = np.array([1.25, 2.0, 3.5, 5.25])
    which_fit = np.where(z_centers == z)[0][0]
    this_fit = fits[fit_to_set][which_fit]
    
    #Now get the value from the fit
    if fit_x == 'actual':
        #Pull the parameters out of the fit
        A = this_fit[0][0]
        B = this_fit[0][1]
        #Invert the function since X is the actual
        act = (biases - B) / (A + 1.)
    else:
        #Pull the parameters out of the fit
        A = this_fit[0][0]
        if fixed_slope:
            B=0.73
        else:
            B = this_fit[0][1]
        act = (1. - A) * biases - B
    
    return act
    
#==========================================================================
#==========================================================================
def pointwise_bias_from_ratio(thetas, cf, z, **kwargs):
    #Gets the bias by taking the ratio of CFs.
    # b(theta) = sqrt(w(theta)/w_dm(theta))

    #First, get the w_dm
    w_dm = dmc.dark_matter_w(thetas, z, **kwargs)

    #Turn it into bias
    b = np.sqrt(cf * 300 / w_dm)
    return b

#==========================================================================
#==========================================================================
def limber_equation(Nz, r0, gamma, integral_min=0, integral_max= 100):
    #Convert the 3D r0 and gamma to the angular CF A and beta
    #Nz is a function(!), N(z), the redshift selection window function 
    #for the galaxies
    #Expects units of Mpc for r0
    beta = gamma-1
 
    def num_integrand(z, Nz):
        #This is the function we'll be integrating over in the numerator
        #z window function squared
        integrand = (Nz(z))**2.   
        #change in comoving distance with angle, DA in Mpc
        integrand *= ((1.+z)*cos.DA(z))**(1.-gamma)  
        #change in comoving distance with redshift, c in km/s and H in km/s/Mpc
        integrand *= (c/cos.H(z))**(-1.)  
        return integrand

    num_integral=ig.quad(num_integrand, integral_min, integral_max, 
                         args=(Nz))[0]
    denom_integral=ig.quad(Nz, integral_min, integral_max)[0]

    A = r0**gamma * sp.beta(0.5, (gamma-1)/2.) * num_integral
    A /= denom_integral**2.

    return A, beta


#==========================================================================
#==========================================================================
def inverted_limber_equation(A, beta, minimization_tol=1.e-5, 
                             use_Nz="custom", ignore_failure=False, 
                             return_z_center = True, 
                             r0_guess=1., **kwargs):
    """
    Get r0 and gamma from the Limber equation by minimizing the difference
    between the A provided and the A returned by the Limber equation.  Note
    that A and beta must be for theta in *radians*

    Parameters
    ----------
    A : scalar
        Amplitude of the power law fit to an angular correlation function
        A*theta^(-beta) for theta in radians.
    beta : scalar
        Power in a power law fit to an angular correlation function
    minimization_tol : scalar (optional)
        The tolerance to pass to the scipy optimize as tol
    return_z_center : bool (optional)
        If True, returns r0, gamma, z_center.  If False, returns r0, gamma.
        Default is True.
    use_Nz : string (optional)
        A string denoting which option you want to use for the N(z).
        Options are ['custom', 'gaussian', 'tophat'].  Each of these need
        more information.

        If you use use_Nz='custom', you must provide a function that gives
        N(z) with the argument Nz=func(z).

        If you set use_Nz='gaussian', you must pass mu=<central redshift>, 
        sigma=<standard deviation>.

        If you set use_Nz='tophat', you must pass z_low=<lower bound>, 
        z_high=<upper bound>.
    ignore_failure : bool
        If ignore_failure==True, then the closest value obtained will be 
        returned rather than raising an error.  The default is
        ignore_failure==False, which raises an error when the optimization
        routine fails.
    Nz : function (optional)
        A function that returns the number of galaxies as a function of
        redshift.  Only needed if use_Nz='custom', which is the default
    mu : scalar
        The central value of the Gaussian distribution of N(z).  Only 
        needed if use_Nz=='gaussian'
    sigma : scalar
        The standard deviation of the N(z) distribution.  Only needed if
        use_Nz=='gaussian'
    z_low : scalar
        The lower limit of the top hat N(z) distribution.  Only needed if
        use_Nz=='tophat'
    z_high : scalar
        The upper limit of the top hat N(z) distribution.  Only needed if
        use_Nz=='tophat'

    Returns
    -------
    r0 : scalar
        The scale length of the 3D power law corresponding to the input 
        angular correlation function fit.  In Mpc.
    gamma : scalar
        The power of the 3D power law corresponding to the input angular
        correlation function fit.
    z_center : scalar (optional)
        Only returned if return_z_center == True.  This is the median 
        redshift for the N(z) distribution given.
    """

    #Convert the power.
    gamma = beta+1

    #Figure out what's going on with N(z)
    if (use_Nz.lower()=='custom') or (use_Nz.lower()=='custom_dict'):
        if "Nz" not in kwargs.keys():
            raise TypeError("When calling inverted_limber_equation with "
                            "use_Nz='custom', which is the default, you "
                            "must also pass Nz=func(z).")
        else:
            N = kwargs['Nz']
            
        #Make sure that the limits are extant
        if "z_low" not in kwargs.keys():
            kwargs["z_low"] = 0
        if "z_high" not in kwargs.keys():
            kwargs["z_high"] = 100

        #Find the median
        z_grid = np.linspace(kwargs["z_low"], kwargs["z_high"], int(1e5))
        fcn_grid = N(z_grid)
        if "z_center_weighting" in kwargs.keys():
            z_center_weighting = kwargs["z_center_weighting"]
        else:
            z_center_weighting = None
        if z_center_weighting == "squared":
            fcn_grid = fcn_grid **2.
        sums = np.cumsum(fcn_grid)  #This is basically a poor-man's integral to each point in z space
        diff = abs(sums - sums[-1]/2.)
        index = np.where(diff == min(diff))[0][0]
        z_center = z_grid[index]
            
    elif use_Nz.lower()=='gaussian':
        if ("mu" not in kwargs.keys()) or ("sigma" not in kwargs.keys()):
            raise TypeError("When calling inverted_limber_equation with "
                            "use_Nz='gaussian', you must also pass "
                            "mu=z_center and sigma=z_width, which are the "
                            "center and width of the Gaussian redshift "
                            "distribution, respectively.")
        mu=kwargs['mu']
        sigma=kwargs['sigma']
        
        #Make sure that the limits are extant
        if "z_low" not in kwargs.keys():
            kwargs["z_low"] = 0
        if "z_high" not in kwargs.keys():
            kwargs["z_high"] = 100
        z_center = mu
            
        def N(z):
            return np.exp(- (z-mu)**2./(2.*sigma**2))
        
    else:
        have_zlow= "z_low" in kwargs.keys()
        have_zhigh= "z_high" in kwargs.keys()
        if (not have_zlow) or (not have_zhigh):
            raise TypeError("When calling inverted_limber_equation with "
                            "use_Nz='tophat', you must also pass "
                            "z_low=lower and z_high=upper, which are the "
                            "lower and upper bounds of the top hat "
                            "function")
        z_center = (kwargs["z_low"] + kwargs["z_high"])/2.
        def N(z):
            return 1

    #Now define the function that we're minimizing.  This is different for
    #top hat and non-top hat N(z)s
    print "Integrating from ",  kwargs["z_low"], "to", kwargs["z_high"], " for the Limber stuff"
    def func_to_minimize(r0):
        A_guess, beta = limber_equation(N, r0, gamma, 
                                        integral_min = kwargs["z_low"], 
                                        integral_max = kwargs["z_high"])
        return abs(A-A_guess)
    
    res = opt.minimize(func_to_minimize, r0_guess, method='Powell', 
                       tol=minimization_tol)
    # print "scipy.optimize.minimize output: minimizing abs(A-A_guess)"
    # print res
    
    if res.success or ignore_failure:
        r0 = float(res.x)
        to_return = [r0, gamma]
        if return_z_center:
            to_return.append(z_center)
        return to_return
    else:
        print ("inverted_limber_equation says: the minimization routine "
               "doesn't think it found a solution with the given tolerance."
               "  The best A it could come up with is"+str(res.fun/A * 100.)
               +"% different from the A value you fed it.  If this is "
               "satisfactory, run again with ignore_failure=True.")
        raise RuntimeError("Inverting the Limber equation failed.")

#==========================================================================
#==========================================================================
def sigma_z(m, z):
    #Get the RMS mass fluctuations for a sphere of mass M (in Msun/h) at redshift z
    # print "in sigma_z"
    #----------------------------------
    #- Figure out what exactly we have
    #----------------------------------
    # Make sure that even if we have only one redshift, we can treat it as an array
    try:
        z=np.array(z)
        n_zs=len(z)
    except TypeError:
        z=np.array([z])
        n_zs=len(z)
        
    #Now do the same for the masses unless it can be cast as a scalar
    try:
        m=np.float(m)
    except TypeError:
        m=np.array(m)
        n_ms=len(m)
    else:
        n_ms=1

    #Check that we have the same number of ms and zs if n_ms>1
    if (n_ms != 1) and (n_ms != n_zs):
        raise ValueError("You must give sigma_z either the same number of ms and zs or only one m.")

    #--------------------------------------------
    #- Figure out mass parameters for the HMF mf
    #--------------------------------------------
    #If we only have one mass, make an array the length of z that's the same mass everywhere
    if n_ms==1:
        m = m*np.ones(n_zs)
    m_min = np.log10(m)
    m_max = m_min+1
    step = 0.4
    
    #--------------------------------------------
    #- Compute and return sigmas
    #--------------------------------------------
    sigma = np.zeros(n_zs)
    for i in np.arange(n_zs):
        mf.update(Mmin=m_min[i], Mmax=m_max[i], dlog10m=step, z=z[i])
        sigma[i] = mf.sigma[0]

    return sigma


#==========================================================================
#==========================================================================
def sigma_m(m, z):
    #Get the RMS mass fluctuations for a sphere of mass M (in Msun/h) at redshift z
    #This can have an array of masses but only one z
    # print "in sigma_m"
    #--------------------------------------------
    #- Figure out mass parameters for the HMF mf
    #--------------------------------------------
    #Make sure that even if we have only one mass, we can treat it as an array
    try:
        m=np.array(m)
        n_ms=len(m)
    except TypeError:
        m=np.array([m])
        n_ms=len(m)
    #Get the min and max (The hmf routine expects log(M))
    m_min = np.log10(m.min())
    m_max = np.log10(m.max())
    #Make sure that we're ok if we only have one guy
    if m_min == m_max:
        m_max = m_min + 1
    #Make our step size such that we have 500 masses and we'll include m_max
    #so that our interpolation doesn't get sad
    step = (m_max - m_min + .2) / 500.

    #--------------------------------------------
    #- Compute and return sigmas
    #--------------------------------------------
    #Update the properties of the mass function to what I want to use now
    mf.update(Mmin=m_min-.1, Mmax=m_max+.1, dlog10m=step, z=z)

    #Pull out the masses and sigmas that we have
    log_m_mf = np.log10(mf.M)
    sigma_mf = mf.sigma

    #Make an interpolation function to get sigma given an M
    interp_fcn = interp1d(log_m_mf, sigma_mf)
    sigmas_requested = interp_fcn(np.log10(m))

    #Return the interpolated sigmas
    return sigmas_requested


#==========================================================================
#==========================================================================
def sigma_r(r, z):
    #Get the RMS mass fluctuations for a sphere of radius r (in Mpc/h) at redshift z
    #This can have an array of masses but only one z
    # print "in sigma_r"
    mf.update(z=z)
    
    #----------------------------------------------
    #- Figure out radius parameters for the HMF mf
    #----------------------------------------------
    #Make sure that even if we have only one mass, we can treat it as an array
    try:
        r=np.array(r)
        n_rs=len(r)
    except TypeError:
        r=np.array([r])
        n_rs=len(r)
    #Get the min and max
    r_min = r.min()
    r_max = r.max()

    #Since we can't actually directly change r, see if they're already in the range mf has
    mf_has_min=min(mf.radii)
    mf_has_max=max(mf.radii)
    #If they're not, walk the range out to include them
    while r_min < mf_has_min:
        mf.update(Mmin = mf.Mmin-.5)
        mf_has_min=min(mf.radii)
    while r_max > mf_has_max:
        mf.update(Mmax = mf.Mmax+.5)
        mf_has_max=max(mf.radii)
        
    #Pull out the radii and sigmas that we have
    r_mf = mf.radii
    sigma_mf = mf.sigma

    #Make an interpolation function to get sigma given an r
    interp_fcn = interp1d(r_mf, sigma_mf)
    sigmas_requested = interp_fcn(r)
    # print "returning"
    #Return the interpolated sigmas
    return sigmas_requested


#==========================================================================
#==========================================================================
def nu(m, z):
    #This is the mass-esque parameter used in the Tinker et al 2005 papers
    #Can be asked for a range of masses xor redshifts (not both)

    #----------------------------------------------------------------
    #- Figure out whether we're doing a range of masses or of zs
    #----------------------------------------------------------------
    #How many ms do we have?
    # print m, type(m)
    if (type(m) == type(np.zeros(1))):
        n_ms = len(m)
    elif (type(m) == type([0])):
        m = np.asarray(m)
        n_ms = len(m)
    else:
        n_ms = 1
        
    #How many zs do we have?
    if (type(z) == type(np.zeros(1))):
        n_zs = len(z)
    elif (type(z) == type([0])):
        z = np.asarray(z)
        n_zs = len(z)
    else:
        n_zs = 1

    if (n_ms < 1) or (n_zs < 1):
        #We got an empty array for either the mass or the redshift
        raise ValueError("You cannot have less than one of either m or z")
    elif (n_zs == 1):
        #We have only 1 z and we've already checked that we have at least 1 m
        axis='m'
    elif (n_ms == 1):
        #We have only 1 m and we've already checked that we have at least 1 z
        axis='z'
    elif (n_zs == n_ms):
        #We have more than one of each, but the two arrays are the same length
        axis='z'
    else:
        #If we're here, we have two arrays of length >1 and the lengths
        #don't match
        raise ValueError("If you have multiple ms and multiple zs, you must have the same number of each")


    #--------------------------------------------------
    #- Use the appropriate routine and return the nu
    #--------------------------------------------------
    #Get the RMS fluctuations
    if axis=='m':
        sigma=sigma_m(m, np.float(z))
    if axis=='z':
        sigma=sigma_z(m, z)

    #Return the ratio
    return delta_c/sigma


#==========================================================================
#==========================================================================
def halo_bias_nu(nu, Delta):
    #Return the bias as a function of nu: delta_c(z)/sigma(M, z),
    #delta_c, and the overdensity Delta- Tinker et al 2010, equation 6
    
    #Definitions from table 2
    y=np.log10(Delta)
    exponential = np.exp(-(4./y)**4.)
    A = 1. + 0.24 * y * exponential
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * exponential
    c = 2.4

    #Write out the bias
    fraction = (nu**a) / (nu**a + delta_c**a)
    bias = 1. - (A*fraction) + (B * nu**b) + (C * nu**c)

    return bias

#==========================================================================
#==========================================================================
def halo_bias_M(M, z):
    #Return the bias as a function of halo mass and redshift- Tinker et al 2005
    #This is a more accessible front end to halo_bias_nu

    #Params
    v = nu(M, z)

    return halo_bias_nu(v, 200.)

#==========================================================================
#==========================================================================
# def cheating_bias_to_halo_mass(bias, z, minimization_tol=1.e-5, ignore_failure=False,
#                                logmass_guess=10):
#     #Return the halo mass that has the same bias as given for the redshift given.
#     #It's the cheating way because doing it properly would use an HOD model and this
#     #just assumes that all the galaxies in the bin are the same halo mass

#     def func_to_minimize(logM):
#         #halo bias given halo mass
#         bias_guess = halo_bias_M(10.**logM, z)
#         return abs(bias-bias_guess)
    
#     res = opt.minimize(func_to_minimize, logmass_guess, method='Powell', tol=minimization_tol)
#     print "scipy.optimize.minimize output: minimizing abs(bias-bias_guess)"
#     print res
    
#     if res.success or ignore_failure:
#         logmass = float(res.x)
#         return 10.**logmass
#     else:
#         print "cheating_bias_to_halo_mass says: the minimization routine doesn't think it found a solution with the given tolerance.  The best bias it could come up with is", res.fun/bias * 100., "% different from the A value you fed it.  If this is satisfactory, run again with ignore_failure=True."
#         return 0

#==========================================================================
#==========================================================================
def cheating_bias_to_halo_mass(b, z):
    #Return the halo mass that has the same bias as given for the redshift given.
    # Do this by interpolating the bias(Mhalo) relation to the halo mass you care about.
    #It's the cheating way because doing it properly would use an HOD model and this
    #just assumes that all the galaxies in the bin are the same halo mass

    #Get the bias(Mhalo) relation
    logM=np.linspace(6, 16, 500)
    biases = halo_bias_M(10.**logM, z)

    #Make a function that interpolates the bias(Mhalo) relation
    interp_fcn = interp1d(biases, logM)
    # print interp_fcn(2)
    # print b
    logmass = interp_fcn(b)
    # print "logmass is ", logmass
    
    return 10.**logmass

    
#==========================================================================
#==========================================================================
def mhalo_from_A_beta(A, beta, return_bias=True, **kwargs):
    """
    Get r0 and gamma from the Limber equation by minimizing the difference
    between the A provided and the A returned by the Limber equation.  Note
    that A and beta must be for theta in *radians*

    Parameters
    ----------
    A : scalar
        Amplitude of the power law fit to an angular correlation function
        A*theta^(-beta) for theta in radians.
        
    beta : scalar
        Power in a power law fit to an angular correlation function

    return_bias : bool
        If True, the function will return mhalo, bias.  If False, the
        function will return just mhalo
        
    **kwargs
        Further keyword arguments are passed to inverted_limber_equations.
        There are a lot.  I'm not typing them twice.
    """

    A_rad, beta_rad = fit_params_degrees_to_radians(A, beta)
    # print "new A, beta for radians: ", A_rad, beta_rad
    r0, gamma, z_center = inverted_limber_equation(A_rad, beta_rad, **kwargs)
    print "r0: ", r0, "  gamma: ", gamma, "z_center: ", z_center
    b = bias(r0, gamma, z_center)
    print "bias: ", b
    print ""
    mh = cheating_bias_to_halo_mass(b, z_center)

    to_return=mh
    if return_bias:
        to_return = to_return, b
    return to_return

#==========================================================================
#==========================================================================

def bias(r0, gamma, z):
    #This is the bias as given by sigma_8_gal/sigma_8
    #I will have sigma(R) soon, so I can do sigma_8 that way.

    denom=(3-gamma)*(4-gamma)*(6-gamma) * 2.**gamma
    sigma_squared = 72. * (r0/8.)**gamma / (denom)
    #This assumes that r0 is given in comoving Mpc/h, which is what the Limber eqn seems to give you (if we trust Barone-Nugent)
    sigma_8_gal=np.sqrt(sigma_squared)
    print "done with gal: sigma is", sigma_8_gal
    sigma_8=sigma_r(8., z)[0]
    print "cosmological sigma 8: ", sigma_8

    return sigma_8_gal/sigma_8

#==========================================================================
#==========================================================================
def fit_params_degrees_to_radians(A, beta):
    #Given the fit for theta in degrees, return the fit for theta in radians.
    return A*(np.pi/180.)**beta, beta
