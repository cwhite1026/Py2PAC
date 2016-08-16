#For things that might not want to be built into the angular catalog

import numpy as np
import numpy.ma as ma
import scipy.special as sp
import scipy.integrate as ig
import scipy.optimize as opt
from scipy.interpolate import interp1d
import cosmology as cos
import hmf
from copy import *
#Speed of light
c=3.e5  #km/s

options={}
options['fname']="planck_for_hmf_transfer_out.dat"
hmf_cosmo = hmf.cosmo.get_cosmo("Planck15")
mf = hmf.MassFunction(z=0, transfer_model = hmf.transfer_models.FromFile, transfer_params={'cosmo': hmf_cosmo, 'model_parameters':options})
# mf = hmf.MassFunction(z=0, transfer_model = hmf.transfer_models.FromFile, cosmo=hmf_cosmo, transfer_params=options)
delta_c = mf.delta_c


#==========================================================================
#==========================================================================
def fit_results_to_halo_mass(fit_results, zbin, **limber_kwargs):
    """
    Take the dictionary from fitting.bootstrap_fit and turn it into 
    corrected and uncorrected biases and halo masses.
    """
    
    #Start a dictionary to save things in
    res_dict = {}
    
    #Do one inversion just to get the z_actual (center of the bin)
    __, __, z_actual, = inverted_limber_equation(fit_results['median_A'], 
                                                fit_results['median_beta'], 
                                                **limber_kwargs)
    
    #Turn the bootstrapped As and betas and turn them into biases and halo 
    #masses.    
    have_keys = fit_results.keys()
    if ('bootstrap_As' in have_keys) and ('bootstrap_betas' in have_keys):
        #We actually have bootstraps to do- do them
        nboots = len(fit_results['bootstrap_As'])
        boot_biases = -np.ones(nboots)
        for i in np.arange(nboots):
            try:
                boot_biases[i] = bias_from_A_beta(fit_results['bootstrap_As'][i], fit_results['bootstrap_betas'][i], **limber_kwargs)
            except RuntimeError:
                print "boot", i, "failed Limber Eqn inversion- moving on"
        res_dict['boot_biases'] = boot_biases
        
        res_dict['boot_bias_props'] = bootstrap_biases_to_props(boot_biases, zbin, z_actual)
    
    #Turn the median A and beta pair into biases and halo masses.
    #First do the uncorrected version
    temp = mhalo_from_A_beta(fit_results['median_A'], 
                             fit_results['median_beta'], return_bias=True, 
                             **limber_kwargs)
    res_dict['mhalo_from_medians_uncorr'], res_dict['bias_from_medians_uncorr'] = temp
    #Now correct
    res_dict['bias_from_medians_corr'] = apply_zbinned_linear_bias_correction(
                res_dict['bias_from_medians_uncorr'], zbin)
    res_dict['mhalo_from_medians_corr'] = cheating_bias_to_halo_mass(res_dict['bias_from_medians_corr'], z_actual)
    
    #Turn the unweighted A and beta pair into bias and halo mass
    temp = mhalo_from_A_beta(fit_results['unweighted_A'], 
                             fit_results['unweighted_beta'], return_bias=True, 
                             **limber_kwargs)
    res_dict['mhalo_from_unwt_uncorr'], res_dict['bias_from_unwt_uncorr'] = temp
    #Now correct
    res_dict['bias_from_unwt_corr'] = apply_zbinned_linear_bias_correction(
                res_dict['bias_from_unweighteds_uncorr'], zbin)
    res_dict['mhalo_from_unwt_corr'] = cheating_bias_to_halo_mass(res_dict['bias_from_unwt_corr'], z_actual)
            
    return res_dict
    
#==========================================================================
#==========================================================================
#==========================================================================
def bootstrap_biases_to_props(boot_biases, zbin, z_actual):
    """
    Takes the biases from each of the bootstraps, correct, and turn into
    halo masses with error bars.
    """
    #Get stats from uncorrected biases and see if we have anything in this
    #bin
    temp = np.percentile(boot_biases, (16, 50, 84))
    uncorr_bias_low, uncorr_bias_median, uncorr_bias_high = temp
    
    #Only do other things if we have a not -1 bias
    if uncorr_bias_median <= 0:
        uncorr_bias_median, uncorr_bias_lowerr, uncorr_bias_higherr= -1,0,0
        corr_bias_median, corr_bias_lowerr, corr_bias_higherr= -1,0,0
        uncorr_mass_median, uncorr_mass_lowerr, uncorr_mass_higherr= -1,0,0
        corr_mass_median, corr_mass_lowerr, corr_mass_higherr= -1,0,0        
        
    else:
        #We have OK biases: correct the biases
        corrected_boot_biases = apply_zbinned_linear_bias_correction(boot_biases, zbin)
    
        #Get stats from corrected biases
        temp = np.percentile(corrected_boot_biases, (16, 50, 84))
        corr_bias_low, corr_bias_median, corr_bias_high = temp
    
        uncorr_bias_lowerr = uncorr_bias_median - uncorr_bias_low
        uncorr_bias_higherr = uncorr_bias_high - uncorr_bias_median
        corr_bias_lowerr = corr_bias_median - corr_bias_low
        corr_bias_higherr = corr_bias_high - corr_bias_median
    
        #Convert both sets to masses
        n_boots = len(boot_biases)
        boot_masses = -np.ones(n_boots)
        corrected_boot_masses = -np.ones(n_boots)
        for i in np.arange(n_boots):
            try:
                boot_masses[i] = np.log10(cheating_bias_to_halo_mass(boot_biases[i], z_actual))
            except ValueError:
                print "Couldn't convert uncorrected bias", i, ", ", boot_biases[i]
                
            try:
                corrected_boot_masses[i] = np.log10(cheating_bias_to_halo_mass(corrected_boot_biases[i], z_actual))
            except ValueError:
                print "Couldn't convert corrected bias", i, ", ", corrected_boot_biases[i]
                    
        #Get stats for masses
        temp = np.percentile(boot_masses[boot_masses>0], (16, 50, 84))
        uncorr_mass_low, uncorr_mass_median, uncorr_mass_high = temp
        temp = np.percentile(corrected_boot_masses[corrected_boot_masses>0], (16, 50, 84))
        corr_mass_low, corr_mass_median, corr_mass_high = temp
    
        uncorr_mass_lowerr = uncorr_mass_median - uncorr_mass_low
        uncorr_mass_higherr = uncorr_mass_high - uncorr_mass_median
        corr_mass_lowerr = corr_mass_median - corr_mass_low
        corr_mass_higherr = corr_mass_high - corr_mass_median  
    
    return_dict = {}
    return_dict['uncorrected'] = {'bias_median' : uncorr_bias_median, 
                                  'bias_lowerr' : uncorr_bias_lowerr,
                                  'bias_higherr': uncorr_bias_higherr,
                                  'mass_median' : uncorr_mass_median,
                                  'mass_lowerr' : uncorr_mass_lowerr,
                                  'mass_higherr': uncorr_mass_higherr,}
    return_dict['corrected'] = {'bias_median' : corr_bias_median, 
                                'bias_lowerr' : corr_bias_lowerr,
                                'bias_higherr': corr_bias_higherr,
                                'mass_median' : corr_mass_median,
                                'mass_lowerr' : corr_mass_lowerr,
                                'mass_higherr': corr_mass_higherr,}                                  
    return return_dict

#==========================================================================
#==========================================================================
def apply_zbinned_linear_bias_correction(biases, z):
    """
    Takes a derived bias and a redshift and corrects it according to the 
    SHAM calculations.  This version got rid of the use of the file by
    replacing it with the default dictionary.  It's fit with y=mx+b where
    x is the derived bias, it's fit to the finely binned masses, without
    a fixed slope and without outliers.
    """

    #From the file '/Users/cathy/Dropbox/plots/plots_for_notes/bias_correction/linear_fits_x_derived_no_outliers.PICKLE', 
    #fit_x='derived', fit_to_set = 'fine', fixed_slope=False, 
    #outliers = False
    fits =  {0: (np.array([ 0.64637876, -0.48714656]),
                 np.array([[  6.89301047e-05,  -1.22486285e-04],
                        [ -1.22486285e-04,   2.91442887e-04]])),
            1: (np.array([ 0.68028928, -0.69763767]), 
                np.array([[ 0.0001957 , -0.00051472],
                          [-0.00051472,  0.00187745]])),
            2: (np.array([ 0.68591907, -1.13984185]), 
                np.array([[ 0.00061458, -0.00248462],
                          [-0.00248462,  0.01204042]])),
            3: (np.array([ 0.67302948, -1.7106559 ]), 
                np.array([[ 0.00119667, -0.00944455],
                [-0.00944455,  0.07910462]]))}
    
    #Figure out the fit to use
    z_centers = np.array([1.25, 2.0, 3.5, 5.25])
    which_fit = np.where(z_centers == z)[0][0]
    this_fit = fits[which_fit]
    
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
def make_Nz_func_from_table(nzs, redshifts = np.linspace(0, 10, 1001)):
    """
    Takes a list of redshifts and the corresponding N(z)s to make a
    function N(z) that can be passed to the Limber equation functions
    
    Parameters
    ----------
    nzs : 1D array-like
        A list of N(z) values corresponding to the redshifts given
        
    redshifts : 1D array-like
        A list of redshifts for the N(z) values.  The defaults corresponds
        to the CANDELS PDF N(z)s
    """
    
    #Make a function that returns the correct value(s) of N(z)
    def temp(z, redshifts = redshifts, nzs = nzs):
        #See if we have one z or many
        try:
            n = len(z)
        except TypeError:
            have_array=False
        else:   
            have_array= True
		
        if have_array:
            #If we have multiple N(z)s, get all the N(z)s we can.  For the
            #ones outside our range, set them to be zero
            ans = np.zeros(n)
            zero_mask = ma.masked_outside(z, redshifts.min(), redshifts.max()).mask
            for i, thisz in enumerate(z):
                diff = redshifts-thisz
                index = np.where(abs(diff) == abs(diff).min())[0][0]
                ans[i] = nzs[index]
            ans[zero_mask] = 0.
        else:
            #If we only have one, return the right N(z) for that point.
            if (z < redshifts.min()) or (z > redshifts.max()):
                ans = 0 
            else:
                diff = redshifts - z
                index = np.where(abs(diff) == abs(diff).min())[0][0]
                ans = nzs[index]
        return ans
        
    #Return the function
    return temp


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
    Get halo mass from the A and beta (fit with theta in degrees)

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
def bias_from_A_beta(A, beta, **kwargs):
    """
    Get just a bias from A and beta, fit with theta in degrees

    Parameters
    ----------
    A : scalar
        Amplitude of the power law fit to an angular correlation function
        A*theta^(-beta) for theta in radians.
        
    beta : scalar
        Power in a power law fit to an angular correlation function
        
    **kwargs
        Further keyword arguments are passed to inverted_limber_equations.
        There are a lot.  I'm not typing them twice.
    """

    A_rad, beta_rad = fit_params_degrees_to_radians(A, beta)
    r0, gamma, z_center = inverted_limber_equation(A_rad, beta_rad, **kwargs)
    print "r0: ", r0, "  gamma: ", gamma, "z_center: ", z_center
    b = bias(r0, gamma, z_center)
    print "bias: ", b
    print ""

    return b

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
