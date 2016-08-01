#This is for the power law fitting the right way
import numpy as np
import numpy.ma as ma
import numpy.random as rand
import matplotlib.pyplot as plt
import cf_useful_things as cu
from scipy import optimize as opt
import bias_tools as t
import scipy.integrate as intg
import scipy.stats as stat
import diagnostics as diag
import statsmodels.robust as robust
from copy import deepcopy

#ALL ANGLES ARE ASSUMED TO BE IN DEGREES

#==========================================================================
def bootstrap_candels_fields(before_field, after_field, **kwargs):
    #Read in the 5 CFs and bootstrap sample and fit them to get
    #an estimate of the median and MAD on the parameter estimates

    #First, read in the set of CFs
    if 'hyphen' in kwargs.keys():
        fields = cu.load_candels_cf_set(before_field, after_field, 
                            hyphen=kwargs['hyphen'])
    else:
        fields = cu.load_candels_cf_set(before_field, after_field)

    #Then run the bootstrap thing on a set of catalogs.
    return bootstrap_fit(fields, **kwargs)

#==========================================================================
def bootstrap_fit(fields, IC_method="offset", n_fit_boots=200, 
                  return_envelope=True, return_boots=True, **kwargs):
    #Organize the overall work flow based on what I want
    
    #Check that we have a valid IC type
    IC_method = IC_method.lower()
    if IC_method not in ["offset", "adelberger"]:
        raise ValueError("You must choose either 'offset' or 'adelberger'"
                        " for IC_method.  Capitalization is ignored.")
        
    #See how many CFs we have
    try:
        n_cfs = len(fields)
    except TypeError:
        fields = [fields]
        n_cfs = len(fields)
    #And make sure we can index it conveniently
    fields= np.array(fields)
    thetas, cfs, errs = get_info_from_catalog_set(fields, **kwargs)
    
    #First, fit just the main set- the unweighted average
    if IC_method == 'adelberger':
        fit_to_unweighted = iterative_fit(fields, **kwargs)
    else:
        cf = np.mean(cfs, axis=0)
        error_bars = np.sqrt(np.sum(errs**2., axis=0))/n_cfs
        fit_to_unweighted = minimize_fit_to_cf(thetas, cf, 
                                                error_bars, **kwargs)
        
    #Now do the bootstraps
    if IC_method == 'adelberger':
        boots = bootstrap_fit_with_adelberger(fields, n_fit_boots=n_fit_boots,
                                                 **kwargs)
    else:
        boots = bootstrap_fit_with_offset(fields, n_fit_boots=n_fit_boots,
                                            **kwargs)
    
    #Process bootstraps to error bars and medians
    theta_range = [thetas.min(), thetas.max()]
    results = process_bootstraps_to_errors(theta_range, boots, 
                                        return_envelope=return_envelope, 
                                        return_boots=return_boots, 
                                        **kwargs)
    
    #Add the unweighted fit to the boot dictionary 
    #A[-1], beta[-1], ICs, chi2[-1]
    results["unweighted_A"] = fit_to_unweighted[0]
    results["unweighted_beta"] = fit_to_unweighted[1]
    results["unweighted_IC"] = fit_to_unweighted[2]
    results["unweighted_chi2"] = fit_to_unweighted[3]
        
    return results
    
#==========================================================================
def process_bootstraps_to_errors(theta_range, boots, return_envelope=True, 
                                    return_boots=True, **kwargs):
    #Take the bootstrap results and return the median and MAD for all the
    #things
    As, betas, offsets, chi2 = boots

    #Now we have the n_fit_boots estimates and their errors
    #Get the median and median absolute deviation (MAD)
    median_A = np.median(As)
    median_A_mad = robust.scale.mad(As, center=median_A)
    median_beta = np.median(betas)
    median_beta_mad = robust.scale.mad(betas, center=median_beta)
    median_offset = np.median(offsets, axis=0)
    median_offset_mad = robust.scale.mad(offsets, center=median_offset, 
                                        axis=0)

    #Compile into easier to return lists
    medians = [median_A, median_beta, median_offset]
    median_mads = [median_A_mad, median_beta_mad, median_offset_mad]

    to_return = {"median_A"       : median_A, 
                "median_beta"     : median_beta,
                "median_IC"       : median_offset,
                "median_A_mad"    : median_A_mad,
                "median_beta_mad" : median_beta_mad,
                "median_IC_mad"   : median_offset_mad, 
                }
                 
    if return_boots:
        to_return["bootstrap_As"] = As
        to_return["bootstrap_betas"] = betas
        to_return["bootstrap_ICs"] = offsets
        to_return["bootstrap_chi2s"] = chi2
    if return_envelope:
        env = envelope(As, betas, offsets, theta_range)
        #theta_grid, median_fit - mads, median_fit, median_fit + mads
        to_return["envelope_theta_grid"] = env[0]
        to_return["envelope_lower"] = env[1]
        to_return["envelope_median"] = env[2]
        to_return["envelope_upper"] = env[3]            
    return to_return        
    
#==========================================================================
def bootstrap_fit_with_offset(fields, n_fit_boots=200, fixed_beta=None, 
                              allowed_failure_rate=0.1, **kwargs):
    #Run the bootstrapping with the IC fit as an offset.  Returns the 
    #bootstraps
    
    thetas, cfs, errs = get_info_from_catalog_set(fields, **kwargs)

    #See how many CFs we have to bootstrap
    n_cfs = cfs.shape[0]

    #How many failures do we allow?
    n_allowed_errors = n_fit_boots * allowed_failure_rate
    n_errors=0

    #Set up arrays to hold the parameters
    As = -np.ones(n_fit_boots)
    chi2 = -np.ones(n_fit_boots)      
    offsets = -np.ones(n_fit_boots)    
    if fixed_beta:
        betas = np.full(n_fit_boots, fixed_beta)
    else:
        betas = -np.ones(n_fit_boots)

    #Do the bootstrap loop
    for i in np.arange(n_fit_boots, dtype=np.int):
        indices = rand.randint(0, n_cfs, n_cfs)

        #Average the selected correlation functions
        avg_cf = np.mean(cfs[indices], axis=0)
        #Get the error on the mean
        error_bars = np.sqrt(np.sum(errs[indices]**2., axis=0))/n_cfs
        #Fit the average 
        try:
            res = minimize_fit_to_cf(thetas, avg_cf, error_bars, 
                                offset=True, fixed_beta=fixed_beta,
                                **kwargs)
        except RuntimeError:
            res = (-1, -1, -1, -1)  
            n_errors += 1
            if n_errors > n_allowed_errors:
                print "Too many errors caught: aborting."
                raise RuntimeError("minimize_bootstrap_cf_fit says: "
                                   "while iterating the fit over the "
                                   "bootstrapped CFs, the fitter failed to"
                                   " find a good fit to too many "
                                   "combinations of CFs.  Examine your CFs"
                                   " to see if they're too ugly.")
            else:
                print ("Minimization failure caught on boot "+str(i)+
                           " of " + str(n_fit_boots))
                           
        #Store what we have
        As[i] = res[0]
        betas[i] = res[1]
        offsets[i] = res[2]
        chi2[i] = res[3]
        
    #Screen down to just the ones that succeeded
    successful_fits = ma.masked_not_equal(As, -1).mask
    to_mask = [As, betas, offsets, chi2]
    for guy in to_mask:
        guy = guy[successful_fits]
            
    return As, betas, offsets, chi2
    
#==========================================================================
def bootstrap_fit_with_adelberger(fields, n_fit_boots=200, fixed_beta=None, 
                                  theta_min=0., theta_max=360., 
                                  allowed_failure_rate=0.1, **kwargs):
    #Run the bootstrapping with the IC from Adelberger.  Returns the 
    #bootstraps    

    #See how many CFs we have to bootstrap
    try:
        n_cfs = len(fields)
    except TypeError:
        fields = [fields]
        n_cfs = len(fields)
    #And make sure we can index it conveniently
    fields= np.array(fields)
    
    #How many failures do we allow?
    n_allowed_errors = n_fit_boots * allowed_failure_rate
    n_errors=0

    #Set up arrays to hold the parameters
    As = -np.ones(n_fit_boots)
    offsets = -np.ones((n_fit_boots, n_cfs))
    chi2 = -np.ones(n_fit_boots)
    if fixed_beta:
        betas = np.full(n_fit_boots, fixed_beta)
    else:
        betas = -np.ones(n_fit_boots)

    for i in np.arange(n_fit_boots, dtype=np.int):
        indices = rand.randint(0, n_cfs, n_cfs)
        temp_fields = fields[indices]
        try:
            res = iterative_fit(temp_fields, fixed_beta=fixed_beta, 
                                    **kwargs)
        except RuntimeError:
            n_errors += 1
            if n_errors > n_allowed_errors:
                print "Too many errors caught: aborting."
                raise RuntimeError("minimize_bootstrap_cf_fit says: "
                                   "while iterating the fit over the "
                                   "bootstrapped CFs, the fitter failed to"
                                   " find a good fit to too many "
                                   "combinations of CFs.  Examine your CFs"
                                   " to see if they're too ugly.")
            res = (-1, -1, -np.ones(n_cfs), -1)
        As[i] = res[0]
        betas[i] = res[1]
        offsets[i,:] = res[2]
        chi2[i] = res[3]
            
    #Screen down to just the ones that succeeded
    successful_fits = ma.masked_not_equal(As, -1).mask
    to_mask = [As, betas, offsets, chi2]
    for guy in to_mask:
        guy = guy[successful_fits]
            
    return As, betas, offsets, chi2


#==========================================================================
def iterative_fit(catalogs, fixed_beta=None, which_cf='bootstrap', 
                  A_change_tol=.01, max_iterations = 50, **kwargs):
    #Take a set of catalogs and iteratively fit the power law and IC

    #Make sure we're using Powell minimization if it's not specified
    if 'method' not in kwargs.keys():
        kwargs['method']="Powell"

    #Make sure that the catalogs are in a list/something with a length
    try:
        n_cats=len(catalogs)
    except TypeError:
        catalogs=[catalogs]
        n_cats=len(catalogs)
    
    #Pull out the CFs and errors
    cf_thetas = np.empty(n_cats, dtype=object)
    cf_theta_bins = np.empty(n_cats, dtype=object)
    original_cf = np.empty(n_cats, dtype=object)
    cf_error = np.empty(n_cats, dtype=object)
    for i in np.arange(n_cats):
        cf_thetas[i], cf_theta_bins[i], original_cf[i], cf_error[i] = extract_cf_from_catalog(catalogs[i], which_cf)
        if not (cf_thetas[i] == cf_thetas[0]).all():
            raise ValueError("iterative_multi_field_fit says: You've given"
                            " me a set of CFs that don't have matching "
                            "theta binning.  You should fix that.")
        
    #Make sure that we leave the CFs in the structures alone- 
    #  copy it to a local one
    cf=deepcopy(original_cf)

    #Now we do the actual fitting- average the CFs together and fit
    done=False
    n_iter = 0
    A=[]
    beta=[]
    ICs=np.zeros(n_cats)
    chi2=[]
    while not done:
        #Average together the CFs and propagate error
        avg_cf=np.zeros(len(cf_thetas[0]))
        avg_cf_err=np.zeros(len(cf_thetas[0]))
        for i in np.arange(n_cats):
            avg_cf += cf[i] + ICs[i]
            avg_cf_err += cf_error[i]**2.
        avg_cf /= n_cats
        avg_cf_err = np.sqrt(avg_cf_err) / n_cats

        #Fit the average and add it to the list
        result = minimize_fit_to_cf(cf_thetas[0], avg_cf, avg_cf_err, 
                                    fixed_beta=fixed_beta, offset=False,
                                    **kwargs)
        #Unpack things
        A_temp, beta_temp, __, chi2_temp = result
        A.append(A_temp)
        beta.append(beta_temp)
        chi2.append(chi2_temp)
            
        #Calculate the ICs
        for i in np.arange(n_cats):
            ICs[i] = integral_constraint(catalogs[i]._thetas_for_rr, 
                                    catalogs[i]._G_p, A_temp, beta_temp)
            
        #If we've done this enough times (must be more than 1), then 
        #declare ourselves done
        if len(A)>1:
            percent_change = (A[-1] - A[-2])/A[-2]
            if abs(percent_change) < A_change_tol:
                done=True
                print "All done!"
            else:
                print "Iteration", n_iter, ", A has changed by ", percent_change
                
        #See if we're stuck
        n_iter += 1
        if n_iter > max_iterations:
            print "Iterated too many times.  Aborting."
            raise RuntimeError("Iterated too many times.  Aborting.")

    #Set the IC to be what we ended up with
    for i, cat in enumerate(catalogs):
        cat._IC[which_cf] = ICs[i]
                
    return A[-1], beta[-1], ICs, chi2[-1]
    
    
#==========================================================================
def minimize_fit_to_cf(thetas, cf, cf_err, fixed_beta=None, offset=True,
                       result_plot=None, distance_metric="chi2", **kwargs):
    #This guy takes the points of a CF and fits a powerlaw.
    #It can either do the Adelberger IC or a power law plus an offset
    #to estimate the IC.
    #The IC has a large variance, so it's better to fit it (according to 
    #Jeff Newman and Bernstein 1994).

    #Make sure we're using Powell minimization if it's not specified
    if 'method' not in kwargs.keys():
        kwargs['method']="Powell"

    #If we aren't given estimates, choose some more reasonable than all 1s
    if 'x0' not in kwargs.keys():
        kwargs['x0'] = [0.001]
        if not fixed_beta:
            kwargs['x0'].append(.8)
        if offset:
            kwargs['x0'].append(0.)
        kwargs['x0'] = np.array(kwargs['x0'])

    #Define the function that we want to fit to
    if offset:
        def powerlaw(theta, A, beta, offset):
            return A*(theta**(-beta)) - offset
    else:   
        def powerlaw(theta, A, beta):
            return A*(theta**(-beta))
            
    #Define the distance metric to minimize
    if distance_metric == "chi2":
        def distance_measure(params):
            if fixed_beta:
                if offset:
                    fit_vals = powerlaw(thetas, params[0], fixed_beta, 
                                    params[1])
                else:
                    fit_vals = powerlaw(thetas, params[0], fixed_beta)
            else:
                fit_vals = powerlaw(thetas, *params)
            distances = (fit_vals - cf)/cf_err
            measure = sum(distances**2)
            return measure          
    else:
        raise ValueError("I don't recognize that distance measure.  "
                         "Options are 'chi2'.")

    #Make a kwarg dictionary that only has kwargs relevant to minimize()
    valid_kws = ['method', 'jac', 'hess', 'hessp', 'bounds', 'constraints',
                 'tol', 'callback', 'options', 'x0', 'args']
    kwargs_to_minimize = {}
    for k in kwargs.keys():
        if k in valid_kws:
            kwargs_to_minimize[k]=kwargs[k]

    #Minimize the distance metric and unpack the results
    res = opt.minimize(distance_measure, **kwargs_to_minimize)

    if not res.success:
        raise RuntimeError("The minimization fitter failed to find an "
                               "acceptable fit")

    if fixed_beta:
        beta = fixed_beta
        offset_ind = 1
        if not offset:
            A=float(res.x)
        else:
            A = res.x[0]
    else:
        A = res.x[0]
        beta = res.x[1]
        offset_ind = 2
    if offset:
        offset = res.x[offset_ind]
    else:
        offset = 0

    distance = res.fun
    
    #If we've been given a result plot name, make a plot with the CF and 
    #fit
    if result_plot:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('w(theta)')
        diag.put_cf_and_fit_on_axis(ax, thetas, cf, cf_err, A, beta, 
                                    IC=offset)
        ax.set_xscale('log')
        ax.set_yscale('log')
        handles, labels=ax.get_legend_handles_labels()
        legnd=ax.legend(handles, labels, loc=0, labelspacing=.15, 
                        fancybox=True, prop={'size':9}, handlelength=2)
        legnd.get_frame().set_alpha(0.)
        plt.savefig(result_plot, bbox_inches='tight')
        plt.close()

    #Return the values that we've gotten and the value of the distance 
    #measure
    return A, beta, offset, distance


        
#==========================================================================
def envelope(As, betas, offsets, theta_range, n_thetas=100):
    #Takes random draws from the distribution of parameters given by
    #bootstrap_params and extracts the ci confidence interval for the
    #power law fit from it

    #How many boots do we have?
    n_fit_boots = len(As)
    
    #Do we have more than 1 fields' worth of ICs?  If we do, we'll return
    #the NON-IC-CORRECTED envelope.
    offsets = np.array(offsets)
    if len(offsets.shape)==2:
        if offsets.shape[1] > 1:
            use_offsets = np.zeros(n_fit_boots)
        else:
            use_offsets = offsets.reshape(n_fit_boots)
    else:
        use_offsets = offsets
        
    #Make the theta grid and calculate all the curves on it
    theta_grid=np.logspace(np.log10(theta_range[0]), np.log10(theta_range[1]), n_thetas)
    all_curves = np.zeros((n_fit_boots, n_thetas))
    for i in np.arange(n_fit_boots):
        all_curves[i] = As[i] * theta_grid**(-betas[i]) - use_offsets[i]

    #Find the ci confidence interval for each theta
    median_fit=np.median(all_curves, axis=0)
    mads = robust.scale.mad(all_curves, center=median_fit, axis=0)
    
    return theta_grid, median_fit - mads, median_fit, median_fit + mads

#==========================================================================
def bootstrap_params(means, covariance, nboots=1000, fixed_beta=None):
    #Use the numpy random multivariate normal distribution to pull
    #bootstrap samples from the fit parameters (A, beta, offset).
    #If the beta was fixed, then it'll fill in the beta column with
    #the fixed beta value.

    #Check some basic things
    mean_shape=np.array(means).shape
    cov_shape=np.array(covariance).shape
    #Do we have 2 or 3 parameters?  If not, we don't know what this came
    #from
    if mean_shape[0] not in [2, 3]:
        raise ValueError("bootstrap_params says: You gave me a number of "
                         "parameters that isn't 2 or 3.  I don't know "
                         "what you're fitting")
    #Do we only have 2 parameters if the beta was fixed?
    if fixed_beta and mean_shape[0]==3:
        raise ValueError("bootstrap_params says: You've given me a mean"
                         "beta in addition to a fixed beta.  I need one "
                         "or the other.")

    #Now that we're ok there, let's go ahead and draw the parameters
    boots = rand.multivariate_normal(means, covariance, nboots)

    #If we have a fixed beta, stick it in as column 1
    if fixed_beta:
        betas=np.full(nboots, fixed_beta)
        temp=np.zeros((nboots, 3))
        temp[:,0]=boots[:,0]
        temp[:,1]=betas
        temp[:,2]=boots[:,1]
        boots = temp

    #Return the values
    return boots
    
#==========================================================================
def moments(x, x_range=None):
    #Calculates all the moments of the data set x within the x_range

    #Make sure the data is a numpy array
    x=np.asarray(x)
    if x_range is None:
        x_range=[x.min(), x.max()]

    #Mask down to the range we care about
    use_mask = ma.masked_inside(x, x_range[0], x_range[1]).mask 
    use_x = x[use_mask]

    #Use the stats shortcut for all the moments I care about
    description=stat.describe(use_x)
    #Do the tests to see if the skewness and kurtosis are normal
    zscore, skew_pval = stat.skewtest(use_x)
    zscore, kurt_pval = stat.kurtosistest(use_x)

    #Put everything together in a dictionary
    moments= {'x_range':             np.array(description.minmax),
              'mean':                description.mean,
              'variance':            description.variance,
              'skewness':            description.skewness,
              'pval_skew_is_normal': skew_pval,
              'kurtosis':            description.kurtosis,
              'pval_kurt_is_normal': kurt_pval
              }

    return moments


#==========================================================================
def poisson_errors(theta_centers, gp_thetas, gp, A, beta, n_data_points):
    #Computes the Poisson error bars a la LS93 
    
    #First compute <d> (Eqn 46 in LS93)
    w_Om = w_Omega(gp_thetas, gp, A, beta)
    d = (1. + A * theta_centers**(-beta)) / (1.+w_Om)

    #Now compute p (Eqn 43 in LS93, using the approximation)
    new_bin_edges = theta_bins_from_centers(theta_centers)
    gp_bins = integrate_to_bins(gp_thetas, gp, new_bin_edges)
    p = 2. / (n_data_points * (n_data_points - 1) * gp_bins)

    #Compute the variance and return sigma
    variance = d**2. * p

    return variance**0.5

#==========================================================================
def poisson_errors_actual_counts(theta_centers, cf, gp_thetas, gp, n_data_points):
    #Computes the Poisson error bars a la LS93

    #Now compute p (Eqn 43 in LS93, using the approximation)
    new_bin_edges = theta_bins_from_centers(theta_centers)
    gp_bins = integrate_to_bins(gp_thetas, gp, new_bin_edges)
    p = 2. / (n_data_points * (n_data_points - 1) * gp_bins)

    #Compute the variance and return sigma
    variance = cf**2. * p

    return variance**0.5

#==========================================================================
def w_Omega(gp_theta_centers, gp, A, beta):
    #Compute the constant w_Omega that appears in Landy and Szalay
    # w_Omega = integral[ G_p(theta) * w(theta) d(theta)]
    #This is just a special case of integrate_gp_times_powerlaw
    # ASSUMES THETA IN DEGREES (and A and beta for theta in degrees)

    return integrate_gp_times_powerlaw(gp_theta_centers, gp, A, beta, 0., 
                                        10.*gp_theta_centers[-1])

#==========================================================================
def integral_constraint(gp_thetas, gp, A, beta):
    #This is the amount that you add to the data w(theta) or subtract from
    #A*theta^(-beta) before you fit.  This is just an alias for w_Omega from LS93

    return w_Omega(gp_thetas, gp, A, beta)

#==========================================================================
def integrate_to_bins(original_theta_centers, fcn, new_bin_edges):
    #Uses the integrate_gp function to integrate a g_p/bin_width to what
    #actually gets used.

    orig_edges = theta_bins_from_centers(original_theta_centers)
    n_new_bins = len(new_bin_edges)-1
    binned = np.zeros(n_new_bins)
    for i in range(n_new_bins):
        binned[i] = integrate_gp(original_theta_centers, fcn,
                                 new_bin_edges[i], new_bin_edges[i+1])

    return binned

#==========================================================================
def integrate_gp(gp_theta_centers, gp, lowlim, highlim):
    #Integrate G_p over a portion of its range by hijacking 
    #integrate_gp_times_powerlaw

    return integrate_gp_times_powerlaw(gp_theta_centers, gp, 1., 0., 
                                        lowlim, highlim)

#==========================================================================
def integrate_gp_times_powerlaw(gp_theta_centers, gp, A, beta, lowlim, highlim):
    #Compute the integral of G_p * A * theta^-beta from lowlim to highlim
    #Again assumes that G_p is defined as (integrated Gp in bin)/bin width
    #Assumes theta in degrees (Note that this also requires that the A and 
    #beta be for theta in degrees)

    #Get basic info
    nbins_gp = len(gp)
    gp_theta_edges = theta_bins_from_centers(gp_theta_centers)

    #If it's completely outside the G_p range, we assume that G_p is 0 and 
    #return 0
    if (lowlim >= gp_theta_edges[-1]) or (highlim <= gp_theta_edges[0]):
        return 0
    
    #Figure out which bins are all the way inside [lowlim, highlim]
    above_lowlim = gp_theta_edges >= lowlim
    min_edge_inside_val = gp_theta_edges[above_lowlim][0]
    min_edge_inside_index = np.arange(nbins_gp+1)[above_lowlim][0]
    
    below_highlim = gp_theta_edges <= highlim
    max_edge_inside_val = gp_theta_edges[below_highlim][-1]
    max_edge_inside_index = np.arange(nbins_gp+1)[below_highlim][-1]

    #Make sure that the limits aren't above or below the limits where G_p 
    #is defined
    if lowlim < gp_theta_edges[0]:
        min_edge_inside_val = gp_theta_edges[0]
        lowlim = gp_theta_edges[0]
        min_edge_inside_index = 0
    if highlim > gp_theta_edges[-1]:
        max_edge_inside_val = gp_theta_edges[-1]
        highlim = gp_theta_edges[-1]
        max_edge_inside_index = nbins_gp
        
    #Initialize the holder for the result
    result=0
    
    #Start by integrating the two partial bins if we didn't happen upon
    #an integration limit that corresponds exactly to an edge
    if min_edge_inside_val != lowlim:
        this_gp_val = gp[min_edge_inside_index-1]
        w_int = integrate_powerlaw(A, beta, lowlim, min_edge_inside_val)
        result += this_gp_val * w_int
        
    if max_edge_inside_val != highlim:
        this_gp_val = gp[max_edge_inside_index]
        w_int = integrate_powerlaw(A, beta, max_edge_inside_val, highlim)
        result += this_gp_val * w_int        

    #Now loop through the bins that are inside
    nbins_inside = max_edge_inside_index - min_edge_inside_index
    for i in np.arange(nbins_inside):
        this_index = min_edge_inside_index + i
        this_gp_val = gp[this_index]
        w_int = integrate_powerlaw(A, beta, gp_theta_edges[this_index], 
                                    gp_theta_edges[this_index+1])
        result += this_gp_val * w_int

    return result
    
#==========================================================================
def integrate_powerlaw(A, beta, lowlim, highlim):
    #Do the integral of a power law A*theta^-beta from lowlim to highlim
    A=np.float(A)
    beta=np.float(beta)
    
    #First define the function to be integrated
    def powerlaw(theta):
        return A * theta ** (-beta)

    #Do the integral
    pl_int, pl_int_err= intg.quad(powerlaw, lowlim, highlim)

    return pl_int

#==========================================================================
def extract_cf_from_catalog(catalog, which_cf):
    #This bit of code is showing up enough that it makes sense to make
    #it a function

    if which_cf not in catalog.cfs.keys():
        raise KeyError("This catalog doesn't have a CF by that name")
    else:
        cf_thetas, cf_theta_bins = catalog.cfs[which_cf].get_thetas(unit='d')
        cf, cf_error = catalog.cfs[which_cf].get_cf() 

    return np.array(cf_thetas), np.array(cf_theta_bins), np.array(cf), np.array(cf_error)

#==========================================================================
def theta_bins_from_centers(centers):
    #Look at the centers of the theta bins to get the bin edges
    #How many bins?
    nbins=len(centers)
    edges=np.zeros(nbins+1)
    #Log or linear?
    if abs((centers[1]-centers[0]) - (centers[2]-centers[1])) < (centers[1]-centers[0])*1.e-6:
        #If the first two bins are the same size, we have linear bins 
        #(Round-off error might be trouble!)
        step=(centers[1]-centers[0])    #Figure out the step size
        edges[0:nbins]=centers-step/2.  #Shift the centers down 
        edges[-1]=edges[-2]+step        #Get the upper edge of the last bin
    else:
        #Otherwise we have log bins
        lcenters=np.log10(centers)          #Shift into log space
        lstep=lcenters[1]-lcenters[0]       #Find the log step
        ledges=np.zeros(nbins+1)            #Figure out the log edges
        ledges[0:nbins]=lcenters-lstep/2.
        ledges[-1]=ledges[-2]+lstep
        edges=10.**ledges                   #Shift back to linear space
    #Store the bin edges
    return edges

#==========================================================================
def get_info_from_catalog_set(fields, which_cf, theta_min=0., 
                              theta_max=360., **kwargs):
    #Arrange the correlation functions, thetas, and errors from N catalogs
    #into a digestible format.
    
    #Make sure we have the CF in all the fields
    for field in fields:
        if which_cf not in field.cfs.keys():
            raise KeyError("The correlation function name you've specified"
                           " doesn't exist in all the fields")
                
    thetas=[]
    cfs=[]
    errs=[]
    for field in fields:
        #Grab the CF information
        temp = extract_cf_from_catalog(field, which_cf)
        cf_thetas, cf_theta_bins, cf, cf_error = temp

        #Mask down to bins that are entirely within the theta range
        #First get the bin edges
        bin_mask = ma.masked_inside(cf_theta_bins, theta_min, theta_max).mask
        cf_theta_bins = cf_theta_bins[bin_mask]

        #Now mask to centers that are within the range of the masked-down
        #theta bin edges
        this_min = cf_theta_bins[0]
        this_max = cf_theta_bins[-1]
        center_mask = ma.masked_inside(cf_thetas, this_min, this_max).mask
        cf_thetas = cf_thetas[center_mask]
        cf = cf[center_mask]
        cf_error = cf_error[center_mask]

        #Add the CFs et al to the list
        thetas.append(cf_thetas)
        cfs.append(cf)
        errs.append(cf_error)

        #Check to make sure we have the same theta binning in all of them
        if not ((cf_thetas-thetas[0])/thetas[0] < 1e-5).all():
            raise ValueError("minimize_bootstrap_catalog_set says: you "
                             "have given me a set of CFs that don't all"
                             " have the same theta binning.  I need the "
                             "theta bins to be the same so I can average "
                             "the CFs together.")

    #Convert to numpy arrays so we can do vector things
    thetas=np.array(thetas[0])
    cfs=np.array(cfs)
    errs=np.array(errs)
    
    return thetas, cfs, errs   
    

    
