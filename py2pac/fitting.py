#This is for the power law fitting the right way
import numpy as np
import numpy.ma as ma
from scipy import optimize as opt
import matplotlib.pyplot as plt
from copy import deepcopy

# import numpy.random as rand
# import cf_useful_things as cu
# import bias_tools as t
# import scipy.integrate as intg
# import scipy.stats as stat
# import statsmodels.robust as robust


#==========================================================================
def bootstrap_fit(cf_objects, IC_method="offset", n_fit_boots=200, 
                  return_envelope=True, return_boots=True, **kwargs):
    """
    
    """
    
    #Check that we have a valid IC type
    IC_method = IC_method.lower()
    if IC_method not in ["offset", "adelberger"]:
        raise ValueError("You must choose either 'offset' or 'adelberger'"
                        " for IC_method.  Capitalization is ignored.")
        
    #See how many CFs we have and make sure we have an array-like cf_objects
    try:
        n_cfs = len(cf_objects)
    except TypeError:
        cf_objects = [cf_objects]
        n_cfs = len(cf_objects)

    #And make sure we can index it conveniently
    cf_objects= np.array(cf_objects)
    thetas, cfs, errs = get_info_from_cf_set(cf_objects, **kwargs)
    
    #First, fit just the main set- the unweighted average
    if IC_method == 'adelberger':
        fit_to_unweighted = iterative_fit(cf_objects, **kwargs)
    else:
        cf = np.mean(cfs, axis=0)
        error_bars = np.sqrt(np.sum(errs**2., axis=0))/n_cfs
        fit_to_unweighted = minimize_fit_to_cf(thetas, cf, 
                                                error_bars, **kwargs)
        
    #Now do the bootstraps
    if IC_method == 'adelberger':
        boots = bootstrap_fit_with_adelberger(cf_objects, n_fit_boots=n_fit_boots,
                                                 **kwargs)
    else:
        boots = bootstrap_fit_with_offset(cf_objects, n_fit_boots=n_fit_boots,
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
def bootstrap_fit_with_offset(cf_objects, n_fit_boots=200, fixed_beta=None, 
                              allowed_failure_rate=0.1, **kwargs):
    #Run the bootstrapping with the IC fit as an offset.  Returns the 
    #bootstraps
    
    thetas, cfs, errs = get_info_from_cf_set(cf_objects, **kwargs)

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
def bootstrap_fit_with_adelberger(cf_objects, n_fit_boots=200, fixed_beta=None, 
                                  theta_min=0., theta_max=360., 
                                  allowed_failure_rate=0.1, **kwargs):
    #Run the bootstrapping with the IC from Adelberger.  Returns the 
    #bootstraps    

    #See how many CFs we have to bootstrap
    try:
        n_cfs = len(cf_objects)
    except TypeError:
        cf_objects = [cf_objects]
        n_cfs = len(cf_objects)
    #And make sure we can index it conveniently
    cf_objects= np.array(cf_objects)
    
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
        temp_cf_objects = cf_objects[indices]
        try:
            res = iterative_fit(temp_cf_objects, fixed_beta=fixed_beta, 
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
def iterative_fit(cf_objects, fixed_beta=None, A_change_tol=.01, 
                  max_iterations = 50, **kwargs):
    #Take a set of cf_objects and iteratively fit the power law and IC
    """
    This is what you use when you want to fit with the Adelberger IC.
    
    Iteratively fit a power law to a set of correlation functions stored in
    a list of cf_objects.  This uses the Adelberger integral constraint and
    requries a Gp to be stored in the CorrelationFunctions.
    
    Parameters
    ----------
    cf_objects : array-like
        1D list or array of CorrelationFunction objects to fit as a set
        
    fixed_beta : float (optional)
        If you wish to fit with a fixed value of the slope, pass it with
        this keyword argument.  If not, the default is None, which fits 
        both A and beta.
    
    A_change_tol : float (optional)
        The fractional change in A at which to declare victory.  If this 
        iteration is i, the fractional change is (A_i - A_(i-1))/A_i.
        The routine stops iterating once that fractional change is less 
        than A_change_tol.  Default it 0.01.
    
    max_iterations : int (optional)
        Maximum number of iterations to try.  If the routine iterates this
        many times and still is above the A_change_tol, it gives up and
        raises an error.  Default is 50.
            
    **kwargs
        Other optional arguments.  The main one is method, which gets 
        passed to scipy.optimize.minimize.  If method isn't specified, the
        method will be set to "Powell."  These get passed to 
        minimize_fit_to_cf and then on to scipy.optimize.minimize.
        
    Returns
    -------
    A : float
        Amplitude fit.
    
    beta : float
        Slope either fit or provided
    
    ICs : 1D numpy array
        Integral constraint correction for each field in the list of 
        cf_objects
        
    distance : float 
        The value of the distance metric (such as chi^2) for the fit
    """

    #Make sure we're using Powell minimization if it's not specified
    if 'method' not in kwargs.keys():
        kwargs['method']="Powell"

    #Make sure that the cf_objects are in a list/something with a length
    try:
        n_cats=len(cf_objects)
    except TypeError:
        cf_objects=[cf_objects]
        n_cats=len(cf_objects)
    
    #Pull out the CFs and errors
    temp = get_info_from_cf_set(cf_objects, **kwargs)
    cf_thetas, original_cf, cf_error = temp
        
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
        avg_cf=np.zeros(len(cf_thetas))
        avg_cf_err=np.zeros(len(cf_thetas))
        for i in np.arange(n_cats):
            avg_cf += cf[i] + ICs[i]
            avg_cf_err += cf_error[i]**2.
        avg_cf /= n_cats
        avg_cf_err = np.sqrt(avg_cf_err) / n_cats

        #Fit the average and add it to the list
        result = minimize_fit_to_cf(cf_thetas, avg_cf, avg_cf_err, 
                                    fixed_beta=fixed_beta, offset=False,
                                    **kwargs)
        #Unpack things
        A_temp, beta_temp, __, chi2_temp = result
        A.append(A_temp)
        beta.append(beta_temp)
        chi2.append(chi2_temp)
            
        #Calculate the ICs
        for i in np.arange(n_cats):
            ICs[i] = cf_objects[i].integral_constraint(A_temp, beta_temp)
            
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
                
    return A[-1], beta[-1], ICs, chi2[-1]
    
#==========================================================================
def minimize_fit_to_cf(thetas, cf, cf_err, fixed_beta=None, offset=True,
                       result_plot=None, distance_metric="chi2", **kwargs):
    """
    This is what you use when you want to fit with the offset IC.
    
    This guy takes the points of a CF and fits a power law. It can either
    fit to just a simple power law (A*theta^(-beta)) or fit the IC as a 
    free-parameter offset (A*theta^(-beta) - IC).
    
    Parameters
    ----------
    thetas : 1D array-like
        The theta values at which to evaluate the power law function.  This
        can generally be assumed to be the linear center of the bin.  To be
        super rigorous, you'd want to weight the separations by the value 
        of the power law over the whole bin and use that weighted average,
        but the center of the bin works just fine.
    
    cf : 1D array-like
        A single set of w(theta) values.  If you're averaging together 
        several fields, do that before this step.
        
    cf_err : 1D array-like
        The error on cf.
        
    fixed_beta : float (optional)
        If you do not wish to have the slope as a free parameter, pass the
        value you want to fixed_beta.  If it is left blank or None is 
        passed, the slope will be a free parameter.  Default it None.
    
    offset : bool (optional)
        Fit with a free-parameter offset to the power law if offset==True.
        Default is True.
    
    result_plot : string (optional)
        File name to which to save a plot of the CF with errors and the
        fit.  Default is None, which won't save a plot.  If you want a plot
        in any directory other than the current directory, the path should
        be specified from /.
    
    distance_metric : "chi2" (optional)
        The distance metric to minimize.  At the moment, the only option is
        "chi2".  Default is therefore "chi2".
        
    **kwargs
        Catch-all for arguments to other functions.  Most importantly for 
        this function, includes keyword arguments to 
        scipy.optimize.minimize.
    
    Returns
    -------
    A : float
        Amplitude for the power law
        
    beta : float
        Slope of the power law.  Returned even if fixed_beta is set, it's
        just the same as the value of fixed_beta.
    
    offset : float
        The best fit value of the offset if offset==True or 0 if 
        offset==False.
        
    distance : float
        Value of the distance metric for the best fit 
    """

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
        ax.errorbar(thetas, cf, yerr=cf_err, fmt='o')
        theta_grid = np.logspace(np.log10(thetas[0]), np.log10(thetas[-1]), 100)
        fit_vals = A * theta_grid**(-beta) - offset
        ax.plot(theta_grid, fit_vals, lw=2, color='r')
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
def get_info_from_cf_set(cf_objects, theta_min=0., theta_max=360.,
                              **kwargs):
    #Arrange the correlation functions, thetas, and errors from N catalogs
    #into a digestible format.
    """
    Pull the correlation function information for *identically named* 
    CorrelationFunctions stored in a list of AngularCatalogs.  Places the
    pulled information into lists to be used by the fitting routines.
    
    Parameters
    ----------
    cf_objects : array-like
        1D list or array of CorrelationFunction instances
    
    theta_min : float (optional)
        If you only want to fit a specific range of angles, this sets the 
        minimum separation in degrees.  Default is 0
    
    theta_max : float (optional)
        If you only want to fit a specific range of angles, this sets the 
        maximum separation in degrees.  Default is 360
        
    **kwargs
        None of these get used in this function, but they're here to let 
        the rest of the code be lazy and just pass around a huge heap of 
        keyword arguments.
        
    Returns
    -------
    thetas : 1D numpy array
        An array containing the centers of the bins in separation
    
    cfs : 2D numpy array
        A numpy array containing a list of the w(theta) values for each 
        field.  Indices go cfs[field_index, theta_bin_index].
    
    errs : 2D numpy array
        A numpy array containing a list of the errors on w(theta) for each 
        field.  Indices go errs[field_index, theta_bin_index].    
    """
    
    thetas=[]
    cfs=[]
    errs=[]
    for cf_obj in cf_objects:
        #Grab the CF information for this guy
        cf_thetas, cf_theta_bins = cf_obj.get_thetas(unit='d')
        cf, cf_error = cf_obj.get_cf() 

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
            raise ValueError("get_info_from_cf_set says: you "
                             "have given me a set of CFs that don't all"
                             " have the same theta binning.  I need the "
                             "theta bins to be the same so I can average "
                             "the CFs together.")

    #Convert to numpy arrays so we can do vector things
    thetas=np.array(thetas[0])
    cfs=np.array(cfs)
    errs=np.array(errs)
    
    return thetas, cfs, errs   
    

