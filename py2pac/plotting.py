#Plotting things

#=====================================================================
#=====================================================================
def put_cf_and_fit_on_axis(ax, thetas, cf, errors, A, beta,
                           sigma_A=None, sigma_beta=None, IC=None,
                           names=None):
    #This is a routine that puts one or a collection of CFs on an
    #axis.  It doesn't manage the configuration of the axes
    
    cf=np.array(cf)
    thetas=np.array(thetas)

    #First, get info about what we're doing.
    if (len(cf.shape) == 2) or (type(cf[0]) == type(np.zeros(1))):
        #We have more than 1 CF.  How many?
        n_cfs = cf.shape[0]
        #Do we have the same thetas for all the guys?
        if (len(thetas.shape) == 2) or (type(thetas[0]) == type(np.zeros(1))):
            #We have more than one theta array
            n_thetas = thetas.shape[0]
            if n_thetas != n_cfs:
                print "put_cf_and_fit_on_axis says: you've given me different numbers of thetas and CFs"
                return
            fit_thetas = np.linspace(min([t.min() for t in thetas]), max([t.max() for t in thetas]), 50)
        else:
            n_thetas = 1
            fit_thetas = np.linspace(thetas.min(), thetas.max(), 50)
        
        #Do we have ICs?
        if IC is not None:
            try:
                n_ICs = len(IC)
            except TypeError:
                n_ICs = 1
        else:
            n_ICs = 0

        #What does the number of ICs imply?  Subtract IC off if applicable
        if (n_ICs == 0):
            #We don't have any ICs
            print "put_cf_and_fit_on_axis says: I have no ICs.  Proceeding without."
            average = False
            IC_label=', no IC'
            plot_cf = np.array(cf)
        elif (n_ICs == 1):
            #We have 1 IC for multiple CFs, which means 1 field
            print "put_cf_and_fit_on_axis says: I have one IC and multiple CFs.  Assuming all have the same IC and are therefore the same field."
            average = True
            IC_label='+IC'
            plot_cf = np.array(cf) + IC
        elif (n_ICs != n_cfs):
            #We have a wrong number of ICs.  Abort IC correction
            print "put_cf_and_fit_on_axis says: WARNING- You've given me a different number of ICs and CFs.  Proceeding without."
            plot_cf = np.array(cf)
            average = False
            IC_label=', no IC'
        else:
            #We have the same number of ICs and CFs- assume different fields
            print "put_cf_and_fit_on_axis says: Adding ICs to the CFs"
            average = True
            IC_label='+IC'
            plot_cf = np.array(cf)
            for i in range(n_cfs):
                plot_cf[i] += IC[i]
            
    else:
        #We only have 1 CF
        n_cfs = 1
        n_thetas = 1
        fit_thetas = np.linspace(thetas.min(), thetas.max(), 50)
        average=False
        if IC is not None:
            plot_cf = np.array(cf) + IC
            IC_label='+IC'
        else:
            IC_label=', no IC'

    #Figure out the range of CF values that we can get at each theta
    options=None

    #First calculate the CF at +/- whatever sigmas we have
    if (sigma_A is not None) and (sigma_beta is not None):
        options=np.zeros((4, 50))
        options[0, :]=(A+sigma_A)*fit_thetas**(-(beta+sigma_beta))
        options[1, :]=(A+sigma_A)*fit_thetas**(-(beta-sigma_beta))
        options[2, :]=(A-sigma_A)*fit_thetas**(-(beta+sigma_beta))
        options[3, :]=(A-sigma_A)*fit_thetas**(-(beta-sigma_beta))
    elif (sigma_A is not None):
        options=np.zeros((2, 50))
        options[0, :]=(A+sigma_A)*fit_thetas**(-beta)
        options[1, :]=(A-sigma_A)*fit_thetas**(-beta)
    elif (sigma_beta is not None):
        options=np.zeros((2, 50))
        options[0, :]=A*fit_thetas**(-(beta+sigma_beta))
        options[1, :]=A*fit_thetas**(-(beta-sigma_beta))

    #Then go through and figure out what the highest and lowest possible
    #CF values are
    if options is not None:
        cf_min=np.amin(options, axis=0)
        cf_max=np.amax(options, axis=0)

    #Now we're actually going to plot.  Start with some silly things
    ax.set_xlabel('theta (degrees)')
    ax.set_ylabel('w(theta)')

    #plot the fit with a shaded region for the possibilities
    ax.plot(fit_thetas, A*fit_thetas**(-beta), label='Fit', color='r', lw=2, zorder=5)
    if options is not None:
        ax.fill_between(fit_thetas, cf_min, cf_max, color='r', alpha=.3)
        
    #Plot the data
    if n_cfs > 1:
        colors=['BlueViolet', 'CornflowerBlue', 'DarkBlue', 'DodgerBlue', 'Indigo', 'LightSkyBlue', 'MediumSlateBlue', 'Orchid', 'RebeccaPurple']
        for nc in range(n_cfs):
            if len(names) == n_cfs:
                lbl=names[nc]
            else:
                lbl='CF'
            if n_thetas > 1:
                ax.errorbar(thetas[nc], plot_cf[nc], yerr=errors[nc],
                            label=lbl+IC_label, color=colors[nc],
                            lw=1.2, capthick=2)
            else:
                ax.errorbar(thetas, plot_cf[nc], yerr=errors[nc],
                                label=lbl+IC_label, color=colors[nc],
                                lw=1.2, capthick=2)
        if average and (n_thetas==1):
            ax.plot(thetas, np.mean(plot_cf, axis=0), color='Orange',
                    lw=2, label='Average')
    else:
        ax.errorbar(thetas, plot_cf, yerr=errors,
                    label='CF'+IC_label, color='k',
                    lw=2, capthick=2)


    #Try to make it log-log
    ax.set_xscale('log')
    try:
        ax.set_yscale('log')
    except ValueError:
        print "put_cf_and_fit_on_axis says: I have negative Y values so I'm not going to log scale the Y axis."

    #Make the legend and save
    handles, labels=ax.get_legend_handles_labels()
    legnd=ax.legend(handles, labels, loc=1, labelspacing=.15, fancybox=True, fontsize=9, handlelength=2.5)
    legnd.get_frame().set_alpha(0.)

    