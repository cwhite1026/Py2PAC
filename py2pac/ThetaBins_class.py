#External code
import numpy as np

#Py2PAC code
import miscellaneous as misc

class ThetaBins(object):
    """
    This is just a little guy to keep the info about theta
    binning together and guarantee consistency.

    Parameters
    ----------
    min_theta : float
        The minimum of the theta bin edges
              
    max_theta : float
        The maximum of the theta bin edges
              
    nbins : float
        The number of theta bins
          
    unit : string (optional)
        The unit that min and max theta are in.  The options
        are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
        'r', 'rad', 'radians'.  Default is 'arcseconds'
         
    logbins : boolean (optional)
        If logbins == True, the bins are evenly spaced in log space.
        If logbins == False, the bins are evenly spaced in linear
        space.  Default is True.
    """

    #======================================================================

    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, min_theta, max_theta, nbins,
                 unit='arcseconds', logbins=True):
        """
        Take the basic parameters and use them to construct bins.
        """
        #Record whether or not we're log binning
        self._logbins=logbins
        
        #Call the generation routine with the info we have
        self._make_theta_bins(min_theta, max_theta, nbins,
                              unit=unit)

    #======================================================================
        
    #----------------------------------#
    #- Reverse-engineer existing bins -#
    #----------------------------------#
    #Class method as an alternate way to initialize- give it bin centers
    # and have it figure out the details.
    @classmethod
    def from_centers(cls, centers, unit='d'):
        """
        Class method that constructs a ThetaBins object with all the
        relevant parameters from the centers of bins.

        **Syntax**

        new_bins = ThetaBins_class.ThetaBins.from_centers(centers, unit=<unit>)

        Parameters
        ----------
        centers : array-like
            The centers of the bins that you want to reconstruct in
            the given unit
                
        unit : string (optional)
            A string saying what units the centers are in.  The options
            are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
            'r', 'rad', 'radians'.  Default is 'degrees'

        Returns
        -------
        new_bins : ThetaBins instance
            A ThetaBins instance that matches the given centers
            and contains all the auxiliary information as well.
        """
        
        #Look at the centers of the theta bins to get the bin edges

        #How many bins?
        centers=np.array(centers)
        nbins=len(centers)

        #Log or linear?
        first_bin_width = (centers[1]-centers[0])
        second_bin_width = (centers[2]-centers[1])
        difference_between_widths = abs(first_bin_width - second_bin_width)
        #If the first two bins are different sizes, we have log bins
        logbins = difference_between_widths > (first_bin_width)*1.e-6

        #Figure out where the min and max edges will be
        if logbins:
            centers = np.log10(centers)
            first_bin_width = centers[1] - centers[0]
        min_theta = centers[0] - first_bin_width/2.  
        max_theta = centers[-1] + first_bin_width/2.  

        #If we were in log space, come back
        if logbins:
            min_theta = 10.** min_theta
            max_theta = 10.** max_theta

        #Now that we have the info, make the actual object
        return cls(min_theta, max_theta, nbins,
                   unit=unit, logbins=logbins)

    #----------------------------------------------------------------------

    @classmethod
    def from_edges(cls, edges, unit='d'):
        """
        Class method that constructs a ThetaBins object with all the
        relevant parameters from the edges of bins.

        **Syntax**

        new_bins = ThetaBins_class.ThetaBins.from_edges(edges, unit=<unit>)

        Parameters
        ----------
        edges : array-like
            The edges of the bins that you want to reconstruct in
            the given unit
        unit : string (optional)
            A string saying what units the centers are in.  The options
            are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
            'r', 'rad', 'radians'.  Default is 'degrees'

        Returns
        -------
        new_bins : ThetaBins instance
            A ThetaBins instance that matches the given edges
            and contains all the auxiliary information as well.
        """
        
        #Look at the centers of the theta bins to get the bin edges
        
        #How many bins?
        nbins=len(edges) - 1

        #Log or linear?
        first_bin_width = (edges[1]-edges[0])
        second_bin_width = (edges[2]-edges[1])
        difference_between_widths = abs(first_bin_width - second_bin_width)
        #If the first two bins are different sizes, we have log bins
        logbins = difference_between_widths > (first_bin_width)*1.e-6

        #Pull out the 
        min_theta = edges[0]
        max_theta = edges[-1]

        #Now that we have the info, make the actual object
        return cls(min_theta, max_theta, nbins,
                   unit=unit, logbins=logbins)

    #======================================================================        
        
    #-----------------------#
    #- Generate theta bins -#
    #-----------------------#
    def _make_theta_bins(self, min_theta, max_theta, nbins,
                         unit='arcseconds'):
        """
        Internally sets the edges and centers of the angular bins

        Parameters
        ----------
        min_theta : float
                 The minimum of the theta bin edges
                 
        max_theta : float
                 The maximum of the theta bin edges
                 
        nbins : float
              The number of theta bins
              
        unit : string (optional)
             The unit that min and max theta are in.  Default is 'arcseconds'
        """
        
        #Make sure the min and max are in degrees
        min_theta, max_theta = misc.put_thetas_in_degrees([min_theta, max_theta],
                                                           unit)

        #Record things in degrees
        self._min_theta = min_theta
        self._max_theta = max_theta
        self._nbins = nbins

        #Make the bins
        if self._logbins:
            if min_theta<=0:
                print ("make_theta_bins says: you've asked for log theta bins "
                       "and a min of theta<=0 which makes logs unhappy.  "
                       "Changed to 1e-4")
                min_theta=0.0001
            edges = np.logspace(np.log10(min_theta), np.log10(max_theta), nbins+1)
            lcenters=misc.centers(np.log10(edges))
            centers=10.**lcenters
        else:
            edges=np.linspace(min_theta, max_theta, nbins+1)
            centers=misc.centers(edges)

        #Record the bins and return
        self._theta_bins=edges
        self._thetas=centers
        return

    #----------------------------------------------------------------------

    #--------------#
    #- I/O things -#
    #--------------#

    def set_logbins(self, logbins):
        """
        Used to change whether the bins are even in log space or linear
        space.  Recalculates the bins when logbins is changed

        Parameters
        ----------
        logbins : boolean
            If logbins == True, the bins are evenly spaced in log space.
            If logbins == False, the bins are evenly spaced in linear
        """
        #If it's actually changing, record and rerun the bins
        if self._logbins != logbins:
            self._logbins=logbins
            self._make_theta_bins(self._min_theta, self._max_theta,
                                 self._nbins, unit='d')

    #----------------------------------------------------------------------

    def get_logbins(self):
        """
        Returns whether or not the bins are logarithmic or linear

        Returns
        -------
        logbins : boolean
            If logbins == True, the bins are evenly spaced in log space.
            If logbins == False, the bins are evenly spaced in linear
        """
        #Return whether or not we have the log bins
        return self._logbins

    #----------------------------------------------------------------------

    def set_new_bins(self, min_theta, max_theta, nbins, unit='a'):
        """
        Redo the range and number of bins.

        Parameters
        ----------
        min_theta : float
                 The minimum of the theta bin edges
                 
        max_theta : float
                 The maximum of the theta bin edges
                 
        nbins : float
              The number of theta bins
              
        unit : string (optional)
             The unit that min and max theta are in.  Default is 'arcseconds'
        """
        #Set new params and rerun if different

        #First, convert to degrees if we need to
        min_theta, max_theta = misc.put_thetas_in_degrees([min_theta, max_theta],
                                                           unit)

        #Are they different?
        min_different = self._min_theta != min_theta
        max_different = self._max_theta != max_theta
        n_different = self._nbins != nbins

        #If so, record and rerun bins
        if min_different or max_different or n_different:
            self._min_theta = min_theta
            self._max_theta = max_theta
            self._nbins = nbins
            self._make_theta_bins(min_theta, max_theta, nbins, unit='d')
            
        return
        
    #----------------------------------------------------------------------

    def get_bins(self, unit='a'):
        """
        Return the centers and edges of the bins in the requested unit

        Parameters
        ----------
        unit : string (optional)
            The unit that the thetas will be returned in.
            Default is 'arcseconds'

        Returns
        -------
        centers : numpy array
            The locations of the centers of the bins in the requested unit
            
        edges : numpy array
            The locations of the edges of the bins in the requested unit
        """
        
        #Convert to whatever unit requested and return
        if unit in misc.arcsec_opts:
            thetas = self._thetas * 3600.
            bins = self._theta_bins * 3600.
        elif unit in misc.radian_opts:
            thetas = np.radians(self._thetas)
            bins = np.radians(self._theta_bins)
        elif unit in misc.degree_opts:
            thetas = self._thetas
            bins = self._theta_bins
        else:
            print "ThetaBins.get_bins says: you have chosen unit=", unit
            print "This is not an option."
            print "For arcseconds, use 'arcseconds', 'arcsecond', 'arcsec', or 'a'."
            print "For radians, use 'radians', 'radian', 'rad', or 'r'."
            print "For degrees, use 'degrees', 'degree', 'deg', or 'd'."
            raise ValueError("You chose an invalid value of unit.")
            
        return thetas, bins
        
