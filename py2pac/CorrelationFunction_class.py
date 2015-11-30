#External code
import numpy as np
import time

#Py2PAC code
import ThetaBins_class as binclass
import Gp_class as gpclass

class CorrelationFunction(object):
    """
    This class holds all the info about a correlation function.
    It has the CF values, theta binning, etc.  It also has the saving
    and loading of correlation function files.  Also will have
    plotting routines eventually.

    Parameters
    ----------
    name : string (optional)
        String for remembering what you did when you calculated the
        correlation function, e.g 'z1_1.5_mstar8_9'.
         
    cf_type : string (optional)
        String for describing what error calculation type you used
            
    estimator : string (optional)
        The CF estimator used.  Options are 'standard', which is DD/RR
        and 'landy-szalay', which is (DD - 2DR + RR)/RR
        
    ngals : scalar (optional)
        Number of galaxies the data sample
        
    theta_bin_object : ThetaBins instance (optional)
        The instance of ThetaBins used to calculate the CF. This form of
        theta information supercedes all other formats.
        
    theta_bins : array (optional)
        The edges of the theta bins used to calculate the CF.  This will be
        used to calculate the theta bins only if theta_bin_object is not
        given.
    thetas : array (optional)
        The centers of the theta bins that were used to calculate the CF.
        This will only be used if both theta_bin_object and theta_bins are
        not given. The options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
        'degrees'; 'r', 'rad', 'radians'.  Default is 'arcseconds'
        
    theta_unit : string (optional)
        The units that the arrays for theta_bins or thetas are in.  Only
        needed if either of those are to be used in making the ThetaBins
        object
        
    verbose : boolean (optional)
        Sets whether or not the __init__ function talks to you.  Default is
        True.
        
    gp_object : Gp instance (optional)
        The Gp instance associated with the geometry of the mask used to
        calculate the CF.  Supercedes gp_filename
        
    gp_filename : string (optional)
        The file from which to read the Gp information.  Only gets used if
        the gp_object is not given.
    """
    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, name='correlation_function', cf_type='unknown',
                 ngals=None, thetas=None, theta_bins=None, estimator=None,
                 theta_bin_object=None, theta_unit='a', verbose=True,
                 gp_object=None, gp_filename=None):
        """
        Organizes all the info we're given, sets up the ThetaBins option
        if we can.
        """
        
        #The name doesn't really matter- it's just an identifier for you
        self._name = name
        #The type is how the error was calculated
        self._cf_type = cf_type
        #The number of galaxies used to calculate the CF
        self._ngals = ngals
        #The time it was created
        self._time_created = 'Created ' + time.strftime("%c")
        #What estimator was used for the CF?
        self._estimator = estimator

        #Set up the Gp if we can
        if gp_object is not None:
            if verbose:
                   print ("CorrelationFunction.__init__ says: you gave me a"
                          " valid Gp object.  Storing")
            self._Gp = gp_object
        elif gp_filename is not None:
            if verbose:
                   print ("CorrelationFunction.__init__ says: you gave me a"
                          " file name for Gp.  Loading")
            self.load_gp(gp_filename)
        else:
            if verbose:
                   print ("CorrelationFunction.__init__ says: you did not "
                          "provide any info on Gp.  Leaving it for later")
            self._Gp = None
        
        #The thetas
        if theta_bin_object is not None:
            if verbose:
                   print ("CorrelationFunction.__init__ says: you gave me a"
                          " valid ThetaBin object.  Storing")
            self.set_theta_object(theta_bin_object)
        elif theta_bins is not None:
            if verbose:
                   print ("CorrelationFunction.__init__ says: constructing "
                          "the ThetaBin object from the theta_bins")
            self.set_thetas_from_edges(theta_bins, unit=theta_unit)
        elif thetas is not None:
            if verbose:
                   print ("CorrelationFunction.__init__ says: constructing "
                          "the ThetaBin object from the bin centers, thetas")
            self.set_thetas_from_centers(thetas, unit=theta_unit)
        else:
            if verbose:
                   print ("CorrelationFunction.__init__ says: no valid theta "
                          "information given.  Leaving theta bins for later.")
            self._theta_bins = None
            
        #The CF
        self._cf=None
        self._error=None
        self._iterations={} #If we're estimating error, this is a
                            #place to keep the iterations
        #Counts
        self._DD=None
        self._DR=None
        self._RR=None
        
    #----------------------------------------------------------------------
        
    #-------------------------#
    #- Bring in CF from file -#
    #-------------------------#
    @classmethod
    def from_file(cls, filename, name=None):
        """
        Creates a CorrelationFunction object from a saved CF file.

        Parameters
        ----------
        filename : string
            The file that contains the CF.  Should be the a path from /
            
        name : string (optional)
            A new description of the correlation function to replace the
            name that was saved.
        """
        #Make an empty guy
        temp = cls(verbose=False)
        #Load the info from file
        temp.load(filename)
        #If an alternative name is given, set it
        if name is not None:
            temp.set_name(name)
            
        return temp
        
        
    #======================================================================
        
    #----------------------------------------#
    #- Functions to set/retrieve properties -#
    #----------------------------------------#

    def set_thetas(self, min_theta, max_theta, nbins, unit='a',
                   logbins=None):
        """
        Generates a ThetaBins instance from parameters

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
            If logbins == True, the bins are evenly spaced in log
            space. If logbins == False, the bins are evenly spaced
            in linear space.  Default is True.
        """
        #If we have a ThetaBins instance, set new bin parameters
        if self._theta_bins is None:
            #Create a new ThetaBins instance with these parameters
            if logbins is None:
                logbins = True
            self._theta_bins = binclass.ThetaBins(min_theta, max_theta, nbins,
                                                 theta_unit=unit,
                                                 logbins=logbins)
        else:
            #Make sure we have logbins right
            if logbins is not None:
                self._theta_bins.set_logbins(logbins)
            #Adjust the range and nbins
            self._theta_bins.set_new_bins(min_theta, max_theta, nbins, theta_unit=unit)

    #----------------------------------------------------------------------

    def set_thetas_from_edges(self, edges, unit='a'):
        """
        Set the ThetaBins object from the bin edges.  Bins must be even in
        either linear or log space.

        Parameters
        ----------
        edges : array
           The edges of the theta bins desired in the given unit
           (default is arcseconds)
           
        unit : string (optional)
            The unit that min and max theta are in.  The options
            are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
            'r', 'rad', 'radians'.  Default is 'arcseconds'
        """
        self._theta_bins = binclass.ThetaBins.from_edges(edges,
                                                        unit=unit)
        
    #----------------------------------------------------------------------
        
    def set_thetas_from_centers(self, centers, unit='a'):
        """
        Set the ThetaBins object from the bin centers.  The centers must be
        of bins that are equally sized in either linear or log space.

        Parameters
        ----------
        centers : array
            The centers of the theta bins desired in the given unit
            (default is arcseconds)
           
        unit : string (optional)
            The unit that min and max theta are in.  The options
            are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
            'r', 'rad', 'radians'.  Default is 'arcseconds'
        """
        self._theta_bins= binclass.ThetaBins.from_centers(centers,
                                                         unit=unit)

    #----------------------------------------------------------------------

    def set_theta_object(self, bin_object):
        """
        Set the ThetaBins to an existing ThetaBins instance

        Parameters
        ----------
        bin_objects : ThetaBins instance
            An instance of ThetaBins that matches the correlation function
        """
        
        if isinstance(bin_object, binclass.ThetaBins):
            self._theta_bins = bin_object
        else:
            raise TypeError("CorrelationFunction.set_theta_object says: you"
                            " have given me something that isn't a ThetaBins"
                            " instance.")

    #----------------------------------------------------------------------
    def get_points(self, unit='a'):
        """
        Return what you'd need to plot the CF
        
        Parameters
        ----------
        unit : string (optional)
            The unit in which you would like the thetas to be returned.
            The options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
            'degrees'; 'r', 'rad', 'radians'.  Default is 'arcseconds'

        Returns
        -------
        centers : array
            The centers of the theta bins in the unit specified
            
        cf : array
            The correlation function values
            
        error : array
            The error estimation on the correlation function
        """
        centers, edges = self._theta_bins.get_bins(unit=unit)
        return centers, self._cf, self._error
           
    #----------------------------------------------------------------------

    def get_thetas(self, unit='a'):
        """
        Return the theta binning in the unit requested

        Parameters
        ----------
        unit : string (optional)
            The unit in which you would like the thetas to be returned.
            The options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
            'degrees'; 'r', 'rad', 'radians'.  Default is 'arcseconds'
            
        Returns
        -------
        centers : array
            The centers of the theta bins in the unit requested.
            
        edges : array
            The edges of the theta bins in the unit requested.
        """
        return self._theta_bins.get_bins(unit=unit)
        
    #----------------------------------------------------------------------
        
    def set_cf(self, cf, cf_err, iterations={}):
        """
        Store the correlation function information

        Parameters
        ----------
        cf : array
            The correlation function
            
        error : array
            The estimated error on the correlation function
            
        iterations : dictionary of arrays (optional)
            The iterations of the CF calculation done to estimate the
            error.
        """
        self._cf = np.array(cf)
        self._error = np.array(cf_err)
        if iterations != {}:
            self._iterations = iterations

    #----------------------------------------------------------------------

    def get_cf(self, get_iterations=False):
        """
        Give back the CF information.  Usually don't want the iterations
        as well as the average, so only give that if it's requested.

        Parameters
        ----------
        get_iterations: boolean (optional)
            If get_iterations is True, return the dictionary containing
            the iterations along with the CF and error

        Returns
        -------
        cf : array
            The correlation function
            
        error : array
            The error estimate on the correlation function
            
        iterations : dictionary of arrays (optional)
            A dictionary of the iterations used to calculate the CF and
            error.  Only returned if get_iterations is True
        """
        to_return=[self._cf, self._error]
        if get_iterations:
            to_return.append(self._iterations)
            
        return to_return

    #----------------------------------------------------------------------

    def set_name(self, name):
        """
        Set the name of the CF if it needs renaming from the init call

        Parameters
        ----------
        name : string
            Name describing the correlation function
        """
        self._name = name
        
    #----------------------------------------------------------------------
        
    def get_name(self):
        """
        Returns the name describing what the CorrelationFunction is of

        Returns
        -------
        name : string
            Name describing the correlation function
        """
        return self._name

    #----------------------------------------------------------------------

    def set_counts(self, DD=None, DR=None, RR=None):
        """
        Set the DD, DR, and RR of the CF

        Parameters
        ----------
        DD : array (optional)
           Data-data pair counts
           
        DR : array (optional)
           Data-random pair counts
           
        RR : array (optional)
           Random-random pair counts
        """
        
        if DD is not None:
            self._DD = DD
        if DR is not None:
            self._DR = DR
        if RR is not None:
            self._RR = RR
        
    #----------------------------------------------------------------------
        
    def get_counts(self):
        """
        Returns the DD, DR, and RR counts.

        Returns
        -------
        DD : array
           Data-data counts.
           
        DR : array (or None)
           Data-random counts.  None if the estimator is 'standard', array
           if the estimator is 'landy-szalay'
           
        RR : array
           Random-random counts.
        """
        return self._DD, self._DR, self._RR
    
    #----------------------------------------------------------------------

    def set_cf_type(self, type):
        """
        Set the type of the CF if it needs renaming from the init call

        Parameters
        ----------
        cf_type : string
                A string describing the method of generation (e.g. jackknife,
                single galaxy bootstrapping)
        """
        self._cf_type = type
        
    #----------------------------------------------------------------------
        
    def get_cf_type(self):
        """
        Return the type of correlation function the object contains

        Returns
        -------
        cf_type : string
                A string describing the method of generation (e.g. jackknife,
                single galaxy bootstrapping)
        """
        #For symmetry purposes...
        return self._cf_type
    
    #======================================================================
    
    #-------------------#
    #- Save/load files -#
    #-------------------#

    def save(self, filename):
        """
        Use the numpy save routine to save all the information in the
        CorrelationFunction object

        Parameters
        ----------
        filename : string
              The name of the saved Gp object.  This should be the path
              from /
        """
        centers, edges = self._theta_bins.get_bins()
        logbins = self._theta_bins.get_logbins()
        to_save = {'created'     : self._time_created,
                   'name'        : self._name,
                   'type'        : self._cf_type,
                   'estimator'   : self._estimator,
                   'ngals'       : self._ngals,
                   'theta_bins'  : edges,
                   'thetas'      : centers,
                   'log_bins'    : np.array(logbins),
                   'theta_unit'  : 'arcseconds',
                   'cf'          : self._cf,
                   'error'       : self._error,
                   'DD'          : self._DD,
                   'DR'          : self._DR,
                   'RR'          : self._RR
                   }

        #If we have the iterations, save each one separately
        #so they can be parsed easily on re-entry
        if self._iterations != {}:
            for k in self._iterations.keys():
                to_save['iter_'+str(k)]=self._iterations[k]

        #Bring in the Gp dictionary if we have it
        if self._Gp is not None:
            gp_dict = self._Gp.get_dictionary()
            #Store it, tagged as the Gp info
            for k in gp_dict.keys():
                to_save['gp_'+k] = gp_dict[k]

        #Make sure we don't try to save anything that's empty
        for k in to_save.keys():
            if to_save[k] is None:
                del to_save[k]
            #also make sure even the one-element things are arrays
            elif type(to_save[k]) != type(np.zeros(1)):
                to_save[k] = np.array([to_save[k]])

        np.savez(filename, **to_save)
        return

    #----------------------------------------------------------------------

    def load(self, filename):
        """
        Use the numpy load routine to pull in a saved file

        Parameters
        ----------
        filename : string
              The name of the saved CorrelationFunction object.
              This should be the path from /
        """

        #Pull in the file and get the keys it has
        saved = np.load(filename)
        keys = saved.files

        #Put things in their correct locations
        #There are some things we shouldn't be able to save without
        self._time_created = saved['created'][0]
        self._name = saved['name'][0]
        self._cf_type = saved['type'][0]
        self._cf = saved['cf']
        #Pull the theta info and set up the ThetaBins object
        edges = saved['theta_bins']
        unit = saved['theta_unit'][0]
        self.set_thetas_from_edges(edges, unit=unit)

        #There are some that don't have to be included
        if 'ngals' in keys:
            self._ngals = saved['ngals'][0]
        if 'error' in keys:
            self._error = saved['error']
        if 'DD' in keys:
            self._DD = saved['DD']
        if 'DR' in keys:
            self._DR = saved['DR']
        if 'RR' in keys:
            self._RR = saved['RR']
        if 'estimator' in keys:
            self._estimator = saved['estimator'][0]

        #If we saved the individual iterations, pull them back in
        #First pull out any keys that correspond to the iterations
        iter_keys = [k for k in keys if k[0:5]=='iter_']
        #Now put them away if they exist
        self._iterations={}
        if iter_keys != []:
            for k in iter_keys:
                self._iterations[k.lstrip('iter_')] = saved[k]

        #If we saved the Gp with the CF, pull those keys and use to
        #reconstruct the Gp
        gp_keys = [k for k in keys if k[0:3]=='gp_']
        if gp_keys != []:
            time_created = saved['gp_created'][0]
            Gp = saved['gp_Gp']
            n_randoms = saved['gp_n_randoms'][0]
            n_chunks = saved['gp_n_chunks'][0]
            edges = saved['gp_theta_bins']
            unit = saved['gp_theta_unit'][0]
            nbins = len(edges) - 1
            logbins = saved['gp_log_bins'][0]
            if 'gp_RR' in gp_keys:
                RR = saved['gp_RR']
            else:
                RR = None
            #Create the Gp object
            self._Gp = gpclass.Gp(edges[0], edges[-1], nbins, Gp,
                                  n_randoms, n_chunks, logbins = logbins,
                                  unit = unit, RR = RR,
                                  creation_string = time_created)
        else:
            self._Gp = None

        #All done!
        return
            
    #----------------------------------------------------------------------
    def load_gp(self, filename):
        """
        Loads in a Gp object from a file.

        Parameters
        ----------
        filename : string
                 The file, with the path from /, where the Gp save file is
                 located.
        """

        self._Gp = gpclass.Gp.from_file(filename)
