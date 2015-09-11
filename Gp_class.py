#External code
import numpy as np
import time
import scipy.integrate as intg

#Py2PAC code
import miscellaneous as misc
import ThetaBins_class as binclass

class Gp(object):
    """
    This object keeps all the info about the G_p function together.
    Also provides functions to calculate the integral constraint and
    rebinning G_p.

    Parameters
    ----------
    min_theta : float
              The minimum of the theta bin edges for constructing the
              theta bins.
    max_theta : float
              The maximum of the theta bin edges for constructing the
              theta bins
    nbins : float
          The number of theta bins
    Gp : array of length nbins
       The Gp function from Landy and Szalay 1993, with the modification
       that the bin width is divided out of how they tell you to calculate
       it from randoms.  The given formula has the property that
       Sum[G_p,i]=1.  This is not equivalent to Integral[G_p d(theta)]=1,
       which is what they assume everywhere else.  Dividing out the bin
       width gives you that and lets you pretend G_p is a continuous but
       chunky-looking function.
    n_randoms : scalar
              The total number of randoms used to calculate G_p
    n_chunks : scalar
             The number of pieces into which n_randoms was broken into in
             order to keep the integration time down.  If n_chunks > 1,
             G_p was calculated n_chunks time for n_randoms/n_chunks randoms
             in each and then averaged.
    logbins : True | False (optional)
            Sets theta bins to be even in log space if True and and linear
            space if False.  Default is True.
    unit : string (optional)
         Tells the theta bin object what unit min_theta and max_theta are in.
         The options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
         'r', 'rad', 'radians'.  Default is 'arcseconds'
    RR : array of length nbins (optional)
       The non-normalized RR counts that the G_p came from.  This does not
       have the bin width divided out because it doesn't need to be
       integrated anywhere.  Default is None.
    creation_string : string (optional)
                    A time stamp of when the G_p function was actually
                    calculated.  Default is None, at which point it's
                    generated to be the time and date when the object is
                    created.    
    """

    #======================================================================
    
    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, min_theta, max_theta, nbins, Gp, n_randoms,
                 n_chunks, logbins=True, unit='arcsec', RR=None,
                 creation_string=None):
        """
        Creates the ThetaBins object, stores the values given, and
        creates the timestamp if creation_string isn't defined.
        """
        #Make the ThetaBins object and store things
        self._theta_bins = binclass.ThetaBins(min_theta, max_theta, nbins,
                                              unit=unit, logbins=logbins)
        self._Gp = np.array(Gp)
        self._n_randoms = n_randoms
        self._RR = RR
        self._n_chunks = n_chunks
        if creation_string is None:
            self._time_created = 'Created ' + time.strftime("%c")
        else:
            self._time_created = creation_string

    #======================================================================

    #--------------------#
    #- Make from a file -#
    #--------------------#
    @classmethod
    def from_file(cls, filename):
        """
        An alternative way to create a Gp object.  You give it a file name
        and it reads the file and then creates the Gp object

        Syntax
        ------
        new_gp = Gp_class.Gp.from_file('/file/name/from/root.npz')

        Parameters
        ----------
        filename : string
              The name of the saved Gp object.  This should be the path
              from /

        Returns
        -------
        new_gp : Gp class instance
               A Gp instance that matches the saved Gp
        """

        #Pull in the file and get the keys it has
        saved = np.load(filename)
        keys = saved.files

        #Put things in their correct locations
        #There are some things we shouldn't be able to save without
        time_created = saved['created'][0]
        Gp = saved['Gp']
        n_randoms = saved['n_randoms'][0]
        n_chunks = saved['n_chunks'][0]
        #Pull the theta info and set up the ThetaBins object
        edges = saved['theta_bins']
        unit = saved['theta_unit'][0]
        nbins = len(edges) - 1
        logbins = saved['log_bins'][0]

        #There are some that don't have to be included
        if 'RR' in keys:
            RR = saved['RR']
        else:
            RR = None

        #Create the Gp object
        temp = cls(edges[0], edges[-1], nbins, Gp, n_randoms, n_chunks,
                   logbins = logbins, unit = unit, RR = RR,
                   creation_string = time_created)

        return temp
    
    #======================================================================

    #----------------#
    #- I/O routines -#
    #----------------#

    def get_thetas(self, unit='a'):
        """
        Pulls the theta centers and edges from the ThetaBins object

        Parameters
        ----------
        unit : string (optional)
             The unit that you want the thetas to be in when they're
             returned.  The options are 'a', 'arcsec', 'arcseconds';
             'd', 'deg', 'degrees'; 'r', 'rad', 'radians'.
             Default is 'arcseconds'
             
        Returns
        -------
        centers : array
                A list of the bin centers in the unit requested
        edges : array
              A list of the bin edges in the unit requested
        """
        
        #Return the theta binning in the unit requested
        return self._theta_bins.get_bins(unit=unit)

    #----------------------------------------------------------------------

    def get_points(self, unit='a'):
        """
        Returns the bin centers and the values of G_p in each bin.

        Parameters
        ----------
        unit : string (optional)
             The unit that you want the thetas to be in when they're
             returned.  The options are 'a', 'arcsec', 'arcseconds';
             'd', 'deg', 'degrees'; 'r', 'rad', 'radians'.
             Default is 'arcseconds'

        Returns
        -------
        centers : array
                A list of the bin centers in the unit requested
        Gp : array
           The values of G_p in each bin
        """
        
        #Return the centers of the bins and the Gp
        centers, edges = self._theta_bins.get_bins(unit=unit)
        return centers, self._Gp
    
    #----------------------------------------------------------------------

    def get_gp(self):
        """
        Return the Gps

        Parameters
        ----------

        Returns
        -------
        Gp : array
           The values of G_p in each bin
        """
        
        #Return the Gp
        return self._Gp

    #----------------------------------------------------------------------

    def stats(self, print_only=True):
        """
        Shows and/or returns some information about when and how the G_p
        was calculated

        Parameters
        ----------
        print_only : True | False (optional)
                  If True, no information is returned.  If False, returns
                  N_randoms, N_chunks, and time_created.

        Returns
        -------
        N_randoms : integer
                  The total number of randoms used to calculate G_p.  Only
                  returned if print_only == False
        N_chunks : integer
                 The number of chunks into which N_randoms was broken.  G_p
                 was calulated in N_chunks instances of N_randoms/N_chunks
                 random points and then averaged.  Only returned if
                 print_only == False
        time_created : string
                     The date and time that the G_p was initially calculated.
                     Only returned if print_only == False
        """
        
        #Return info about how the Gp was calculated
        print "N_randoms =", self._n_randoms
        print "N_chunks =", self._n_chunks
        print "Created", self._time_created

        if not print_only:
            return self._n_randoms, self._n_chunks, self._time_created
    
    #======================================================================
    
    #----------------#
    #- Save to file -#
    #----------------#

    def save(self, filename):
        """
        Saves the information in the object to a .npz file

        Parameters
        ----------
        filename : string
                 Where to save the file.  This should be a path from /
                 and can either include .npz or not
        """
        #Grab the dictionary form
        to_save = self.get_dictionary()
        
        #Make sure even the one-element things are arrays
        for k in to_save.keys():
            if type(to_save[k]) != type(np.zeros(1)):
                to_save[k] = np.array([to_save[k]])
                
        np.savez(filename, **to_save)
        return

    #----------------------------------------------------------------------

    def get_dictionary(self):
        """
        Returns all the information that the object contains as a
        dictionary

        Returns
        -------
        Gp_dict : python dictionary
                A dictionary that contains all the information that the
                Gp object contains.
        """

        #Put everything into dictionary form
        centers, edges = self._theta_bins.get_bins()
        logbins = self._theta_bins.get_logbins()
        gp_dict = {'created'     : self._time_created,
                   'n_randoms'   : self._n_randoms,
                   'n_chunks'    : self._n_chunks,
                   'theta_bins'  : edges,
                   'thetas'      : centers,
                   'log_bins'    : logbins,
                   'theta_unit'  : 'arcseconds',
                   'Gp'          : self._Gp,
                   'RR'          : self._RR
                   }

        #Inspect all the things that are in the dictionary
        for k in gp_dict.keys():
            #Delete any that are None
            if gp_dict[k] is None:
                del gp_dict[k]

        return gp_dict

    #======================================================================
        
    #--------------------#
    #- Gp manipulations -#
    #--------------------#
        
    def convert_A_to_degrees(self, A, beta, unit):
        """
        Takes A for A*theta^-beta for theta in one unit and converts
        it to what it should be for theta in degrees

        Parameters
        ----------
        A : float
          The amplitude of the power law (A*theta ^-beta)
        beta : float
             The power in the power law (A*theta ^-beta)

        Returns
        -------
        deg_A : float
              The amplitude for theta in degrees
        """

        #figure out which unit we're in and set the conversion factor
        if unit in misc.arcsec_opts:
            factor = 1/3600. #(3600 arcseconds per degree)
        elif unit in misc.degree_opts:
            factor = 1 #We're already there
        elif unit in misc.radian_opts:
            factor = 180. / np.pi  #pi radians per 180 degrees
        else:
            print "Gp.convert_A_to_degrees says: you have chosen unit=", unit
            print "This is not an option."
            print "For arcseconds, use 'arcseconds', 'arcsecond', 'arcsec', or 'a'."
            print "For radians, use 'radians', 'radian', 'rad', or 'r'."
            print "For degrees, use 'degrees', 'degree', 'deg', or 'd'."
            raise ValueError("You chose an invalid value of unit.")

        return A * factor**beta

    #----------------------------------------------------------------------        
        
    def integrate_powerlaw(self, A, beta, lowlim, highlim, param_unit='d',
                           theta_unit='d'):
        """
        A funtion that integrates the power law A * theta^-beta over a
        range of thetas

        Parameters
        ----------
        A : float
          The amplitude of the power law for theta in units param_unit
        beta : float
             The power in the power law
        lowlim : float
               The lower limit of the theta range to integrate over
        highlim : float
                The upper limit of the theta range to integrate over
        param_unit : string (optional)
                   The units of theta for which A was calculated.  The
                   options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
                   'degrees'; 'r', 'rad', 'radians'.  Default is 'd'
        theta_unit : string (optional)
                   The units of lowlim and highlim.  The
                   options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
                   'degrees'; 'r', 'rad', 'radians'.  Default is 'd'

        Returns
        -------
        integral : float
                 The integral of the power law over the specified range
        """
        
        #Do the integral of a power law A*theta^-beta from lowlim to highlim
        A = np.float(A)
        beta=np.float(beta)
        
        #Make sure the A is for theta in degrees
        A = self.convert_A_to_degrees(A, beta, param_unit)

        #Convert the limits to degrees
        lowlim, highlim = misc.put_thetas_in_degrees([lowlim, highlim],
                                                     unit=theta_unit)

        #First define the function to be integrated
        def powerlaw(theta):
            return A * theta ** (-beta)

        #Do the integral
        pl_int, pl_int_err= intg.quad(powerlaw, lowlim, highlim)

        return pl_int
    
    #----------------------------------------------------------------------
    
    def integrate_gp_times_powerlaw(self, A, beta, lowlim, highlim,
                                    theta_unit='d', param_unit='d'):
        """
        Integrates G_p * A * theta^-beta over a range in thetas.  Useful
        for calculating the integral constraint.

        Parameters
        ----------
        A : float
          The amplitude of the power law for theta in units param_unit
        beta : float
             The power in the power law
        lowlim : float
               The lower limit of the theta range to integrate over
        highlim : float
                The upper limit of the theta range to integrate over
        param_unit : string (optional)
                   The units of theta for which A was calculated.  The
                   options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
                   'degrees'; 'r', 'rad', 'radians'.  Default is 'd'
        theta_unit : string (optional)
                   The units of lowlim and highlim.  The
                   options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
                   'degrees'; 'r', 'rad', 'radians'.  Default is 'd'

        Returns
        -------
        integral : float
                 The integral of G_p * A * theta^-beta
        """
        #Compute the integral of G_p * A * theta^-beta from lowlim to
        #highlim.  A and beta are for the unit specified 

        #Pull the theta properties
        centers, edges = self._theta_bins.get_bins(unit='d')
        
        #Get basic info and convert everything for theta in degrees
        nbins_gp = len(self._Gp)
        lowlim, highlim = misc.put_thetas_in_degrees([lowlim, highlim],
                                                     unit=theta_unit)
        A = self.convert_A_to_degrees(A, beta, param_unit)

        #If it's completely outside the G_p range, we assume that G_p is 0
        #and return 0
        if (lowlim >= edges[-1]) or (highlim <= edges[0]):
            return 0
    
        #Figure out which bins are all the way inside [lowlim, highlim]
        above_lowlim = edges >= lowlim
        min_edge_inside_val = edges[above_lowlim][0]
        min_edge_inside_index = np.arange(nbins_gp+1)[above_lowlim][0]
    
        below_highlim = edges <= highlim
        max_edge_inside_val = edges[below_highlim][-1]
        max_edge_inside_index = np.arange(nbins_gp+1)[below_highlim][-1]

        #Make sure that the limits aren't above or below the limits where G_p is defined
        if lowlim < edges[0]:
            min_edge_inside_val = edges[0]
            lowlim = edges[0]
            min_edge_inside_index = 0
        if highlim > edges[-1]:
            max_edge_inside_val = edges[-1]
            highlim = edges[-1]
            max_edge_inside_index = nbins_gp
        
        #Initialize the holder for the result
        result=0
    
        #Start by integrating the two partial bins if we didn't happen upon
        #an integration limit that corresponds exactly to an edge
        if min_edge_inside_val != lowlim:
            this_gp_val = self._Gp[min_edge_inside_index-1]
            w_int = self.integrate_powerlaw(A, beta, lowlim, min_edge_inside_val)
            result += this_gp_val * w_int
        
        if max_edge_inside_val != highlim:
            this_gp_val = self._Gp[max_edge_inside_index]
            w_int = self.integrate_powerlaw(A, beta, max_edge_inside_val, highlim)
            result += this_gp_val * w_int        

        #Now loop through the bins that are inside
        nbins_inside = max_edge_inside_index - min_edge_inside_index
        for i in np.arange(nbins_inside):
            this_index = min_edge_inside_index + i
            this_gp_val = self._Gp[this_index]
            w_int = self.integrate_powerlaw(A, beta, edges[this_index], edges[this_index+1])
            result += this_gp_val * w_int

        return result
    
    #----------------------------------------------------------------------

    def integrate_gp(self, lowlim, highlim, theta_unit='d'):
        """
        Integrates G_p over a range in thetas

        Parameters
        ----------
        lowlim : float
               The lower limit of the theta range to integrate over
        highlim : float
                The upper limit of the theta range to integrate over
        theta_unit : string (optional)
                   The units of lowlim and highlim.  The
                   options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
                   'degrees'; 'r', 'rad', 'radians'.  Default is 'd'

        Returns
        -------
        integral : float
                 The integral of G_p from lowlim to highlim
        """
        #Integrate G_p over a portion of its range by hijacking
        #integrate_gp_times_powerlaw
        return self.integrate_gp_times_powerlaw(1., 0., lowlim, highlim,
                                                theta_unit=theta_unit,
                                                param_unit='d')

    #----------------------------------------------------------------------
    
    def integrate_to_bins(self, new_bin_edges, theta_unit='a'):
        """
        Uses integrate_gp to rebin the existing G_p.  Note that the new G_p
        does not have the new bin widths divided out so it follows
        Sum{G_p,i} = 1, not integral(G_p(theta) d theta) = 1

        Parameters
        ----------
        new_bin_edges: array
                     The values of theta defining the bin edges that you
                     want to integrate to.  The theta values are in
                     units theta_unit
        theta_unit : string (optional)
                   The unit in which new_bin_edges are given.  The options
                   are 'a', 'arcsec', 'arcseconds'; 'd', 'deg', 'degrees';
                   'r', 'rad', 'radians'.  Default is 'arcseconds'

        Returns
        -------
        new_gp : array
               Gp integrated over the new bins
        """

        #Convert the thetas to degrees
        new_bin_edges = misc.put_thetas_in_degrees(new_bin_edges, theta_unit)

        #Integrate Gp over each bin
        n_new_bins = len(new_bin_edges)-1
        binned = np.zeros(n_new_bins)
        for i in range(n_new_bins):
            binned[i] = self.integrate_gp(new_bin_edges[i], new_bin_edges[i+1])

        return binned
        
    #----------------------------------------------------------------------
        
    def w_Omega(self, A, beta, param_unit='d'):
        """
        Compute the constant w_Omega that appears in Landy and Szalay
             w_Omega = integral[ G_p(theta) * w(theta) d(theta)]
        This is just a special case of integrate_gp_times_powerlaw
        over the entire range of thetas.

        Parameters
        ----------
        A : float
          The amplitude of the power law for theta in units param_unit
        beta : float
             The power in the power law
        param_unit : string (optional)
                   The units of theta for which A was calculated.  The
                   options are 'a', 'arcsec', 'arcseconds'; 'd', 'deg',
                   'degrees'; 'r', 'rad', 'radians'.  Default is 'd'

        Returns
        -------
        w_Omega : float
                The constant w_Omega from Landy and Szalay.
                w_Omega = integral[ G_p(theta) * w(theta) d(theta)]
        """

        #Get thetas to figure out what range to integrate over
        centers, edges = self.get_thetas(unit='d')
        
        res = self.integrate_gp_times_powerlaw(A, beta, 0., edges[-1] + 1,
                                          theta_unit='d',
                                          param_unit=param_unit)

        return res





