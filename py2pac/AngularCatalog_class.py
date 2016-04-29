
#---------------------------------------------------------------------#
#- This is the file that contains the main class that Py2PAC is      -#
#- built around, the AngularCatalog, which holds RAs and Decs and    -#
#- does the actual calculations of correlation functions.            -#
#---------------------------------------------------------------------#

# External code
import copy
import warnings
import numpy as np
import numpy.ma as ma
import numpy.random as rand
from scipy import optimize as opt
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt

#Py2PAC code
import correlations as corr
import ImageMask_class as imclass
import miscellaneous as misc
import ThetaBins_class as binclass
import CorrelationFunction_class as cfclass
import Gp_class as gpclass

#==========================================================================
#==========================================================================
#==========================================================================
class AngularCatalog(object):
    """
    This class is the workhorse of Py2PAC.  It manages the actual catalogs
    of objects, it creates various objects to hold information, and it
    performs the correlation function calculations on the catalogs.
    AngularCatalogs are single-bin objects, so if you want to sub-divide
    your data set, do so before you pull it into AngularCatalogs.  Future
    releases of Py2PAC will include a MultiCatalog that manages slicing a
    catalog into bins.

    Parameters
    ----------
    ra : array-like
        A list of RAs for your objects in degrees
        
    dec : array-like
        A list of declinations for your objects in degrees

    properties : dictionary (optional)
        Any additional properties that you want to carry around with the
        angular positions.  This isn't used at all by AngularCatalog, but
        makes it easier to access things.
        
    weight_file : string (optional)
        The file name of a FITS file to read in.

    image_mask : ImageMask instance (optional)
        An instance of an ImageMask object to be used as this catalog's 
        image mask.
        
    Returns
    -------
    cat : AngularCatalog instance
        The AngularCatalog instance with all the properties that you gave
        it.
    """

    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, ra, dec, properties=None, weight_file=None, 
        image_mask=None, fits_file_type='weight'):
        """
        The init function for the AngularCatalog class
        """

        #Make sure we have Numpy arrays
        ra = np.array(ra)
        dec = np.array(dec)

        #Check to make sure we have sensible values for RA and Dec
        if ra.ndim != 1:
            raise ValueError('RA list must be a 1D array')
        if dec.ndim != 1:
            raise ValueError('Dec list must be a 1D array')
        if dec.size != ra.size:
            raise ValueError('RA and Dec arrays must be the same length')

        #Now store the RA and Dec information
        self._ra = ra
        self._dec = dec
        self._ra_range = np.array([ra.min(), ra.max()])
        self._ra_span = np.diff(self._ra_range)[0]
        self._dec_range = np.array([dec.min(), dec.max()])
        self._dec_span = np.diff(self._dec_range)[0]
        self._input_n_objects = ra.size
        self._n_objects = None

        #Store the info from keywords
        self._weight_file_name = weight_file
        self._fits_file_type = fits_file_type
        self._image_mask = image_mask
        self._properties = properties

        #Store some defaults/holders
        self._theta_bins=None
        self.cfs={}

        #Make blank things so I can ask "is None" rather than "exists"
        self._data_tree=None
        self._random_tree=None
        self._ra_random=None
        self._dec_random=None
        self._random_number_type=None
        self._random_quantity=None
        self._random_gen_kwargs={}
        self._Gp=None
        self._completeness=None
        self._use=None
        self._use_random=None
        self._subregion_number=None

        #If we have a weight file name and a fits file type, set that up
        if (weight_file is not None) and (fits_file_type is not None):
            self.mask_from_FITS_file(weight_file, fits_file_type) 
                    
        #Setup the mask if we have one.
        if self._image_mask:
            self.setup_mask()

    #----------------------------------------------------------------------
    #--------------------------------------------#
    #- Class method for making a random catalog -#
    #--------------------------------------------#
    @classmethod
    def random_catalog(cls, n_randoms, image_mask = None, ra_range=None,
                       dec_range=None, **kwargs):
        """
        Creates an AngularCatalog populated with RAs and Decs placed
        randomly within the mask.  This can be passed either an image
        mask or an RA and Dec range

        **Syntax**

        * cat = ac_class.AngularCatalog.random_catalog(n_randoms,
                                     image_mask=ImageMask_object)
        OR
        * cat = ac_class.AngularCatalog.random_catalog(n_randoms,
                        ra_range=[min, max], dec_range=[min, max])

        Parameters
        ----------
        n_randoms : scalar
            The number of randoms that you want in you catalog
            
        image_mask : ImageMask object (optional)
            An ImageMask object with the outline that you want for your
            randoms.  This is one option.
            
        ra_range : array-like (optional, length=2)
            The minimum and maximum RA you would like your randoms to have.
            This is an alternative to the image_mask option.  This must be
            combined with the dec_range argument as well.
            
        dec_range : array-like (optional, length=2)
            The minimum and maximum Dec you would like your randoms to have.
            This is an alternative to the image_mask option.  This must be
            combined with the ra_range argument.
            
        **kwargs : keyword arguments (optional)
            Keyword arguments to be passed to 
            ImageMask.generate_random_sample().

        Returns
        -------
        cat : AngularCatalog object
            An AngularCatalog instance with n_randoms distributed over
            either the image_mask or over the RA and Dec range.  If the
            generate_random_sample call returns magnitudes and radii, they
            will be stored in cat._properties with keys "magnitude" and
            "radius".
        """

        #Make an image mask from the RA and Dec ranges if we don't have an
        #image mask already
        need_image_mask = image_mask is None
        if need_image_mask:
            image_mask = imclass.ImageMask.from_ranges(ra_range, dec_range)

        #Use the ImageMask to create random RAs and Decs and make them into
        #an AngularCatalog with the corresponding mask.
        temp = image_mask.generate_random_sample(n_randoms, )
        ra, dec, comp, mags, radii = temp
        props = {}
        if mags is not None:
            props['magnitude'] = mags
        if radii is not None:
            props['radius'] = radii
        return AngularCatalog(ra, dec, image_mask=image_mask, 
                                properties=props)
    
#==========================================================================
#==========================================================================

    #----------------------------------------------------------------------
    #  Image mask creation and manipulation
    #----------------------------------------------------------------------

    #-----------------------------------------#
    #- Set the image mask from a weight file -#
    #-----------------------------------------#
    def mask_from_FITS_file(self, filename, type):
        """
        Set the weight file name and make an image mask out of the file.
        This is specifically for weight files, where the array value at a
        particular position is related to the RMS- I need to ask exactly 
        how...

        Parameters
        ----------
        filename : string
            The location of the FITS file that you want to process to
            a weight mask.  The file name should be specified from /
        """
        self._weight_file_name=filename
        self._fits_file_type = type
        self._image_mask= imclass.ImageMask.from_FITS_file(filename,
                                    fits_file_type = self._fits_file_type)
        self.setup_mask()
        return
        
    #----------------------------------------------------------------------       
    #--------------------------------------#
    #- Set the image mask from a flag map -#
    #--------------------------------------#
#     def mask_from_levels_file(self, filename):
#         """
#         This routine sets up an ImageMask with a levels file- a FITS file
#         that contains integers indicating the exposure level at that 
#         position
# 
#         Parameters
#         ----------
#         filename : string
#             The location of the FITS file that you want to process to
#             a flag map.  The file name should be specified from /
#         """
#         self._weight_file_name=filename
#         self._fits_file_type = 'levels'
#         self._image_mask= imclass.ImageMask.from_FITS_file(filename,
#                                     fits_file_type = self._fits_file_type)
#         self.setup_mask()
#         return        
        

    #----------------------------------------------------------------------
    #----------------------------------#
    #- Set the image mask from ranges -#
    #----------------------------------#
    def mask_from_ranges(self, ra_range, dec_range):
        """
        Creates an image mask that is a rectangle in RA-Dec space with
        the ranges given.

        Parameters
        ----------
        ra_range : array-like (length 2)
            Minimum and maxium RA for the mask in degrees.
        dec_range : array-like (length 2)
            Minimum and maxium Dec for the mask in degrees.
        """

        self._image_mask = imclass.ImageMask.from_ranges(ra_range,
                                                         dec_range)
        self.setup_mask()
        return

    #----------------------------------------------------------------------
    #------------------------------------#
    #- Set the image mask from an array -#
    #------------------------------------#
    def mask_from_array(self, mask, ra_range, dec_range):
        """
        WARNING:
        This does not necessarily do the thing you think it does.  See the
        documentation for ImageMask_class.from_array.

        -------

        Creates an image mask from a mask array with values from 0 to 1,
        inclusive, and the RA and Dec ranges that the array should cover on
        the sky.

        For more information on exactly how the image mask routine works,
        see the ImageMask class documentation under the classmethod
        from_array.

        Parameters
        ----------
        mask : 2D array
            2D matrix with values between 0 and 1 inclusive denoting the
            completeness in each cell.  1 indicates 100% random placement,
            0 for no randoms.
        ra_range : array-like (length 2)
            Minimum and maxium RA for the mask in degrees.
        dec_range : array-like (length 2)
            Minimum and maxium Dec for the mask in degrees.
        """

        self._image_mask = imclass.ImageMask.from_array(mask, ra_range,
                                                        dec_range)
        self.setup_mask()
        return

    #----------------------------------------------------------------------
    #------------------------------------#
    #- Set up mask-dependent properties -#
    #------------------------------------#
    def setup_mask(self):
        """
        This routine runs through the creation/reassignment of some mask-
        dependent properties.

        It checks which galaxies fall inside the
        mask and screens out the ones that fall in areas with
        completeness==0.  Then the data BallTree is recomputed if the data
        points to use have been changed and record how many objects we have
        inside the mask.
        """
        
        #Ask the mask for the completenesses of each data object
        self._completeness=self._image_mask.return_completenesses(self._ra,
                                       self._dec, use_mags_and_radii=False)

        #Store the old _use array if we have one- if not, create a bogus one
        if self._use is not None:
            old_use = copy.deepcopy(self._use)
        else:
            old_use = -np.ones(self._input_n_objects)

        #Update the _use array and compare to the old one
        self._use = self._completeness > 0
        use_is_same = (self._use == old_use).all()

        if not self._use.any():
            print ""
            warnings.warn("setup_mask says: WARNING- You're trying to set"
                " up a mask that doesn't intersect with your data points "
                "at all.")

        #If the _use array is different, update some things
        if not use_is_same:
            #Set up the data tree now that we have a mask
            if self._use.any():
                self._make_data_tree()
            #Record how many objects we're actually using
            self._n_objects=len(self._ra[self._use])

    #----------------------------------------------------------------------
    #-------------#
    #- Move mask -#
    #-------------#
    def move_mask_on_sky(self, delta_ra=0, delta_dec=0, theta_degrees=0,
                         preview=False):
        """
        Calls the image mask's translation/rotation routine:
        
        If preview=True, the catalog's instance won't be changed and a new
        ImageMask with the altered coordinates will be returned. Otherwise,
        the catalog's ImageMask will be changed and nothing is returned.

        Parameters
        ----------
        delta_ra : scalar (optional)
            Amount to change the central RA.  Defaults to 0
            
        delta_dec : scalar (optional)
            Amount to change the central Dec.  Defaults to 0
            
        theta_degrees : scalar (optional)
            Amount to rotate the WCS instance in degrees.  Defaults to 0
            
        preview : boolean (optional)
            Default it False.  If True, returns an ImageMask instance with
            the same mask as the ImageMask that generated it but with the
            WCS instance altered.  The calling instance won't be changed.
            If False, the calling instance's WCS instance will be changed
            and nothing will be returned.

        Returns
        -------
        altered_mask : ImageMask instance
            This is only returned if preview==False.  It is a copy of the
            calling ImageMask instance with a changed WCS instance.
        """
        
        return self._image_mask.move_mask_on_sky(delta_ra=delta_ra,
                                                 delta_dec=delta_dec,
                                                 theta_degrees=theta_degrees,
                                                 preview=preview)
    
    #----------------------------------------------------------------------

    #--------------------#
    #- Generate randoms -#
    #--------------------#
    def generate_random_sample(self, number_to_make=None, 
                     multiple_of_data=None, density_on_sky=None, **kwargs):
        """
        This routine calls the image mask's masked random generation to
        create a random sample to compare the data to.  The number of
        randoms can be specified several ways.  The number of randoms can
        be 1) a set number, 2) a multiple of the number of data objects,
        or 3) a number density on the sky (in objects per square degree).

        Parameters
        ----------
        
        number_to_make : int (optional)
            Specify this if you want a set number of randoms placed.
            Ignored if either multiple_of_data or density_on_sky is given.

        multiple_of_data : float (optional)
            The number of randoms to place as a multiple of how many data
            objects are in the catalog.  If there are N objects in the
            catalog, this will produce np.ceil(multiple_of_data * N)
            randoms.  Ignored if density_on_sky is given.

        density_on_sky : float (optional)
            The number of randoms to place per square degree on the sky.
            This is not the most exact procedure because the calculation
            of the mask's footprint is approximate.
            
        **kwargs : (optional)
            Keyword arguments to be passed to 
            ImageMask.generate_random_sample.  See documentation for 
            options.
        """
        
        #Store how we called generate random sample
        self._random_gen_kwargs = kwargs

        #Figure out how many randoms we're making
        if density_on_sky is not None:
            self._random_number_type = 'density'
            self._random_quantity = density_on_sky
            num_per_steradian = density_on_sky * 360.**2 / (4*np.pi**2)
            area_steradians = self._image_mask.masked_area_solid_angle()
            N_make = num_per_steradian * area_steradians
            
        elif multiple_of_data is not None:
            self._random_number_type = 'multiple'
            self._random_quantity = multiple_of_data
            N_make = np.ceil(multiple_of_data * self._n_objects)

        elif number_to_make is not None:
            self._random_number_type='number'
            self._random_quantity = number_to_make
            N_make = number_to_make

        else:
            raise ValueError("To generate randoms, you must specify one "
                             "of the three options: density_on_sky in "
                             "number per square degree, multiple_of_data, "
                             "or number_to_make.")

        #Make sure we have an integer
        N_make = int(N_make)

        #Get the RAs and Decs from the ImageMask and store
        ra, dec, __ = self._image_mask.generate_random_sample(N_make, 
                                                                **kwargs)
        self._ra_random = ra
        self._dec_random = dec

        #Rerun the random tree
        self._make_random_tree()
        
    #----------------------------------------------------------------------

    #-----------------------#
    #- Re-generate randoms -#
    #-----------------------#
    def rerun_generate_randoms(self):
        """
        Takes the stored parameters from the last time that we ran
        generate_random_sample and reruns it with the same arguments.
        """

        #Use the random_number_type to determine which of the arguments
        #random_quantity should be passed to.
        if self._random_number_type == 'density':
            self.generate_random_sample(density_on_sky =
                                        self._random_quantity, 
                                        **self._random_gen_kwargs)
            
        elif self._random_number_type == 'multiple':
            self.generate_random_sample(multiple_of_data =
                                        self._random_quantity,
                                        **self._random_gen_kwargs)
            
        elif self._random_number_type == 'number':
            self.generate_random_sample(number_to_make =
                                        self._random_quantity,
                                        **self._random_gen_kwargs)

        else:
            raise ValueError("rerun_randoms says: Not sure how you managed"
                             " this, but you have an invalid random_number"
                             "_type.")
                             
    #----------------------------------------------------------------------
    #------------------#
    #- Subdivide mask -#
    #------------------#        
    def subdivide_mask(self, **kwargs):
        """
        Calls the image mask's subdivide_mask routine- see the 
        ImageMask_class documentation for all the options.
        """
        self._image_mask.subdivide_mask(**kwargs)
        self._subregion_number = self._image_mask.return_subregions(
                                                    self._ra, self._dec)
        
#==========================================================================
#==========================================================================     

    #----------------------------------------------------------------------
    #  Ball tree creation
    #----------------------------------------------------------------------

    #---------------------------------#
    #- Compute BallTree for the data -#
    #---------------------------------#
    def _make_data_tree(self):
        """
        This routine creates and stores a BallTree (from sklearn) on the
        data points so this doesn't have to be computed every time you run
        a correlation function.
        """
        #Transform to the funny cartesian coords that astroML uses and make
        #the tree
        data = np.asarray(corr.ra_dec_to_xyz(self._ra[self._use],
                                             self._dec[self._use]),
                                             order='F').T
        self._data_tree = BallTree(data, leaf_size=2)
        return

    #----------------------------------------------------------------------

    #------------------------------------------#
    #- Compute BallTree for the random sample -#
    #------------------------------------------#
    def _make_random_tree(self):
        """
        This routine creates and stores a BallTree (from sklearn) on the
        random points so this doesn't have to be computed every time you
        run a correlation function.
        """
        
        #Make sure we have the random data made
        if (self._ra_random is None) or (self._dec_random is None):
            raise ValueError("You must generate randoms before you can "
                             "make the random tree.")

        #Make the tree
        print "make_randomtree says: Computing the BallTree for randoms."
        random_data=np.asarray(corr.ra_dec_to_xyz(self._ra_random,
                                                  self._dec_random),
                                                  order='F').T
        self._random_tree = BallTree(random_data, leaf_size=2)                
                
        return          


            
#==========================================================================
#==========================================================================     

    #----------------------------------------------------------------------
    #  Calculating the correlation functions
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------

    #-----------------------------------------------------#
    #- Calculate the correlation function without errors -#
    #-----------------------------------------------------# 
    def cf(self, estimator='landy-szalay', n_iter=1, clobber=False,
          save_file_base=None, name='cf'):
        """
        Calculates a correlation function without estimating errors.  This
        procedure can be iterated with different instances of the random
        catalog to reduce noise in the CF with less additional computation
        time than increasing the number of randoms in the random catalog.

        The correlation function information is stored in self.cfs[name],
        with the default name 'cf'.

        Parameters
        ----------
        estimator : string (optional)
            Which correlation function estimator to use.  The default is
            'landy-szalay', which does CF = (DD - 2DR + RR)/RR.  The other
            Option is 'standard' which does CF = DD/RR - 1.

        n_iter : int (optional)
            The number of iterations to perform with different random
            catalogs.  The correlation functions are computed and
            averaged together.

        name : string (optional)
            The name to assign this correlation function object in the
            AngularCatalog's list of correlation functions.  Defaults to
            'cf'.

        clobber : bool (optional)
            Whether or not to overwrite a correlation function already
            stored with the same name.  If clobber=True, existing CFs will
            be overwritten.  If False, trying to use an already-existing
            name will result in a ValueError.  Default is False.

        save_file_base : string (optional)
            Base of the file name to save the correlation function to as it 
            is computed.  The final result will also be saved.  NOTE:  
            `save_file_base` is not the full file name!  The saving routine 
            saves files as `file_base + name + '.npz'`, so if name="cf" and
            `save_file_base = "/path/to/file/"`, then the saved file will be
            /path/to/file/cf.npz.  Default is None, so no file is saved.
        """

        if (name in self.cfs.keys()) and not clobber:
            raise ValueError("AngularCatalog.cf says: There's already"
                             " a CF by that name.  Please choose another or"
                             " overwrite by calling with clobber=True")

        #Make sure that we have everything we need and fix anything missing
        #that's fixable
        self.__check_cf_setup(need_subregions=False, check_trees=True)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'no_error',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'        : estimator
             }
        self.cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self.cfs[name].get_thetas(unit='degrees')
        nbins=len(centers)

        #Do the calculation
        cf=np.zeros(nbins)
        DD=np.zeros(nbins)
        print ("AngularCatalog.cf says: doing a CF calculation without "
               "error estimation")
        iterations={}
        for it in np.arange(n_iter):
            #Calculate the CF
            this_cf, this_dd = corr.two_point_angular(self._ra[self._use], 
                                                     self._dec[self._use], 
                                                     edges,
                                                     BT_D=self._data_tree, 
                                                     BT_R=self._random_tree,
                                                     method=estimator, 
                                                     ra_R=self._ra_random,
                                                     dec_R=self._dec_random,
                                                     return_DD=True)
            #Add to the cumulative CF and DD
            iterations[it]=this_cf
            cf += this_cf
            DD += this_dd/2.
            if save_file_base is not None:
                #Set the CF and DD to the averages so far and save.
                self.cfs[name].set_cf(cf/(it+1), np.zeros(nbins),
                                       iterations=iterations)
                self.cfs[name].set_counts(DD=DD/(it+1))
                self.save_cf(save_file_base, cf_keys=[name])
            if n_iter >1:
                #Make a new set of randoms
                self.rerun_generate_randoms()

        #Divide out the number of iterations
        cf/=n_iter
        DD/=n_iter
        
        #Make sure we've stored everything properly even if we're not
        #saving
        self.cfs[name].set_cf(cf, np.zeros(nbins), iterations=iterations)
        self.cfs[name].set_counts(DD)

    #----------------------------------------------------------------------

    #----------------------------------------------------#
    #- Find the CF and error by single-galaxy bootstrap -#
    #----------------------------------------------------#
    def cf_bootstrap(self, n_boots=10, clobber=False, 
                     estimator='landy-szalay', save_file_base=None,
                     name='galaxy_bootstrap'):
        """
        Calculate the correlation function and error with single-galaxy
        bootstrapping.  This means that the correlation function is
        calculated repeatedly.  Each calculation is done on a random
        sampling of the data (with replacement).  The correlation function
        is then the average of the bootstraps and the error at each theta
        separation is the standard deviation of the bootstraps at that
        separation.  There is some evidence that single galaxy
        bootstrapping may be a bad idea, but I haven't had any trouble with
        it in my own work.

        Parameters
        ----------
        n_boots : int (optional)
            Number of times the bootstrap resampling and correlation
            function measurement will be done.  Default is 10.

        estimator : string (optional)
            Which correlation function estimator to use.  The default is
            'landy-szalay', which does CF = (DD - 2DR + RR)/RR.  The other
            Option is 'standard' which does CF = DD/RR - 1.
            
        name : string (optional)
            The name to assign this correlation function object in the
            AngularCatalog's list of correlation functions.  Defaults to
            'galaxy_bootstrap'.

        clobber : bool (optional)
            Whether or not to overwrite a correlation function already
            stored with the same name.  If clobber=True, existing CFs will
            be overwritten.  If False, trying to use an already-existing
            name will result in a ValueError.  Default is False.

        save_file_base : string (optional)
            Base of the file name to save the correlation function to as it 
            is computed.  The final result will also be saved.  NOTE:  
            `save_file_base` is not the full file name!  The saving routine 
            saves files as `file_base + name + '.npz'`, so if name="cf" and
            `save_file_base = "/path/to/file/"`, then the saved file will be
            /path/to/file/cf.npz.  Default is None, so no file is saved.
        """

        if (name in self.cfs.keys()) and not clobber:
            raise ValueError("AngularCatalog.cf_bootstrap says: "
                             "There's already a CF by that name.  Please "
                             "choose another or overwrite by calling with "
                             "clobber=True")
        
        #Check that everything is set up
        self.__check_cf_setup(need_subregions=False, check_trees=False)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'single_galaxy_bootstrap',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'        : estimator
             }
        self.cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self.cfs[name].get_thetas(unit='degrees')
        nbins=len(centers)

        #Make an array so it's easy to average over the boots
        temp = np.zeros((n_boots, nbins))
        
        #This RR will keep track of the RR counts so you don't have to
        #calculate them every time.
        rr=None
        
        #A holder for the boots that will be passed to the
        #CorrelationFunction as the iterations
        bootstrap_boots={}
        
        print ("AngularCatalog.cf_bootstrap says: doing a bootstrap "
               "CF calculation")

        #Loop through the boots
        for i in np.arange(n_boots):
            #Give a progress report
            print "calculating boot", i
            
            #Choose the right number of galaxies *with replacement*
            ind=np.random.randint(0, self._n_objects, self._n_objects)
            ra_b=self._ra[self._use][ind]
            dec_b=self._dec[self._use][ind]
            
            #Calculate this boot
            boot, rr = corr.two_point_angular(ra_b, dec_b, edges,
                                              BT_D=self._data_tree,
                                              BT_R=self._random_tree,
                                              method=estimator,
                                              ra_R=self._ra_random,
                                              dec_R=self._dec_random,
                                              RR=rr, return_RR=True)
            bootstrap_boots[i] = boot
            
            #Store what we have
            temp[i]=bootstrap_boots[i]
            if (save_file_base is not None):
                bootstrap_cf=np.nanmean(temp[0:i+1], axis=0)
                #The denominator should have an Nboots-1 in it, hence
                # the ddof=1.  See Norberg et al 2009, section 2.4
                bootstrap_cf_err=np.nanstd(temp[0:i+1], axis=0, ddof=1)
                self.save_cf(save_file_base, cf_keys=[name])
                
        #Now we're done- do the final storage.
        bootstrap_cf=np.nanmean(temp, axis=0)
        bootstrap_cf_err=np.nanstd(temp, axis=0, ddof=1)
        self.cfs[name].set_cf(bootstrap_cf, bootstrap_cf_err,
                               iterations=bootstrap_boots)
        self.cfs[name].set_counts(RR=rr)
        self.save_cf(save_file_base, cf_keys=[name])
        
    #----------------------------------------------------------------------

    #----------------------------------------#
    #- Find the CF and error by jackknifing -#
    #----------------------------------------#
    def cf_jackknife(self, ignore_regions=[], estimator='landy-szalay',
                     save_file_base=None, name='jackknife',
                     clobber=False):
        """
        Computes a correlation function with error through jackknifing.
        Jackknifing takes a divided mask and performs the correlation
        function calculation on the field with each sub-region
        removed in turn.  The correlation function is then taken to be
        average of the individual jackknifes and the error is the
        standard deviation.

        Note that this requires a mask that's subdivided into blocks. You
        can do this with subdivide_mask.  The way you subdivide the mask
        matters if you're trying to avoid the concerns about bias in the
        error estimates.  Try to make your boxes ~10x larger than the
        largest scale you want to measure the CF on and try to make them
        all approximately equally populated.  If one region is mostly
        empty, you can exclude it by figuring out what region number it is
        and adding that to the ignore_regions array.

        Parameters
        ----------
        ignore_regions : 1D array-like (optional)
            Which subregions to leave out when calculating the jackknife
            CF.  None of the galaxies in these regions will be used in
            any of the jackknife correlation function measurements.
        
        estimator : string (optional)
            Which correlation function estimator to use.  The default is
            'landy-szalay', which does CF = (DD - 2DR + RR)/RR.  The other
            Option is 'standard' which does CF = DD/RR - 1.
        
        name : string (optional)
            The name to assign this correlation function object in the
            AngularCatalog's list of correlation functions.  Defaults to
            'jackknife'.

        clobber : bool (optional)
            Whether or not to overwrite a correlation function already
            stored with the same name.  If clobber=True, existing CFs will
            be overwritten.  If False, trying to use an already-existing
            name will result in a ValueError.  Default is False.

        save_file_base : string (optional)
            Base of the file name to save the correlation function to as it 
            is computed.  The final result will also be saved.  NOTE:  
            `save_file_base` is not the full file name!  The saving routine 
            saves files as `file_base + name + '.npz'`, so if name="cf" and
            `save_file_base = "/path/to/file/"`, then the saved file will be
            /path/to/file/cf.npz.  Default is None, so no file is saved.
        """

        if (name in self.cfs.keys()) and not clobber:
            raise ValueError("AngularCatalog.cf_jackknife says: "
                             "There's already a CF by that name.  Please "
                             "choose another or overwrite by calling with "
                             "clobber=True")

        #Check to make sure we have everything we need
        self.__check_cf_setup(need_subregions=True, check_trees=False)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'jackknife',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'        : estimator
             }
        self.cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self.cfs[name].get_thetas(unit='degrees')
        nbins=len(centers)
        
        #pull out the unique subregion numbers and figure out which to use
        regions=np.asarray(list(set(self._subregion_number)))
        use_regions=[r for r in regions if (r not in ignore_regions) and
                     (r != -1)]
        use_regions=np.array(use_regions)
        n_jacks=len(use_regions)

        #Figure out which subregions the randoms are in
        random_subregions=self._image_mask.return_subregions(
                                    self._ra_random, self._dec_random)

        #Make masks that exclude the galaxies outside the mask
        #(subregion == -1) or are in ignored regions
        valid_subregion = ma.masked_not_equal(self._subregion_number, 
                                                            -1).mask
        random_valid_subregion=ma.masked_not_equal(random_subregions, 
                                                                -1).mask
        for bad_reg in ignore_regions:
            #Change the masks so that it masks out the guys in this ignored
            #region too
            this_mask = ma.masked_not_equal(self._subregion_number, 
                                                            bad_reg).mask
            valid_subregion = valid_subregion & this_mask
            this_mask = ma.masked_not_equal(random_subregions, bad_reg).mask
            random_valid_subregion = random_valid_subregion & this_mask        

        #Now loop through the regions that you should be using 
        #and calculate the correlation function leaving out each
        jackknife_jacks = {}
        temp = np.zeros((n_jacks, nbins))
        for i, r in enumerate(use_regions):
            #Make the mask for the data
            not_region_r = ma.masked_not_equal(self._subregion_number, 
                                                            r).mask  
            this_jackknife = valid_subregion & not_region_r & self._use  
            
            #Make the mask for the randoms
            random_not_region_r = ma.masked_not_equal(random_subregions, 
                                                            r).mask
            random_this_jackknife = (random_not_region_r & 
                                            random_valid_subregion)

            #Do the calculation for this jackknife and store it
            print "calculating jackknife", i
            this_random_ra = self._ra_random[random_this_jackknife]
            this_random_dec = self._dec_random[random_this_jackknife]
            jack = corr.two_point_angular(self._ra[this_jackknife],
                                          self._dec[this_jackknife],
                                          edges, method=estimator,
                                          ra_R = this_random_ra,
                                          dec_R = this_random_dec)
            jackknife_jacks[r] = jack
            temp[i]=jackknife_jacks[r]
            
            #Do the saving if we have a file to save to
            jackknife_cf=np.nanmean(temp[0:i+1], axis=0)
            #The factor of np.sqrt(n_jacks - 1.) is to take into account
            #the fact that the different samples aren't all independent.
            #Taken from Norberg et al 2009, section 2.3
            jackknife_cf_err=(np.sqrt(n_jacks - 1.) * 
                                    np.nanstd(temp[0:i+1], axis=0))
            self.cfs[name].set_cf(jackknife_cf, jackknife_cf_err,
                                   iterations=jackknife_jacks)
            if (save_file_base is not None):
                    self.save_cf(save_file_base, cf_keys=[name])

    #----------------------------------------------------------------------

    #--------------------------------------------#
    #- Find the CF and error by block bootstrap -#
    #--------------------------------------------#
    def cf_block_bootstrap(self, n_boots=10, estimator='landy-szalay', 
                           save_file_base=None, name='block_bootstrap', 
                           clobber=False, ignore_regions=[]):
        """
        Compute the correlation function with errors from block
        bootstrapping.  Block bootstrapping is similar to single-galaxy
        bootstrapping, but instead of selecting from the set of galaxies,
        it selects from spatial blocks on the image.  This routine 
        requires that the mask be subdivided into blocks.
        

        Parameters
        ----------
        n_boots : int (optional)
            Number of iterations of the bootstrapping process to do.
            Default is 10.
            
        ignore_regions : 1D array-like (optional)
            Which subregions to leave out when calculating the block
            boostrap CF.  None of the galaxies in these regions will be
            used in any of the block bootstrap correlation function
            measurements.
            
        estimator : string (optional)
            Which correlation function estimator to use.  The default is
            'landy-szalay', which does CF = (DD - 2DR + RR)/RR.  The other
            Option is 'standard' which does CF = DD/RR - 1.
        
        name : string (optional)
            The name to assign this correlation function object in the
            AngularCatalog's list of correlation functions.  Defaults to
            'block_bootstrap'.

        clobber : bool (optional)
            Whether or not to overwrite a correlation function already
            stored with the same name.  If clobber=True, existing CFs will
            be overwritten.  If False, trying to use an already-existing
            name will result in a ValueError.  Default is False.

        save_steps_file : string (optional)
            Base of the file name to save the correlation function to as it is
            computed.  The final result will also be saved.  NOTE:  
            `save_file_base` is not the full file name!  The saving routine 
            saves files as `file_base + name + '.npz'`, so if name="cf" and
            `save_file_base = "/path/to/file/"`, then the saved file will be
            /path/to/file/cf.npz.  Default is None, so no file is saved.
        """

        #Make sure we have a legit name
        if (name in self.cfs.keys()) and not clobber:
            raise ValueError("AngularCatalog.cf_block_bootstrap says: "
                             "There's already a CF by that name.  Please "
                             "choose another or overwrite by calling with "
                             "clobber=True")

        #Check to make sure I have everything that I need
        self.__check_cf_setup(need_subregions=True, check_trees=False)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'block_bootstrap',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'        : estimator
             }
        self.cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self.cfs[name].get_thetas(unit='degrees')
        nbins = len(centers)

        #Figure out which subregions we should be using
        regions=np.asarray(list(set(self._subregion_number)))
        use_regions=[r for r in regions if (r not in ignore_regions) and
                     (r != -1)]
        use_regions=np.array(use_regions)

        #Figure out where the randoms are
        rnd_subregions=self._image_mask.return_subregions(self._ra_random,
                                                          self._dec_random)

        #Make a dictionary of arrays containing the indices of the members
        #of each sub-region we need
        indices={}
        random_indices={}
        for r in use_regions:
            indices[r]=np.where(self._subregion_number == r)[0]
            random_indices[r]=np.where(rnd_subregions == r)[0]

        #Loop through the bootstraps
        block_bootstrap_boots = {}
        n_choose = len(use_regions)
        temp = np.zeros((n_boots, nbins))
        print "block boots looping through boots"
        for i in np.arange(n_boots):
            #Which regions will be using?
            this_boot=rand.choice(use_regions, size=n_choose)
            this_boot_indices=np.array([], dtype=np.int)
            this_boot_random_indices=np.array([], dtype=np.int)
            
            #Which indices correspond to the regions we want?
            for region in this_boot:
                this_boot_indices=np.concatenate((this_boot_indices,
                                                  indices[region]))
                this_boot_random_indices=np.concatenate(
                        (this_boot_random_indices, random_indices[region]))

            print "calculating boot", i
            #Calculate this boot's CF
            this_random_ra = self._ra_random[this_boot_random_indices]
            this_random_dec = self._dec_random[this_boot_random_indices]
            temp[i] = corr.two_point_angular(self._ra[this_boot_indices], 
                                             self._dec[this_boot_indices], 
                                             edges, method=estimator, 
                                             ra_R = this_random_ra,
                                             dec_R = this_random_dec)
            block_bootstrap_boots[i] = temp[i]
            cf=np.nanmean(temp[0:i+1], axis=0)
            #The denominator should have an Nboots-1 in it, hence
            # the ddof=1.  See Norberg et al 2009, section 2.4
            cf_err=np.nanstd(temp[0:i+1], axis=0, ddof=1)
            self.cfs[name].set_cf(cf, cf_err, 
                    iterations=block_bootstrap_boots)
            if (save_file_base is not None):
                self.save_cf(save_file_base, cf_keys=[name])


#==========================================================================
#==========================================================================     

    #----------------------------------------------------------------------
    #  Book-keeping
    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------

    def set_theta_bins(self, min_theta, max_theta, nbins,
                       unit='a', logbins=True):
        """
        Create and store the theta bins to use when calculating a
        correlation function.

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
        
        #Make a ThetaBins class and save it.
        self._theta_bins = binclass.ThetaBins(min_theta, max_theta, nbins,
                                              unit=unit, logbins=logbins)

    #----------------------------------------------------------------------

    #---------------------------------------------------------------------#
    #- Check to make sure we have all the info needed for CF calculation -#
    #---------------------------------------------------------------------#         
    def __check_cf_setup(self, need_subregions=False, check_trees=True):
        """
        Make sure that we have all the things we need to do a
        correlation function properly
    
        Parameters
        ----------
        need_subregions : bool
            If True, requires that subregions are defined.  Default is
            False- no subregions required.

        check_trees : bool
            If True, will generate the BallTrees for the data and randoms
            if they're not already generated.  If False, it won't check.
            Default is True.
        """
        
        #Check that we have the bins 
        if not isinstance(self._theta_bins, binclass.ThetaBins):
            raise ValueError("CF calculations need separation bins.  Use "
                             "catalog.set_theta_bins(min_theta, max_theta,"
                             "nbins, unit='arcsec', logbins=True)")

        #Check the existence of a random sample
        if (self._ra_random is None) or (self._dec_random is None):
            raise ValueError("You must generate a random sample before you"
                             " run a correlation function.  Use "
                             "catalog.generate_random_sample number_to_"
                             "make=None, multiple_of_data=None, density"
                             "_on_sky=None)")
            
        #Check to make sure we have the trees for the appropriate guys
        if check_trees:
            if self._data_tree is None:
                self._make_data_tree()
            if self._random_tree is None:
                self._make_random_tree()

        #Check to make sure that the subdivisions have happened
        #if need_subregions.  If not, throw an error because it's
        #too specific to fill it in automatically
        if need_subregions:
            if self._subregion_number is None:
                raise ValueError("Jackknife and block bootstrap require "
                                "that you subdivide the field.  Call the "
                                "catalog.subdivide_mask() routine first.")
            
    #----------------------------------------------------------------------

    #-------------------------------------#
    #- Read in previously calculated CFs -#
    #-------------------------------------#
    def load_cf(self, filen, clobber=False, remove_prefix='',
                add_prefix='', custom_names=None, only_keys=None):
        """
        Load in correlation functions from files.  The files that will be
        read in are

        <filen>*.npz

        The keys to the AngularCatalog.cfs dictionary that the files will
        be read in to are the entire file name.  So, if you have a
        directory that contains /sample/path/this_cf.npz and
        /sample/path/this_other_cf.npz, by default they will be read in
        with keys 'this_cf' and 'this_other_cf'.

        The manipulations of the keys deserve an example.  If you have the
        two files, /sample/path/this_cf.npz and
        /sample/path/this_other_cf.npz, and add the argument
        remove_prefix='this_', they will be read in as 'cf' and 'other_cf'.
        If you set remove_prefix='this_' and add_prefix='some_', they will
        have keys 'some_cf' and 'some_other_cf'.  If you don't set
        remove_prefix or add_prefix but set custom_names={'this_cf':
        'cf_A', 'this_other_cf': 'cf_B'}, then they will have keys
        'cf_A' and 'cf_B'.

        Parameters
        ----------
        filen : string
            The beginning of the file names to read in.  If multiple files
            match this, they will all be considered.

        clobber : bool (optional)
            If True, existing correlation functions are overwritten if you
            read in a file with a duplicate key.  If False, duplicate CFs
            will raise a warning and not read in that CF.  Default is False

        remove_prefix : string (optional)
            If remove_prefix is defined, it will begin stripped from the
            left side of the keys.  For instance, if you had a directory,
            /path/, that contained a bunch of files but you just wanted
            /path/example_cf.npz and /path/example_boostrap_cf.npz, you
            would set
            
            filen = '/path/example_' and remove_prefix = 'example_'

            in order to read in correlation functions called 'cf' and
            'bootstrap_cf'.  By default, the entire file name is used as
            the key (without the path).

        add_prefix : string (optional)
            Used to add a string to the left side of keys (in case there
            are duplicate names or you want to be more specific than the
            file name).  For instance, if you have /path/cf.npz and
            /path/boostrap_cf.npz and wanted them to be read in with the
            keys 'older_cf' and 'older_bootstrap_cf', you would set
            add_prefix = 'older_'.

            add_prefix is applied after remove_prefix.  Default is no
            add_prefix.

        custom_names : dictionary (optional)
            If the prefix manipulation is insufficient for your purposes,
            you can specify custom names.  This is set up as a dictionary
            whose keys correspond to the file names (without the path or
            '.npz') and entries are strings containing the key you want
            to be used.  This is done after processing the remove_prefix
            and add_prefix, so if you set those as well, you should make
            the keys to this dictionary the altered names.

        only_keys : 1D array-like of strings (optional)
            A list of keys to use.  For instance, if you had two files,
            /path/cf.npz and /path/boostrap_cf.npz and set
            only_keys=["cf"], it would load /path/cf.npz and not the
            bootstrap.npz file.
        """

        #First, what files start with filen?
        file_list = misc.files_starting_with(filen)
        nfiles = len(file_list)

        #Only take the .npz files, without the .npz
        candidate_files = [f for f in file_list if f[-4:]==".npz"]
        names = [f.rstrip(".npz") for f in candidate_files]

        #Get just the file names without the path or .npz
        for i, n in enumerate(names):
            n = n.split('/')
            n = n[-1]
            names[i]=n

        #Remove the remove_prefix and add the add_prefix
        for i, n in enumerate(names):
            n = n.lstrip(remove_prefix)
            names[i] = add_prefix + n

        #Subset down to just the files and key names that we want to load
        if only_keys is None:
            only_keys = names
        files_to_load = [candidate_files[i] for i, n in enumerate(names)
                             if (n in only_keys)]
        print files_to_load
            
        #If there's a custom name dictionary, use it to make the name
        #substitutions
        if custom_names:
            names = [custom_names[n] for n in names]

        #Now load the guys that we've been asked for
        for i, key in enumerate(only_keys):
            load_allowed = (key not in self.cfs.keys()) | clobber
            if load_allowed:
                print "Loading: ", files_to_load[i]
                self.cfs[key] = cfclass.CorrelationFunction.from_file(
                                            files_to_load[i], name=key)
            else:
                warnings.warn("You already have a correlation function "
                              "loaded called " + key + ".  This CF will "
                              "not be read in.  Either set clobber = True "
                              "to overwrite or change the key somehow.")
                
    #----------------------------------------------------------------------

    #--------------------------------------------#
    #- Save the correlation functions to a file -#
    #--------------------------------------------#
    def save_cf(self, file_base, cf_keys=None, clobber=False):
        """
        Takes all the CF information we have and saves to a file
        per CF.  The file names will be of the form
            file_base + cf_key + '.npz'

        Parameters
        ----------
        file_base : string
            Path from / where the file will be saved plus any sort of
            beginning to the file name if you want it to be something
            other than <cf_key>.npz

        cf_keys : array-like (optional)
            Keys to which of the CFs in the self._cf dictionary to save.

        clobber : bool (optional)
            If clobber == False, save_cf() won't overwrite existing files.
            If clobber == True, it won't check.  Default is False.
        """

        #Check to see if the cf_key is a string- if it is, make it a list
        if type(cf_keys) == type(""):
            cf_keys = [cf_keys]

        #If they didn't say which ones specifically, save all
        if cf_keys is None:
            cf_keys=self.cfs.keys()

        for k in cf_keys:
            filen = file_base + k
            self.cfs[k].save(filen)
        
    #----------------------------------------------------------------------

    #-----------------------------------------------------------------#
    #- Read in previously calculated random-random counts for the IC -#
    #-----------------------------------------------------------------#
    def load_gp(self, filename, clobber=False):
        """
        Take the saved files with the normed random-random counts
        and read it in to self._Gp.

        Parameters
        ----------
        filename : string
            Path from / to the saved Gp file.

        clobber : bool (optional)
            If clobber == True, it will overwrite any existing Gp in the
            catalog.  If clobber == False, it will only read in the Gp if
            there isn't one already in existence.  Default is False.
        """

        if (self._Gp is None) or clobber:
            self._Gp = gpclass.Gp.from_file(filename)
        else:
            print ("angular_catalog.load_rr says: You've asked me not "
                   "to overwrite the existing RR counts and there's "
                   "already Gp information .")

    #----------------------------------------------------------------------

    #--------------------------------------------#
    #- Save the random-random counts for the IC -#
    #--------------------------------------------#
    def save_gp(self, filename):
        """
        If we have done the random-random counts for the integral
        constraint, save to a file

        Parameters
        ----------
        filename : string
            Path from / to save the Gp to.
        """
        
        if self._Gp:
            self._Gp.save(filename)
        else:
            print ("Sorry, I can't do that, Dave.  There is no Gp in "
                   "this catalog.")
        
#==========================================================================
#==========================================================================     

    #----------------------------------------------------------------------
    #  Plotting
    #----------------------------------------------------------------------

    def scatterplot_points(self, sample="data", save_to=None, 
                           max_n_points=1.e4, masked_data=True):
        """
        Scatterplots the data points and/or the random points.  By
        default, just shows the plot bit can save it.

        Parameters
        ----------
        save_to : string (optional)
            The file name to save the plot to.  If not specified, the plot
            will not be saved, just shown with show().

        sample : "data" | "random" | "both"  (optional)
            Which of the samples to plot.  If "data", plots the data
            points in blue.  If "random", plots the randoms in red.  If
            "both", plots the data in blue and the randoms in red.  
            Default is "data".

        max_n_points : int (optional)
            If the number of points in the sample to plot is larger than
            max_n_points, a random subsample is taken.  This is helpful
            for very large random samples so it doesn't take forever to 
            plot and/or save.  The default value is 1.e4.

        masked_data : bool (optional)
            Only used if sample is "data" or "both".  If True, plots only 
            the data points inside the mask.  If False, plots all data
            points.  Default is True.
        """

        #Check that we have a valid option for sample
        if sample not in ["data", "random", "both"]:
            raise ValueError("You have chosen an invalid option for "
                "sample.  It must be 'data', 'random', or 'both'")

        #Set up the plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("RA (degrees)")
        ax.set_ylabel("Dec (degrees)")

        #Add the data and/or randoms
        if (sample == 'data') | (sample == 'both'):
            if masked_data:
                ax.scatter(self._ra[self._use], self._dec[self._use],
                    color='b', label="Masked data")
            else:
                ax.scatter(self._ra, self._dec, color='b', 
                    label="Unmasked data")

        if (sample == 'random') | (sample == 'both'):
            ax.scatter(self._ra_random, self._dec_random, color='r',
                label="Randoms")

        #Make a legend
        handles, labels=ax.get_legend_handles_labels()
        legnd=ax.legend(handles, labels, loc=4, labelspacing=.15, 
            fancybox=True, fontsize=8, handlelength=3)
        legnd.get_frame().set_alpha(0.)

        #Show or save
        if save_to:
            print "saving"
            plt.savefig(save_to)
            plt.close()
        else:  
            print "showing"
            plt.show()
        
    #----------------------------------------------------------------------
    def plot_cfs(self, ax=None, save_to=None, return_axis=None, 
                colors=None, which_cfs=None, labels=None, 
                make_legend=True, **kwargs):
        """
        Plots up the correlation functions (all or a subset) currently 
        loaded in the catalog.  They can be put on an existing axis or 
        on a new plot.
        
        Parameters
        ----------
        ax : instance of a matplotlib axis (optional)
            If ax is specified, the correlation functions will be added to
            that axis.  If `ax==None`, the correlation function will be 
            plotted on a new axis.  Default is `ax=None`.
            
        which_cfs : array-like (optional)
            If you don't want to use all of the correlation functions saved
            in the catalog, pass the list of keys to the catalog.cfs 
            dictionary that you want to use.  Default is `which_cfs=None`, 
            which plots all of the correlation functions.
            
        colors : array-like (optional)
            Colors for the correlation functions.  Default is 
            `colors=None`, which uses an internal list of colors.
            
        labels : array-like (optional)
            A list of strings that contains the labels for the correlation
            functions to plot.  This list must be the same length as 
            `which_cfs`.  If `which_cfs==None`, `labels` must be the same
            length as the list of keys to `catalog.cfs`.
            
        make_legend : bool (optional)
            If `make_legend==True`, the routine will make a legend before 
            returning, saving, or showing the plot.  If False it won't.
            Default is True.
            
        save_to : string (optional)
            If save_to is specified, the plot will be saved to `save_to` at 
            the end of the routine.  If not saving to the current 
            directory, `save_to` should be the path from /.  Note that if 
            `ax` and `save_to` are both specified, the whole plot will be 
            saved.  If `save_to` and `return_axis` are both true, the
            plot will be saved and not returned.  If both are False, 
            plt.show() will be called at the end of the routine.
            
        return_axis : bool (optional)
            If True, this routine will return the axis that has been 
            plotted to after adding the correlation function. This is true 
            whether this routine was given the axis or created it.  Default 
            is False.  If `save_to` and `return_axis` are both true, the
            plot will be saved and not returned.  If both are False, 
            plt.show() will be called at the end of the routine.
            
        **kwargs : (optional)
            Keyword arguments to be passed to the CorrelationFunction class
            plotting routine, `CorrelationFunction.plot`.  You can specify
            `theta_unit`, which dictates the unit on the x axis, and 
            `log_yscale`, which sets the scale of the Y axis to logarithmic
            if it's True and linear if it's False.  The remaining keyword 
            arguments will be passed to ax.errorbar().
        
        Returns
        -------
        ax : instance of a matplotlib axis
            If return_axis is True, the axis plotted on will be returned.
        """
        
        #First, do some bookkeeping
        if which_cfs is None:
            which_cfs = self.cfs.keys()
            
        if labels is None:
            labels = which_cfs
        elif len(labels) != len(which_cfs):
            raise ValueError("If you specify labels, the array must be"
                        " the same length as which_cfs.  Which_cfs ="
                        " "+str(which_cfs))
        
        default_colors = ['r', 'Orange', 'Lime', "Blue", "Purple", 
                        "Magenta", "Cyan", "Green", "Yellow", "DarkCyan", 
                        "DimGray", "HotPink", "MediumPurple"]
        if colors is None:
            colors = default_colors
        if len(colors) < len(which_cfs):
            raise ValueError("I don't have enough colors.  If you gave me "
                    "the colors, you need to add more.  If you used the "
                    "default colors, then you have more than " + 
                    str(len(default_colors)) + " correlation functions.  "
                    "Cathy didn't think you'd need that many, so you'll "
                    "need to define your own colors.")
        
        #Now we can make our plot
        for i, cf_name in enumerate(which_cfs):
            ax = self.cfs[cf_name].plot(return_axis=True, ax=ax, 
                            label=labels[i], color=colors[i], **kwargs)
        
        #Make the legend
        if make_legend:
            handles, labels=ax.get_legend_handles_labels()
            legnd=ax.legend(handles, labels, loc=0, labelspacing=.15, 
                                fancybox=True, fontsize=8)
            legnd.get_frame().set_alpha(0.)
        
        #Do something with the plot
        if save_to:
            plt.savefig(save_to, bbox_inches="tight")
            plt.close()
        elif return_axis:
            return ax
        else:
            plt.show()

    #----------------------------------------------------------------------
    def plot_errors(self, which_cfs=None, labels=None, save_to=None, 
                    colors=None, log_yscale=True, **kwargs):
        """
        A plotting routine to compare the size of the error bars on 
        different correlation functions stored in the catalog.
        
        Parameters
        ----------
        which_cfs : array-like (optional)
            If you don't want to use the errors on all of the correlation 
            functions saved in the catalog, pass the list of keys to the 
            catalog.cfs  dictionary that you want to use.  Default is 
            `which_cfs=None`, which plots all of the correlation functions.
            
        labels : array-like (optional)
            A list of strings that contains the labels for the errors
            to plot.  This list must be the same length as 
            `which_cfs`.  If `which_cfs==None`, `labels` must be the same
            length as the list of keys to `catalog.cfs`.
        
        colors : array-like (optional)
            Colors for the correlation functions.  Default is 
            `colors=None`, which uses an internal list of colors.
            
        log_yscale : bool (optional)
            If True, the y axis will be log-scaled.  If False, the y axis 
            will be linearly scaled.  Default is True.            
            
        save_to : string (optional)
            If save_to is specified, the plot will be saved to `save_to` at 
            the end of the routine.  If not saving to the current 
            directory, `save_to` should be the path from /.  If save_to is 
            None, the plot will be shown with plt.show().  Default is None.
        
        **kwargs : (optional)
            keyword arguments to matplotlib.axes.Axes.plot
        """
        
        #Bookkeeping
        if which_cfs is None:
            which_cfs = self.cfs.keys()
            
        if labels is None:
            labels = which_cfs
        elif len(labels) != len(which_cfs):
            raise ValueError("If you specify labels, the array must be"
                        " the same length as which_cfs.  Which_cfs ="
                        " "+str(which_cfs))    

        default_colors = ['r', 'Orange', 'Lime', "Blue", "Purple", 
                        "Magenta", "Cyan", "Green", "Yellow", "DarkCyan", 
                        "DimGray", "HotPink", "MediumPurple"]
        if colors is None:
            colors = default_colors
        if len(colors) < len(which_cfs):
            raise ValueError("I don't have enough colors.  If you gave me "
                    "the colors, you need to add more.  If you used the "
                    "default colors, then you have more than " + 
                    str(len(default_colors)) + " correlation functions.  "
                    "Cathy didn't think you'd need that many, so you'll "
                    "need to define your own colors.")
            
        #Make the figure
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlabel('theta (arcsec)')
        ax.set_ylabel("error size")
        ax.set_xscale('log')
        if log_yscale:
            ax.set_yscale('log')
        for i, k in enumerate(which_cfs):
            thetas, __ = self.cfs[k].get_thetas()
            cf, err = self.cfs[k].get_cf()
            if log_yscale:
                if (err==0).all():
                    print "Skipping", k, "because it has no non-zero errors"
                else:
                    msk = err != 0
                    ax.plot(thetas[msk], err[msk], label=labels[i], 
                        color=colors[i], **kwargs)
            else:
                ax.plot(thetas, err, label=labels[i], 
                            color=colors[i], **kwargs)
        
        #Make the legend
        handles, labels=ax.get_legend_handles_labels()
        legnd=ax.legend(handles, labels, loc=0, labelspacing=.15, 
                            fancybox=True, fontsize=8)
        legnd.get_frame().set_alpha(0.)
        
        #Save or show
        if save_to:
            plt.savefig(save_to, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
