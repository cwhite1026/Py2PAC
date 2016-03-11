#External code
import numpy as np
import numpy.ma as ma
import scipy.integrate as intg
import scipy.spatial as spatial
import numpy.random as rand
import astropy.wcs as wcs
import astropy.io.fits as fits
from copy import deepcopy
import matplotlib.pyplot as plt

#Py2PAC code
import miscellaneous as misc
import image_mask_cybits as cybits
import correlations as corr
from CompletenessFunction_class import CompletenessFunction
from mags_and_radii import get_mags_and_radii


#===============================================================================
#===============================================================================
#===============================================================================

class ImageMask:
    """This is a class that handles the arrangement of the observations on
    the sky. The default constructor is probably not what you will want to
    access directly.  There are more user-friendly class methods that allow
    mask definition from various sources.
    
    Class methods are ImageMask.from_FITS_file, ImageMask.from_array, ImageMask.from_ranges

    Parameters
    ----------
    mask : 2D array
        This is an array describing completeness as a function of position
        on the sky
        
    wcs_instance : astropy.wcs.WCS instance
        Encodes the information about how large the field is and where it
        is on the sky.

    wcs_moveable : bool
        True if the WCS instance can be moved on the sky.  False if it
        can't.  The WCS instances that are created froms scratch by the
        class methods from_ranges and from_array cannot be moved.  This is
        a bug that will be fixed later.  Default is True
    """
    
    #----------------------------------------------------------------------
    #                  Routines for making ImageMasks
    #----------------------------------------------------------------------
        
    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, mask, wcs_instance, wcs_moveable=True, completeness_dict=None, levels=None):
        """
        Takes the mask and WCS instance and stores them, along with
        extracting some useful information from them
        """
        
        #Store what we were given
        self._mask = np.array(mask)
        self._wcs_moveable = wcs_moveable
        if isinstance(wcs_instance, wcs.WCS):
            self._wcs_instance = wcs_instance
        else:
            raise TypeError("The WCS instance you pass to ImageMask must "
                            "be an instance of astropy.wcs.WCS.  This one "
                            "was not.")
        self._nx_pixels, self._ny_pixels = self._mask.shape
        nx, ny = self._mask.shape

        #Get the RA and Dec range that the mask covers.
        self._calc_footprint()
        self._ra_bin_size, self._ra_edges=self._make_bin_edges(self._ra_range,
                                                              self._nx_pixels)
        self._dec_bin_size, self._dec_edges=self._make_bin_edges(self._dec_range,
                                                                self._ny_pixels)

        #Set up some holders for things we might use later
        self._subregion_rotation=None
        self._subregion_rotated_xedges=None
        self._subregion_rotated_yedges=None
        self._completeness_dict=completeness_dict
        self._levels=levels

    #----------------------------------------------------------------------
    #--------------------------------#
    #- Make a mask from a FITS file -#
    #--------------------------------#
    @classmethod
    def from_FITS_file(cls, fits_file, fits_file_type='weight'):
        """
        Class method to generate an image mask from a weight or levels file
        in FITS format.  If the FITS file is large, this routine can take
        some time.

        **Syntax**
        
        immask = ImageMask.from_FITS_weight_file(fits_file, mask_type)

        Parameters
        ----------
        
        fits_file : string
            The file name including the path from / that contains the
            FITS file to mask with

        fits_file_type : string
            Valid options: 'weight' or 'levels'

        Returns
        -------
        new_mask : ImageMask instance
            An image mask that corresponds to the FITS file specified
        """
        
        #Get the mask info from the fits file via a Cython routine (because
        #it's super slow in plain python)
        if fits_file_type=='weight':
            mask_info = cybits.make_mask_from_weights(fits_file)
            nx_pixels, ny_pixels, approx_frac_nonzero, mask = mask_info
        elif fits_file_type=='levels':
            mask = fits.getdata(fits_file)
        else:
            raise ValueError("'fits_file_type' kwarg must be either "
                             "'weight' or 'levels'")
        
        #Make a WCS instance and get useful things from it
        wcs_instance=wcs.WCS(fits_file)

        #filetype
        if fits_file_type == 'levels':
            print 'Getting level values'
            levels=sorted(np.unique(mask))
        else:
            levels=None

        #Make and return the mask
        immask = cls(mask, wcs_instance, levels=levels)
        return immask

    #----------------------------------------------------------------------
    #-----------------------------------------------#
    #- Make a mask from an array and RA/Dec ranges -#
    #-----------------------------------------------#
    @classmethod
    def from_array(cls, mask, ra_range, dec_range):
        """
        Used to create an image mask from an array and RA and Dec ranges.
        The RA and Dec ranges should be the ranges covered by the entire
        mask, not just the part that has galaxies on it.  The main purpose
        of this class method is to create a WCS instance for you when you
        don't have one already.

        The mask orientation is important.  The [0, 0] corner corresponds
        to the minimum of both RA and Dec.  The first index increasing
        corresponds to increasing the declination.  The second index
        increasing corresponds to increasing the RA.

        WARNING:  This routine assumes that the region of the sky the mask
        covers is small enough that things like pixel scale in RA is the
        same over the whole image.  This will break for very large masks.

        Syntax
        ------
        immask = ImageMask.from_array(mask_array, ra_range, dec_range)

        Parameters
        ----------
        mask : 2D array
            An array of completenesses as a function of position on the
            sky.  0 means that nothing in that pixel will be used for CFs.
            1 means that everything will be.  Numbers between are
            partially complete.
            
        ra_range : array-like, length 2
            The RAs of the left and right sides of the mask.  (In degrees.)
            
        dec_range : array-like, length 2
            The Decs of the top and bottom of the mask.  (In degrees.)

        Returns
        -------
        new_mask : ImageMask instance
            An image mask that has the mask provided and the corresponding
            WCS instance.
            
        """

        #Check some basic stuff
        if (len(ra_range) !=2) or (len(dec_range) !=2):
            raise ValueError("ImageMask.from_array says:  The RA and Dec "
                             "ranges must be array-like objects of length "
                             "two.")
        
        #Get some basic info from the mask and make the WCS instance accordingly
        x_pix, y_pix = mask.shape
        center_x_pixel = (x_pix+1)/2.
        center_y_pixel = (y_pix+1)/2.
        center_RA = sum(ra_range)/2.
        center_Dec = sum(dec_range)/2.
        RA_span = np.diff(ra_range)[0]
        Dec_span = np.diff(dec_range)[0]
        delta_Dec = Dec_span / float(x_pix)
        delta_RA = RA_span / float(y_pix)

        #Make the WCS instance and fill in the blanks
        wcs_inst = wcs.WCS(naxis=2)
        wcs_inst.wcs.crpix = [center_y_pixel, center_x_pixel]
        wcs_inst.wcs.cdelt = np.array([delta_RA, delta_Dec])
        wcs_inst.wcs.crval = [center_RA, center_Dec]
        wcs_inst.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        #Make the image mask and return
        immask=cls(mask, wcs_inst, wcs_moveable=False)
        return immask

    #----------------------------------------------------------------------
    #---------------------------#
    #- Make a rectangular mask -#
    #---------------------------#
    @classmethod
    def from_ranges(cls, ra_range, dec_range):
        """
        Used to create an image mask from RA and Dec ranges.  This is
        useful for things like mock catalogs that are rectangular (in
        the spherical coordinate sense) 
        
        WARNING:  This routine assumes that the region of the sky the mask
        covers is small enough that things like pixel scale in RA is the
        same over the whole image.  This will break for very large masks.

        Syntax
        ------
        immask = ImageMask.from_ranges(ra_range, dec_range)

        Parameters
        ----------
        ra_range : array-like, length 2
            The min and max RAs to be covered.  (In degrees.)
            
        dec_range : array-like, length 2
            The min and max Decs to be covered.  (In degrees.)

        Returns
        -------
        new_mask : ImageMask instance
            An image mask that is a rectangle that spans the RA and Dec
            ranges specified
        """
        
        #Make a really simple mask and pass the job off to from_array
        x_pix = y_pix = 11
        mask = np.ones((x_pix, x_pix))
        immask = cls.from_array(mask, ra_range, dec_range)
        return immask

#==========================================================================
#==========================================================================

    #----------------------------------------------------------------------
    #                 Hidden routines
    #----------------------------------------------------------------------
            
    #---------------------------#
    #- Edge definition routine -#
    #---------------------------#
    def _make_bin_edges(self, rng, n_pix):
        """
        Takes a range and a number of bins and returns the bin size and
        the edges of the bins.

        Parameters
        ----------
        rng : array-like, length 2
            The range over which you want your bins
            
        n_pix : scalar
            The number of bins (pixels) that you want to divide the range
            into

        Returns
        -------
        bin_size : scalar
            The size of the bin/pixel in whatever units rng was in
            
        bin_edges : np.ndarray
            An array of n_pix+1 values with where the edges of the bins are
        """
        
        bin_size=np.float(np.diff(rng)[0])/n_pix
        return bin_size, np.linspace(rng[0], rng[1], n_pix+1)

    #----------------------------------------------------------------------
    #---------------------------------------------------#
    #- Calculate the RA and Dec ranges the mask covers -#
    #---------------------------------------------------#
    def _calc_footprint(self):
        """
        This function replaces the built-in WCS instance's calc_footprint.
        The problem is that if the WCS instance isn't informed of the range
        of the mask properly, then it can give a range that is too small,
        causing the randoms to be generated over too small an area.

        This function just checks which rows and colums of the mask have
        any non-zero elements in them and computes the RA and Dec positions
        of the corners of that x-y rectangle.  The min and max of the
        corner RAs and Decs are stored as the RA and Dec ranges.
        """

        print "Calculating the footprint of the mask"
        
        #See which rows and columns have pixels that are nonzero
        column_has_any = np.sum(self._mask, axis=0).astype(bool)
        row_has_any = np.sum(self._mask, axis=1).astype(bool)

        #Figure out the ranges of xs and ys
        xs = np.where(row_has_any)[0]
        ys = np.where(column_has_any)[0]
        x_min=xs.min()-.5
        x_max=xs.max()+.5
        y_min=ys.min()-.5
        y_max=ys.max()+.5

        #Convert the corners of that footprint to RA and Dec
        pass_xs = [x_min, x_min, x_max, x_max]
        pass_ys = [y_min, y_max, y_max, y_min]
        ras, decs = self.xy_to_ra_dec(pass_xs, pass_ys)

        #Make sure that we don't have any slightly negative guys rolling
        #over into the 359s
        def correct_pair(arr, i, j):
            if abs(arr[i] - arr[j]) > 359:
                if arr[i] > arr[j]:
                    arr[i] = arr[i]-360
                else:
                    arr[j] = arr[j]-360
        correct_pair(ras, 0, 3)
        correct_pair(ras, 1, 2)
        correct_pair(decs, 0, 1)
        correct_pair(decs, 2, 3)
        
        #Store the RA and Dec ranges
        self._ra_range = np.array([ras.min(), ras.max()])
        self._dec_range = np.array([decs.min(), decs.max()])        
        
        
#==========================================================================
#==========================================================================

    #----------------------------------------------------------------------
    #                 Generate randoms on the mask
    #----------------------------------------------------------------------

    #-----------------------------#
    #- Generate unmasked randoms -#
    #-----------------------------#
    def unmasked_random_sample(self, number_to_make, ra_range, dec_range):
        """
        Generates a given number of randomly placed points in a specified
        RA and Dec range.  This is really just an intermediary for a call
        to correlations.uniform_sphere(RAlim, DEClim, size=1).  This
        routine does not allow for the RA and Dec ranges to be completely
        outside the mask.  If you want that, go to the function in
        correlations.

        Parameters
        ----------
        number_to_make : scalar
            The number of random objects to place in the RA and Dec range
            
        ra_range : array-like
            Randoms will be placed between ra_range[0] and ra_range[1].
            Units are degrees.
            
        dec_range : array-like
            Randoms will be placed between dec_range[0] and dec_range[1].
            Units are degrees.

        Returns
        -------
        ra : numpy ndarray
            RA coords of the randomly placed objects.  Array shape is
            (number_to_make,)
            
        dec : numpy ndarray
            Dec coords of the randomly placed objects.  Array shape is
            (number_to_make,)
        """

        #Check that we have an integer number of objects
        if int(number_to_make) != number_to_make:
            raise ValueError("You must give me an integer number of "
                             "objects.  You entered "+str(number_to_make))
        
        #Check that we'll have at least some points in the mask
        ra_range_high = ra_range[0] >= self._ra_range[1]
        print ra_range_high
        ra_range_low = ra_range[1] <= self._ra_range[0]
        print ra_range_low
        bad_ra = ra_range_high | ra_range_low
        print bad_ra
        dec_range_high = dec_range[0] >= self._dec_range[1]
        print dec_range_high
        dec_range_low = dec_range[1] <= self._dec_range[0]
        print dec_range_low
        bad_dec = dec_range_high | dec_range_low
        print bad_dec
        if bad_ra or bad_dec:
            raise ValueError("You have given me a range that doesn't "
                             "overlap with the RA and Dec range of the "
                             "mask.  Please correct that or use the "
                             "correlations.uniform_sphere function.")

        #Return the call to uniform_sphere
        return corr.uniform_sphere(ra_range, dec_range,
                                   size=number_to_make)

    #----------------------------------------------------------------------
    #--------------------------------#
    #- Generate randoms on the mask -#
    #--------------------------------#
    def generate_random_sample(self, number_to_make, complicated_completeness = False):
        """
        Generate a given number of random points within the mask.

        Parameters
        ----------
        number_to_make : scalar
            Number of randomly placed objects within the mask area
            returned.

        Returns
        -------
        ra : numpy ndarray
            The RAs of the randomly placed objects within the mask.  Unit
            is degrees.  The array shape is (number_to_make,)
            
        dec : numpy ndarray
            The Decs of the randomly placed objects within the mask.  Unit
            is degrees.  The array shape is (number_to_make,)
        """
        
        #Check that we have an integer number of objects
        if int(number_to_make) != number_to_make:
            raise ValueError("You must give me an integer number of "
                             "objects.  You entered "+str(number_to_make))

        #Make the first pass of randoms
        ra_R, dec_R = corr.uniform_sphere(self._ra_range, self._dec_range,
                                          size=number_to_make)
        
        #----------------------------------
        #- Mask and add more if undershot
        #----------------------------------
        #Get completenesses and see which to use
        if complicated_completeness:
            mags, radii = get_mags_and_radii(number_to_make)
        else:
            mags = None
            radii = None
        random_completeness = self.return_completenesses(ra_R, dec_R, mags, radii, complicated_completeness)
        compare_to = rand.random(size=len(ra_R))
        use = compare_to < random_completeness
        #Mask down to the ones that survived
        ra_R = ra_R[use]
        dec_R = dec_R[use]
        compare_to = compare_to[use]
        random_completeness = random_completeness[use]

        #How many do we have?
        number_we_have = len(ra_R)
        print ("ImageMask.generate_random_sample says: "
               " We made "+str(number_we_have))
        print "      We need", number_to_make, "total"
        print "      We will make", number_to_make - number_we_have, "more"

        #Check to see by how many we've overshot
        number_left_to_make = number_to_make - number_we_have
        
        #If we've actually made too few, make more    
        # else:
        if number_left_to_make > 0:
            print ("ImageMask.generate_random_sample says: I have "
                   "made too few objects within the target area. Making "
                   "more.")
            #Figure out what fraction of the guys that we made were used
            #so if my mask is teeny and in a big field, it won't take
            #forever to get to where we want to be
            fraction_used_last_time = float(number_we_have)/number_to_make
            if fraction_used_last_time < 1.e-2:
                fraction_used_last_time = 1e-2
                
            #Ask for exactly how many more we need.
            # new_multiplier = 1. / fraction_used_last_time
            # ask_for = np.ceil(number_left_to_make * new_multiplier)
            ask_for = number_left_to_make
            newguys = self.generate_random_sample(ask_for, complicated_completeness)
            #Unpack
            more_ras, more_decs, more_comps, more_mag, more_rad = newguys
            
            #Add these galaxies to the existing arrays
            ra_R = np.concatenate((ra_R, more_ras))
            dec_R = np.concatenate((dec_R, more_decs))
            random_completeness = np.concatenate((random_completeness,
                                                  more_comps))
            number_we_have = len(ra_R)
            number_left_to_make = number_to_make - number_we_have
            if number_left_to_make > 0:
                raise RuntimeError("Cathy screwed up something major this "
                                   "time.  We didn't make the right number"
                                   " after falling short and calling "
                                   "generate_randoms again.")

        #If we overshot, cut some off
        elif number_left_to_make < 0:
            print ("ImageMask.generate_random_sample says: "
                  "Cutting down to exactly as many objects as we need.")
            ra_R =ra_R[0:number_to_make]
            dec_R =dec_R[0:number_to_make]
            random_completeness = random_completeness[0:number_to_make]
        else:
            print ("ImageMask.generate_random_sample says: "
                  "I made exactly the right number!  It's like winning "
                  "the lottery but not actually fun...")
                
        #Return things!
        return ra_R, dec_R, random_completeness, mags, radii
        

#==========================================================================
#==========================================================================

    #----------------------------------------------------------------------
    #                 Transforms
    #----------------------------------------------------------------------

    #-----------------------------------------------#
    #- Returns the x-y coords of given RA and Decs -#
    #-----------------------------------------------#
    def ra_dec_to_xy(self, ra, dec):
        """
        Given a list of RA and Dec, returns the XY position on the image.

        Parameters
        ----------
        ra : array-like
            A list of RA coordinates to transform.  (In degrees.)
            
        dec : array-like
            A list of Dec coordinates to transform.  (In degrees.)

        Returns
        -------
        x : 1D np.ndarray
            X positions of the input coordinates
            
        y : 1D np.ndarray
            Y positions of the input coordinates
        """
        pairs=np.transpose([ra, dec])
        positions=self._wcs_instance.wcs_world2pix(pairs, 0)
        x=np.array(positions[:,1])
        y=np.array(positions[:,0])
        return x, y

    #----------------------------------------------------------------------
    #----------------------------------------------#
    #- Returns the RA and dec of given x-y coords -#
    #----------------------------------------------#
    def xy_to_ra_dec(self, x, y):
        """
        Given a list of X and Y coordinates, returns the RA and Dec.

        Parameters
        ----------
        x : array-like
            X positions of the input coordinates
            
        y : array-like
            Y positions of the input coordinates

        Returns
        -------
        ra : np.ndarray
            RAs of the input coordinates in degrees
            
        dec : array-like
            Decs of the input coordinates in degrees
        """
        pairs=np.transpose([y, x])
        positions=self._wcs_instance.wcs_pix2world(pairs,0)
        ra=np.array(positions[:,0])
        dec=np.array(positions[:,1])
        return ra, dec

#==========================================================================
#==========================================================================

    #----------------------------------------------------------------------
    #                 Mask manipulation
    #----------------------------------------------------------------------

    #------------------------------#
    #- Set subregions within mask -#
    #------------------------------# 
    def subdivide_mask(self, n_shortside=3, n_longside=4, preview=False,
                       rotation_angle=None, padding=None, only_show=None,
                       save_plot=None):
        """
        Subdivide mask takes the image mask, draws a rectangle around the
        valid region of the mask, rotated to be as small as possible, and
        subdivides that rectangle.  The rotation angle of the box can be
        set manually and space can be trimmed or added from the sides of
        the box.

        Parameters
        ----------
        n_shortside : integer (optional)
            The number of cells along the short side of the rectangle.
            Default is 3.
            
        n_longside : integer (optional)
            The number of cells along the long side of the rectangle.
            Default is 4.
            
        preview : bool (optional)
            Should the mask show you what this set of parameters looks like
            but not store the results?  If preview == True, either a plot
            will be shown on your screen (no value given for save_plot) or
            saved to a file (save_plot specified).  The image mask object
            will not keep the division information.  If preview == False,
            which is the default, the values will be stored and a plot will
            only be made if save_plot is specified.
            
        rotation_angle : scalar (optional)
            Fixes the rotation angle that the mask's bounding box will be
            defined at with respect to the XY coordinate system of the
            mask.  By default, the routine will choose the integer angle in
            degrees which minimizes the area of the bounding box.
            
        padding : 4 element array-like (optional)
            How many pixels extra to allow in the rotated coordinates of
            the bounding box.  The order is [bottom, top, left, right].
            Positive padding corresponds to moving the edges outwards and
            leaving extra room.  Negative padding will move the bounding
            box inward and cut off part of the mask.
            
        only_show : integer (optional)
            If you only want to see the random points that fall into one of
            the cells in the subdivided mask, you set only_show to that
            cell number.  This will only matter if you have preview=True
            or have specified save_plot.
            
        save_plot : string (optional)
            Name with path from '/' of the file where you would like the
            plot of the subdivided mask saved.  If only_show is set to an
            integer that corresponds to a cell number, only the points in
            that cell will be shown.
        """
        
        #Start by putting down a bunch of randoms.
        ra, dec, __ = self.generate_random_sample(5.e4)
        x1, y1=self.ra_dec_to_xy(ra, dec)

        #Set the padding on each side
        if padding:
            try:
                pad_bottom, pad_top, pad_left, pad_right=padding
            except:
                #If we can't unpack the padding, either raise an
                #informative error (if we're doing the subdivision
                #permanently) or just a warning if we're only previewing
                says_str = "subdivide_mask says: "
                message_str = ("You have given me something I can't use "
                               "for padding.  Padding must be a 4-element "
                               "1D array in the format [bottom, top, left,"
                               " right].")
                if preview == True:
                    print (says_str + "WARNING!  " + message_str +
                           "  No padding used this time")
                    pad_bottom, pad_top, pad_left, pad_right=[0,0,0,0]
                else:
                    raise ValueError(says_str + "ERROR!  " + message_str)
        else:
            pad_bottom, pad_top, pad_left, pad_right=[0,0,0,0]
        
        #If we don't have an already chosen angle, choose a bunch of angles
        #and transform the coordinates to rotated systems and get the areas
        #of the rectangle enclosing all the data points at this angle. Take
        #The angle with the minimum area for the enclosing rectangle.
        if rotation_angle is None:
            thetas=np.radians(np.arange(90, dtype=np.float))
            areas=[]
            corners=[]
            for th in thetas:
                x2, y2= misc.rotate_coords(x1, y1, th)
                #Don't take padding into account to determine the angle.
                x2min=x2.min()
                x2max=x2.max()
                y2min=y2.min()
                y2max=y2.max()
                areas.append((x2.max()-x2.min()) * (y2.max()-y2.min()))

            areas=np.asarray(areas)
            which_theta=np.where(areas==areas.min())[0][0]
            use_theta=thetas[which_theta]
        #Else, use the given angle
        else:
            use_theta=np.radians(rotation_angle)

        #Define the edges of the regions
        x2, y2= misc.rotate_coords(x1, y1, use_theta)
        x2min=x2.min() - pad_left
        x2max=x2.max() + pad_right
        y2min=y2.min() - pad_bottom
        y2max=y2.max() + pad_top

        #Figure out the x2 and y2 bin divisions
        if (x2max-x2min) < (y2max-y2min):
            nx=n_shortside
            ny=n_longside
        else:
            ny=n_shortside
            nx=n_longside

        x2edges = np.linspace(x2min, x2max, nx+1)
        y2edges = np.linspace(y2min, y2max, ny+1)

        #-----------------#
        #- Make the plot -#
        #-----------------#
        if preview or save_plot:
            #Figure out what subregions we have
            subregions=self.return_subregions(ra, dec, theta=use_theta,
                                              rot_xedges=x2edges,
                                              rot_yedges=y2edges)
            outside = subregions == -1
            inside= np.invert(outside)
            if only_show is not None:
                this_box = subregions==only_show
                inside = inside & this_box

            #Make a figure and plot the random points
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.scatter(x1[outside], y1[outside], c='LightGray')
            ax.scatter(x1[inside], y1[inside], c='Blue')  
                      
            #Plot the vertical lines
            for ix in range(nx+1):
                x2=[x2edges[ix], x2edges[ix]]
                y2=[y2min, y2max]
                x1, y1=misc.rotate_coords(x2, y2, -use_theta)
                ax.plot(x1, y1, color='Red', lw=2)
            #Plot the horizontal lines
            for iy in range(ny+1):
                x2=[x2min, x2max]
                y2=[y2edges[iy], y2edges[iy]]
                x1, y1=misc.rotate_coords(x2, y2, -use_theta)
                ax.plot(x1, y1, color='Red', lw=2)

            #Figure out the dimensions of the boxes in angular space
            x2=[x2edges[0], x2edges[0], x2edges[1]]
            y2=[y2edges[0], y2edges[1], y2edges[0]]
            ra_box, dec_box=self.xy_to_ra_dec(x2, y2)
            y_side=misc.ang_sep(ra_box[0], dec_box[0], ra_box[1], dec_box[1],
                               radians_in=False, radians_out=False) * 3600
            x_side=misc.ang_sep(ra_box[0], dec_box[0], ra_box[2], dec_box[2],
                               radians_in=False, radians_out=False) * 3600

            #Print out the parameters
            ax.text(.05, .95, "theta= "+str(np.degrees(use_theta))[0:5],
                    transform=ax.transAxes, fontsize=8)
            ax.text(.05, .925, "padding="+str(padding),
                    transform=ax.transAxes, fontsize=8)
            ax.text(.05, .9, "n_longside="+str(n_longside),
                    transform=ax.transAxes, fontsize=8)
            ax.text(.05, .875, "n_shortside="+str(n_shortside),
                    transform=ax.transAxes, fontsize=8)
            ax.text(.5, .05, ("box size: "+str(x_side)[0:5]+" by "+
                              str(y_side)[0:5]+" arcsec"),
                    transform=ax.transAxes, fontsize=8)

            #Label the subregions
            y_label_coord=.85
            avg_ngals=float(len(ra[np.invert(outside)]))/(nx*ny)
            ax.text(0.8, 0.95, "N_bin/N_avg", transform=ax.transAxes,
                    fontsize=12)
            ax.text(0.8, 0.9, "outside-> "+str(float(len(ra[outside]))/avg_ngals)[0:4],
                    transform=ax.transAxes, fontsize=9)
            for ix in range(nx):
                for iy in range(ny):
                    #What bin number is this?
                    bin_number= nx*iy + ix
                    #Where's the center of the box?
                    text_x2=(x2edges[ix] + x2edges[ix+1])/2.
                    text_y2=(y2edges[iy] + y2edges[iy+1])/2.
                    text_x1, text_y1=misc.rotate_coords(text_x2, text_y2,
                                                        -use_theta)
                    #Print the bin number at the center of the box
                    ax.text(text_x1, text_y1, str(bin_number), fontsize=20,
                            color='Lime', horizontalalignment='center',
                            verticalalignment='center')

                    #Print the number of galaxies in the upper right corner
                    thisbin= subregions==bin_number
                    n_thisbin= float(len(ra[thisbin]))
                    print ("bin "+str(bin_number)+" has "+str(n_thisbin)+
                           " randoms in it")
                    display_string=('bin '+str(bin_number)+'-> '+
                                    str(n_thisbin/avg_ngals)[0:4])
                    ax.text(0.85, y_label_coord, display_string,
                            transform=ax.transAxes, fontsize=9)
                    y_label_coord-=0.05

            if save_plot:
                plt.savefig(save_plot, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        #If we want to store the information, do so
        if not preview:
            #Store the subregion information
            self.set_subregions(use_theta, x2edges, y2edges)

    #----------------------------------------------------------------------            
    #---------------------------------#
    #- Sets up subregion information -#
    #---------------------------------#
    def set_subregions(self, theta, xedges, yedges):
        """
        Set subregion information.  Theta is the rotation of the enclosing
        rectangle and the x and y edges are the locations in the rotated
        coordinate system of the rectangle.  This is mainly useful if you
        calculated these quantities with subdivide_mask previously and
        would prefer to just set these by hand instead of replicating the
        subdivide_mask call.

        Parameters
        ----------
        theta : scalar
            The angle in *radians* of the bounding box rotation
            
        xedges : array-like
            The x coordinates in the rotated coordinate system of the cell
            boundaries
            
        yedges : array-like
            The y coordinates in the rotated coordinate system of the cell
            boundaries
        """
        self._subregion_rotation=theta
        self._subregion_rotated_xedges=np.array(xedges)
        self._subregion_rotated_yedges=np.array(yedges)
        return
    
    #----------------------------------------------------------------------
    #-----------------------------------------------#
    #- Translate and/or rotate the mask on the sky -#
    #-----------------------------------------------#
    def move_mask_on_sky(self, delta_ra=0, delta_dec=0, theta_degrees=0,
                         preview=False):
        """
        Move the mask around on the sky by changing the parameters in the
        WCS instance.  If preview=True, the calling instance won't be
        changed and a copy with the altered WCS instance will be returned.
        Otherwise, the ImageMask calling this function will be changed and
        nothing is returned.

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

        #First, check to make sure that the instance is such that it can be
        #moved on the sky without causing problems
        if not self._wcs_moveable:
            raise ValueError("This mask has a WCS instance that is flagged"
                             " as not moveable.  If it's something that "
                             "came from Py2PAC and not from the user, it's"
                             " because the astropy WCS package is uncooperative"
                             " and I couldn't figure out how to get it to "
                             "move the masks that I created by hand "
                             "(in the from_array class method) "
                             "properly.  The scales ended up messed up.  "
                             "If you know how to fix this, I would be more"
                             " than happy to hear your suggestions.")

        #If we're previewing, make a copy to work with.  If not, use the
        #WCS instance that we have already
        if preview:
            work_with = self._wcs_instance.deepcopy()
        else:
            work_with = self._wcs_instance
    
        #Update the center
        work_with.wcs.crval[0] += delta_ra
        work_with.wcs.crval[1] += delta_dec

        #Apply the rotation
        sine = np.sin(np.radians(theta_degrees))
        cosine = np.cos(np.radians(theta_degrees))
        rotation_matrix=np.array([[cosine, -sine] , [sine, cosine]])
        work_with.wcs.cd = np.dot(rotation_matrix,
                                           work_with.wcs.cd)

        #If this is a preview, send the altered instance back.  If not,
        #update the RA and Dec ranges.  
        if not preview:
            self._calc_footprint()
        else:
            return ImageMask(self._mask, work_with)

#==========================================================================
#==========================================================================

    #----------------------------------------------------------------------
    #                 Get info from mask
    #----------------------------------------------------------------------
        
    #----------------------------------------------------------------#
    #- Returns the solid angle subtended by the "true" region in sr -#
    #----------------------------------------------------------------#
    def masked_area_solid_angle(self):
        """
        Returns the solid angle subtended by the masked region- just the
        part where data is allowed.  This is a pretty dumb way of doing
        it, so it should only be taken as an approximation.

        **Method**

        The WCS instance is queried to get the RA and Dec coordinates of
        the image.  Those are converted to X and Y coordinates- note that
        this involves some distortion.  If the number of coordinates at the
        bottom and top are different, a warning is printed but the routine
        proceeds. The area is treated as a trapezoid to get the total
        number of effective pixels.  This is compared to the number of
        pixels on the mask where the mask value is greater than 0,
        returning the fraction of the image that is to be considered.

        Then, a solid angle integral is performed over the ranges in RA and
        Dec and multiplied by the fraction of the image to be considered.
        This is returned as the solid angle covered.

        Returns
        -------
        solid_angle : scalar
            The solid angle in steradians covered by the nonzero elements
            of the mask array.
        """
        
        #Locations of the corners in RA-Dec, and x-y
        ra_list=[self._ra_range[0], self._ra_range[1], self._ra_range[0],
                 self._ra_range[1]]
        dec_list=[self._dec_range[0], self._dec_range[0], self._dec_range[1],
                  self._dec_range[1]]
        x, y=self.ra_dec_to_xy(ra_list, dec_list)

        #Now pretend it's a trapezoid to get the total area
        npix_top = y[2:].max() - y[2:].min()
        npix_bottom = y[0:2].max() - y[0:2].min()
        if npix_top != npix_bottom:
            print ("masked_area_solid_angle says: WARNING!  There is "
                   "enough distortion from the projection to a flat image "
                   "over the RA and Dec ranges that this mask spans that "
                   "the number of pixels across the RA span at the top and"
                   " bottom are different.  There may be distortion "
                   "effects strong enough to mess with things.  There are "
                   +str(npix_top)+" pixels at the max Dec and "+
                   str(npix_bottom)+" pixels at the min Dec.")
        npix_h = x.max() - x.min()
        total_npix = (npix_top+npix_bottom) * npix_h / 2.

        #How many pixels are in the valid part of the image?
        true_false=np.ceil(self._mask)  #accounts for non-binary masking
        npix_true=np.sum(true_false)

        #What solid angle is covered by the whole RA and Dec range?
        #Convert the RA and Decs to theta and phi (in radians)
        #Dec is pi/2 at theta=0 and -pi/2 at theta=pi
        theta_range= - np.radians(self._dec_range) + np.pi/2. 
        theta_range.sort()
        phi_range= np.radians(self._ra_range)
        phi_range.sort()
    
        #Do the integral
        phi_int, phi_int_err= intg.quad(lambda phi: 1, phi_range[0],
                                        phi_range[1])
        theta_int, theta_int_err= intg.quad(lambda theta: np.sin(theta),
                                            theta_range[0], theta_range[1])
        solid_angle = phi_int * theta_int
        print solid_angle
        print npix_true
        print total_npix

        #Return the fraction of the solid angle that's covered
        #by the true part
        return solid_angle * npix_true/total_npix
        
    #----------------------------------------------------------------------        
    #------------------------------------------#
    #- Queries completeness for given catalog -#
    #------------------------------------------#
    def make_completeness_dict(self, *args):
        """
        Takes a list of CompletenessFunction instances and returns a 
        dictionary of them.

        Parameters
        ----------
        *args : CompletenessFunction instances
            As many completeness function instances as you have

        Returns
        -------
        completeness_dict : dictionary
            A dictionary of completeness functions
        """

        #Check that the lists are the same length and convert to np arrays
        completeness_dict = {}
        for arg in args:
            if not isinstance(arg, CompletenessFunction):
                raise TypeError("Arguments passed to make_completeness_dict "
                                "must be CompletenessFunction instances")
            elif not hasattr(arg, '_level'):
                raise ValueError("No level specified in CompletenessFunction "
                                 "instance.")
            else:
                completeness_dict[str(arg._level)] = arg
        self._completeness_dict = completeness_dict
        return

    #----------------------------------------------------------------------        
    #------------------------------------------#
    #- Queries completeness for given catalog -#
    #------------------------------------------#
    def return_completenesses(self, ra_list, dec_list, mag_list=None, rad_list=None, complicated_completeness=False):
        """
        Takes a list of RAs and Decs and returns the completenesses for
        each point.  This version only supports completeness with no
        dependence on object properties.

        Parameters
        ----------
        ra_list : 1D array-like
            A list of the RAs in degrees for the objects to query for
            completeness.
            
        dec_list : 1D array-like
            A list of the Decs in degrees for the objects to query for
            completeness.

        Returns
        -------
        completeness : 1D numpy array
            A list of completenesses for the objects in the input lists.
            Completenesses are between 0 and 1 (inclusive).  To see if the
            object should be used, draw a random uniform number on [0,1]
            if that number is less than or equal to the completeness,
            include the object.
        """

        #Check that the lists are the same length and convert to np arrays
        ra_list = np.array(ra_list)
        dec_list = np.array(dec_list)
        if len(ra_list) != len(dec_list):
            raise ValueError("The RA and Dec lists must be the same length")

        #Mask down to just the guys that are inside the RA and Dec ranges
        in_ranges = ma.masked_inside(ra_list, self._ra_range[0],
                                     self._ra_range[1]).mask
        in_ranges = in_ranges & ma.masked_inside(dec_list,
                                                 self._dec_range[0],
                                                 self._dec_range[1]).mask
        
        #Get the pixel numbers for all the objects
        pairs=np.transpose([ra_list[in_ranges], dec_list[in_ranges]])
        float_inds=self._wcs_instance.wcs_world2pix(pairs, 0)
        indices=np.asarray(float_inds, dtype=np.int)
        xinds=indices[:,1]
        yinds=indices[:,0]

        #Figure out which things are even on the image (this is needed if
        #the mask is rotated on the sky- the corners of the RA and Dec
        #square hang off the image mask)
        nx, ny= self._mask.shape
        inside_x=ma.masked_inside(xinds, 0, nx-1).mask
        inside_y=ma.masked_inside(yinds, 0, ny-1).mask
        on_image= inside_x & inside_y
        
        #Now make the completeness array
        complete=np.zeros(len(ra_list))
        temp_complete = np.zeros(len(ra_list[in_ranges]))
        if np.asarray(on_image).any():
            print ("return_completenesses says: I have " +
                   str(len(xinds[on_image])) +
                   " points that are actually on the image")
            on_mask_bits = self._mask[xinds[on_image],yinds[on_image]]
            # use completeness functions if they exist
            # this doesn't work with all options yet
            if complicated_completeness:
                # iterate over levels
                for level in self._levels:
                    level_string = str(int(level))
                    if level_string in self._completeness_dict.keys():
                        # get completeness object corresponding to level
                        cf = self._completeness_dict[level_string]
                        # find all galaxies located within level
                        at_level = np.where(on_mask_bits == int(level))
                        num_to_generate = len(temp_complete[at_level])
                        if num_to_generate > 0:
                            if mag_list is not None:
                                mags_in_ranges = mag_list[in_ranges]
                                mags = mags_in_ranges[at_level]
                                if rad_list is not None:
                                    rads_in_ranges = rad_list[in_ranges]
                                    rads = rads_in_ranges[at_level]
                                temp_complete[at_level] = cf.query(mags, r_list=rads)
            else:
                temp_complete[on_image] = on_mask_bits
            
            complete[in_ranges] = temp_complete

        return complete
        

    #----------------------------------------------------------------------    
    #-----------------------------------------------#
    #- Returns subregions for a list of RA and Dec -#
    #-----------------------------------------------#
    def return_subregions(self, ra, dec, theta=None, rot_xedges=None,
                          rot_yedges=None):
        """
        Returns the subregion number for each pair of RA and Dec given the
        parameters either stored or given as function parameters.

        Parameters
        ----------
        ra : array-like
            A list of RAs to get subregion numbers for (in degrees)
            
        dec : array-like
            A list of Decs to get subregion numbers for (in degrees)
            
        theta : scalar (optional)
            Rotation angle that the mask's bounding box will be defined at
            with respect to the XY coordinate system of the mask.  If not
            given, the routine will look for a stored theta.  Units are
            degrees
            
        rot_xedges : array-like (optional)
            The x coordinates of the cell boundaries in the rotated
            coordinate system.  If not given, the routine will look for a
            stored theta.
            
        rot_yedges : array-like (optional)
            The x coordinates of the cell boundaries in the rotated
            coordinate system.  If not given, the routine will look for a
            stored theta.

        Returns
        -------
        subregions : numpy ndarray
            The subregion number for each of the points input.  The array
            shape is (len(ra),).  The subregion -1 is outside the bounding
            box (this only happens if you've set negative padding somewhere
            or have asked for things outside the mask).
        """
        #Check to make sure we have what we need, pull to local if we have
        #stored values but no given values
        if (theta is None):
            if (self._subregion_rotation is None):
                raise ValueError("ImageMask.return_subregions says: "
                                 "ERROR!  I don't have the rotation "
                                 "angle.  Please provide one.")
            else:
                theta=self._subregion_rotation
                
        if (rot_xedges is None):
            if (self._subregion_rotated_xedges is None):
                raise ValueError("ImageMask.return_subregions says: "
                                 "ERROR!  I don't have rotated x edges."
                                 "  Please provide them.")
            else:
                rot_xedges=self._subregion_rotated_xedges
                
        if (rot_yedges is None):
            if (self._subregion_rotated_yedges is None):
                raise ValueError("ImageMask.return_subregions says: "
                                 "ERROR!  I don't have rotated y edges.  "
                                 "Please provide them.")
            else:
                rot_yedges=self._subregion_rotated_yedges

        #Now that we know we have everything, put the ra and decs into x
        #and y coords
        x1, y1=self.ra_dec_to_xy(ra, dec)

        #Transform to the rotated coordinate system
        x2, y2=misc.rotate_coords(x1, y1, theta)

        #Now make masks for each row and column
        nx=len(rot_xedges)-1
        ny=len(rot_yedges)-1
        ymasks={}
        xmasks={}
        for i in range(nx):
            xmasks[i]=ma.masked_inside(x2, rot_xedges[i],
                                       rot_xedges[i+1]).mask
        for i in range(ny):
            ymasks[i]=ma.masked_inside(y2, rot_yedges[i],
                                       rot_yedges[i+1]).mask

        #Now use the masks to put numbers to each galaxy
        #No subregion defaults to -1
        subregion=-np.ones(len(ra))
        for ix in range(nx):
            for iy in range(ny):
                bin_number= nx*iy + ix
                thismask = xmasks[ix] & ymasks[iy]
                subregion[thismask]=bin_number

        return subregion

