#External code
import numpy as np
import numpy.ma as ma
import warnings
from copy import deepcopy

#==========================================================================
#==========================================================================
#==========================================================================

class CompletenessFunction:
    """
    This is a class that holds a completeness function for use in generating the
    randoms on the sky.

    Parameters
    ----------
    completeness_array : 1- or 2D array
        This is an array describing completeness as a function of magnitude 
        only if 1D and of magnitude and radius if 2D.
        
    mag_range : 1D array-like
        Range of magnitudes 

    r_range : 1D array-like, optional
        Range of radii

    Notes
    -----
    Can be constructed so as to either be only dependent on magnitude, or 
    on both magnitude and radius.
    """

    #----------------------------------------------------------------------
    #              Routines for making completeness functions
    #----------------------------------------------------------------------
        
    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, completeness_array, mag_range, r_range=None, level=None, galtype=None):
        completeness_array = np.asarray(completeness_array)
        # make sure completeness values are within acceptable range
        if (np.min(completeness_array) < 0) or \
        (np.max(completeness_array) > 1):
            warnings.warn("Your completeness array contains values less "
                          "than zero or greater than one. Setting values "
                          "to zero or one.")
        completeness_array[completeness_array < 0] = 0
        completeness_array[completeness_array > 1] = 1
        # store relevant values
        self._completeness_array = completeness_array
        self._mag_range = np.asarray(mag_range)
        self._min_mag = np.min(mag_range)
        self._max_mag = np.max(mag_range)
        self._mag_bin_size = mag_range[1] - mag_range[0]
        if r_range is not None:
            self._r_range = np.asarray(r_range)
            self._min_r = r_range.min()
            self._max_r = r_range.max()
            self._r_bin_size = r_range[1] - r_range[0]
        if level is not None:
            self._level = int(level)
        if galtype is not None:
            self._galtype = galtype

    #----------------------------------------------------------------------
    #-----------------------------------------------#
    #- Make a magnitude-only completeness function -#
    #-----------------------------------------------#
    @classmethod
    def from_1d_array(cls, completeness_array, mag_range, **kwargs):
        """Class method to generate a completeness function dependent only
        on magnitude.

        **Syntax**

        completeness_function = CompletenessFunction.from_1D_array(completeness_array, mag_range)

        Parameters
        ----------

        completeness_array : 1D array-like
            A 1D array describing completeness over a set of magnitudes

        mag_range : 1D array-like
            A 1D array containing the magnitude range over which
            completeness is known. Can either be of length 2, containing
            the minimum and maximum magnitude, or one greater than the
            length of the completeness array, containing completeness
            histogram bin edges.

        Returns
        -------
        completeness_function_1D : CompletenessFunction instance
            An object that describes completeness over the specified
            magnitude range.

        Notes
        -----
        Assumes equal bin widths.
        """

        mag_length = len(completeness_array) + 1
        # check for appropriately sized magnitude input
        if len(mag_range) not in [2, mag_length]:
            raise ValueError("The length of your magnitude array must be "
                             "either 2 (min, max) or one greater than the "
                             "length of the completeness array.")
        # create magnitude bin edges if min and max given
        if len(mag_range) == 2:
            mag_range = np.linspace(mag_range.min(), mag_range.max(),
                mag_length)
        # construct class instance
        completeness_function_1D = cls(completeness_array, mag_range, **kwargs)
        return completeness_function_1D

    #----------------------------------------------------------------------
    #-----------------------------------------------------#
    #- Make a magnitude-and-radius completeness function -#
    #-----------------------------------------------------#
    @classmethod
    def from_2d_array(cls, completeness_array, mag_range, r_range, **kwargs):
        """Class method to generate a completeness function dependent on both
        magnitude and radius.

        **Syntax**

        completeness_function = CompletenessFunction.from_2D_array(completeness_array, mag_range, r_range)

        Parameters
        ----------

        completeness_array : 2D array-like
            A 2D array describing completeness over a set of magnitudes

        mag_range : 1D array-like
            A 1D array containing the magnitude range over which
            completeness is known. Can either be of length 2, containing
            the minimum and maximum magnitude, or one greater than the
            length of the completeness array, containing completeness
            histogram bin edges. If it is of length 2, the method will
            assume equal bin sizes within the specified range.

        r_range : 1D array-like
            A 1D array containing the half-light radius range over which
            completeness is known. Can either be of length 2, containing
            the minimum and maximum radius, or one greater than the
            length of the completeness array, containing completeness
            histogram bin edges. If it is of length 2, the method will
            assume equal bin sizes within the specified range.

        Returns
        -------
        completeness_function_2D : CompletenessFunction instance
            An object that describes completeness over the specified
            magnitude and radius ranges.
        
        Notes
        -----
        Assumes equal bin widths.
        """
        
        mag_length = completeness_array.shape[1] + 1
        r_length = completeness_array.shape[0] + 1
        # check for appropriately sized mag and radius input
        if (len(mag_range) not in [2, mag_length]) or \
        (len(r_range) not in [2, r_length]):
            raise ValueError("The length of your magnitude and radius "
                             "arrays must be either 2 (min, max) or one "
                             "greater than the corresponding dimension "
                             "of the completeness array.")
        # create magnitude and radius bin edges if min and max given
        if len(mag_range) == 2:
            mag_range = np.linspace(np.min(mag_range), np.max(mag_range),
                mag_length)
        if len(r_range) == 2:
            r_range = np.linspace(np.min(r_range), np.max(r_range),
                r_length)
        # construct class instance
        completeness_function_2D = cls(completeness_array, mag_range,
            r_range=r_range)
        return completeness_function_2D

#----------------------------------------------------------------------
    #-------------------------------------------------#
    #- Make a completeness function from a .npz file -#
    #-------------------------------------------------#
    @classmethod
    def from_npz_file(cls, npz_file, **kwargs):
        """Class method to generate a completeness function dependent on
        both magnitude and radius from .npz completeness files.

        **Syntax**

        completeness_function = CompletenessFunction.from_npz_file(npz_file)

        Parameters
        ----------

        npz_file : string
            Path to the .npz file containing completeness information.
            Assumes file with ['X', 'Y', 'H'] arrays, where 'X' is
            magnitude, 'Y' is radius, and 'H' is completeness.

        Returns
        -------
        completeness_function_npz : CompletenessFunction instance
            An object that describes completeness over the specified
            magnitude and radius ranges.
        
        Notes
        -----
        Assumes equal bin widths. Mostly for creator convenience.
        """
        
        completeness_data = np.load(npz_file)
        
        #Get the magnitude bin edges
        if len(completeness_data['X'].shape)==1:
            mag_range = completeness_data['X']
        elif len(completeness_data['X'].shape)==2:
            mag_range = completeness_data['X'][0,:]
        else:
            raise ValueError("I don't recognize the shape of the 'X' "
                             "matrix.  It can either be a 1D array or a "
                             "2D matrix where the rows are copies of the "
                             "magnitude bin edges.")
                             
        #Get the radius bin edges
        if len(completeness_data['Y'].shape)==1:
            r_range = completeness_data['Y']
        elif len(completeness_data['Y'].shape)==2:
            r_range = completeness_data['Y'][:,0]
        else:
            raise ValueError("I don't recognize the shape of the 'Y' "
                             "matrix.  It can either be a 1D array or a "
                             "2D matrix where the columns are copies of "
                             "the magnitude bin edges.")
                             
        #Make the completeness function
        completeness_array = completeness_data['H']
        completeness_function_npz = cls(completeness_array, mag_range,
            r_range=r_range, **kwargs)
        return completeness_function_npz

    #----------------------------------------------------------------------
    #--------------------------------------------#
    #- Find completenesses for mag/radius lists -#
    #--------------------------------------------#
    def query(self, mag_list, r_list=None):
        """Queries an instance of CompletenessFunction to get completenesses.

        **Syntax**

        completeness = completeness_function.query(mag, radius=radius)

        Parameters
        ----------

        mag_list : 1D array-like
            List of magnitudes to query completeness object for

        r_list : 1D array-like, optional
            List of radii to query completeness object for

        Returns
        -------
        completeness : 1D array of equal length to mag/radius arrays
            Completeness values corresponding to each mag/radius pair in
            input.

        Notes
        -----
        Assumes equal bin widths and increasing bin edges.
        """

        # Mask down to just guys that fall in our completeness function.
        #The ones that aren't will be returned as 0 completeness
        mag_list = np.asarray(mag_list)
        msk = ma.masked_inside(mag_list, self._min_mag, self._max_mag).mask
        if hasattr(self, '_r_range'):
            # check if mag and radius input are same size
            if len(r_list) != len(mag_list):
                raise ValueError("Your magnitude and radius arrays are "
                                 "not the same size:", len(mag_list),
                                 len(r_list))
            rmsk = ma.masked_inside(r_list, self._min_r, self._max_r).mask
            msk= msk & rmsk
        all_completeness = np.zeros(len(mag_list))
        inside_range_maglist = mag_list[msk]
        
        #If we don't have any guys inside both the magnitude and radius
        #bounds, return all zeros
        try:
            n_mags = len(inside_range_maglist)
        except TypeError:
            print "returning all zeros"
            return all_completeness

        # find magnitude bin that each input mag falls into
        mag_condition = np.hstack([np.where((mag >= self._mag_range) &
            (mag < self._mag_range + self._mag_bin_size))[0][0] for mag in inside_range_maglist])
            
        # if any values equal the last magnitude value, set their index to
        # one smaller so they fit in the completeness array
        # this is a hack
        mag_condition[mag_condition == len(self._mag_range)] -= 1
        mag_condition = mag_condition.ravel()
        if hasattr(self, '_r_range'):
            r_list = np.array(r_list)
            # do the same processing for radius as magnitude
            r_condition = np.hstack([np.where((r >= self._r_range) &
                (r < self._r_range + self._r_bin_size))[0][0] for r in r_list])
            r_condition[r_condition == len(self._r_range)] = len(self._r_range) - 1
            r_condition = r_condition.ravel()
            # select completeness values corresponding to correct bins
            try:
                completeness = self._completeness_array[r_condition, mag_condition]
            except:
                print "r_condition:", r_condition
                print "mag_condition:", mag_condition 
                print "completeness_array.shape:", self._completeness_array.shape
                completeness = np.array([0])
        else:
            completeness = self._completeness_array[mag_condition]
        # flatten out the array
        completeness = completeness.ravel()
        all_completeness[msk] = completeness
        return all_completeness
