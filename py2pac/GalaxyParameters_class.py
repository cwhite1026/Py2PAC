#External code
import numpy as np
from astropy.cosmology import default_cosmology
from schechter_functions import get_schechter_mags, get_radii

#===============================================================================
#===============================================================================
#===============================================================================

class GalaxyParameters:
    """
    This is a class that holds a completeness function for use in generating the
    randoms on the sky.

    Parameters
    ----------
    completeness_array : 1- or 2D array
        This is an array describing completeness as a function of magnitude only
        if 1D and of magnitude and radius if 2D.
        
    mag_range : 1D array-like
        Range of magnitudes 

    r_range : 1D array-like, optional
        Range of radii

    Notes
    -----
    Can be constructed so as to either be only dependent on magnitude, or on
    both magnitude and radius.
    """

    #----------------------------------------------------------------------
    #              Routines for generating mags and radii
    #----------------------------------------------------------------------
        
    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, mags, radii=None):
        mags = np.asarray(mags)
        self.mags = mags
        self._min_mag = np.min(mags)
        self._max_mag = np.max(mags)
        if radii is not None:
            radii = np.asarray(radii)
            self.radii = radii
            self._min_rad = np.min(radii)
            self._max_rad = np.max(radii)

    #----------------------------------------------------------------------
    #-----------------------------------------------#
    #- Make a magnitude-only completeness function -#
    #-----------------------------------------------#
    @classmethod
    def get_mags_and_radii(cls, size, radii=True, min_mag = 20, max_mag = 28, z = 1.7):
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
        df = default_cosmology.get_cosmology_from_string('Planck13')
        distmod = df.distmod(z).value
        mags = get_schechter_mags(size, distmod, min_mag, max_mag)
        radii = get_radii(mags)
        return cls(mags + distmod, radii)


