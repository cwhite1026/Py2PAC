#Cython code specific to the image mask code

#Normal python imports
import numpy as np
import astropy.io.fits as fits
import scipy.stats as stat
import sys

#Cython specific imports
cimport numpy as np  #This apparently brings in compile-time info about numpy
cimport cython

#Define data types for numpy float32 and int
DTYPE_float32= np.float32
ctypedef np.float32_t DTYPE_t_float32
DTYPE_int= np.int
ctypedef np.int_t DTYPE_t_int
# DTYPE_bool= np.bool
# ctypedef np.bool_t DTYPE_t_bool



#do a thing for some reason (the reason being that Jake Vanderplas does it)
#I think it takes some of the pythonic safeties off the array handling for speed
@cython.boundscheck(False)
@cython.wraparound(False)


#Define some functions to help with the image mask procedures
#----------------------------------------------------------------------
#----------------------------------------------------------------------
def make_mask_from_weights(str filename):
    print "make_mask_from_weights says: generating an image mask from a weight file"
    #-----------------------------------
    #- Read in the fits file
    #-----------------------------------
    wmap=fits.open(filename)
    data_dtype_string=str(wmap[0].data.dtype)

    #Make the weights array with the correct byte order
    sys_is_le= sys.byteorder == 'little'
    native_ordering= sys_is_le and '<' or '>'
    print "make_mask_from_weights says: switching FITS file endianness"
    cdef np.ndarray[DTYPE_t_float32, ndim=2, mode='c'] weights=np.asarray(wmap[0].data, dtype=native_ordering+data_dtype_string[1:])

    #-------------------------------------------
    #- Figure out where the cutoff should be
    #-------------------------------------------
    #First select a subsample
    n_subset=np.int(1e6)  #How many points do we want in the subset?
    print "make_mask_from_weights says: getting random subsample and calculating stats on random subsample"
    cdef np.int nx=weights.shape[0]
    cdef np.int ny=weights.shape[1]

    #Define the x indices and y indices
    cdef np.ndarray[DTYPE_t_int, ndim=1, mode='c'] xinds=np.random.randint(0, nx, n_subset)
    cdef np.ndarray[DTYPE_t_int, ndim=1, mode='c'] yinds=np.random.randint(0, ny, n_subset)

    #Pull out the list of pixel values at the randomly selected locations
    cdef np.ndarray[DTYPE_t_float32, ndim=1, mode='c'] subsamp = weights[xinds, yinds]    

    # cdef np.ndarray[DTYPE_t_float32, ndim=1, mode='c'] subsamp=random_subset_of_f32_pixels(weights, n_subset)

    #Mask to nonzero guys
    # cdef np.ndarray[DTYPE_t_int, ndim=1, mode='c'] subsamp_nonzero_mask=subsamp>0
    subsamp_nonzero_mask=subsamp>0
    cdef np.ndarray[DTYPE_t_float32, ndim=1, mode='c'] subsamp_nonzero=subsamp[subsamp_nonzero_mask]
    # print "make_mask_from_weights says: number of nonzeros in random subsample= ", len(subsamp_nonzero)
    approx_frac_nonzero=float(len(subsamp_nonzero))/float(n_subset)

    #Do scipy stats score at percentile to get the 5th percentile weight
    cutoff=stat.scoreatpercentile(subsamp_nonzero, 5)
    # print "make_mask_from_weights says: minimum weight= ", cutoff
    
    #-------------------------------------------
    #- Return an array for the mask
    #-------------------------------------------
    where_good=weights>cutoff
    cdef np.ndarray[DTYPE_t_float32, ndim=2, mode='c'] completeness_mask=np.zeros(where_good.shape, dtype=DTYPE_float32)
    completeness_mask[where_good]=1.
    return nx, ny, approx_frac_nonzero, completeness_mask


#----------------------------------------------------------------------
#----------------------------------------------------------------------

def random_subset_of_f32_pixels(np.ndarray[DTYPE_t_float32, ndim=2, mode='c'] arr, np.int n_subset):
    #We have been given a 2D array and the number of values to return.  Pick n_subset x indices and y indices, then snag them from the array
    #Pick out the number of pixels in each direction
    cdef np.int nx=arr.shape[0]
    cdef np.int ny=arr.shape[1]

    #Define the x indices and y indices
    cdef np.ndarray[DTYPE_t_int, ndim=1, mode='c'] xinds=np.random.randint(0, nx, n_subset)
    cdef np.ndarray[DTYPE_t_int, ndim=1, mode='c'] yinds=np.random.randint(0, ny, n_subset)
    # cdef np.ndarray[DTYPE_t_int, ndim=2, mode='c'] pair_inds=np.transpose([xinds, yinds])

    #Pull out the list of pixel values at the randomly selected locations
    cdef np.ndarray[DTYPE_t_float32, ndim=1, mode='c'] arr_subset = arr[xinds, yinds]

    return arr_subset