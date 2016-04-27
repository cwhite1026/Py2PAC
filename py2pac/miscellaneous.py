import numpy as np
import subprocess
import os
import glob

import CompletenessFunction_class as compclass


#--------------------------------------------------------------------------
#Some constants
arcsec_opts = ['arcseconds', 'arcsecond', 'arcsec', 'a']
radian_opts = ['radians', 'radian', 'rad', 'r']
degree_opts = ['degrees', 'degree', 'deg', 'd']


#--------------------------------------------------------------------------
def ang_sep(ra1, dec1, ra2, dec2, radians_out=True, radians_in=False):
    """
    Calculates the angular separation between point in RA/Dec space.  If
    the RAs and Decs given are scalars, it computes the single separation
    and returns the result as a scalar.  If the RAs and Decs are arrays,
    they must all be the same length and the returned value will be an
    array where theta[i] = separation between ra1[i], dec1[i] and ra2[i],
    dec2[i].

    Parameters
    ----------
    ra1: array-like or scalar
        The right ascensions of the first points
    dec1: array-like or scalar
        The declinations of the first points
    ra2: array-like or scalar
        The right ascensions of the second points
    dec2: array-like or scalar
        The declinations of the second points
    radians_in: boolean (optional)
        If True, assumes the RAs and Decs are in radians.  If False,
        assumes they are in degrees.  Default is False.
    radians_out: boolean (optional)
        If True, returns the angular separations in radians.  If False,
        returns the angular separations in degrees.

    Returns
    -------
    theta: array-like or scalar (matches input)
        The angular separations between the points given.  
    """
    #Convert to radians 
    if radians_in==False:
        ra1 = np.radians(ra1)
        ra2 = np.radians(ra2)
        dec1 = np.radians(dec1)
        dec2 = np.radians(dec2)
        
    #Define the numerator for the arctan.  It's gross
    numer=(np.cos(dec2)**2.) * (np.sin(ra2-ra1)**2.)
    numer+=(np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2.
    numer=numer**.5
    
    #Define the denominator. Also gross
    denom=np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
    
    #Find the separation
    theta = np.arctan(numer/denom)
    if not radians_out:
        theta=np.degrees(np.arctan(numer/denom))
        
    return theta


#--------------------------------------------------------------------------
#-----------------------------#
#- Rotate coordinate systems -#
#-----------------------------#
def rotate_coords(x1, y1, theta):
    """
    Applies a rotation matrix to rotate the coordinate frame by an angle
    theta in RADIANS

    Parameters
    ----------
    x1: array-like or scalar
        The x coordinate(s) to rotate
    y1: array-like or scalar
        The y-coordinate(s) to rotate
    theta: scalar
        The angle to rotate the coordinates, given in radians

    Returns
    -------
    x2: array-like or scalar (matches x1)
        The rotated x coordinates
    y2: array-like or scalar (matches y1)
        The rotated y coordinates
    """
    
    #If we have an array, make sure it's a numpy array
    try:
        nx=len(x1)
    except:
        pass
    else:
        x1=np.array(x1)
        y1=np.array(y1)

    #Apply the rotation matrix
    x2 = x1 * np.cos(theta) - y1 * np.sin(theta)
    y2 = x1 * np.sin(theta) + y1 * np.cos(theta)

    return x2, y2

#--------------------------------------------------------------------------

def centers(arr):
    """
    Takes an N+1 length array and returns an N length array that has
    the (arr[i]+arr[i+1])/2 in element i (if you have the edges of
    histogram bins, this will give you the centers)

    Parameters
    ----------
    arr : array-like
        The array that you want the centers of

    Returns
    -------
    centers : numpy array
            An array with length 1 shorter than arr that is
            centers[i] = (arr[i]+arr[i+1])/2.
    """
    arr=np.array(arr)
    
    #Make sure we can actually do this
    if len(arr)<2:
        raise ValueError("centers says: You've given me a 0 or "
                         "1 length array.  I can't work with that")
    
    #Find the centers and return
    return arr[0:-1] + np.diff(arr)/2.

#--------------------------------------------------------------------------

def files_starting_with(file_base):
    """
    This takes file_base and returns all the files that start with that

    We have to look at everything in the directory and then pare down to
    just what you care about because check_output is being a butt and
    Popen will do it but is insecure for these purposes: you could put
    in 'not_filename; rm -rf /*' and you would be very, very sad.

    Parameters
    ----------
    file_base : string
              The path from / that you want to complete.  For instance
              if I wanted files /a/b/thing1.txt and /a/b/thing2.txt, I
              would put in '/a/b/thing' even if the working directory is
              /a/b/.

    Returns
    -------
    files : python list
          A list of strings that specify the files that complete
          file_base.
    """

    #If it's a directory, just ls the directory
    if os.path.isdir(file_base):
        use_direc = file_base
        start_of_filename = ''
    else:
        #If it's not a directory already, strip off the start of the
        #file name from the directory.
        stripped_file_base = file_base.strip('/')
        parts = stripped_file_base.split('/')
        start_of_filename = parts[-1]
        use_direc = file_base.rstrip(start_of_filename)

    #Ask for all the file names in the directory we care about
    files = subprocess.check_output(['ls', use_direc])
    files = files.split()

    #Now see which start with what I want
    nchars = len(start_of_filename)
    use_files = []
    for f in files:
        if f[0:nchars] == start_of_filename:
            use_files.append(use_direc+'/'+f)

    return use_files

#--------------------------------------------------------------------------
        
#-----------------------------#
#- Convert angles to degrees -#
#-----------------------------#
def put_thetas_in_degrees(thetas, unit):
    """
    Takes thetas and turns them from whatever unit they're in
    and converts everything to degrees

    Parameters
    ----------
    thetas : array-like
            An array of angles.
    unit : string
         The unit that the thetas are in.  The options are 'a',
         'arcsec', 'arcseconds'; 'd', 'deg', 'degrees'; 'r', 'rad',
         'radians'.

    Returns
    -------
    deg_thetas : numpy array
              The thetas you put in now converted into degrees
    """
    
    #Take a min and a max and what unit they're in and convert
    #to degrees.
    
    thetas=np.array(thetas, dtype=np.float)
    
    #What input unit do we have?
    in_arcsec = unit.lower() in arcsec_opts
    in_radians = unit.lower() in radian_opts
    in_degrees = unit.lower() in degree_opts
        
    #Deal with each option
    if in_arcsec:
        thetas/=3600.
    elif in_radians:
        thetas = np.degrees(min_theta)
    elif not in_degrees:
        print "miscellaneous.put_thetas_in_degrees says: you have chosen unit=", unit
        print "This is not an option."
        print "For arcseconds, use 'arcseconds', 'arcsecond', 'arcsec', or 'a'."
        print "For radians, use 'radians', 'radian', 'rad', or 'r'."
        print "For degrees, use 'degrees', 'degree', 'deg', or 'd'."
        raise ValueError("You chose an invalid value of unit.")
        
    return thetas
