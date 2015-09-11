import numpy as np
import subprocess
import os

#--------------------------------------------------------------------------
#Some constants
arcsec_opts = ['arcseconds', 'arcsecond', 'arcsec', 'a']
radian_opts = ['radians', 'radian', 'rad', 'r']
degree_opts = ['degrees', 'degree', 'deg', 'd']

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
              would put in '/a/b/thing' even if I'm in /a/b/.

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
