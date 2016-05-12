import numpy as np
import subprocess
import os
import glob

import CompletenessFunction_class as compclass

#--------------------------------------------------------------------------
def completeness_list_from_file(filestr):
    """
    Takes a string that describes the file names of the completeness 
    functions, reads them in, and turns them into something you can feed
    to ImageMask.make_completeness_dict. 
    
    **Usage**
    immask = ImageMask.from_FITS_file(flag_map, fits_file_type='levels') 
    compfcns = miscellaneous.completeness_list_from_file("/path/to/files/*_s_*_expdisk_XYH.npz")
    immask.make_completeness_dict(*cfs)
    """

    compfcns = []
    for i in glob.glob(filestr):
        compfcn = compclass.CompletenessFunction.from_npz_file(i, level=i.split('_')[5])
        compfcns.append(compfcn)
    
    return compfcns
