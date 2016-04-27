
#---------------------------------------------------------------------#
#- A class that inherits AngularCatalog and manages the division of  -#
#- a catalog into bins of properties                                 -#
#---------------------------------------------------------------------#

# External code
import copy
import warnings
import numpy as np
import numpy.ma as ma
import numpy.random as rand
import matplotlib.pyplot as plt

#Py2PAC code
import miscellaneous as misc
import AngularCatalog_class as ac

class MultiCatalog(ac.AngularCatalog):
    
    def __init__(self, ra, dec, properties=None, weight_file=None, 
        image_mask=None):
        """
        The init function for the MultiCatalog class- just calls the 
        init function for the AngularCatalog class.
        """
        
        #Nothing special so far, just call the AngularCatalog class
        AngularCatalog.__init__(self, ra, dec, properties=None, 
                                weight_file=None, image_mask=None)
                                
