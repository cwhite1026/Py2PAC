#-------------------------------------------------------------------#
# This is the file that contains the main class that Py2PAC is      #
# built around, the AngularCatalog, which holds RAs and Decs and    #
# does the actual calculations of correlation functions.            #
#-------------------------------------------------------------------#

# External code
import copy
import numpy as np
import numpy.ma as ma
import numpy.random as rand
from scipy import optimize as opt
from sklearn.neighbors import BallTree

#Py2PAC code
import correlations as corr
import cosmology as cos
import ImageMask_class as imclass
import miscellaneous as misc
import ThetaBins_class as binclass
import CorrelationFunction_class as cfclass
import Gp_class as gpclass

#===============================================================================
#===============================================================================
#===============================================================================
class AngularCatalog(object):
    #------------------#
    #- Initialization -#
    #------------------#Your data and masked data will be the same
    def __init__(self, ra, dec, generate_randoms=False, default_oversample=1.,
                 properties=None, weight_file=None, image_mask=None):

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
        self._ra_range =np.array([ra.min(), ra.max()])
        self._ra_span = np.diff(self._ra_range)[0]
        self._dec_range = np.array([dec.min(), dec.max()])
        self._dec_span = np.diff(self._dec_range)[0]
        #Store how many objects are in the whole sample
        self._input_n_objects = ra.size

        #Store the info from keywords
        self._image_mask = image_mask
        self._weight_file_name = weight_file
        self._properties = properties
        self._random_oversample_factor = default_oversample

        #Store some defaults/holders
        self._theta_bins=None
        self._cfs={}
        self._powerlaw_A={}
        self._powerlaw_beta={}
        self._powerlaw_offset={}
        self._IC={}

        #Make blank things so I can ask "is None" rather than "exists"
        self._n_objects=None
        self._data_tree=None
        self._ra_random=None
        self._dec_random=None
        self._random_tree=None
        #Things for the correlation functions
        self._cf_theta_bins=None
        self._cf_thetas=None           
        #Things needed for the fitting/integral constraint
        self._rr_counts=None
        self._thetas_for_rr=None
        self._rr_ngals=None
        self._G_p=None               
        #Completenesses
        self._completeness=None
        self._use=None             
        self._use_random=None      
        self._subregion_number=None

        #Set up the mask and generate the randoms if asked
        self.setup_mask()
        if generate_randoms:
            self.generate_random_sample()

    #------------------------------------------------------------------------------------------
    #--------------------------------------------#
    #- Class method for making a random catalog -#
    #--------------------------------------------#
    @classmethod
    def random_catalog(cls, n_randoms, image_mask = None, ra_range=None,
                       dec_range=None):
        """
        Creates an AngularCatalog populated with RAs and Decs placed
        randomly within the mask.  This can be passed either an image
        mask or an RA and Dec range

        Syntax
        ------
        * cat = ac_class.AngularCatalog.random_catalog(n_randoms, image_mask=ImageMask_object)
        OR
        * cat = ac_class.AngularCatalog.random_catalog(n_randoms, ra_range=[min, max],
                                                       dec_range=[min, max])

        Parameters
        ----------
        n_randoms : scalar
            The number of randoms that you want in you catalog
        image_mask : ImageMask object (optional)
            An ImageMask object with the outline that you want for your
            randoms.  This is one option.
        ra_range : two-element array-like (optional)
            The minimum and maximum RA you would like your randoms to have.
            This is an alternative to the image_mask option.  This must be
            combined with the dec_range argument as well.
        dec_range : two-element array-like (optional)
            The minimum and maximum Dec you would like your randoms to have.
            This is an alternative to the image_mask option.  This must be
            combined with the ra_range argument.

        Returns
        -------
        cat : AngularCatalog object
            An AngularCatalog instance with n_randoms distributed over either
            the image_mask or over the RA and Dec range.
        """

        #Make an image mask from the RA and Dec ranges if we don't have an
        #image mask already
        need_image_mask = image_mask is None
        if need_image_mask:
            image_mask = imclass.ImageMask(forced_ra_range=ra_range,
                                           forced_dec_range=dec_range)

        #Make a dummy catalog and generate the randoms we want
        dummy_cat = cls([0], [0], generate_randoms=False,
                        image_mask=image_mask)
        temp = dummy_cat.generate_random_sample(store=False,
                                                make_exactly=n_randoms)
        rand_ras, rand_decs = temp[0:2]

        #Return the angular catalog with the RAs and Decs
        return AngularCatalog(rand_ras, rand_decs, image_mask=image_mask)
        
    #------------------------------------------------------------------------------------------

    #----------------------------#
    #- Set the weight file name -#
    #----------------------------#            
    def set_mask_to_weight_file(self, filename):
        """
        Set the weight file name and process the file to an image mask

        Parameters
        ----------
        filename : string
            The location of the FITS file that you want to process to
            a weight mask.  The file name should be specified from /
        """
        self._weight_file_name=filename
        self.setup_mask(force_remake=True)
        return

    #------------------------------------------------------------------------------------------

    #-------------------------------------------#
    #- Make an image mask from the weight file -#
    #-------------------------------------------#
    def setup_mask(self, force_remake=False):
        #Create an image mask (from the weight file if given one)
        if (self._image_mask is None) or force_remake:
            self._image_mask = imclass.ImageMask(angular_catalog=self,
                                                  weight_file=self._weight_file_name)
        
        #Ask the mask for the completenesses of each data object
        self._completeness = self._image_mask.return_completenesses(self._ra, self._dec)

        #Generate random numbers- this is basically for when we have non-binary completeness
        compare_to = rand.random(size=self._n_objects)

        #Use the random numbers to figure out which guys in the data to use
        self._use = compare_to < self._completeness

        #Set up the data tree now that we have a mask
        self.make_data_tree()

        #Record how many objects we're actually using
        self._n_objects=len(self._ra[self._use])

    #------------------------------------------------------------------------------------------

    #-------------#
    #- Move mask -#
    #-------------#
    def move_mask(self, delta_ra=None, delta_dec=None,
                  theta_degrees=None, preview=False):
        #Calls the image mask's translation/rotation routine.
        if preview:
            newmask=self._image_mask.move_mask_on_sky(delta_ra=delta_ra,
                                                      delta_dec=delta_dec,
                                                      theta_degrees=theta_degrees,
                                                      preview=preview)
            return newmask
        else:
            self._image_mask.move_mask_on_sky(delta_ra=delta_ra,
                                              delta_dec=delta_dec,
                                              theta_degrees=theta_degrees,
                                              preview=preview)
    
    #------------------------------------------------------------------------------------------

    #------------------------------#
    #- Set subregions within mask -#
    #------------------------------# 
    def subdivide_mask(self, n_shortside=3, n_longside=4, preview=False, rotation_angle=None, padding=None, only_show=None):
        #draw a box that fits around all the data as snugly as possible and subdivide.
        #Make an image mask that corresponds to each box
        #If we don't have an image mask, then I am too lazy to code it at the moment.  I want the WCS transforms
        
        #Start by putting down a bunch of randoms.
        self.generate_random_sample(make_exactly=5.e4, masked=True)
        ra=self._ra_random
        dec=self._dec_random
        x1, y1=self._image_mask.ra_dec_to_xy(ra, dec)

        #Set the padding on each side
        if padding:
            try:
                pad_bottom, pad_top, pad_left, pad_right=padding
            except:
                print "subdivide_mask says: WARNING!  You have given me something I can't use for padding.  Padding must be a 4-element 1D array in the format [bottom, top, left, right].  No padding used this time"
                pad_bottom, pad_top, pad_left, pad_right=[0,0,0,0]
        else:
            pad_bottom, pad_top, pad_left, pad_right=[0,0,0,0]
        
        #If we don't have an already chosen angle, choose a bunch of angles and 
        #transform the coordinates to rotated systems and get the 
        #areas of the rectangle enclosing all the data points at this angle.  Take
        #The angle with the minimum area for the enclosing rectangle.
        if rotation_angle is None:
            thetas=np.radians(np.arange(90, dtype=np.float))
            areas=[]
            corners=[]
            for th in thetas:
                x2, y2= u.rotate_coords(x1, y1, th)
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
        x2, y2= u.rotate_coords(x1, y1, use_theta)
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

        #If we're just previewing, then plot the randoms with the grid overlaid
        if preview:
            #Figure out what subregions we have
            subregions=self._image_mask.return_subregions(ra, dec, theta=use_theta, rot_xedges=x2edges, rot_yedges=y2edges, hypothetical=True)
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
                x1, y1=u.rotate_coords(x2, y2, -use_theta)
                ax.plot(x1, y1, color='Red', lw=2)
            #Plot the horizontal lines
            for iy in range(ny+1):
                x2=[x2min, x2max]
                y2=[y2edges[iy], y2edges[iy]]
                x1, y1=u.rotate_coords(x2, y2, -use_theta)
                ax.plot(x1, y1, color='Red', lw=2)

            #Figure out the dimensions of the boxes in angular space
            x2=[x2edges[0], x2edges[0], x2edges[1]]
            y2=[y2edges[0], y2edges[1], y2edges[0]]
            ra_box, dec_box=self._image_mask.xy_to_ra_dec(x2, y2)
            y_side=cos.ang_sep(ra_box[0], dec_box[0], ra_box[1], dec_box[1],
                               radians_in=False, radians_out=False) * 3600
            x_side=cos.ang_sep(ra_box[0], dec_box[0], ra_box[2], dec_box[2],
                               radians_in=False, radians_out=False) * 3600
            print x_side, y_side

            #Print out the parameters
            ax.text(.05, .95, "theta= "+str(np.degrees(use_theta))[0:5],
                    transform=ax.transAxes)
            ax.text(.05, .9, "padding="+str(padding),  transform=ax.transAxes)
            ax.text(.05, .85, "n_longside="+str(n_longside),  transform=ax.transAxes)
            ax.text(.05, .8, "n_shortside="+str(n_shortside),  transform=ax.transAxes)
            ax.text(.5, .05, "box size: "+str(x_side)[0:5]+" by "+str(y_side)[0:5]+" arcsec",
                    transform=ax.transAxes)

            #Label the subregions
            y_label_coord=.85
            avg_ngals=float(len(ra[inside]))/(nx*ny)
            ax.text(0.8, 0.95, "N_bin/N_avg", transform=ax.transAxes, fontsize=12)
            ax.text(0.8, 0.9, "outside-> "+str(float(len(ra[outside]))/avg_ngals)[0:4],
                    transform=ax.transAxes, fontsize=9)
            for ix in range(nx):
                for iy in range(ny):
                    #What bin number is this?
                    bin_number= nx*iy + ix
                    #Where's the center of the box?
                    text_x2=(x2edges[ix] + x2edges[ix+1])/2.
                    text_y2=(y2edges[iy] + y2edges[iy+1])/2.
                    text_x1, text_y1=u.rotate_coords(text_x2, text_y2, -use_theta)
                    #Print the bin number at the center of the box
                    ax.text(text_x1, text_y1, str(bin_number), fontsize=20, color='Lime',
                            horizontalalignment='center', verticalalignment='center')

                    #Print the number of galaxies in the upper right corner
                    thisbin= subregions==bin_number
                    n_thisbin= float(len(ra[thisbin]))
                    print "bin ", bin_number, " has ", n_thisbin, " randoms in it"
                    display_string='bin '+str(bin_number)+'-> '+str(n_thisbin/avg_ngals)[0:4]
                    ax.text(0.85, y_label_coord, display_string,  transform=ax.transAxes, fontsize=9)
                    y_label_coord-=0.05
            
            plt.show()

        #Otherwise (not preview)...
        else:
            #Store the subregion number for each galaxy and store subregion info in the image mask
            self._subregion_number = self._image_mask.return_subregions(self._ra, self._dec,
                                                                        theta=use_theta,
                                                                        rot_xedges=x2edges,
                                                                        rot_yedges=y2edges)

    #------------------------------------------------------------------------------------------

    #------------------------#
    #- Create random sample -#
    #------------------------#
    def generate_random_sample(self, store=True, oversample_factor=None,
                               make_exactly=None, multiplier=None):
        #Pulls a random sample from our section of RA and Dec
        #   on a uniform sphere
        #If store==True, it puts them right into the object, if
        #   not, it returns them

        #Make sure we have the right oversample here and stored
        if oversample_factor is not None:
            self._random_oversample_factor = oversample_factor
            
        oversample_factor = self._random_oversample_factor


        #How many do we want to end up with?
        if make_exactly is not None:
            number_to_make = make_exactly
        else:
            number_to_make = oversample_factor * self._n_objects 

        #The the factor more than I actually need to make (to account for
        #what gets lost to the mask)
        if multiplier is None:
            multiplier = 1. / self._image_mask._approx_frac_nonzero

        number_to_start_with = np.int(number_to_make * multiplier)  
            
        #----------------------------
        #- Generate a random sample
        #----------------------------      
        ra_min=min(self._image_mask._ra_range[0], self._ra_range[0]) #-.05
        ra_max=max(self._image_mask._ra_range[1], self._ra_range[1]) #+.05
        dec_min=min(self._image_mask._dec_range[0], self._dec_range[0]) #-.05
        dec_max=max(self._image_mask._dec_range[1], self._dec_range[1]) #+.05
        ra_R, dec_R= corr.uniform_sphere((ra_min, ra_max),
                                              (dec_min, dec_max),
                                              number_to_start_with)
        
        #----------------------------------
        #- Mask and add more if undershot
        #----------------------------------
        #Get completenesses and see which to use
        random_completeness = self._image_mask.return_completenesses(ra_R, dec_R)
        compare_to = rand.random(size=len(ra_R))
        use_random = compare_to < random_completeness
        number_we_have = len(ra_R[use_random])
        print ("AngularCatalog.generate_random_sample says: At first pass,"
               " we made "+str(number_we_have))
        print "      We need ", number_to_make
                
        #Check to see by how many we've overshot
        number_excess = number_we_have - number_to_make
        
        #If we've actually made too few, make more
        if number_excess<0:
            print ("AngularCatalog.generate_random_sample says: I have made too "
                  "few objects within the target area. Making more.")
            #Figure out what fraction of the guys that we made were used so if my mask is 
            #teeny and in a big field, it won't take forever to get to where we want to be
            fraction_used_last_time = float(number_we_have) / len(ra_R)
            if fraction_used_last_time < 1.e-3:
                fraction_used_last_time = 1e-3
                
            #Recursive things are fun!  Ask for exactly how many more we need.
            new_multiplier = 1. / fraction_used_last_time
            newguys = self.generate_random_sample(store=False,
                                                  multiplier=new_multiplier,
                                                  make_exactly=abs(number_excess))
            #Unpack
            more_ras, more_decs, more_comps, more_use = newguys
            
            #Add these galaxies to the existing arrays
            ra_R= np.concatenate((ra_R, more_ras))
            dec_R= np.concatenate((dec_R, more_decs))
            random_completeness = np.concatenate((random_completeness, more_comps))
            use_random= np.concatenate((use_random, more_use))
            number_we_have = len(ra_R[use_random])
            number_excess = number_we_have - number_to_make

        #If we overshot, cut some off
        if number_excess > 0:
            print ("AngularCatalog.generate_random_sample says: "
                  "Cutting down to exactly as many unmasked objects as the data")
            #Now we want to cut down to exactly n_objects in our masked random.
            object_indices=np.arange(len(ra_R)) 
            #What's the index that corresponds to the n_objects'th *true* element
            end_index= int(object_indices[use_random][int(number_to_make)])
            #Cut down the arrays
            ra_R=ra_R[0:end_index]
            dec_R=dec_R[0:end_index]
            random_completeness= random_completeness[0:end_index]
            use_random= use_random[0:end_index]
        else:
            print ("AngularCatalog.generate_random_sample says: "
                  "I made exactly the right number!  It's like winning "
                  "the lottery but not actually fun...")
            
                
        #-----------------------------------------------
        #- Figure out what to do with what we've made
        #-----------------------------------------------
        if store:                
                #We're not storing the data separately and we've already set up the
                #_random_completeness and the _use_random
                # so we just have to make the tree.  Since this is completely
                #independent of the unmasked RA and Dec set, we'll
                # store the catalog
                self._ra_random = ra_R[use_random]
                self._dec_random = dec_R[use_random]
                self.make_random_tree()
        else:
            return (ra_R[use_random], dec_R[use_random],
                   random_completeness[use_random],
                   np.ones(sum(use_random)).astype(bool))
        
    #------------------------------------------------------------------------------------------
        
    #---------------------------------#
    #- Compute BallTree for the data -#
    #---------------------------------#
    def make_data_tree(self):
        #The astroML correlation function methods want a cartesian position
        #instead of the angular positions- this does the conversion
        
        print "make_datatree says: Computing the BallTree for data."
        data = np.asarray(corr.ra_dec_to_xyz(self._ra[self._use], self._dec[self._use]), order='F').T
        self._data_tree = BallTree(data, leaf_size=2)
        
        return

    #------------------------------------------------------------------------------------------

    #------------------------------------------#
    #- Compute BallTree for the random sample -#
    #------------------------------------------#
    def make_random_tree(self):
        #Make sure we have the random data made
        if (self._ra_random is None) or (self._dec_random is None):
            print "make_random_tree says: no random sample found.  Generating one."
            self.generate_random_sample()

        #Make the tree
        print "make_randomtree says: Computing the BallTree for the randoms."
        random_data=np.asarray(corr.ra_dec_to_xyz(self._ra_random, self._dec_random), order='F').T
        self._random_tree = BallTree(random_data, leaf_size=2)                
                
        return          

    #------------------------------------------------------------------------------------------

    def set_theta_bins(self, min_theta, max_theta, nbins,
                       unit='a', logbins=True):
        #Make a ThetaBins class and save it.
        self._theta_bins = binclass.ThetaBins(min_theta, max_theta, nbins,
                                              unit=unit, logbins=logbins)

    #------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------#
    #- Check to make sure we have all the info needed for CF calculation -#
    #---------------------------------------------------------------------#         
    def __check_cf_setup(self, need_subregions=False,
                         random_oversample=None, check_trees=True):
        #Make sure that we have all the things we need to do a
        #correlation function properly (I got tired of the redundant
        #code in the different CF calculation routines)
        
        #Check that we have the bins 
        if not isinstance(self._theta_bins, binclass.ThetaBins):
            raise ValueError("CF calculations need separation bins.  Use "
                             "catalog.set_theta_bins(min_theta, max_theta,"
                             "nbins, unit='arcsec', logbins=True)")
        
        #Change/store the random oversampling factor if it's given
        if random_oversample is not None:
            self._random_oversample_factor=random_oversample

        #Check the existence of a random sample
        if self._ra_random is None:
            self.generate_random_sample()

        #See if we're properly oversampled.
        nR=len(self._ra_random)
        if nR != len(self._ra)*self._random_oversample_factor:
            self.generate_random_sample()
            
        #Check to make sure we have the trees for the appropriate guys
        if check_trees:
            if self._data_tree is None:
                self.make_data_tree()
            if self._random_tree is None:
                self.make_random_tree()

        #Check to make sure that the subdivisions have happened
        #if need_subregions.  If not, throw an error because it's
        #too specific to fill it in automatically
        if need_subregions:
            if self._subregion_number is None:
                raise ValueError("Jackknife and block bootstrap require "
                                "that you subdivide the field.  Call the "
                                "catalog.subdivide_mask() routine first.")

    #------------------------------------------------------------------------------------------

    #-----------------------------------------------------#
    #- Calculate the correlation function without errors -#
    #-----------------------------------------------------# 
    def cf(self, estimator='landy-szalay', n_iter=1, clobber=False,
          random_oversample=None, save_steps_file=None, name='cf'):
        #This uses the info we have plus the astroML correlation package
        #   to compute the angular correlation function.
        #The idea is that this function will figure out what information
        #   is available and call the appropriate (most efficient) function
        #   with all the relevant information.
        #This function will store the values it calculates for missing info

        if (name in self._cfs.keys()) and not clobber:
            raise ValueError("CorrelationFunction.cf says: There's already"
                             " a CF by that name.  Please choose another or "
                             "overwrite by calling with clobber=True")

        #Make sure that we have everything we need and fix anything missing that's fixable
        self.__check_cf_setup(random_oversample=random_oversample,
                              need_subregions=False, check_trees=True)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'no_error',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'           : estimator
             }
        self._cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self._cfs[name].get_thetas(unit='degrees')
        nbins=len(centers)

        #Do the calculation
        cf=np.zeros(nbins)
        DD=np.zeros(nbins)
        print "AngularCatalog.cf says: doing a CF calculation without error estimation"
        iterations={}
        for it in np.arange(n_iter):
            this_cf, this_dd = corr.two_point_angular(self._ra[self._use], 
                                                     self._dec[self._use], 
                                                     edges,
                                                     BT_D=self._data_tree, 
                                                     BT_R=self._random_tree,
                                                     estimator=estimator, 
                                                     ra_R=self._ra_random,
                                                     dec_R=self._dec_random,
                                                     return_DD=True)
            iterations[it]=this_cf
            cf += this_cf
            DD = this_dd/2.
            if save_steps_file is not None:
                self._cfs[name].set_cf(cf/(it+1), np.zeros(nbins), iterations=iterations)
                self._cfs[name].set_DD(DD)
                self.save_cf(save_steps_file, cf_keys=name)
            if n_iter >1:
                self.generate_random_sample()

        #Divide out the number of iterations
        cf/=n_iter

        #Make sure we've stored everything properly even if we're not saving
        self._cfs[name].set_cf(cf, np.zeros(nbins), iterations=iterations)

    #------------------------------------------------------------------------------------------

    #----------------------------------------------------#
    #- Find the CF and error by single-galaxy bootstrap -#
    #----------------------------------------------------#
    def cf_bootstrap(self, n_boots=10, bootstrap_oversample=1,
                     random_oversample=None, estimator='landy-szalay',
                     save_steps_file=None, name='galaxy_bootstrap',
                     clobber=False):
        #Calculate the  correlation function with single-galaxy bootstrapping

        if (name in self._cfs.keys()) and not clobber:
            raise ValueError("CorrelationFunction.cf_bootstrap says: "
                             "There's already a CF by that name.  Please "
                             "choose another or overwrite by calling with "
                             "clobber=True")
        
        #Check that everything is set up
        self.__check_cf_setup(need_subregions=False, check_trees=False,
                              random_oversample=random_oversample)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'single_galaxy_bootstrap',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'           : estimator
             }
        self._cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self._cfs[name].get_thetas(unit='degrees')
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
            ind=np.random.randint(0, self._n_objects,
                                  bootstrap_oversample*self._n_objects)
            ra_b=self._ra[self._use][ind]
            dc_b=self._dec[self._use][ind]
            
            #Calculate this boot
            bootstrap_boots[i], rr = corr.two_point_angular(ra_b, dec_b, edges, 
                                                            BT_D=self._data_tree, 
                                                            BT_R=self._random_tree,
                                                            estimator=estimator, 
                                                            ra_R=self._ra_random, 
                                                            dec_R=self._dec_random, 
                                                            RR=rr, return_RR=True)
            #Store what we have
            temp[i]=bootstrap_boots[i]
            if (save_steps_file is not None):
                bootstrap_cf=np.nanmean(temp[0:i+1], axis=0)
                bootstrap_cf_err=np.nanstd(temp[0:i+1], axis=0)
                self.save_cfs(save_steps_file, cf_keys=[name])
                
        #Now we're done- do the final storage.
        bootstrap_cf=np.nanmean(temp, axis=0)
        bootstrap_cf_err=np.nanstd(temp, axis=0)
        self._cfs[name].set_cf(bootstrap_cf, bootstrap_cf_err,
                               iterations=bootstrap_boots)
        self._cfs[name].set_counts(RR=rr)
        
    #------------------------------------------------------------------------------------------

    #----------------------------------------#
    #- Find the CF and error by jackknifing -#
    #----------------------------------------#
    def cf_jackknife(self, ignore_regions=[], estimator='landy-szalay',
                     random_oversample=None, save_steps_file=None,
                     name='jackknife', clobber=False):
        #This takes a divided mask and performs the correlation
        #function calculation on the field with each sub-region
        #removed in turn.

        if (name in self._cfs.keys()) and not clobber:
            raise ValueError("CorrelationFunction.cf_jackknife says: "
                             "There's already a CF by that name.  Please "
                             "choose another or overwrite by calling with "
                             "clobber=True")

        #Check to make sure we have everything we need
        self.__check_cf_setup(need_subregions=True, check_trees=False,
                              random_oversample=random_oversample)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'jackknife',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'           : estimator
             }
        self._cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self._cfs[name].get_thetas(unit='degrees')
        
        #pull out the unique subregion numbers and figure out which to use
        regions=np.asarray(list(set(self._subregion_number)))
        use_regions=[r for r in regions if (r not in ignore_regions) and (r != -1)]
        use_regions=np.array(use_regions)
        n_jacks=len(use_regions)

        #Figure out where the randoms are
        random_subregions=self._image_mask.return_subregions(self._ra_random,
                                                             self._dec_random)
        
        #Now loop through the regions that you should be using 
        #and calculate the correlation function leaving out each
        jackknife_jacks = {}
        #Make a mask that takes out all the galaxies that aren't in use_regions
        valid_subregion = ma.masked_not_equal(self._subregion_number, -1).mask
        random_valid_subregion=ma.masked_not_equal(random_subregions, -1).mask
        for bad_reg in ignore_regions:
            this_mask = ma.masked_not_equal(self._subregion_number, bad_reg).mask
            valid_subregion = valid_subregion & this_mask
            this_mask = ma.masked_not_equal(random_subregions, bad_reg).mask
            random_valid_subregion = random_valid_subregion & this_mask        

        temp = np.zeros((n_jacks, len(self._cf_thetas)))
        for i, r in enumerate(use_regions):
            #Make the mask for the data
            not_region_r = ma.masked_not_equal(self._subregion_number, r).mask  
            this_jackknife = valid_subregion & not_region_r & self._use  
            
            #Make the mask for the randoms
            random_not_region_r = ma.masked_not_equal(random_subregions, r).mask
            random_this_jackknife = random_not_region_r & random_valid_subregion

            #Do the calculation for this jackknife and store it
            print "calculating jackknife", i
            jackknife_jacks[r] = corr.two_point_angular(self._ra[this_jackknife], 
                                                        self._dec[this_jackknife], 
                                                        edges, estimator=estimator, 
                                                        ra_R = self._ra_random[random_this_jackknife],
                                                        dec_R = self._dec_random[random_this_jackknife])
            temp[i]=jackknife_jacks[r]
            if (save_steps_file is not None):
                    jackknife_cf=np.nanmean(temp[0:i+1], axis=0)
                    jackknife_cf_err=np.nanstd(temp[0:i+1], axis=0)
                    self._cfs[name].set_cf(jackknife_cf, jackknife_cf_err,
                                           iterations=bootstrap_boots)
                    self.save_cfs(save_steps_file, cf_keys=[name])
            
        #Now that we have all of the jackknifes (jackknives?), calculate the mean
        # and variance.
        jackknife_cf=np.nanmean(temp, axis=0)
        jackknife_cf_err=np.nanstd(temp, axis=0)
        self._cfs[name].set_cf(jackknife_cf, jackknife_cf_err,
                               iterations=bootstrap_boots)

    #------------------------------------------------------------------------------------------

    #--------------------------------------------#
    #- Find the CF and error by block bootstrap -#
    #--------------------------------------------#
    def cf_block_bootstrap(self, n_boots=10, ignore_regions=None,
                           estimator='landy-szalay', random_oversample=None,
                           bootstrap_oversample=1, save_steps_file=None,
                           name='block_bootstrap', clobber=False):
        #Use the subdivided mask to bootstrap on blocks rather than
        #single galaxies.

        if (name in self._cfs.keys()) and not clobber:
            raise ValueError("CorrelationFunction.cf_block_bootstrap says: "
                             "There's already a CF by that name.  Please "
                             "choose another or overwrite by calling with "
                             "clobber=True")

        #Check to make sure I have everything that I need
        self.__check_cf_setup(masked=True, need_subregions=True,
                              random_oversample=random_oversample,
                              check_trees=False)

        #Make a new CorrelationFunction instance and set the basic info
        #First make a dictionary of the arguments to pass because it's ugly
        info={'name'            : name,
             'cf_type'          : 'jackknife',
             'ngals'            : self._n_objects,
             'theta_bin_object' : copy.deepcopy(self._theta_bins),
             'estimator'           : estimator
             }
        self._cfs[name] = cfclass.CorrelationFunction(**info)
        centers, edges = self._cfs[name].get_thetas(unit='degrees')
        nbins = len(centers)
        
        print "block boots done with setup"

        #Figure out which subregions we should be using
        regions=np.asarray(list(set(self._subregion_number)))
        use_regions=[r for r in regions if (r not in ignore_regions) and (r != -1)]
        use_regions=np.array(use_regions)

        #Figure out where the randoms are
        random_subregions=self._image_mask.return_subregions(self._ra_random,
                                                             self._dec_random)

        #Make a dictionary of arrays containing the indices of the members of each sub-region we need
        indices={}
        random_indices={}
        for r in use_regions:
            indices[r]=np.where(self._subregion_number == r)[0]
            random_indices[r]=np.where(random_subregions == r)[0]

        #Loop through the bootstraps
        block_bootstrap_boots={}
        n_choose=len(use_regions)*bootstrap_oversample
        temp = np.zeros((n_boots, nbins))
        print "block boots looping through boots"
        for i in np.arange(n_boots):
            this_boot=rand.choice(use_regions, size=n_choose)
            this_boot_indices=np.array([], dtype=np.int)
            this_boot_random_indices=np.array([], dtype=np.int)
            
            for region in this_boot:
                this_boot_indices=np.concatenate((this_boot_indices, indices[region]))
                this_boot_random_indices=np.concatenate((this_boot_random_indices,
                                                         random_indices[region]))

            # this_boot_indices=np.array(
            print "calculating boot", i
            temp[i] = corr.two_point_angular(self._ra[this_boot_indices], 
                                             self._dec[this_boot_indices], 
                                             edges, estimator=estimator, 
                                             ra_R=self._ra_random[this_boot_random_indices],
                                             dec_R=self._dec_random[this_boot_random_indices])
            block_bootstrap_boots[i] = temp[i]
            cf=np.nanmean(temp[0:i+1], axis=0)
            cf_err=np.nanstd(temp[0:i+1], axis=0)
            self._cfs[name].set_cf(cf, cf_err, iterations=bootstrap_boots)
            if (save_steps_file is not None):
                self.save_cfs(save_steps_file, cfkeys=[name])

    #------------------------------------------------------------------------------------------

    #----------------------------------------------------------------#
    #- Generate the random-random counts required to compute the IC -#
    #----------------------------------------------------------------#
    def generate_rr(self, set_nbins=None, logbins=True, min_sep=0.01, 
                    force_n_randoms=None, save_to=None, n_chunks=1):
        #Do random-random counts over the entire field.  If set_nbins is declared,
        #generate_rr will not go looking for the correlation functions so that the
        #RR counts for the IC calculation and the CF calculation can be done in parallel.
 
        #Figure out how many randoms we need.  This was calculated by playing with
        #the number of randoms in the GOODS-S field and seeing when the RR counts converged
        #to the "way too many" curve.  27860 per 1.43e-5 steradians was what I settled on.
        #If there's a forced number, it will ignore my estimate.
        #  Amendment 4/15- this minimum number seems to be somewhat too small for fields that 
        #                  aren't as smooth as GOODS-S, so I'm multiplying it by 5.  This looks ok.
        #  Amendment 8/15- added the capability to do this in several chunks.
        
        if force_n_randoms is None:
            surface_density_required = 27860.*5./1.43e-5
            area = self._image_mask.masked_area_solid_angle()
            number_needed = surface_density_required * area
        else:
            number_needed=force_n_randoms

        #If we're doing more than one chunk, divide the number we need into n_chunks chunks
        if n_chunks > 1:
            number_needed = np.ceil(float(number_needed)/n_chunks).astype(int)
        total_number = number_needed * n_chunks
        print "total number: ",  total_number
        print "number per iteration: ", number_needed
        print "number of chunks: ", n_chunks

        #Range of separations to make bins over
        min_ra = self._ra[self._use].min()
        min_dec = self._dec[self._use].min()
        max_ra = self._ra[self._use].max()
        max_dec = self._dec[self._use].max()
        max_sep=cos.ang_sep(min_ra, min_dec, max_ra, max_dec,
                           radians_in=False, radians_out=False)

        #Choose how many bins
        if set_nbins is None:
            #Get our theta bin info from the CF if we can.  Error if we can't
            if self._theta_bins is None:
                raise ValueError("AngularCatalog.generate_rr says: I need"
                                " either a set number of bins (set_nbins=N)"
                                " or thetas from a CF to extrapolate. "
                                " You have given me neither.")
            centers, edges = self._cfs[name].get_thetas(unit='degrees')
            nbins= np.ceil( len(centers) * 2. * max_sep/edges.max())
        else:
            nbins=set_nbins

        #Make the bins
        rr_theta_bins = binclass.ThetaBins(min_sep, max_sep, nbins,
                                           unit='d', logbins=logbins)
        use_centers, use_theta_bins = rr_theta_bins.get_thetas(unit='degrees')

        #Do the loop
        G_p=np.zeros(nbins)
        rr_counts=np.zeros(nbins)
        for n_i in np.arange(n_chunks):
            print "doing chunk #", n_i
            #Remake the random sample so we're sure we have the right oversample factor            
            self.generate_random_sample(masked=True, make_exactly=number_needed)
        
            #Code snippet shamelessly copied from astroML.correlations
            xyz_data = corr.ra_dec_to_xyz(self._ra_random,
                                         self._dec_random)
            data_R = np.asarray(xyz_data, order='F').T
            bins = corr.angular_dist_to_euclidean_dist(use_theta_bins)
            Nbins = len(bins) - 1
            counts_RR = np.zeros(Nbins + 1)
            for i in range(Nbins + 1):
                counts_RR[i] = np.sum(self._random_tree.query_radius(data_R, bins[i],
                                                                            count_only=True))
            rr = np.diff(counts_RR)
            #Landy and Szalay define G_p(theta) as <N_p(theta)>/(n(n-1)/2)
            G_p += rr/(number_needed*(number_needed-1)) 
            rr_counts += rr

        print "Dividing out the theta bin sizes and number of chunks"
        
        #I divide out the bin width because just using the method
        #that L&S detail gives you a {G_p,i} with the property that
        #Sum[G_p,i]=1.  This is not equivalent to Integral[G_p d(theta)]=1,
        #which is what they assume everywhere else.
        #Dividing out the bin width gives you that and lets you pretend
        #G_p is a continuous but chunky-looking function.
        G_p /= np.diff(use_theta_bins)                    
        G_p /= n_chunks                                   
        self._rr_ngals=[total_number, n_chunks]
        self._Gp = gpclass.Gp(min_sep, max_sep, nbins, G_p, total_number,
                              n_chunks, logbins=logbins, unit='d',
                              RR=rr_counts)

        if save_to is not None:
            self.save_gp(save_to)
        
    #------------------------------------------------------------------------------------------

    #-----------------------------------------------------------#
    #- Determine the integral constraint given a power law fit -#
    #-----------------------------------------------------------#
    def one_shot_integral_constraint(self, which_cf='all', A=None, beta=None, random_oversample=100, set_nbins=None, logbins=True, min_sep=0.01, allow_powerlaw_offset=False, use_thetas=None):
        #Calculates the integral constraint given A and beta (as defined in 
        #the structure for each estimator or as in the keywords)
        # self._powerlaw_A
        # self._powerlaw_beta

        #Make use_thetas an all-true mask if it's none
        if use_thetas is None:
            use_thetas = ma.masked_greater(self._cf_thetas, -100.).mask
        
        #Make sure we have a valid which_cf
        if which_cf not in ['all', 'cf', 'bootstrap', 'block_bootstrap', 'jackknife']:
            raise ValueError("one_shot_integral_constraint says: You have given me an invalid value of which_cf.  Your options are ['all', 'cf', 'bootstrap', 'block_bootstrap', 'jackknife']")
        
        #Figure out which ones I'm actually doing
        calculate_keys=[]  #The list of keys to the powerlaw and IC dictionaries that I'll be working with

        have_cf = self._cf is not None
        have_bootstrap = self._bootstrap_cf is not None
        have_block_bootstrap = self._block_bootstrap_cf is not None
        have_jackknife = self._jackknife_cf is not None

        #add all the names that I both have and want to calculate to the list
        if (which_cf == 'cf') or (which_cf == 'all'):
            if have_cf:
                calculate_keys.append('cf')
        if (which_cf == 'bootstrap') or (which_cf == 'all'):
            if have_bootstrap:
                calculate_keys.append('bootstrap')
        if (which_cf == 'block_bootstrap') or (which_cf == 'all'):
            if have_block_bootstrap:
                calculate_keys.append('block_bootstrap')
        if (which_cf == 'jackknife') or (which_cf == 'all'):
            if have_jackknife:
                calculate_keys.append('jackknife')

        #Make sure we have something to do
        if len(calculate_keys)==0:
            raise ValueError("one_shot_integral_constraint says: You have chosen a CF estimate that has not been run.  Try again.")

        #See if we have the RR counts already or if we have to generate them
        if self._rr_counts is None:
            generate_rr=True
        else:
            generate_rr=False

        if generate_rr:
            self.generate_rr(set_nbins=set_nbins, random_oversample=random_oversample, logbins=logbins, min_sep=min_sep)
            
        rr=self._rr_counts
        thetas=self._thetas_for_rr
            
        #Now do all the estimates that I want
        for k in calculate_keys:
            #Do just the plain old CF without errors
            #First get the values of A and beta
            #Check if they're given- that takes priority.  If there isn't any, recalculate by setting to None.
            if A is not None:
                self._powerlaw_A[k]=A
            if beta is not None:
                self._powerlaw_beta[k]=beta

            #If they're not given and we don't have one or both, run a fit
            if (self._powerlaw_A[k] is None) or (self._powerlaw_beta[k] is None):
                # print "am I even doing anything?  yes, something"
                #Pick out the CF and error for this method
                if k == 'cf':
                    cf = self._cf
                    error = None
                elif k == 'bootstrap':
                    cf = self._bootstrap_cf
                    error = self._bootstrap_cf_err
                elif k == 'block_bootstrap':
                    cf = self._block_bootstrap_cf
                    error = self._block_bootstrap_cf_err
                else:
                    cf = self._jackknife_cf
                    error = self._jackknife_cf_err
                    
                print use_thetas
                if allow_powerlaw_offset:
                    # print "in if"
                    A, beta, offset= self.fit_power_law(self._cf_thetas[use_thetas], cf[use_thetas], 
                                                        error[use_thetas], fixed_A=self._powerlaw_A[k],
                                                        fixed_beta=self._powerlaw_beta[k], allow_offset=True)
                    self._powerlaw_A[k]=A
                    self._powerlaw_beta[k]=beta
                    self._powerlaw_offset[k]=offset
                    # print A, beta, offset
                else:
                    # print "in else"
                    A, beta= self.fit_power_law(self._cf_thetas[use_thetas], cf[use_thetas], 
                                                error[use_thetas], fixed_A=self._powerlaw_A[k],
                                                fixed_beta=self._powerlaw_beta[k], allow_offset=False)
                    self._powerlaw_A[k]=A
                    self._powerlaw_beta[k]=beta
                    offset=0
                    self._powerlaw_offset[k]=offset
                    # print A, beta, offset

            #Calculate the integral constraint
            print "A=", A, "beta=", beta, "offset=", offset
            rr_times_powerlaw = rr *  (A * (thetas ** (-beta)) + offset)
            IC= sum(rr_times_powerlaw)/sum(rr)

            #Store it (if the iterative routine is calling this function, 
            # it'll get overwritten on the next iteration)
            self._IC[k]=IC

    #------------------------------------------------------------------------------------------

    #----------------#
    #- Compute bias -#
    #----------------#
    def bias_bootstrap(self, redshift_range):
        #Calculate the bias from the CF and IC (single gal bootstrap for now)

        #Pull in the things that I'll need for the fit
        zmin = redshift_range.min()
        zmax = redshift_range.max()
        x = self._cf_thetas
        y = self._bootstrap_cf - self._IC['bootstrap']
        err = self._bootstrap_cf_err

        #Get a power law fit to the IC corrected CF
        A, beta=self.fit_power_law(x, y, err)

        #Define the redshift window
        def Nz(z):
            if (z<zmin) or (z>zmax):
                return 0
            else:
                return 1

        #Invert the Limber equation to get the 3D params
        r0, gamma = t.inverted_limber_equation(Nz, A, beta, minimization_tol=1.e-5, ignore_failure=False)

        #Call the bias
        z=(zmin+zmax)/2.
        self._bias = t.bias(r0, gamma, z)
        
    #------------------------------------------------------------------------------------------

    #---------------------------------------------#
    #- Fit a correlation function to a power law -#
    #---------------------------------------------#
    def fit_power_law(self, thetas, cf, cf_err, fixed_A=None, fixed_beta=None, return_covariance=False, allow_offset=False):
        #This is a grunt function, mainly intended for use by the integral constraint function
        print "thetas to fit:"
        print thetas
        print "CF to fit:"
        print cf
        #Figure out what we're fitting for and act accordingly
        if (fixed_A is None) and (fixed_beta is None):
            #I'm fitting both A and beta
            #Define my function
            if allow_offset:
                def powerlaw(theta, A, beta, offset):
                    return A*(theta**(-beta)) + offset
            else:
                def powerlaw(theta, A, beta):
                    return A*(theta**(-beta))
            #Fit to the function
            best_vals, covariance = opt.curve_fit(powerlaw, thetas, cf, sigma=cf_err, absolute_sigma=True)
            if allow_offset:
                A, beta, offset = best_vals 
            else:
                A, beta = best_vals 
        elif (fixed_A is None):
            #This case means we have a fixed beta but not A
            #Define the function
            if allow_offset:
                def powerlaw(theta, A, offset):
                    return A*(theta**(-fixed_beta)) + offset
            else:
                def powerlaw(theta, A):
                    return A*(theta**(-fixed_beta))
            #Fit to the function
            beta = fixed_beta
            best_vals, covariance = opt.curve_fit(powerlaw, thetas, cf, sigma=cf_err, absolute_sigma=True)
            A = best_vals[0]
            if allow_offset:
                offset=best_vals[1]
            
        elif (fixed_beta is None):
            #We have a fixed A but not beta
            #Define the function
            if allow_offset:
                def powerlaw(theta, beta, offset):
                    return fixed_A*(theta**(-beta)) + offset
            else:
                def powerlaw(theta, beta):
                    return fixed_A*(theta**(-beta))
            #Fit to the function
            A=fixed_A
            best_vals, covariance = opt.curve_fit(powerlaw, thetas, cf, sigma=cf_err, absolute_sigma=True)
            beta = best_vals[0]
            if allow_offset:
                offset=best_vals[1]
            
        else:
            #Both are fixed- this is not fittable (yes, I made that a word)
            print "fit_power_law says:  You have called me with both parameters fixed.  There's nothing to fit"
            raise ValueError("No free parameters")

        #Return the fit values
        returning=[A, beta]
        if allow_offset:
            returning.append(offset)
        if return_covariance:
            returning.append(covariance)
            
        return returning
            
        
    #------------------------------------------------------------------------------------------

    #-------------------------------------#
    #- Read in previously calculated CFs -#
    #-------------------------------------#
    def load_cf(self, filen, overwrite_existing=False, name_prefix=''):
        #Load in a CF from a file or set of files

        #First, what files start with filen?
        file_list = misc.files_starting_with(filen)
        nfiles = len(file_list)

        #Generate the names
        names = copy.copy(file_list)
        for i, n in names:
            names[i] = name_prefix + n.lstrip(filen)
        
    #------------------------------------------------------------------------------------------

    #--------------------------------------------#
    #- Save the correlation functions to a file -#
    #--------------------------------------------#
    def save_cf(self, file_base, cf_keys=None):
        #Takes all the CF information we have and saves to a file
        #per CF

        #If they didn't say which ones specifically, save all
        if cf_keys is None:
            cf_keys=self._cfs.keys()

        for k in cf_keys:
            filen = file_base + k
            self._cfs[k].save(filen)
        
    #------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------#
    #- Read in previously calculated random-random counts for the IC -#
    #-----------------------------------------------------------------#
    def load_gp(self, filename, overwrite_existing=False):
        #Take the ASCII files with the normed random-random counts calculated and read it in

        if (self._Gp is None) or overwrite_existing:
            self._Gp = gpclass.Gp.from_file(filename)
        else:
            print ("angular_catalog.load_rr says: You've asked me not "
                   "to overwrite the existing RR counts and there's "
                   "already Gp information .")

    #------------------------------------------------------------------------------------------

    #--------------------------------------------#
    #- Save the random-random counts for the IC -#
    #--------------------------------------------#
    def save_gp(self, filename):
        #If we have done the random-random counts for the integral
        #constraint, save to a file
        self._Gp.save(filename)
        
