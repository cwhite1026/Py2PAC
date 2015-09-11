#External code
import numpy as np
import numpy.ma as ma
import scipy.integrate as intg
import numpy.random as rand
import astropy.wcs as wcs
import astropy.io.fits as fits
from copy import deepcopy

#Py2PAC code
import miscellaneous as misc
import cosmo as cos
import image_mask_cybits as cybits

class ImageMask:
    #------------------#
    #- Initialization -#
    #------------------#
    def __init__(self, angular_catalog=None, nx_pix=10, ny_pix=10, weight_file=None, forced_ra_range=None, forced_dec_range=None):
        #If we have a weight file to make the mask from, read it in and make mask
        #Forced RA and Dec ranges only used if we don't have a weight file.
        if weight_file is not None:
            #Get the number of x and y pixels and the image mask from the cython routine
            self._nx_pixels, self._ny_pixels, self._approx_frac_nonzero, self._mask=cybits.make_mask_from_weights(weight_file)
            
            #Make a WCS instance and get useful things from it
            self._wcs_instance=wcs.WCS(weight_file)
            #The astropy WCS library changed from one of these to the other.  
            #This makes sure older versions of astropy don't complain
            try:
                image_corners=self._wcs_instance.calc_footprint()
            except AttributeError:
                image_corners=self._wcs_instance.calcFootprint()
            self._ra_range=[image_corners[:,0].min(), image_corners[:,0].max()]
            self._dec_range=[image_corners[:,1].min(), image_corners[:,1].max()]
            
        #If not, make some things up, basically
        else:
            self._wcs_instance=None
            self._nx_pixels=nx_pix
            self._ny_pixels=ny_pix
            self._mask=np.ones([nx_pix, ny_pix]) #make an dummy mask
            self._approx_frac_nonzero=1.

            if angular_catalog is not None:
                if forced_ra_range is None:
                    self._ra_range=angular_catalog._ra_range
                else:
                    self._ra_range=forced_ra_range
                    
                if forced_dec_range is None:
                    self._dec_range=angular_catalog._dec_range
                else:
                    self._dec_range=forced_dec_range
            else:
                self._ra_range=forced_ra_range
                self._dec_range=forced_dec_range
                if (forced_ra_range is None) or (forced_dec_range is None):
                    print "image_mask.__init__ says: You will need to define ranges in RA and Dec before using this object"
                    
            self._ra_bin_size, self._ra_edges=self.make_bin_edges(self._ra_range, nx_pix)
            self._dec_bin_size, self._dec_edges=self.make_bin_edges(self._dec_range, ny_pix)


        #Set up some holders for things we might use later
        self._subregion_rotation=None
        self._subregion_rotated_xedges=None
        self._subregion_rotated_yedges=None

    #------------------------------------------------#
    #- Allow the initiation of a mask with an array -#
    #------------------------------------------------#
    @classmethod
    def mask_explicitly(cls, mask, ra_range, dec_range):
        #If we want to manually set the image mask, this is how we do it.
        #You'll call this guy with 
        #  immask=image_mask.mask_explicitly(mask_array, ra_range, dec_range)
        x_pix, y_pix = mask.shape

        immask=cls(nx_pix=x_pix, ny_pix=y_pix, forced_ra_range=ra_range, forced_dec_range=dec_range)
        immask._mask=mask
        return immask
    
        
    #---------------------------#
    #- Edge definition routine -#
    #---------------------------#
    def make_bin_edges(self, rng, n_pix):
        #This guy will return the bin edges given a particular range and number of bins
        bin_size=np.diff(rng)[0]/n_pix
        return bin_size, np.arange(rng[0], rng[1]+bin_size, bin_size)

    #----------------------------------------------------------------#
    #- Returns the solid angle subtended by the "true" region in sr -#
    #----------------------------------------------------------------#
    def masked_area_solid_angle(self):
        #Returns the solid angle subtended by the masked region- just the
        #part where data is allowed

        if self._wcs_instance is not None:
            #Figure out how many effective pixels the image covers (I say effective because
            #the image could be rotated on the sky.  We want to know how many pixels
            #an image that covers the entire RA and Dec range would be)
            ra_list=[self._ra_range[0], self._ra_range[1], self._ra_range[0], self._ra_range[1]]
            dec_list=[self._dec_range[0], self._dec_range[0], self._dec_range[1], self._dec_range[1]]
            x, y=self.ra_dec_to_xy(ra_list, dec_list)

            #Now pretend it's a trapezoid to get the total area
            npix_top = y[2:].max() - y[2:].min()
            npix_bottom = y[0:2].max() - y[0:2].min()
            if npix_top != npix_bottom:
                print "masked_area_solid_angle says: WARNING!  There is enough distortion from the projection to a flat image over the RA and Dec ranges that this mask spans that the number of pixels across the RA span at the top and bottom are different.  There may be distortion effects strong enough to mess with things.  There are", npix_top, "pixels at the max Dec and", npix_bottom, "pixels at the min Dec."
            npix_h = x.max() - x.min()
            total_npix = (npix_top+npix_bottom) * npix_h / 2.

            #How many pixels are in the valid part of the image?
            true_false=np.ceil(self._mask)  #This accounts for non-binary masking
            npix_true=np.sum(true_false)
        else:
            npix_true=total_npix=1.
            

        #What solid angle is covered by the whole RA and Dec range?
        #Convert the RA and Decs to theta and phi (in radian)
        theta_range= - np.radians(self._dec_range) + np.pi/2. #Dec is pi/2 at theta=0 and -pi/2 at theta=pi
        theta_range.sort()
        phi_range= np.radians(self._ra_range)
        phi_range.sort()
    
        #Do the integral
        phi_int, phi_int_err= intg.quad(lambda phi: 1, phi_range[0], phi_range[1])
        theta_int, theta_int_err= intg.quad(lambda theta: np.sin(theta), theta_range[0], theta_range[1])
        solid_angle = phi_int * theta_int

        #return the fraction of the solid angle that's covered by the true part
        return solid_angle * npix_true/total_npix
        

    #-----------------------------------------------#
    #- Returns the x-y coords of given RA and Decs -#
    #-----------------------------------------------#
    def ra_dec_to_xy(self, ra, dec):
        #Given a list of RA and Dec, it returns the XY position on the image
        pairs=np.transpose([ra, dec])
        positions=self._wcs_instance.wcs_world2pix(pairs, 0)
        x=np.array(positions[:,1])
        y=np.array(positions[:,0])
        return x, y

    #----------------------------------------------#
    #- Returns the RA and dec of given x-y coords -#
    #----------------------------------------------#
    def xy_to_ra_dec(self, x, y):
        #Given a list of x and y positions, returns RA and Dec
        pairs=np.transpose([y, x])
        positions=self._wcs_instance.wcs_pix2world(pairs,0)
        ra=np.array(positions[:,0])
        dec=np.array(positions[:,1])
        return ra, dec

    #-----------------------------------------------#
    #- Translate and/or rotate the mask on the sky -#
    #-----------------------------------------------#
    def move_mask_on_sky(self, delta_ra=None, delta_dec=None, theta_degrees=None, preview=False):
        #Move the mask around on the sky by changing the parameters in the WCS instance

        #If we're not going to keep the mask this way, store the original values
        if preview:
            # original_self=deepcopy(self)
            original_crval=deepcopy(self._wcs_instance.wcs.crval)
            original_cd=deepcopy(self._wcs_instance.wcs.cd)
            original_ra_range=deepcopy(self._ra_range)
            original_dec_range=deepcopy(self._dec_range)

        #If we have a change in RA, do it
        if delta_ra is not None:
            self._wcs_instance.wcs.crval[0] += delta_ra

        #If we have a change in Dec, do it
        if delta_dec is not None:
            self._wcs_instance.wcs.crval[1] += delta_dec

        #If we have a rotation, apply it
        if theta_degrees is not None:
            sine = np.sin( np.radians(theta_degrees))
            cosine = np.cos(np.radians(theta_degrees))
            rotation_matrix=np.array([[cosine, -sine] , [sine, cosine]])
            self._wcs_instance.wcs.cd = np.dot(rotation_matrix, self._wcs_instance.wcs.cd)

        #Update the corners
        #The astropy WCS library changed from one of these to the other.  
        #This makes sure older versions of astropy don't complain
        try:
            image_corners=self._wcs_instance.calc_footprint()
        except AttributeError:
            image_corners=self._wcs_instance.calcFootprint()
        self._ra_range=[image_corners[:,0].min(), image_corners[:,0].max()]
        self._dec_range=[image_corners[:,1].min(), image_corners[:,1].max()]
        
        #If this is a preview, send the altered instance back and switch things back
        if preview:
            to_return=deepcopy(self)
            # self=original_self
            self._wcs_instance.wcs.crval=original_crval
            self._wcs_instance.wcs.cd=original_cd
            self._ra_range=original_ra_range
            self._dec_range=original_dec_range
            return to_return
        
        
    #------------------------------------------#
    #- Queries completeness for given catalog -#
    #------------------------------------------#
    def return_completenesses(self, ra_list, dec_list):
        #Take a list of RAs and Decs and return completenesses for them
        if self._wcs_instance is not None:
            #Get the pixel numbers for all the objects
            pairs=np.transpose([ra_list, dec_list])
            float_inds=self._wcs_instance.wcs_world2pix(pairs, 0)
            indices=np.asarray(float_inds, dtype=np.int)
            xinds=indices[:,1]
            yinds=indices[:,0]

            #Figure out which things are inside the image
            nx, ny= self._mask.shape
            # print nx, ny
            inside_x=ma.masked_inside(xinds, 0, nx-1).mask
            inside_y=ma.masked_inside(yinds, 0, ny-1).mask
            in_image= inside_x & inside_y
            print "I have ", len(xinds[in_image]), " points that are actually on the image"

            #Now make the completeness array
            complete=np.zeros(len(xinds))
            if np.asarray(in_image).any():
                complete[in_image]=self._mask[xinds[in_image], yinds[in_image]]
            return complete
        
        else:
            if len(ra_list) != len(dec_list):
                print "ERROR: image_mask.return_completenesses says: You have given me a different number of RAs and Decs.  I can't work with that."
                return

            n_objects=len(ra_list)
        
            #make a holder for the completenesses
            complete=np.ones(n_objects, dtype=float)

            #Figure out which bin each guy belongs in
            ra_inds=np.zeros(n_objects, dtype=np.int)
            dec_inds=np.zeros(n_objects, dtype=np.int)
            ra_min=self._ra_range[0]
            # ra_max=self._ra_range[1]
            dec_min=self._dec_range[0]
            # dec_max=self._dec_range[1]
            print "return_completenesses says: looping through all the objects to find their completenesses"
            for i in range(n_objects):
                #Find the ra bin
                ra_inds[i]=np.int(np.floor((ra_list[i]-ra_min)/self._ra_bin_size))
                if (ra_list[i]-ra_min)/self._ra_bin_size == self._nx_pixels:
                    #If we're exactly at the maximum RA, bump it down to the top bin
                    ra_inds[i]=np.int(self._nx_pixels - 1)
                #Find the dec bin
                dec_inds[i]=np.int(np.floor((dec_list[i]-dec_min)/self._dec_bin_size))
                if (dec_list[i]-dec_min)/self._dec_bin_size == self._ny_pixels:
                    #If we're exactly at the maximum DEC, bump it down to the top bin
                    dec_inds[i]=np.int(self._ny_pixels - 1)
                #Store the completeness
                if (ra_inds[i]<0) or (dec_inds[i]<0) or (ra_inds[i]>=self._nx_pixels) or (dec_inds[i]>=self._ny_pixels):
                    #If we're here, then the RA or Dec is outside the range of our mask.
                    #Set completeness to 0
                    complete[i]=0.0
                    # print "return_completenesses says: WARNING- you have asked for a coordinate outside of the mask.  0 completeness returned for this point."
                else:
                    complete[i]=self._mask[ra_inds[i], dec_inds[i]]
                    
            return complete


    #---------------------------------#
    #- Sets up subregion information -#
    #---------------------------------#
    def set_subregions(self, theta, xedges, yedges):
        self._subregion_rotation=theta
        self._subregion_rotated_xedges=xedges
        self._subregion_rotated_yedges=yedges
        return

    
    #-----------------------------------------------#
    #- Returns subregions for a list of RA and Dec -#
    #-----------------------------------------------#
    def return_subregions(self, ra, dec, theta=None, rot_xedges=None, rot_yedges=None, hypothetical=False):
        #This returns the subregion number for each pair of RA and Dec.  The hypothetical option
        #calculates it but doesn't store the theta or edge info.

        #Check to make sure we have what we need, pull to local if we have stored values but no given values
        if (theta is None):
            if (self._subregion_rotation is None):
                print "image_mask.return_subregions says: ERROR!  I don't have the rotation angle.  Please provide one."
                return
            else:
                theta=self._subregion_rotation
                
        if (rot_xedges is None):
            if (self._subregion_rotated_xedges is None):
                print "image_mask.return_subregions says: ERROR!  I don't have rotated x edges.  Please provide them."
                return
            else:
                rot_xedges=self._subregion_rotated_xedges
                
        if (rot_yedges is None):
            if (self._subregion_rotated_yedges is None):
                print "image_mask.return_subregions says: ERROR!  I don't have rotated y edges.  Please provide them."
                return
            else:
                rot_yedges=self._subregion_rotated_yedges

        #If this isn't a hypothetical scenario, store any values we've been given
        if not hypothetical:
            if theta is not None:
                self._subregion_rotation=theta
            if rot_xedges is not None:
                self._subregion_rotated_xedges=rot_xedges
            if rot_yedges is not None:
                self._subregion_rotated_yedges=rot_yedges

        #Now that we know we have everything, put the ra and decs into x and y coords
        x1, y1=self.ra_dec_to_xy(ra, dec)

        #Transform to the rotated coordinate system
        x2, y2=u.rotate_coords(x1, y1, theta)

        #Now make masks for each row and column
        nx=len(rot_xedges)-1
        # print nx, rot_xedges
        ny=len(rot_yedges)-1
        # print ny, rot_yedges
        ymasks={}
        xmasks={}
        for i in range(nx):
            xmasks[i]=ma.masked_inside(x2, rot_xedges[i], rot_xedges[i+1]).mask
        for i in range(ny):
            ymasks[i]=ma.masked_inside(y2, rot_yedges[i], rot_yedges[i+1]).mask

        #Now use the masks to put numbers to each galaxy
        #No subregion defaults to -1
        subregion=-np.ones(len(ra))
        for ix in range(nx):
            for iy in range(ny):
                bin_number= nx*iy + ix
                thismask = xmasks[ix] & ymasks[iy]
                # print "bin ", bin_number
                # print "nguys: ", len(subregion[thismask])
                subregion[thismask]=bin_number

        #We have what we wanted.  Return the subregion numbers
        return subregion
