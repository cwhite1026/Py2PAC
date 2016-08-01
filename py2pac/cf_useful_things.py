#Various convenience routines
import numpy as np
import math as mth
import struct as st
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
import fitting as fit
import cosmo as cos
import matplotlib.pyplot as plt
import angular_catalog as ac
import multiz as mz
import subprocess
import os
import image_mask as im
import bias_tools as t

arcsec_opts = ['arcseconds', 'arcsecond', 'arcsec', 'a']
radian_opts = ['radians', 'radian', 'rad', 'r']
degree_opts = ['degrees', 'degree', 'deg', 'd']

#==========================================================================
#==========================================================================
#==========================================================================
def bootstrap_biases_to_props(boot_biases, zbin, z_actual):

    #Get stats from uncorrected biases and see if we have anything in this
    #bin
    temp = np.percentile(boot_biases, (16, 50, 84))
    uncorr_bias_low, uncorr_bias_median, uncorr_bias_high = temp
    
    #Only do other things if we have a not -1 bias
    if uncorr_bias_median <= 0:
        uncorr_bias_median, uncorr_bias_lowerr, uncorr_bias_higherr= -1,0,0
        corr_bias_median, corr_bias_lowerr, corr_bias_higherr= -1,0,0
        uncorr_mass_median, uncorr_mass_lowerr, uncorr_mass_higherr= -1,0,0
        corr_mass_median, corr_mass_lowerr, corr_mass_higherr= -1,0,0        
        
    else:
        #We have OK biases: correct the biases
        corrected_boot_biases = t.apply_zbinned_linear_bias_correction(boot_biases, zbin)
    
        #Get stats from corrected biases
        temp = np.percentile(corrected_boot_biases, (16, 50, 84))
        corr_bias_low, corr_bias_median, corr_bias_high = temp
    
        uncorr_bias_lowerr = uncorr_bias_median - uncorr_bias_low
        uncorr_bias_higherr = uncorr_bias_high - uncorr_bias_median
        corr_bias_lowerr = corr_bias_median - corr_bias_low
        corr_bias_higherr = corr_bias_high - corr_bias_median
    
        #Convert both sets to masses
        n_boots = len(boot_biases)
        boot_masses = -np.ones(n_boots)
        corrected_boot_masses = -np.ones(n_boots)
        for i in np.arange(n_boots):
            try:
                boot_masses[i] = np.log10(t.cheating_bias_to_halo_mass(boot_biases[i], z_actual))
            except ValueError:
                print "Couldn't convert uncorrected bias", i, ", ", boot_biases[i]
                
            try:
                corrected_boot_masses[i] = np.log10(t.cheating_bias_to_halo_mass(corrected_boot_biases[i], z_actual))
            except ValueError:
                print "Couldn't convert corrected bias", i, ", ", corrected_boot_biases[i]
                    
        #Get stats for masses
        temp = np.percentile(boot_masses[boot_masses>0], (16, 50, 84))
        uncorr_mass_low, uncorr_mass_median, uncorr_mass_high = temp
        temp = np.percentile(corrected_boot_masses[corrected_boot_masses>0], (16, 50, 84))
        corr_mass_low, corr_mass_median, corr_mass_high = temp
    
        uncorr_mass_lowerr = uncorr_mass_median - uncorr_mass_low
        uncorr_mass_higherr = uncorr_mass_high - uncorr_mass_median
        corr_mass_lowerr = corr_mass_median - corr_mass_low
        corr_mass_higherr = corr_mass_high - corr_mass_median  
    
    return_dict = {}
    return_dict['uncorrected'] = {'bias_median' : uncorr_bias_median, 
                                  'bias_lowerr' : uncorr_bias_lowerr,
                                  'bias_higherr': uncorr_bias_higherr,
                                  'mass_median' : uncorr_mass_median,
                                  'mass_lowerr' : uncorr_mass_lowerr,
                                  'mass_higherr': uncorr_mass_higherr,}
    return_dict['corrected'] = {'bias_median' : corr_bias_median, 
                                'bias_lowerr' : corr_bias_lowerr,
                                'bias_higherr': corr_bias_higherr,
                                'mass_median' : corr_mass_median,
                                'mass_lowerr' : corr_mass_lowerr,
                                'mass_higherr': corr_mass_higherr,}                                  
    return return_dict
    

#==========================================================================
#==========================================================================
#==========================================================================

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


#==========================================================================
#==========================================================================
#==========================================================================
def make_dummy_sham_multicats(bin_names):
    #Make image masks from the ranges in the SHAM
    cosmos_mask = im.image_mask(forced_ra_range=[149.988434, 150.272842],
                               forced_dec_range=[1.978589, 2.662248])
    gn_mask = im.image_mask(forced_ra_range=[188.622421, 189.777161],
                            forced_dec_range=[61.96093, 62.49667])
    gs_mask = im.image_mask(forced_ra_range=[52.761482, 53.503708],
                            forced_dec_range=[-28.159304, -27.469843])
    uds_mask = im.image_mask(forced_ra_range=[34.063538, 34.670387],
                             forced_dec_range=[-5.528365, -4.9406350])
    image_masks = [cosmos_mask, gn_mask, gs_mask, uds_mask]
    
    return make_dummy_multi_catalogs(bin_names, image_masks=image_masks)

#==========================================================================
#==========================================================================
#==========================================================================
def make_dummy_candels_multi_catalogs(bin_names):
    cosmos_weight_file_name = ('/astro/candels1/data/cosmos/mosaics/curren'
                               't/60mas/cos_2epoch_wfc3_f160w_060mas_v1.0_'
                               'wht.fits')
    egs_weight_file_name = ('/astro/candels1/data/egs/mosaics/current/egs_'
                            'all_combined_v1.0/egs_all_wfc3_ir_f160w_060mas'
                            '_v1.0_wht.fits')
    gn_weight_file_name = ('/astro/candels1/data/goodsn/mosaics/current/'
                           'goods_n_all_combined_v1.0/goodsn_all_wfc3_ir_'
                           'f160w_060mas_v1.0_wht.fits')
    gs_weight_file_name = ('/astro/candels1/data/goodss/mosaics/current/'
                           'goods_s_all_combined_v0.5/gs_all_candels_ers_'
                           'udf_f160w_060mas_v0.5_wht.fits')
    uds_weight_file_name = ('/astro/candels1/data/uds/mosaics/current/60'
                            'mas/uds_all_wfc3_f160w_060mas_v1.0_wht.fits')
    weight_files = [cosmos_weight_file_name, egs_weight_file_name,
                    gn_weight_file_name, gs_weight_file_name,
                    uds_weight_file_name]
    rr_files = ['/user/caviglia/research/correlations/rrs/cosmos_rr_gp.txt',
                '/user/caviglia/research/correlations/rrs/egs_rr_gp.txt',
                '/user/caviglia/research/correlations/rrs/gn_rr_gp.txt',
                '/user/caviglia/research/correlations/rrs/gs_rr_gp.txt',
                '/user/caviglia/research/correlations/rrs/uds_rr_gp.txt']
    return make_dummy_multi_catalogs(bin_names, rr_files=rr_files,
                                     weight_files=weight_files)


#==========================================================================
#==========================================================================
#==========================================================================
def make_dummy_multi_catalogs(bin_names, rr_files=None, weight_files=None,
                              image_masks=None):
    #Make multi-catalogs with the right image masks and empty catalogs
    
    #Make sure we have something to make masks from
    if (not weight_files) and (not image_masks):
        raise ValueError("You have to give me either already loaded "
                         "image masks or weight file names to load in "
                         "as masks.")

    #Loop through the weight files or image masks and make a multi-cat for
    #each one
    cats =[]
    if weight_files:
        for wf in weight_files:
            cats.append(mz.multi_catalog([0], [0], weight_file=wf))
    else:
        for immsk in image_masks:
            cats.append(mz.multi_catalog([0], [0], image_mask=immsk))

    #Make sure we have the same number of cats and bin names
    if rr_files:
        if len(cats) != len(rr_files):
            raise ValueError("I need one RR file for each mask.  You have "
                            "given me "+str(len(cats))+" masks and " +
                             str(len(rr_files))+" RR files")
            
    #Now loop through our catalogs and make an empty angular catalog for
    #each of the bin names.  They'll need to have the right image mask too
    for i, cat in enumerate(cats):
        for bn in bin_names:
            cat._catalogs[bn]= ac.angular_position_catalog([0], [0],
                                            image_mask = cat._image_mask)
            if rr_files:
                cat._catalogs[bn].load_rr(rr_files[i])
    
    return cats
    
#==========================================================================
#==========================================================================
#==========================================================================
def make_dummy_candels_catalogs_with_masks():
    #Make dummy catalogs with the CANDELS image masks and RRs.

    #Define all the file names for the fits files used for the masks
    cosmos_weight_file_name = ('/astro/candels1/data/cosmos/mosaics/curren'
                               't/60mas/cos_2epoch_wfc3_f160w_060mas_v1.0_'
                               'wht.fits')
    egs_weight_file_name = ('/astro/candels1/data/egs/mosaics/current/egs_'
                            'all_combined_v1.0/egs_all_wfc3_ir_f160w_060mas'
                            '_v1.0_wht.fits')
    gn_weight_file_name = ('/astro/candels1/data/goodsn/mosaics/current/'
                           'goods_n_all_combined_v1.0/goodsn_all_wfc3_ir_'
                           'f160w_060mas_v1.0_wht.fits')
    gs_weight_file_name = ('/astro/candels1/data/goodss/mosaics/current/'
                           'goods_s_all_combined_v0.5/gs_all_candels_ers_'
                           'udf_f160w_060mas_v0.5_wht.fits')
    uds_weight_file_name = ('/astro/candels1/data/uds/mosaics/current/60'
                            'mas/uds_all_wfc3_f160w_060mas_v1.0_wht.fits')

    #Make a catalog with one galaxy (randomly chosen from the actual cats)
    cos = ac.angular_position_catalog([150.091626], [2.2377658],
                                      weight_file=cosmos_weight_file_name)
    egs = ac.angular_position_catalog([150.091626], [2.2377658],
                                      weight_file=egs_weight_file_name)
    gn = ac.angular_position_catalog([189.2794823], [62.18128376],
                                     weight_file=gn_weight_file_name)
    gs = ac.angular_position_catalog([53.1031201], [-27.8554295],
                                     weight_file=gs_weight_file_name)
    uds = ac.angular_position_catalog([34.4579782], [-5.246599],
                                      weight_file=cosmos_weight_file_name)

    #Load in the RR files
    cos.load_rr('/user/caviglia/research/correlations/rrs/cosmos_rr_gp.txt')
    egs.load_rr('/user/caviglia/research/correlations/rrs/egs_rr_gp.txt')
    gn.load_rr('/user/caviglia/research/correlations/rrs/gn_rr_gp.txt')
    gs.load_rr('/user/caviglia/research/correlations/rrs/gs_rr_gp.txt')
    uds.load_rr('/user/caviglia/research/correlations/rrs/uds_rr_gp.txt')
    
    return cos, egs, gn, gs, uds

#==========================================================================
#==========================================================================
#==========================================================================
def make_dummy_candels_catalogs_without_masks():
    #Make empty catalogs for the five fields and read in the RRs
    cos=ac.angular_position_catalog([0], [0])
    cos.load_rr('/user/caviglia/research/correlations/rrs/cosmos_rr_gp.txt')
    egs=ac.angular_position_catalog([0], [0])
    egs.load_rr('/user/caviglia/research/correlations/rrs/egs_rr_gp.txt')
    gn=ac.angular_position_catalog([0], [0])
    gn.load_rr('/user/caviglia/research/correlations/rrs/gn_rr_gp.txt')
    gs=ac.angular_position_catalog([0], [0])
    gs.load_rr('/user/caviglia/research/correlations/rrs/gs_rr_gp.txt')
    uds=ac.angular_position_catalog([0], [0])
    uds.load_rr('/user/caviglia/research/correlations/rrs/uds_rr_gp.txt')

    return cos, egs, gn, gs, uds

#==========================================================================
#==========================================================================
#==========================================================================
def get_key_vals_from_bin_name(bin_name, key):
    part_with_vals = bin_name.split(key)[1]
    val_range = np.array(part_with_vals.split("_")[0:2]).astype(float)
    return val_range

#==========================================================================
#==========================================================================
#==========================================================================
def even_bins(data, nbins, bin_range=None):
    """
    Takes a set of values and returns the bin edges that would put the same
    number of objects in every bin.  For instance, if you had a bunch of
    galaxies with redshifts and wanted to know what redshift bins would
    give you the same number of galaxies in each bin, you would pass
    even_bins(redshifts, number_of_bins).

    Parameters
    ----------
    data: array-like
        The set of values you have for the property you want to bin in
    nbins: integer
        How many bins to divide the data into
    bin_range: array-like
        The range over which you want bins.

    Returns
    -------
    bin_edges: 1D numpy ndarray
        The position of the bin edges that gives you as close to the same
        number of objects per bin as possible.  The evenness of the bins
        depends on the data- if there are 100 of the same value, they have
        to all go in the same bin regardless of whether or not that throws
        off the count.
    """
    #Make sure things are the right format
    data = np.array(data)
    if nbins != int(nbins):
        raise ValueError("You must set the number of bins to be an integer"
                         ".  You have given me something that is not.")
    nbins=int(nbins)
    if bin_range is None:
        bin_range=np.array([data.min(), data.max()])
    
    #Figure out which of the data are in our range
    in_range = ma.masked_inside(data, bin_range[0], bin_range[1]).mask
    use_data = data[in_range]
    
    #How many points are we aiming for per bin?
    npoints = len(use_data)
    points_per_bin = np.floor(npoints/nbins).astype(int)
        
    #How many are left over?
    extra = npoints - points_per_bin * nbins

    #Prep for dividing into bins
    use_data.sort()
    bin_edges=np.zeros(nbins+1)
    bin_edges[0] = bin_range[0]
    bin_edges[-1] = bin_range[1]
    index = 0
    force_include_next_time=False

    #Find bin edges
    for i in np.arange(nbins-1)+1:
        #See what the next points_per_bin look like
        val_right_side = use_data[index+points_per_bin]
        val_left_side = use_data[index+points_per_bin-1]
        guess_seps = use_data[index:index+points_per_bin]
            
        #Have we landed the next edge in some duplicates?
        if val_right_side == val_left_side:
            #Yep!  See whether it's better to grab in a few extra
            #or to pull back a few
            number_already_in = ma.masked_not_equal(guess_seps,
                                                    val_left_side).count()
            number_outside = ma.masked_not_equal(use_data[index+points_per_bin:],
                                                 val_left_side).count()

            #What's our setup look like?
            seriously_low_on_extras = (-extra > number_already_in)
            even_but_have_extras = (number_already_in == number_outside) and (extra >= 0)
            even_no_extras = (number_already_in == number_outside) and (extra < 0)
            more_in = (number_already_in > number_outside)
            more_out = (number_already_in < number_outside)

            #Decide what to do based on how things worked out
            exclude = more_out or even_no_extras or seriously_low_on_extras
            include = more_in or even_but_have_extras

            #Actually do what we decided
            if include or force_include_next_time:
                #Include the duplicates
                msk = ma.masked_less_equal(use_data, val_right_side).mask
                extra -= number_outside
            elif exclude:
                #Don't include duplicates
                msk = ma.masked_less(use_data, val_right_side).mask
                extra += number_already_in
            else:
                raise ValueError("What on Earth did Cathy do?  You "
                                 "got a logical option that she didn't "
                                 "think of in even_bins")

            #Reset the force include flag
            force_include_next_time=False
                
            #Figure out the index that's the last included value in this bin
            if msk.any():
                inds_left_of_edge = np.arange(npoints)[msk]
                last_ind_in = inds_left_of_edge[-1]
                index = last_ind_in +1
                bin_edges[i] = (use_data[last_ind_in] + use_data[index])/2.
            else:
                #We've logicked our way to an empty bin.  That may
                #be something that happens- it just doesn't like
                #having nbins bins.  Let that be ok
                bin_edges[i] = bin_edges[i-1]
                index += points_per_bin
                force_include_next_time=True
        else:
            #We landed between values of separation.  Set the bin
            #edge to be the average of the separations it falls between
            bin_edges[i] = (val_left_side + val_right_side)/2.
            index += points_per_bin
                
    #now we have our bins- make sure none are 0 width
    bin_edges=np.array(list(set(bin_edges)))
    bin_edges.sort()

    return bin_edges


#==========================================================================
#==========================================================================
#==========================================================================

def make_tophat(low_edge, high_edge):
    def tophat_N(z):
        #Figure out if we were handed a scalar- if we were, make it an 
        #array but make a note that we want a scalar out
        z_is_scalar = False
        try:
            nzs = len(z)
        except TypeError:
            z_is_scalar=True
            nzs=1
            z=np.array([z])

        #Make an array of 0s for z outside of the range and 1 inside
        this=np.zeros(nzs)
        in_hat = ma.masked_inside(z, low_edge, high_edge).mask
        this[in_hat]=1

        #Return a scalar if we only had a scalar z, array otherwise
        if z_is_scalar:
            this=this[0]
        return this
    return tophat_N

#==========================================================================
#==========================================================================
#==========================================================================

def load_candels_cf_set(before_field, after_field, hyphen=False):
    #Takes a before_field (path from / plus any prefix on the file name) and
    #and after_field (anything that comes after the field name) and reads in
    #the CFs to dummy catalogs.
    #  An example would be before_field="/Users/caviglia/Desktop/test_"
    #    after_field="_20bins_ros1.txt"
    #    would read in "/Users/caviglia/Desktop/test_cosmos_20bins_ros1.txt"
    #                  "/Users/caviglia/Desktop/test_egs_20bins_ros1.txt"
    #  and so forth for goodss, goodsn, and uds

    #Start up the catalogs for the 5 fields
    cos=ac.angular_position_catalog([0], [0])
    cos.load_rr('/user/caviglia/research/correlations/rrs/cosmos_rr_gp.txt')
    egs=ac.angular_position_catalog([0], [0])
    egs.load_rr('/user/caviglia/research/correlations/rrs/egs_rr_gp.txt')
    gn=ac.angular_position_catalog([0], [0])
    gn.load_rr('/user/caviglia/research/correlations/rrs/gn_rr_gp.txt')
    gs=ac.angular_position_catalog([0], [0])
    gs.load_rr('/user/caviglia/research/correlations/rrs/gs_rr_gp.txt')
    uds=ac.angular_position_catalog([0], [0])
    uds.load_rr('/user/caviglia/research/correlations/rrs/uds_rr_gp.txt')

    #Now make a list of the fields and pull in the CFs
    fields=[cos, egs, gn, gs, uds]
    if hyphen:
        field_names=['cosmos', 'egs', 'goods-n', 'goods-s', 'uds']
    else:
        field_names=['cosmos', 'egs', 'goodsn', 'goodss', 'uds']
    for i, field in enumerate(fields):
        #Read in the CF
        filen = before_field + field_names[i] + after_field
        field.load_cfs(filen, overwrite_existing=True)

    #Return the fields with the CFs loaded
    return fields

#==========================================================================
#==========================================================================
#==========================================================================

#This was one of the earliest python programs I wrote, so I apologize for the 
#clunkiness.
def read_sext(filen, keyadd='', exceptions_to_keyadd=[], make_numpy=True):
    try:
        tabfile=open(filen, 'r')
        print 'read_sext says: reading ', filen
    except IOError:
        print "read_sext says: You fail- I find no file called ", filen

    #Separate the comments from the content of the table
    comments=[]
    print 'read_sext says: pulling in the header'
    for line in tabfile:
        linetemp=line.rstrip('\n')	#Take off the return characters
        linetemp.lstrip().rstrip()      #Take off leading and trailing whitespace
        if linetemp[0]=='#':
            comments.append(linetemp)
        else:
            break
    #Now we close and open the file to reset our location in the file- it's really fast so
        #it's not a problem
    tabfile.close()
    tabfile=open(filen, 'r')

    
    #Split up the comment lines, take the third thing in the row (it goes #, number, name, description)
    #as the tag for the dictionary
    print 'read_sext says: breaking down comments to get variable names'
    names=[]
    for i in range(len(comments)):
       	comline=comments[i].split(' ')
        #Pull out the entries that are just ' '
        linetemp=[]
        for indx in range(len(comline)):
            if (comline[indx] != '') & (comline[indx] != ' '):
                linetemp.append(comline[indx])
        comline=linetemp
        #Now pull off the white space in each entry
       	for j in range(len(comline)):
            comline[j]=comline[j].lstrip().rstrip()
        names.append(comline[2])
    #If there's an addition to the key names specified, add it
    for i in range(len(names)):
        if names[i] not in exceptions_to_keyadd:
            names[i]=names[i]+keyadd
    print 'read_sext says: the dictionary will have keys ', names


    #Ok, now we've gotten our dictionary keys from the header- time to move on to the
    #body of the file
    #Make a dictionary with the appropriate keys to hold things
    tab={}
    for nm in names:
        tab[nm]=[]
    print 'read_sext says: reading in the body of the table, making the dictionary'
    for line in tabfile:
        linetemp=line.rstrip('\n')	#Take off the return characters
        linetemp.lstrip().rstrip()      #Take off leading and trailing whitespace
        if linetemp[0] == '#':
            #skip all the comments- they're either in the header or irrelevant
            pass
        else:
            tabline=linetemp.split(' ')
            linetemp=[]
            #Here, we're getting rid of empty entries- this happens when you have
            #consecutive spaces
            for indx in range(len(tabline)):
                if (tabline[indx] != '') & (tabline[indx] != ' '):
                    linetemp.append(tabline[indx])
            tabline=linetemp
            #Pull off whitespace and convert to a number, stick the line into the the dictionary
            for j, name in enumerate(names):
                if (tabline[j] == 'inf') or (tabline[j] == 'nan'):
                    tabline[j]='99.'
                if tabline[j] == '-inf':
                    tabline[j]='-99.'
                if ('.' in tabline[j]) or ('e' in tabline[j]) or ('E' in tabline[j]) or (tabline[j]=='inf') or (tabline[j]=='-inf'):
                    tab[name].append(float(tabline[j].lstrip().rstrip()))
                else:
                    tab[name].append(int(tabline[j].lstrip().rstrip()))
            
    if make_numpy:
        #Convert lists into numpy arrays
        print "read_sext says: converting lists into numpy arrays because I'm cool like that"
        for name in names:
            tab[name]=np.array(tab[name])
    
    #Close the file and return our dictionary
    tabfile.close()
    return tab

#==========================================================================
#==========================================================================
#==========================================================================


def line2array(line, separator=' ', arrtype='float'):
    #Takes a line from a file with entries, strips it down and returns it as an array
    temp=line.split(separator)
    
    #Now pull off the white space in each entry
    for j in range(len(temp)):
        temp[j]=temp[j].lstrip().rstrip()

    #Get rid of empty entries
    linetemp=[]
    for indx in range(len(temp)):
        if (temp[indx] != '') & (temp[indx] != ' ') & (temp[indx] != separator):
            linetemp.append(temp[indx])
    temp=linetemp

    #Typecast things
    if arrtype=='float':
        temp=np.array(temp, dtype=float)
    elif arrtype=='int':
        temp=np.array(temp, dtype=int)
        
    return temp

#==========================================================================
#==========================================================================
#==========================================================================

def write_ascii(cat, filen):
    #Take a dictionary (must be something that lends itself to such output)

    ks=cat.keys()
    for k in ks:
        if len(cat[k]) != len(cat[ks[0]]):
            print "write_ascii says: This dictionary is not something I can handle."
            return

    #Open the file
    outfile=open(filen, 'w')
    #Write the top comment line with the key names
    commentstr='#'
    for k in ks:
        commentstr= commentstr+str(k)+' '
    outfile.write(commentstr.rstrip(' ') + '\n')

    #Now do each line
    for i in range(len(cat[ks[0]])):
        thisline=''
        for k in ks:
            thisline=thisline+str(cat[k][i])+' '
        thisline=thisline.rstrip(' ') + '\n'
        outfile.write(thisline)
    #Close the file.  All done.
    outfile.close()

#==========================================================================
#==========================================================================
#==========================================================================

#This is another really old program
def read_numtab(filename, separator=' ', comment_char='#', lastcomment=False, nonames=False, toponlycomments=False, verbose=True):
    #Reads in a table of numbers
    try:
        tabfile=open(filename, 'r')
    except IOError:
        print "read_numtab says: You fail- I find no file called ", filename
    #Separate the comments from the content of the table
    table=[]
    comments=[]
    if toponlycomments:
        #Assume we don't have to go searching in the body of the file for commented out 
        #lines- as soon as we hit data, we quit looking
        for line in tabfile:
            linetemp=line.rstrip('\n')	#Take off the return characters
            linetemp.lstrip().rstrip()      #Take off leading and trailing whitespace
            if linetemp[0]==comment_char:
                comments.append(linetemp)
            else:
                table.append(linetemp)
                break
        for line in tabfile:
            linetemp=line.rstrip('\n')	#Take off the return characters
            table.append(linetemp)
    else:  
        for line in tabfile:
            linetemp=line.rstrip('\n')	#Take off the return characters
            if linetemp[0] == comment_char:
                comments.append(linetemp)
            elif linetemp.lstrip().rstrip() == '':
                pass
            else:
                table.append(linetemp)
    #Split up the table, strip all the whitespace 
    for i in range(len(table)):
       	table[i]=line2array(table[i], separator=separator)
    #If the first line of the table is a list of the column titles but is uncommented, pull it out.  If it
    #isn't, we still need column names, so give them the unimaginative names 'col0', 'col1', etc.
    #Alternatively, if lastcomment is set, use the last comment line for the names, stripped of the comment
    #character
    # print table[0][0]
    # print type(table[0][0]),  type('')
    if nonames:
        names=[]
        for i in range(len(table[0])):
            names.append('col'+str(i))
    elif lastcomment:
        names=comments[len(comments)-1].lstrip(comment_char)    #Snag the last line of the comments
        names=names.split(separator)     #Separate the names
        ntemp=[]
        for i in range(len(names)):
            if names[i].lstrip().rstrip() == '':   #Strip the whitespace but don't take empty guys
                pass
            else:
                ntemp.append(names[i].lstrip().rstrip())
        names=ntemp
    elif type(table[0][0]) == type(''):
        names=table[0]
        table=table[1:]
    else:
        names=[]
        for i in range(len(table[0])):
            names.append('col'+str(i))
    if verbose:
        print "read_numtab says: your table will have keys: ", names
    #Now that our table is a list of lists, it's time to make a dictionary.
    tab={}
    for k in range(len(names)):
       	temp=[]
       	for m in range(len(table)):
            temp.append(table[m][k])
       	tab[names[k]]=temp
    #Convert lists into numpy arrays
    for name in names:
       	tab[name]=np.array(tab[name])
    #Close the file and return our dictionary
    tabfile.close()
    return tab
    


#==========================================================================
#==========================================================================
#==========================================================================

def array_safelog10(arr, zero_fill=-99., return_only_nonzero=False):
    #Returns log10(num) for ok numbers, -99 for anything invalid
    msk=ma.masked_greater(arr, 0.).mask
    if return_only_nonzero:
        res=np.log10(arr[msk])
    else:
        res=np.ones(len(arr))*zero_fill
        res[msk]=np.log10(arr[msk])
    return res


#==========================================================================
#==========================================================================
#==========================================================================

def safelog10(num):
    #Returns log10(num) for ok numbers, -99 for anything invalid 
    if num>0:
        lg=mth.log10(num)
    else:
        lg=-99
    return lg

#==========================================================================
#==========================================================================
#==========================================================================

def regular_coords_2d(xs, ys):
    #Returns 2D arrays with the x and y coords of that pixel

    #First get the index numbers and make them floats
    nxs=len(xs)
    nys=len(ys)
    yinds, xinds=np.mgrid[:nys, :nxs]
    xinds = np.asarray(xinds, dtype=np.float)
    yinds = np.asarray(yinds, dtype=np.float)

    #Now convert to the actual coords
    xinds *= (xs[-1] - xs[0])/(nxs-1)
    xinds += xs[0]
    yinds *= (ys[-1] - ys[0])/(nys-1)
    yinds += ys[0]

    # #Check that the coordinates match up with the input
    # if not (xinds[0, :] == xs).all():
    #     print xs == xinds[0, :]
    #     print xs
    #     print xinds[0, :]
    #     raise ValueError("You have given me xs that are not on a linear grid.")
    # if not (yinds[:, 0] == ys).all():
    #     raise ValueError("You have given me xs that are not on a linear grid.")

    return xinds, yinds

#==========================================================================
#==========================================================================
#==========================================================================

def rotate_coords(x1, y1, theta):
    try:
        nx=len(x1)
    except:
        pass
    else:
        x1=np.array(x1)
        y1=np.array(y1)
    x2 = x1 * np.cos(theta) - y1 * np.sin(theta)
    y2 = x1 * np.sin(theta) + y1 * np.cos(theta)

    return x2, y2

#==========================================================================
#==========================================================================
#==========================================================================

def centers(arr):
    #Takes an N+1 length array and returns an N length array that has the
    # (arr[i]+arr[i+1])/2 in element i (if you have the edges of histogram bins,
    # this will give you the centers)
    #some prep stuff
    l=len(arr)
    arr=np.array(arr)
    #Make sure we can actually do this
    if l<2:
        print "centers says: You've given me a 0 or 1 length array.  I can't work with that"
        return 0
    #Find the centers
    res=np.zeros(l-1)
    for i in range(l-1):
        res[i]=(arr[i]+arr[i+1])/2.
    return res

#==========================================================================
#==========================================================================
#==========================================================================

def add_gp_to_rr_file(in_file, save_file):
    #Reads in the RR file and adds in a G_p if needed

    #Read in the file
    rr = read_numtab(in_file, lastcomment=True, verbose=False)
    
    # If there's nothing to do, go ahead and return
    if 'G_p' in rr.keys():
        print "add_gp_to_rr_file says:  You already have G_p in that file."
        return

    #Divide out the bin widths
    bin_edges=fit.theta_bins_from_centers(rr['theta'])
    bin_widths=np.diff(bin_edges)
    gp=rr['rr']/bin_widths

    #Normalize
    integral=fit.integrate_gp(rr['theta'], gp, 0, rr['theta'][-1]*2.)
    gp /= integral

    #Save
    rr['G_p']=gp
    write_ascii(rr, save_file)

#==========================================================================
#==========================================================================
#==========================================================================

def write_scalar_dictionary(dic, filename):
    #Write out a dictionary in the form
    #  # key = str(dic[k])

    #Open the file for writing (this will overwrite existing files)
    outfile=open(filename, 'w')
    
    #Loop through the keys and output
    ks= dic.keys()
    for k in ks:
        if type(dic[k]) == type(np.zeros(1)):
            try:
                temp=list(dic[k])
            except TypeError:
                temp=dic[k]
        else:
            temp=dic[k]
        outstr = '# '+str(k)+' = '+str(temp)+'\n'
        outfile.write(outstr)

    #Close the file.  All done.
    outfile.close()


#==========================================================================
#==========================================================================
#==========================================================================

def read_scalar_dictionary(filename):
    #read a dictionary that has the format
    #  # key = str(dic[k])

    try:
        tabfile=open(filename, 'r')
    except IOError:
        print "read_scalar_dictionary says: You fail- I find no file called ", filename

    out_dic={}
    for line in tabfile:
        #Take off the return character
        linetemp=line.rstrip('\n')	
        #Split the line up
        bits=linetemp.split()
        #Grab the key, which is the second element
        key=bits[1]
        #The value is the rest of it- it can be a couple things
        val_strs=bits[3:]
        #Pull off the array brackets if there are any
        n_vals=len(val_strs)
        val_strs[0] = val_strs[0].lstrip('[')
        val_strs[-1] = val_strs[-1].rstrip(']')
        #Pull off commas
        for i, val in enumerate(val_strs):
            val_strs[i] = val_strs[i].rstrip(',')

        #See if we can make it into a float
        vals=[]
        try:
            if n_vals > 1:
                for val in val_strs:
                    vals.append(np.float(val))
                vals=np.array(vals)
            else:
                vals=np.float(val_strs[0])
        #If not, make it a string
        except ValueError:
            vals=''
            for val in val_strs:
                vals= vals + val + ' '
            vals=vals.lstrip().rstrip()
        
        #Store
        out_dic[key] = vals

    #Close the file and return our dictionary
    tabfile.close()
    return out_dic
    
#==========================================================================
#==========================================================================
#==========================================================================

def avg_in_bins(x, y, nbins, xmin=None, xmax=None):
    #Bins the sample y in nbins bins of x.  Returns the center of the 
    #bin and the average in that bin

    #We like numpy arrays
    x=np.asarray(x, dtype=np.float)
    y=np.asarray(y, dtype=np.float)

    #If we don't have defined xmin and xmax, take whole range
    if xmin==None:
        xmin=x.min()
    if xmax==None:
        xmax=x.max()

    #Make our bins
    edges = np.linspace(xmin, xmax, nbins+1, dtype=np.float)
    xcents=centers(edges)

    #Loop through bins, take average of y in each
    avgs=np.zeros(nbins)
    for i in range(nbins):
        msk = ma.masked_inside(x, edges[i], edges[i+1]).mask
        avgs[i] = y[msk].mean()

    #Return our result
    return xcents, avgs


#==========================================================================
#==========================================================================
#==========================================================================

def median_in_bins(x, y, nbins, xmin=None, xmax=None):
    #Bins the sample y in nbins bins of x.  Returns the center of the 
    #bin and the average in that bin

    #We like numpy arrays
    x=np.asarray(x, dtype=np.float)
    y=np.asarray(y, dtype=np.float)

    #If we don't have defined xmin and xmax, take whole range
    if xmin==None:
        xmin=x.min()
    if xmax==None:
        xmax=x.max()

    #Make our bins
    edges = np.linspace(xmin, xmax, nbins+1, dtype=np.float)
    xcents=centers(edges)

    #Loop through bins, take average of y in each
    meds=np.zeros(nbins)
    for i in range(nbins):
        msk = ma.masked_inside(x, edges[i], edges[i+1]).mask
        meds[i] = np.median(y[msk])

    #Return our result
    return xcents, meds

#==========================================================================
#==========================================================================
#==========================================================================

def fit_rotated_rectangle(x, y, rotation_angle=None, padding=None, 
                          save_preview_to=None, preview=False, image_mask=None):
    #draw a box that fits around all the data as snugly as possible

    #Make sure we're in a suitable format
    if image_mask is not None:
        #If we have an image mask, convert to XY
        x1, y1 = image_mask.ra_dec_to_xy(x, y)
    else:
        #Make sure they're numpy arrays
        x1=np.array(x)
        y1=np.array(y)
        print "fit_rotated_rectangle says: WARNING- I can fit any set of 2 coordinates, but the sizes will not be the angular sized if you gave me spherical coordinates.  I need an image mask to convert there."
    
    #Set the padding on each side
    if padding:
        try:
            pad_bottom, pad_top, pad_left, pad_right=padding
        except:
            print "fit_rotated_rectangle says: WARNING!  You have given me something I can't use for padding.  Padding must be a 4-element 1D array in the format [bottom, top, left, right].  No padding used this time"
            pad_bottom, pad_top, pad_left, pad_right=[0,0,0,0]
    else:
        pad_bottom, pad_top, pad_left, pad_right=[0,0,0,0]
        
    #If we don't have an already chosen angle, choose a bunch of angles and 
    #transform the coordinates to rotated systems and get the 
    #areas of the rectangle enclosing all the data points at this angle.
    #The angle with the minimum area for the enclosing rectangle.
    if rotation_angle is None:
        thetas=np.radians(np.linspace(0, 90, 901, dtype=np.float))
        areas=[]
        corners=[]
        for th in thetas:
            x2, y2= rotate_coords(x1, y1, th)
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
    x2, y2= rotate_coords(x1, y1, use_theta)
    x2min=x2.min() - pad_left
    x2max=x2.max() + pad_right
    y2min=y2.min() - pad_bottom
    y2max=y2.max() + pad_top

    #Figure out what's inside and what's outside
    inside = ma.masked_inside(x2, x2min, x2max).mask & ma.masked_inside(y2, y2min, y2max).mask
    outside = np.invert(inside)

    #What are the vertices for the box?
    vertices=np.array([[x2min, y2max], [x2min, y2min], [x2max, y2min], [x2max, y2max], [x2min, y2max]])

    #Figure out the dimensions of the box
    if image_mask is not None:
        x2_list_convert=[x2min, x2min, x2max]
        y2_list_convert=[y2min, y2max, y2min]
        ra_box, dec_box=image_mask.xy_to_ra_dec(x2_list_convert, y2_list_convert)
        y_side=cos.ang_sep(ra_box[0], dec_box[0], ra_box[1], dec_box[1], radians_in=False, radians_out=False) * 3600
        x_side=cos.ang_sep(ra_box[0], dec_box[0], ra_box[2], dec_box[2], radians_in=False, radians_out=False) * 3600
    else:
        y_side=y2max-y2min
        x_side=x2max-x2min

    #If we're just previewing, then plot the randoms with the grid overlaid
    if preview:
        #Make a figure and plot the random points
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(x1[outside], y1[outside], c='LightGray')
        ax.scatter(x1[inside], y1[inside], c='Blue')  

        #Plot the outline
        ax.plot(np.transpose(vertices)[0], np.transpose(vertices)[1], color='r', lw=2)

        #Print out the parameters
        ax.text(.05, .95, "theta= "+str(np.degrees(use_theta))[0:5], transform=ax.transAxes)
        ax.text(.05, .9, "padding="+str(padding),  transform=ax.transAxes)
        ax.text(.5, .05, "box size: "+str(x_side)[0:5]+" by "+str(y_side)[0:5],  transform=ax.transAxes)
        if save_preview_to is None:
            plt.show()
        else:
            plt.savefig(save_preview_to, bbox_inches='tight')
            plt.close()
    else:
        info = {'theta'   : use_theta,
                'padding' : padding,
                'x_size'  : x_side,
                'y_size'  : y_side
                }
        return info

    
#==========================================================================
#==========================================================================
#==========================================================================

def chi2_for_fit(ys, errs, fit_ys):
    #The physics version of chi^2- add up the squared error bar lengths away
    #from the theoretical function the ys are.

    #convert everything to numpy arrays so we can do vector things
    ys=np.array(ys)
    errs=np.array(errs)
    fit_ys = np.array(fit_ys)

    #Figure out the chi2
    dist_away = ys - fit_ys
    chi2 = sum((dist_away/errs)**2)
    return chi2
    

#==========================================================================
#==========================================================================
#==========================================================================

def files_starting_with(file_base):
    """
    This takes file_base and returns all the files that start with that

    We have to look at everything in the directory and then pare down to
    just what you care about because check_output is being a butt and
    Popen will do it but is insecure: you could put
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

