#Do the variable completeness CFs for the SAM and the data

import CorrelationFunction_class as cfclass
import ThetaBins_class as binclass
import Gp_class as gpclass
import ImageMask_class as imclass
import AngularCatalog_class as ac

import lazytools as lt

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import copy
import brewer2mpl as brew
from os.path import isfile
from reading import *

#==========================================================================
#==========================================================================
# Run the bins in mass and redshift 

#Where are things going?
plot_dir = "/Users/caviglia/Dropbox/plots/plots_for_notes/first_var_comp_grid_05_09/"
sam_cf_dir = "/user/caviglia/research/correlations/var_complete/sams/"
data_cf_dir = "/user/caviglia/research/correlations/var_complete/candels/"

#What are the bins?
z_bins = np.array([1, 1.5, 2.5, 4.5, 6])
mass_bins = np.array([8, 9, 10, 12])

#Things I'll need to read in the catalogs
# cat_names = ['cosmos', 'egs', 'goodsn', 'goodss', 'uds']
cat_names = ['uds']
# completeness_tag = ['cos', 'egs', 'n', 's', 'uds']
completeness_tag = ['uds']
flag_base = "/user/caviglia/candels_catalogs/flag_maps/flag_map_"
# level_files = [flag_base+"cos.fits", flag_base+"egs.fits", flag_base+"goodsn.fits", flag_base+"goodss.fits", flag_base+"uds.fits"]
level_files = [flag_base+"uds.fits"]
# weight_files = ['/astro/candels1/data/cosmos/mosaics/current/60mas/cos_2epoch_wfc3_f160w_060mas_v1.0_wht.fits', '/astro/candels1/data/egs/mosaics/current/egs_all_combined_v1.0/egs_all_wfc3_ir_f160w_060mas_v1.0_wht.fits', '/astro/candels1/data/goodsn/mosaics/current/goods_n_all_combined_v1.0/goodsn_all_wfc3_ir_f160w_060mas_v1.0_wht.fits', '/astro/candels1/data/goodss/mosaics/current/goods_s_all_combined_v0.5/gs_all_candels_ers_udf_f160w_060mas_v0.5_wht.fits', '/astro/candels1/data/uds/mosaics/current/60mas/uds_all_wfc3_f160w_060mas_v1.0_wht.fits']
weight_files = ['/astro/candels1/data/uds/mosaics/current/60mas/uds_all_wfc3_f160w_060mas_v1.0_wht.fits']
sam_base = "/user/caviglia/candels_catalogs/sam/"
sam_files = [sam_base + k+".0/"+k+".fits" for k in cat_names]
data_base = "/Users/caviglia/Box Sync/merged_catalogs/"
# data_files = [data_base+"cos.fits", data_base+"egs.fits", data_base+"gdn.fits", data_base+"gds.fits", data_base+"uds.fits"]
data_files = [data_base+"uds.fits"]

for i, field in enumerate(cat_names):
    print field, level_files[i], completeness_tag[i], weight_files[i]
    #Read in the catalogs
    varcomp_mask = imclass.ImageMask.from_FITS_file(level_files[i], fits_file_type='levels')
    filestr = '/user/mdurbin/candels/for_cathy/for_yotam/completeness_allfields/*_'+completeness_tag[i]+'_*_expdisk_XYH.npz'
    cfs = lt.completeness_list_from_file(filestr)
    varcomp_mask.make_completeness_dict(*cfs)
    flat_mask = imclass.ImageMask.from_FITS_file(weight_files[i], fits_file_type='weight')
    
    #Read in the data, convert
    sam = read_fits(sam_files[i])
    sam['logMstar'] = np.log10(sam['mstar']) + 10.
    data = read_fits(data_files[i])
    data['logMstar'] = data['M_med']
    if 'z_best' not in data.keys():
        data['z_best'] = data['zbest']
    
    #Loop through bins
    sam_mag_mask = ma.masked_less_equal(sam['wfc3f160w_dust'], 26.).mask
    data_mag_mask = ma.masked_less_equal(data['Hmag'], 26.).mask
    for i_z in np.arange(len(z_bins)-1):
        #Make all the masks
        z_min = z_bins[i_z]
        z_max = z_bins[i_z+1]
        z_center = (z_min + z_max)/2.
        sam_zmask = ma.masked_inside(sam['redshift'], z_min, z_max).mask
        data_zmask = ma.masked_inside(data['z_best'], z_min, z_max).mask
        for i_m in np.arange(len(mass_bins)-1):
            m_min = mass_bins[i_m]
            m_max = mass_bins[i_m+1]
            m_center = (m_min + m_max)/2.
            sam_mass_mask = ma.masked_inside(sam['logMstar'], m_min, m_max).mask
            data_mass_mask = ma.masked_inside(data['logMstar'], m_min, m_max).mask
            
            #Mask down to this bin
            sam_bin = {}
            sam_magcut_bin = {}
            data_bin = {}
            data_magcut_bin = {}
            for k in sam.keys():
                sam_bin[k] = sam[k][sam_zmask & sam_mass_mask]
                sam_magcut_bin[k] = sam[k][sam_zmask & sam_mass_mask & sam_mag_mask]
            for k in data.keys():
                data_bin[k] = data[k][data_zmask & data_mass_mask]
                data_magcut_bin[k] = data[k][data_zmask & data_mass_mask & data_mag_mask]
            
            #Make the catalogs that don't need processing
            try:
                n_obj = len(sam_bin['ra'])
            except TypeError:
                n_obj = 0
            if n_obj > 0:
                sam_true = ac.AngularCatalog(sam_bin['ra'], sam_bin['dec'], image_mask = flat_mask, properties = sam_bin)
            else:
                sam_true = None
                
            try:
                n_obj = len(sam_magcut_bin['ra'])
            except TypeError:
                n_obj = 0
            if n_obj > 0:
                sam_magcut = ac.AngularCatalog(sam_magcut_bin['ra'], sam_magcut_bin['dec'], image_mask = flat_mask, properties = sam_magcut_bin)
            else:
                sam_magcut = None
            
            try:
                n_obj = len(data_magcut_bin['RAdeg'])
            except TypeError:
                n_obj = 0
            if n_obj > 0:
                data_magcut = ac.AngularCatalog(data_magcut_bin['RAdeg'], data_magcut_bin['DECdeg'], image_mask = flat_mask, properties = data_magcut_bin)
            else:
                data_magcut = None
            
            #Make the variable completeness SAM 
            sam_bin_mag_mask = ma.masked_inside(sam_bin['wfc3f160w_dust'], 10., 28.).mask
            small_props = {}
            for k in sam_bin.keys():
                small_props[k] = sam_bin[k][sam_bin_mag_mask]
            complete = varcomp_mask.return_completenesses(small_props['ra'], small_props['dec'], mag_list=small_props['wfc3f160w_dust'], rad_list=np.log10(small_props['r_disk']), use_mags_and_radii=True)
            compare_to = rand.random(len(complete))
            use = compare_to <= complete
            for k in small_props.keys():
                small_props[k] = small_props[k][use]
            
            try:
                n_obj = len(small_props['ra'])
            except TypeError:
                n_obj = 0
            if n_obj > 0:
                sam_var = ac.AngularCatalog(small_props['ra'], small_props['dec'], properties= small_props, image_mask = varcomp_mask)
            else:
                sam_var = None
            
            #Full data thing
            data_mag_range = ma.masked_inside(data_bin['Hmag'], 10., 28.).mask
            data_var_masked = {}
            try:
                for k in data_bin.keys():
                    data_var_masked[k] = data_bin[k][data_mag_range]
            except IndexError:
                data_var_masked={'RAdeg':0}

            try:
                n_obj = len(data_var_masked['RAdeg'])
            except TypeError:
                n_obj = 0
            if n_obj > 0:
                data_full = ac.AngularCatalog(data_var_masked['RAdeg'], data_var_masked['DECdeg'], image_mask = varcomp_mask, properties = data_var_masked)
            else:
                data_full = None
            
            #Do the correlation functions
            tag = "_z"+str(z_center)+"_m"+str(m_center)
            if sam_true is not None:
                sam_true.set_theta_bins(1, 350, 12)
                sam_true.generate_random_sample(20000, z = z_center, mstar = m_center, use_mags_and_radii = False)
                sam_true.cf_bootstrap(n_boots=40, clobber=True, save_file_base=sam_cf_dir, name=cat_names[i] + '_all_sam_gals'+tag)
            if sam_magcut is not None:
                sam_magcut.set_theta_bins(1, 350, 12)
                sam_magcut.generate_random_sample(20000, z = z_center, mstar = m_center, use_mags_and_radii = False)
                sam_magcut.cf_bootstrap(n_boots=40, clobber=True, save_file_base=sam_cf_dir, name=cat_names[i] + '_magcut_sam_gals'+tag)
            if sam_var is not None:
                sam_var.set_theta_bins(1, 350, 12)
                sam_var.generate_random_sample(20000, z = z_center, mstar =     m_center, use_mags_and_radii = True)
                sam_var.cf_bootstrap(n_boots=40, clobber=True, save_file_base=sam_cf_dir, name=cat_names[i] + '_varcomp_sam_gals'+tag)
            if data_full is not None:        
                data_full.set_theta_bins(1, 350, 12)
                data_full.generate_random_sample(20000, z = z_center, mstar = m_center, use_mags_and_radii = True)
                data_full.cf_bootstrap(n_boots=40, clobber=True, save_file_base=data_cf_dir, name=cat_names[i] + '_data_varcomp'+tag)
            if data_magcut is not None:
                data_magcut.set_theta_bins(1, 350, 12)
                data_magcut.generate_random_sample(20000, z = z_center, mstar = m_center, use_mags_and_radii = False)
                data_magcut.cf_bootstrap(n_boots=40, clobber=True, save_file_base=data_cf_dir, name=cat_names[i] + '_data_magcut'+tag)
            
            
            
