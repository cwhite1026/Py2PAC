import AngularCatalog_class as ac
import astropy.io.fits as fits
from numpy import ma
import fitting as fit
import bias_tools as t

import CorrelationFunction_class as cfclass
import ThetaBins_class as tb
import Gp_class as gp
from reading import *
from copy import deepcopy
cat_names = ["cosmos", "egs", "goodsn", "goodss", "uds"]
bin_name = "_logMstar9_10_z_best1.5_2.5.txt"
cfdir = "/user/caviglia/research/correlations/data_mass_grid_03_16/cfs/"
ngals = [1591, 2915, 2359, 2112, 2282]
rrdir = "/user/caviglia/research/correlations/rrs/"
rr_names = ["cosmos_rr_gp.txt", "egs_rr_gp.txt", "gn_rr_gp_new.txt", "gs_rr_gp.txt", "uds_rr_gp.txt"]
for i in np.arange(5):
    cf_dict = read_numtab(cfdir+cat_names[0]+bin_name, lastcomment=True)
    this_theta = tb.ThetaBins.from_centers(cf_dict['theta'])
    gp_dict = read_numtab(rrdir + rr_names[i], lastcomment=True)
    gp_thetas = tb.ThetaBins.from_centers(gp_dict['theta'])
    this_gp = gp.Gp(gp_dict['G_p'], 163573, 10, thetabins_object=deepcopy(gp_thetas), RR=gp_dict['rr'], creation_string=None)
    this_cf = cfclass.CorrelationFunction(name='cf', cf_type='single_galaxy_boot', ngals=ngals[i], estimator="landy-szalay", theta_bin_object=deepcopy(this_theta), verbose=True, gp_object=None)


#Where things are
# catalogdir = "/Users/hcferguson/Box Sync/Ferguson/CANDELS Catalog Team Release May 2015/merged_catalogs/"
# imagedir = "/Users/hcferguson/data/candels/goodss/mosaics/current/goods_s_all_combined_v0.5/"
catalogdir = "/Users/caviglia/Box Sync/merged_catalogs/"
imagedir = '/astro/candels1/data/goodss/mosaics/current/'

#Read in the data
data = fits.open(catalogdir + "gds.fits")
data = data[1].data
msk = ma.masked_less(data['Hmag'], 26).mask

#Generate an AngularCatalog with the ImageMask from the appropriate weight file
weight_file = 'gs_all_candels_ers_udf_f160w_060mas_v0.5_wht.fits'
weight_file = imagedir + weight_file
cat = ac.AngularCatalog(data['RAdeg'][msk], data['DECdeg'][msk], weight_file = weight_file)

#Generate the random sample
cat.generate_random_sample(number_to_make=1e5)

#Set the theta binning
cat.set_theta_bins(10, 350, 7)

#Do the correlation function
cat.cf_bootstrap(n_boots=20, clobber=True, name="single_gal_cf", save_file_base="")

#Plot correlation function
cat.plot_cfs(which_cfs=['single_gal_cf'], labels=["Single gal bootstrap"], fmt='o-')

#Fit the correlation function
# fit_results = fit.bootstrap_fit(cat, IC_method="offset", 
#                                 n_fit_boots=200, return_envelope=True, 
#                                 return_boots=True)

