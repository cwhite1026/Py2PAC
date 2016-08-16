import AngularCatalog_class as ac
import CorrelationFunction_class as cfclass
import astropy.io.fits as fits
import numpy as np
from numpy import ma
import fitting as fit
import matplotlib.pyplot as plt
# import bias_tools as t

#============================================================
#============================================================
# Fit correlation functions

#Pull in the correlation functions
cfs = []
file_names = ["cosmos.npz", "egs.npz", "goodsn.npz", "goodss.npz", "uds.npz"]
for fn in file_names:
    cfs.append(cfclass.CorrelationFunction.from_file(fn))
cfs = np.array(cfs)
    
#Plot them   
plot_file = "cf_set.pdf"
fig= plt.figure()
ax=fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("theta (deg)")
ax.set_ylabel("w(theta)")
for i in range(5):
    ax = cfs[i].plot(ax=ax, theta_unit="arcsec", return_axis=True)
plt.savefig(plot_file, bbox_inches='tight')
plt.close()

fit_results = fit.bootstrap_fit(cfs, IC_method="offset", 
                                n_fit_boots=200, return_envelope=True, 
                                return_boots=True, fixed_beta=0.6)



#============================================================
#============================================================
# Calculate correlation functions

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

#============================================================
#============================================================
# Code used to convert old correlation functions to the new format

import CorrelationFunction_class as cfclass
import ThetaBins_class as tb
import Gp_class as gp
from reading import *
from copy import deepcopy
cat_names = ["cosmos", "egs", "goodsn", "goodss", "uds"]
bin_name = "_logMstar9_10_z_best1.5_2.5.txt"
cfdir = "/user/caviglia/research/correlations/data_mass_grid_03_16/cfs/"
ngals = [1591, 2915, 2359, 2112, 2282]
nrands = [163572.317361, 91835.7304948, 145270, 139114.574719, 166153.910042]
nchunks = [1, 1, 10, 1, 1]
rrdir = "/user/caviglia/research/correlations/rrs/"
rr_names = ["cosmos_rr_gp.txt", "egs_rr_gp.txt", "gn_rr_gp_new.txt", "gs_rr_gp.txt", "uds_rr_gp.txt"]
for i in np.arange(5):
    gp_dict = read_numtab(rrdir + rr_names[i], lastcomment=True)
    gp_thetas = tb.ThetaBins.from_centers(gp_dict['theta'])
    this_gp = gp.Gp(gp_dict['G_p'], nrands[i], nchunks[i], thetabins_object=deepcopy(gp_thetas), RR=gp_dict['rr'], creation_string=None)
    
    cf_dict = read_numtab(cfdir+cat_names[i]+bin_name, lastcomment=True)
    this_cf = cfclass.CorrelationFunction(name='cf', cf_type='single_galaxy_boot', ngals=ngals[i], estimator="landy-szalay", verbose=True, gp_object=this_gp)
    this_cf.set_thetas_from_centers(cf_dict['theta'], unit='d')
    #This next line could include the boots as a dictionary passed to
    # iterations, but I didn't feel it was necessary
    this_cf.set_cf(cf_dict['galaxy_boot_cf'], cf_dict['galaxy_boot_err'])
    
    this_cf.save(cat_names[i]+".npz")