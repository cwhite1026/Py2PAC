import AngularCatalog_class as ac
import astropy.io.fits as fits
from numpy import ma
import fitting as fit

#Where things are
# catalogdir = "/Users/hcferguson/Box Sync/Ferguson/CANDELS Catalog Team Release May 2015/merged_catalogs/"
# imagedir = "/Users/hcferguson/data/candels/goodss/mosaics/current/goods_s_all_combined_v0.5/"
catalogdir = ""

#Read in the data
data = fits.open(catalogdir + "gds.fits")
data = data[1].data
msk = ma.masked_less(data['Hmag'], 26).mask

#Generate an AngularCatalog with the ImageMask from the appropriate weight file
weight_file = 'hlsp_candels_hst_wfc3_gs-tot-sect33_f160w_v1.0_wht.fits'
weight_file = imagedir + weight_file
cat = ac.AngularCatalog(data['RAdeg'][msk], data['DECdeg'][msk], weight_file = weight_file)

#Generate the random sample
cat.generate_random_sample(number_to_make=1e5)

#Set the theta binning
cat.set_theta_bins(10, 350, 7)

#Do the correlation function
cat.cf_bootstrap(n_boots=20, clobber=True, name="single_gal_cf")

#Plot correlation function
cat.plot_cfs(which_cfs=['single_gal_cf'], labels=["Single gal bootstrap"], fmt='o-')

#Fit the correlation function
fit_results = fit.bootstrap_fit(cat, IC_method="offset", 
                                n_fit_boots=200, return_envelope=True, 
                                return_boots=True)

