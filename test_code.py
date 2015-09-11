import AngularCatalog_class as ac
import CorrelationFunction_class as cfclass
import ThetaBins_class as binclass
import Gp_class as gpclass

import numpy as np


#Check the cute little theta bin class
tb=binclass.ThetaBins(1, 100, 20)
tb.get_bins()
tb.get_bins(unit='d')
tb.get_bins(unit='r')
tb.set_new_bins(1, 200, 20)
tb.get_bins()
tb.get_logbins()
tb.set_logbins(False)
tb.get_bins()

#Check the CorrelationFunction class
compiles = cfclass.CorrelationFunction(name='test', cf_type='cf', ngals=180, theta_bin_object=tb2) 
centers, edges=tb2.get_bins()
compiles = cfclass.CorrelationFunction(theta_bins=edges)
compiles = cfclass.CorrelationFunction(thetas=centers)
compiles.get_thetas(unit='a')
compiles.set_thetas(1, 200, 20, unit='a', logbins=True)

compiles = cfclass.CorrelationFunction(theta_bins=edges)
centers, edges = compiles.get_thetas(unit='a')
cf = .001 * centers**-.8
err = 1e-5 * np.ones(len(centers))
iterations = {}
for i in range(4):
    iterations[i]=cf
compiles.set_cf(cf, err, iterations=iterations)
compiles.save('/Users/cathyc/Desktop/testcfsave')
saved=np.load('/Users/cathyc/Desktop/testcfsave.npz')
cmtest=cfclass.CorrelationFunction.from_file('/Users/cathyc/Desktop/testcfsave.npz')

#Test the Gp class
min_theta=1
max_theta = 100
nbins=10
Gp= np.arange(nbins)
n_randoms=100
n_chunks=1
test_gp = gpclass.Gp(min_theta, max_theta, nbins, Gp, n_randoms, n_chunks, logbins=True, unit='arcsec', RR=None, creation_string=None)

test_gp.get_thetas(unit='a')
test_gp.stats(print_only=False)
test_gp.get_Gp()
test_gp.get_points(unit='d')

test_gp.save('/Users/cathyc/Desktop/testgpsave')
newgp = gpclass.Gp.from_file('/Users/cathyc/Desktop/testgpsave.npz')
newgp.get_thetas(unit='a')
newgp.stats(print_only=False)
newgp.get_Gp()
newgp.get_points(unit='d')

saved= np.load('/Users/cathyc/Desktop/cosmos_gp.npz')
cosmos = gpclass.Gp.from_file('/Users/cathyc/Desktop/cosmos_gp.npz')
cosmos.save('/Users/cathyc/Desktop/cosmos_gp2.npz')
cosmos=gpclass.Gp.from_file('/Users/cathyc/Desktop/cosmos_gp2.npz')
cosmos.integrate_gp_times_powerlaw(.003, .8, 1., 100., theta_unit='a', param_unit='d')
cosmos.integrate_powerlaw(.003, .8, 1, 100)
cosmos.integrate_gp(1, 100, theta_unit='a')
new_bin_edges = np.logspace(-1, 4, 20)
sum(cosmos.integrate_to_bins(new_bin_edges))

cf=cfclass.CorrelationFunction.from_file('/Users/cathyc/Desktop/testcfsave.npz')
cf.load_gp('/Users/cathyc/Desktop/cosmos_gp_again.npz')




