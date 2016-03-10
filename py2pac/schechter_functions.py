import numpy as np

def schechter(m, phi_star=0.00681, m_star=-19.61, alpha=-1.33):
    fudge_factor = np.log(10)*0.4 * phi_star
    base_exp = 10**(0.4*(m_star - m))
    function = fudge_factor * (base_exp**(1 + alpha)) * np.exp(-base_exp)
    return function

def get_schechter_mags(size, distmod, min_mag, max_mag, z = 1.7):
    nmax = schechter(max_mag - distmod)
    mags = []
    while len(mags) < size:
        x = np.random.uniform(min_mag, max_mag, size=size) - distmod
        y = np.random.uniform(0, high=nmax, size=size)
        mags += list(x[y<=schechter(x)])   
    mags = np.asarray(mags[:size])
    return mags

def get_radii(m, r0 = 0.21 / 0.06, m0 = -21., beta = 0.3, sigma = 0.7):
    exp = 10**(0.4*(m0 - m))
    mean = r0 * exp**beta
    rand = np.random.lognormal(mean, sigma)
    log_rand = np.log10(rand)
    log_rand[log_rand > 2.5] = 2.5
    log_rand[log_rand < -1.] = -1.
    return log_rand
