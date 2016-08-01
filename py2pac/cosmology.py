#Contains routines to calculate cosmological things and other usefulness
#Returns distances in Mpc

import numpy as np
import numpy.ma as ma
import scipy as sc
import scipy.integrate as intg 
import math as mth
import astropy.coordinates as coords
import astropy.units as u

#Define cosmological parameters
omegaM=0.28
omegak=0.
omegaL=0.72
h100=0.70  
f_baryon=0.1658
q0=omegaM/2. - omegaL

#Miscellaneous basic things
torad=sc.pi/180.
G=6.67e-11 #m^3 kg^-1 s^-2
km_per_Mpc=3.086e19  # 1 Mpc in km
kg_per_Msun=1.99e30  # 1 Msun in kg

#Figure out the constant, DH
c=3*10**5   #in km/s
H0=h100*100. #km/s/Mpc
DH=c/H0     #Mpc

#Solid angles of fields for Peter's light cones
Om_goodss=39.*41.*torad**2/60.**2
Om_cosmos=17.*41.*torad**2/60.**2
Om_egs=17.*46.*torad**2/60.**2
Om_goodsn=32.**2.*torad**2/60.**2
Om_uds=36.*35.*torad**2/60.**2



#==============================================================================
#==============================================================================
#==============================================================================

# def Vvir(M, z):
#     #This should give the same answer as vc, but I stole the computations for
#     #this guy from Rachel's C code.

#     virdens=virial_density_approx_for_flat_universe(z)
#     f=( (3 * Omega(z)) / (4. * np.pi * virdens * omegaM * rho_crit(0)) )**(1./3.) * 1./(1.+z)
#     C = H0 * ( 3./(8 * np.pi * rho_crit(0) * f) - (f**2 * (1-omegaM)) / 3. ) ** (0.5)

#     return C * M**(1./3.)

#==============================================================================
#==============================================================================
#==============================================================================

def Vc(M, z):
    #Returns the circular velocity of a given mass halo at the given redshift **** RETURNS m/s *****
    rvir=virial_radius(M, z)  #in m
    # print "rvir  ", rvir
    M_kg= M * kg_per_Msun #Mass from Msun to kg
    # print "M  ", M_kg  # in kg
    standard_units_H0=H0/km_per_Mpc  #This puts H in units of s^-1
    # print (G*M_kg/rvir)/((omegaL * standard_units_H0**2. * rvir**2.) / 3.)
    vc2= G*M_kg/rvir - (omegaL * standard_units_H0**2. * rvir**2.) / 3.
    # print "vc^2  ", vc2
    # fudge_factor=0.056 #I have absolutely no idea why, but I'm getting values of vc a factor of 1.13 off.  This corrects.
    fudge_factor=0
    return vc2**0.5 * (10.**fudge_factor)

#==============================================================================
#==============================================================================
#==============================================================================

def virial_radius(M, z):
    #Returns the virial radius for a given mass at a given redshift- in m
    Delta_c=virial_density_approx_for_flat_universe(z)
    M_kg= M * kg_per_Msun #Mass from Msun to kg
    rvir=3.*M_kg*Omega(z)
    rvir=rvir / (4. *np.pi *Delta_c *omegaM *rho_crit(0))
    return (rvir**(1./3.))/(1.+z)

    
#==============================================================================
#==============================================================================
#==============================================================================

def virial_density_approx_for_flat_universe(z):
    #Returns the approximate virial density (Delta_c in Somerville and Primack 99)- dimensionless
    #  assuming a flat universe.  Changes for an open universe.
    x=Omega(z)-1
    return 18.* np.pi**2. + 82.*x - 39.*x**2.
 
#==============================================================================
#==============================================================================
#==============================================================================

def Omega(z):
    #Returns Omega=rho/rho_crit for redshift z- dimensionless
    rho_0=omegaM*rho_crit(0)  #Current matter density of the universe
    a=1./(1.+z)
    return (rho_0/a**3.)/rho_crit(z)
    

#==============================================================================
#==============================================================================
#==============================================================================

def rho_crit(z):
    #Returns the critical density at the given redshift- units= kg/m^3
    #3H^2/8piG
    standard_units_H=H(z)/km_per_Mpc  #This puts H in units of s^-1
    return 3. * standard_units_H**2 / (8.*np.pi*G)  #meaning this will have units of density


#==============================================================================
#==============================================================================
#==============================================================================

def H(z):
    #Checked this against plots found on google over [0, 2.5] and it looks fine - units= km/s/Mpc
    #Returns the hubble constant as the given redshift
    return H0*E(z)

#==============================================================================
#==============================================================================
#==============================================================================

def h(z):
    #Checked this against plots found on google over [0, 2.5] and it looks fine - units= km/s/Mpc
    #Returns the hubble constant as the given redshift
    return H0*E(z)/100.

#==============================================================================
#==============================================================================
#==============================================================================

def volume(field, z1, z2):
    #Takes the ra and dec spans and redshift interval and calculates the comoving volume in units of Mpc^3
    if field=='goodss':
        Om=Om_goodss
    elif field=='goodssn':
        Om=Om_goodsn
    elif field=='egs':
        Om=Om_egs
    elif field=='uds':
        Om=Om_uds
    elif field=='cosmos':
        Om=Om_cosmos
    else:
        print "volume says: I don't know the field ", field
        return 0
    #Do the integral
    integrand=lambda z: DC(z)**2 * oneoverE(z)
    I, Ierr=intg.quad(integrand, z1, z2)
    #Now put in the constants
    #print "Omega: ", Om
    res=Om * DH * I
    return res

#==============================================================================
#==============================================================================
#==============================================================================


def gen_volume(rarange, decrange, z1, z2):
    #Takes the ra and dec spans and redshift interval and calculates the comoving volume in units of Mpc^3
    delta_ra=abs(rarange[1]-rarange[0])
    delta_dec=abs(decrange[1]-decrange[0])
    Om=delta_ra*delta_dec*torad**2/60.**2
    #Do the integral
    integrand=lambda z: DC(z)**2 * oneoverE(z)
    I, Ierr=intg.quad(integrand, z1, z2)
    #Now put in the constants
    #print "Omega: ", Om
    res=Om * DH * I
    return res

#==============================================================================
#==============================================================================
#==============================================================================

def E(z):
    inside= omegaM*(1.+z)**3 + omegak*(1.+z)**2 + omegaL
    return np.sqrt(inside)


#==============================================================================
#==============================================================================
#==============================================================================

def oneoverE(z):
    #The function to integrate over
    inside= omegaM*(1.+z)**3 + omegak*(1.+z)**2 + omegaL
    return 1./np.sqrt(inside)

#==============================================================================
#==============================================================================
#==============================================================================

def DC(z):
    #Calculates the comoving distance from us to an object at redshift z in Mpc
    I, Ierr=intg.quad(oneoverE, 0, z)
    return DH*I

#==============================================================================
#==============================================================================
#==============================================================================

def DA(z, comoving = False):
    #Checked against astropy.cosmology and found to be accurate to 0.1% to z=7
    #Calculates the angular diameter distance from us to an object at redshift z
        #Returns PHYSICAL Mpc by default.  Comoving=True, then returns comoving.
    
    #Get the comoving distance chi
    chi=DC(z)

    #Get r(chi)
    if omegak < 0:
        print "Not programmed yet"
        return 0
        r= np.sin((-omegak)**0.5 *H0*chi) /(H0*(mth.fabs(omegak))**0.5)
    elif omegak == 0:
        r= chi
    elif omegak > 0:
        print "Not programmed yet"
        return
        r= mth.sinh(omegak**0.5 *H0*chi) /(H0*(mth.fabs(omegak))**0.5)

    #Convert to physical units if asked
    if not comoving:
        r=r/(1.+z)

    return r


#==============================================================================
#==============================================================================
#==============================================================================

 
def ang_sep(ra1, dec1, ra2, dec2, radians_out=True, radians_in=False):
    #Calculates the angular separation between two points 
    #If we're given degrees (i.e. radians_in==False), convert to radians
    if radians_in==False:
        ra1*=mth.pi/180.
        ra2*=mth.pi/180.
        dec1*=mth.pi/180.
        dec2*=mth.pi/180.
    #First find the numerator for the arctan.  It's gross
    numer=(mth.cos(dec2)**2.) * (mth.sin(ra2-ra1)**2.)
    numer+=(mth.cos(dec1)*mth.sin(dec2) - mth.sin(dec1)*mth.cos(dec2)*mth.cos(ra2-ra1))**2.
    numer=numer**.5
    #Find the denominator. Also gross
    denom=mth.sin(dec1)*mth.sin(dec2) + mth.cos(dec1)*mth.cos(dec2)*mth.cos(ra2-ra1)
    #Find the separation
    if radians_out:
        theta=mth.atan(numer/denom)
    else:
        theta=(180/mth.pi)*mth.atan(numer/denom)
    return theta


#==============================================================================
#==============================================================================
#==============================================================================
def separation(ra1, dec1, ra2, dec2, radians_in=False, radians_out=True):
    #Uses the astropy coordinates module to calculate the separation
    if radians_in:
        units=[u.radian, u.radian]
    else:
        units=[u.degree, u.degree]

    c1 = SkyCoord(ra=ra1, dec=dec1, frame='icrs', unit=units)
    c2 = SkyCoord(ra=ra2, dec=dec2, frame='icrs', unit=units)
    sep= c1.separation(c2)
    return sep

            
#==============================================================================
#==============================================================================
#==============================================================================

 
def array_ang_sep(ra1, dec1, ra2, dec2, radians_out=True, radians_in=False):
    #Calculates the angular separation between two points 
    #If we're given degrees (i.e. radians_in==False), convert to radians
    if radians_in==False:
        ra_1 = ra1*mth.pi/180.
        ra_2 = ra2*mth.pi/180.
        dec_1 = dec1*mth.pi/180.
        dec_2 = dec2*mth.pi/180.
    #First find the numerator for the arctan.  It's gross
    numer=(np.cos(dec_2)**2.) * (np.sin(ra_2-ra_1)**2.)
    numer+=(np.cos(dec_1)*np.sin(dec_2) - np.sin(dec_1)*np.cos(dec_2)*np.cos(ra_2-ra_1))**2.
    numer=numer**.5
    #Find the denominator. Also gross
    denom=np.sin(dec_1)*np.sin(dec_2) + np.cos(dec_1)*np.cos(dec_2)*np.cos(ra_2-ra_1)
    #Find the separation
    if radians_out:
        theta=np.arctan(numer/denom)
    else:
        theta=(180/mth.pi)*np.arctan(numer/denom)
    return theta



#==============================================================================
#==============================================================================
#==============================================================================

 
def phys_to_comove_len(ls, zs):
    #Takes any number of physical lengths and either a single z or a z for each
    #and convert the lengths from physical to comoving

    #Make sure that we have numpy arrays if we have arrays
    if len(ls)>1:
        ls=np.array(ls)
    if len(zs)>1:
        zs=np.array(zs)

    #Calculate the scale factor(s)
    a=1./(1.+zs)

    #Return the length(s) in comoving coords
    comov=ls/a
    return comov

#==============================================================================
#==============================================================================
#==============================================================================

 
def comove_to_phys_len(ls, zs):
    #Takes any number of comoving lengths and either a single z or a z for each
    #and convert the lengths from comoving to physical

    #Make sure that we have numpy arrays if we have arrays
    if len(ls)>1:
        ls=np.array(ls)
    if len(zs)>1:
        zs=np.array(zs)

    #Calculate the scale factor(s)
    a=1./(1.+zs)

    #Return the length(s) in comoving coords
    phys=ls*a
    return phys

#==============================================================================
#==============================================================================
#==============================================================================

def double_schechter_function(masses, Phi_star1, M_star1, alpha1, Phi_star2, M_star2, alpha2):
    #Returns double schechter functions
    return schechter_function(masses, Phi_star1, M_star1, alpha1) + schechter_function(masses, Phi_star2, M_star2, alpha2)

    
#==============================================================================
#==============================================================================
#==============================================================================

def schechter_function(masses, Phi_star, M_star, alpha):
    #Returns the Schecter function of given parameters for the masses requested
    #Phi(M)= ln(10) * Phi_star * [10**((M-M_star)*(1+alpha))] * exp[-10**(M-M_star)]
    #M and M_star here are log_10 masses

    masses=np.array(masses)
    const=np.log(10)*Phi_star
    power=10. ** ((masses-M_star) * (1+alpha))
    exponential= np.exp(-10. ** (masses-M_star))

    return const*power*exponential


#==============================================================================
#==============================================================================
#==============================================================================

def microJy_to_monochrome_AB_mag(fluxes):
    #Convert micro Janskys to a magnitude in a rough way.  This is not really
    #very thorough because a true magnitude would take into account the bandpass,
    #but that's way too complicated for me to deal with at the moment
    msk=ma.masked_less_equal(fluxes, 0).mask
    fluxes[msk]=1.e-36  #Make the 0s very small but nonzero
    #Now convert
    fluxes*=1.e-6
    mags=-5./2. * np.log10(fluxes) + 8.9
    return mags
    
