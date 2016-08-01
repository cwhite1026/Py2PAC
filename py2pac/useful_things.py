#This is where useful routines that aren't specific to the dropout selection program will live.

import numpy as np
import math as mth
import struct as st
import numpy.ma as ma
# import pyfits as pf
import cosmology as cos
import reading as r
from reading import *
from writing import *
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
import pickle
#===============================================================================
#===============================================================================
#===============================================================================
#So I don't forget the shortcuts:
def mask_valid(arr):
    m=ma.masked_where(np.isfinite(arr), arr)
    m=m.mask

    return m


def mask_invalid(arr):
    m=ma.masked_where(np.isfinite(arr), arr)
    m=m.mask
    m=np.invert(m)

    return m


#==========================================================================
#==========================================================================
#==========================================================================

def save_pickle(obj, filen):
    with open(filen, 'wb') as f:
        pickle.dump(obj, f)

#==========================================================================
#==========================================================================
#==========================================================================

def load_pickle(filen):
    with open(filen, 'rb') as f:
        obj = pickle.load(f)
    return obj

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

#===============================================================================
#===============================================================================
#===============================================================================
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


#===============================================================================
#===============================================================================
#===============================================================================

def smooth_with_gaussian(x0, y0, sigma, nx=51):
    #Smooths the y0 curve with a gaussian
    x0=np.array(x0)
    y0=np.array(y0)

    #Start by getting an even x grid
    xrng=[x0.min(), x0.max()]
    step=np.diff(xrng)[0]/float(nx-1)  
    #So that round-off errors don't put us outside the range for interpolation
    breathing_room=step/1000.
    xrng_comfy=[x0.min()+breathing_room, x0.max()-breathing_room]
    step=np.diff(xrng_comfy)[0]/float(nx-1)
    xs=np.arange(xrng_comfy[0], xrng_comfy[1]+step/2., step)
        
    #Now get our even spaced ys
    fcn=interp1d(x0, y0)
    ys=fcn(xs)
    
    #Now the smoothing
    return xs, gaussian_filter1d(ys, sigma)

#===============================================================================
#===============================================================================
#===============================================================================
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
    span=xmax-xmin

    #Make our bins
    edges = np.arange(nbins+1, dtype=np.float)
    edges *= span/nbins
    edges += xmin
    xcents=centers(edges)

    #Loop through bins, take average of y in each
    avgs=np.zeros(nbins)
    for i in range(nbins):
        msk = ma.masked_inside(x, edges[i], edges[i+1]).mask
        avgs[i] = y[msk].mean()

    #Return our result
    return xcents, avgs


#===============================================================================
#===============================================================================
#===============================================================================
def array_safelog10(arr, zero_fill=-99., return_only_nonzero=False):
    #Returns log10(num) for ok numbers, -99 for anything invalid
    msk=ma.masked_greater(arr, 0.).mask
    if return_only_nonzero:
        res=np.log10(arr[msk])
    else:
        res=np.ones(len(arr))*zero_fill
        res[msk]=np.log10(arr[msk])
    return res


#===============================================================================
#===============================================================================
#===============================================================================
def safelog10(num):
    #Returns log10(num) for ok numbers, -99 for anything invalid 
    if num>0:
        lg=mth.log10(num)
    else:
        lg=-99
    return lg

#===============================================================================
#===============================================================================
#===============================================================================
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

#===============================================================================
#===============================================================================
#===============================================================================
def ascii_to_binary(direc='', filenames=[], halofile='halos.dat', write_to_direc=''):
    #Takes the ascii files, reads them in, outputs as binary
    if direc!='':
        direc=direc+'/'
    if write_to_direc=='':
        write_to_direc=direc

    #Read in all the files I want to put in the binary file
    files={}
    for f in filenames:
        files[f]=r.read_sext(direc+f)

    #Read in the halo file separately because it's a different length
    halos=r.read_sext(direc+halofile)

    #Consolidate the files without duplicating the columns
    cat={}
    for filekey in files.keys():
        for columnname in files[filekey].keys():
            if columnname not in cat.keys():
                cat[columnname]=files[filekey][columnname]

    #Match up the halo information to the galaxy information
    for halokey in halos.keys():
        if halokey not in cat.keys():
            cat[halokey]=halos[halokey][cat['halo_id']]

    #Write the dictionary to binary
    writebin(cat, direc)


#===============================================================================
#===============================================================================
#===============================================================================
def ascii_to_fits(direc='', filenames=[], halofile='halos.dat', write_to_direc='', outname='catalog.fits'):
    #Takes the ascii files, reads them in, outputs as fits
    if direc!='':
        direc=direc+'/'
    if write_to_direc=='':
        write_to_direc=direc

    #Read in all the files I want to put in the fits file
    files={}
    for f in filenames:
        files[f]=r.read_sext(direc+f)

    #Read in the halo file separately because it's a different length
    if halofile is not None:
        halos=r.read_sext(direc+halofile)

    #Consolidate the files without duplicating the columns
    cat={}
    for filekey in files.keys():
        for columnname in files[filekey].keys():
            if columnname not in cat.keys():
                cat[columnname]=files[filekey][columnname]

    #Match up the halo information to the galaxy information
    if halofile is not None:
        for halokey in halos.keys():
            if halokey not in cat.keys():
                cat[halokey]=halos[halokey][cat['halo_id']]

    #Write the dictionary to fits
    write_fits(cat, direc+outname)



#===============================================================================
#===============================================================================
#===============================================================================
def lightcone_to_fits(filename, outname='catalog.fits'):
    #Takes the ascii lightcone files, reads them in, outputs as fits

    #Read in the file
    cat = r.read_lightcone(filename)

    #Write the dictionary to binary
    write_fits(cat, outname)

#===============================================================================
def SAM_lightcone_to_fits(filename, outname='catalog.fits'):
    #Takes the ascii lightcone files, reads them in, outputs as fits

    #Read in the file
    cat = r.read_sext(filename)

    #Write the dictionary to binary
    write_fits(cat, outname)    
    
    
#===============================================================================
#===============================================================================
#===============================================================================

def make_merged_binary(direc, bin_name='catalog_binary', samelength=True, version=2, fits=True):
    #This executes the process needed to make a binary file of the merged catalogs
    direc+='/'
    print "make_merged_binary says: putting the catalogs together"
    if samelength:
        print 'make_merged_binary says: reading in galprop file'
        gal=read_sext(direc+'galprop.dat', make_numpy=False)
        smoosh=gal
        #Pull some properties of the dictionaries
        gal_keys=gal.keys()
        ngals=len(gal[gal_keys[0]])
        gal=None

        #Add dictionary entries to gal that it doesn't already have.
        print 'make_merged_binary says: reading in galphot file, adding to catalog'
        phot=read_sext(direc+'galphot.dat', make_numpy=False)
        phot_keys=phot.keys()
        for k in phot_keys:
            if k not in smoosh.keys():
                smoosh[k]=phot[k]
        phot=None

        print 'make_merged_binary says: reading in fir file, adding to catalog'
        fir=read_sext(direc+'FIR.dat', make_numpy=False)
        fir_keys=fir.keys()
        for k in fir_keys:
            if k not in smoosh.keys():
                smoosh[k]=fir[k]
        fir=None
                
        print 'make_merged_binary says: reading in galphot_dust file, adding to catalog'
        dust=read_sext(direc+'galphotdust.dat', keyadd='_dust', exceptions_to_keyadd=['halo_id', 'gal_id', 'z'], make_numpy=False)
        dust_keys=dust.keys()
        for k in dust_keys:
            if k not in smoosh.keys():
                smoosh[k]=dust[k]
        dust=None

        #Matching halo properties
        print 'make_merged_binary says: reading in halo file, adding to catalog'
        halo=read_sext(direc+'halos.dat', make_numpy=False)
        halo_keys=halo.keys()
        halo_add=[]        
        for k in halo_keys:
            if k not in smoosh.keys():
                halo_add.append(k)
                smoosh[k]=np.zeros(ngals)
        for i in range(ngals):
            for k in halo_add:
                smoosh[k][i]=halo[k][smoosh['halo_id'][i]]
        halo=None
    else:
        gal=read_sext(direc+'galprop.dat')
        phot=read_sext(direc+'galphot.dat')
        halo=read_sext(direc+'halos.dat')
        dust=read_sext(direc+'galphotdust.dat', keyadd='_dust', exceptions_to_keyadd=['halo_id', 'gal_id', 'z'])
        fir=read_sext(direc+'FIR.dat')
        smoosh=sam_combine_all(gal, halo, phot, dust, fir)

    #Now write the catalog to binary
    if fits:
        print "make_merged_binary says: Writing catalog to fits file"
        #Get some basic info
        ks=smoosh.keys()
        nks=len(ks)
        
        #Set up the columns
        print "make_merged_binary says: setting up the columns"
        col_list=[]
        for i in range(nks):
            testval=smoosh[ks[i]][0]
            if type(testval) == type(np.int64(0)):
                fmt='K'
            elif type(testval) == type(np.float64(0)):
                fmt='D'
            else:
                print "make_merged_binary says: I don't know the type ", type(testval)
            temp=pf.Column(name=ks[i], format=fmt, array=smoosh[ks[i]])
            col_list.append(temp)

        #Create a column definition object
        cols=pf.ColDefs(col_list)
        
        #Make a new binary table HDU object
        hdu_tab=pf.new_table(cols)

        #Write the fits file out
        hdu_tab.writeto(direc+bin_name+'.fits')
    if version==1:
        print "make_merged_binary says: calling the binary file writing routine version 1"
        write_merged_binary(smoosh, direc, write_to=bin_name)
    elif version==2:
        print "make_merged_binary says: calling the binary file writing routine version 2"
        write_merged_binary(smoosh, direc)

#===============================================================================
#===============================================================================
#===============================================================================

def binv1_to_binv2(filen, direc):
    #Reads in a binary version1 catalog and writes out a file in v2 format

    #Read in the original binary
    cat=read_merged_binary(direc+'/'+filen)

    #Write the v2 binary
    write_merged_binary_v2(cat, direc)

#===============================================================================
#===============================================================================
#===============================================================================

def bin2fits(direc, version=2, filen='', outfilen='catalog.fits'):
    #Reads in a binary file and converts it to fits
    
    #Read in the binary catalog
    print "bin2fits says: Reading the binary catalog from ", direc
    if version==1:
        cat=read_merged_binary(direc+'/'+filen)
    elif version==2:
        cat=read_merged_binary_v2(direc)
    else:
        print "bin2fits says: I don't know of a version ", version

    
    #Get some basic info
    ks=cat.keys()
    nks=len(ks)
    ngals=len(cat[ks[0]])

    
    #Set up the columns
    print "bin2fits says: setting up the columns"
    col_list=[]
    for i in range(nks):
        testval=cat[ks[i]][0]
        if type(testval) == type(np.int64(0)):
            fmt='K'
        elif type(testval) == type(np.float64(0)):
            fmt='D'
        else:
            print "bin2fits says: I don't know the type ", type(testval)
        temp=pf.Column(name=ks[i], format=fmt, array=cat[ks[i]])
        col_list.append(temp)
        cat[ks[i]]=[]

    #Create a column definition object
    cols=pf.ColDefs(col_list)

    #Make a new binary table HDU object
    hdu_tab=pf.new_table(cols)

    #Make ourselves an HDU list.  The primary can't be a table, so I make it a 0?
    #hdu=pf.PrimaryHDU()
    #hdulist=pf.HDUList([hdu, hdu_tab])

    #Write the fits file out
    hdu_tab.writeto(direc+'/'+outfilen)


#===============================================================================
#===============================================================================
#===============================================================================

def binary_subset(direc, wantkeys, catfile='', add2filen='_subset', version=2):
    #Takes the whole catalog (reads in from binary) and writes the catalog in binary
    #to a new, smaller binary file

    #Read in the binary catalog
    print "binary_subset says: Reading the binary catalog from ", catfile
    if version==1:
        cat=read_merged_binary(direc+'/'+catfile)
    elif version==2:
        cat=read_merged_binary_v2(direc)
    else:
        print "binary_subset says: I don't know of a version ", version

    #Check to make sure that all the desired keys are in the catalog
    print "binary_subset says: Checking to make sure the catalog has the requested keys"
    ks=cat.keys()
    keepkeys=[]
    for k in wantkeys:
        if k in ks:
            keepkeys.append(k)
        else:
            print "binary subset says: The catalog you gave me doesn't have key ", k, ".  Leaving it out."

    #Now pare down the catalog and pass it to the appropriate binary writing
    smallcat={}
    for k in keepkeys:
        smallcat[k]=cat[k]
        
    if version==1:
        print "binary_subset says: Paring the dictionary down and printing the subset to ", add2filen
        write_merged_binary(smallcat, direc, write_to=add2filen)
    elif version==2:
        print "binary_subset says: Paring the dictionary down and printing the subset to bin_catalog", add2filen
        write_merged_binary_v2(smallcat, direc, add2name=add2filen)

#===============================================================================
#===============================================================================
#===============================================================================

def fits_subset(filen, wantkeys, add2filen='_subset'):
    #Reads a fits file and writes a copy of it with only wantkeys in it

    #Read the whole thing in
    print "fits_subset says: reading in the file"
    cat=read_fits(filen)

    #Check to make sure that all the desired keys are in the catalog
    print "fits_subset says: Checking to make sure the catalog has the requested keys"
    ks=cat.keys()
    keepkeys=[]
    for k in wantkeys:
        if k in ks:
            keepkeys.append(k)
        else:
            print "fits_subset says: The catalog you gave me doesn't have key ", k, ".  Leaving it out."

    #Now pare down the catalog and pass it to the fits writing routine
    smallcat={}
    for k in keepkeys:
        smallcat[k]=cat[k]

    #Create the new file name and call write_fits to write the file
    newfilen=filen.rstrip('.fits')+add2filen+'.fits'
    print "fits_subset says: passing the pared-down catalog to write_fits to ", newfilen
    write_fits(smallcat, newfilen)
    
#===============================================================================
#===============================================================================
#===============================================================================

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

    
#===============================================================================
#===============================================================================
#===============================================================================

def get_binary_info():
    #I wanted the key order and type string in one place so when I change it here,
    #it will change everywhere

    key_order=['halo_id', 'gal_id', 'v_disk', 'r_disk', 'sigma_bulge', 'r_bulge', 'mhalo', 'rhalo', 'vhalo', 'm_strip', 'mstar', 'mstar_burst', 'mcold', 'mbulge', 'mpseudobulge', 'mBH', 'maccdot', 'maccdot_radio', 'Zstar', 'Zcold', 'tau0', 'mstardot', 'sfr_burst', 'sfr_ave', 'meanage', 'tmerge', 'tmajmerge', 'mu_merge', 't_sat', 'r_fric', 'cosi', 't_reheat', 'mstar_diffuse', 'r_cool', 'weight', 'redshift', 'V_vir', 'Z_hot', 'sfr', 'm_vir', 'V_max', 'mcooldot', 'c_NFW', 'halo_id_nbody', 'ngal', 'spin', 'r_vir', 'm_hot', 'BJ', 'CFHTLS_g_bulge', 'musyc_u38_bulge', 'acsf850lp', 'NUV_obs_bulge', 'musyc_z_bulge', 'wfc3f125w_bulge', 'deep_I_bulge', 'J125_rest', 'wfc3f275w', 'RJ', 'RJ_bulge', 'musyc_R_bulge', 'sdss_u_bulge', 'musyc_u', 'wfc3f275w_bulge', 'musyc_z', 'HawkI_K_bulge', 'deep_I', 'BJ_bulge', 'musyc_J_bulge', 'acsf775w', 'deep_R', 'H160_rest_bulge', 'musyc_R', 'musyc_V', 'musyc_B', 'HawkI_J_bulge', 'musyc_K', 'musyc_J', 'musyc_I', 'musyc_H', 'sdss_g_bulge', 'acsf814w', 'musyc_H_bulge', 'vmos_U', 'acsf435w', 'wfc3f336w', 'Y105_rest', 'HawkI_Y_bulge', 'TWOMASS_H_bulge', 'Y105_rest_bulge', 'FUV_rest', 'J125_rest_bulge', 'musyc_u_bulge', 'deep_B_bulge', 'acsf850lp_bulge', 'FUV_rest_bulge', 'wfc3f336w_bulge', 'wfc3f105w_bulge', 'ctio_U_bulge', 'acsf435w_bulge', 'TWOMASS_K', 'TWOMASS_H', 'VJ_bulge', 'wfc3f125w', 'UJ', 'NUV_rest_bulge', 'CFHTLS_z', 'sdss_g', 'sdss_i', 'sdss_i_bulge', 'sdss_r', 'sdss_u', 'musyc_V_bulge', 'sdss_z', 'acsf606w_bulge', 'CFHTLS_r', 'CFHTLS_u', 'wfc3f105w', 'acsf606w', 'ctio_U', 'CFHTLS_g', 'CFHTLS_i', 'HawkI_J', 'CFHTLS_u_bulge', 'HawkI_H', 'IJ_bulge', 'CFHTLS_i_bulge', 'wfc3f160w', 'NUV_rest', 'HawkI_Y', 'acsf814w_bulge', 'z', 'H160_rest', 'musyc_K_bulge', 'vmos_U_bulge', 'NUV_obs', 'CFHTLS_z_bulge', 'acsf775w_bulge', 'HawkI_K', 'musyc_u38', 'FUV_obs_bulge', 'CFHTLS_r_bulge', 'deep_B', 'musyc_B_bulge', 'HawkI_H_bulge', 'TWOMASS_K_bulge', 'VJ', 'deep_R_bulge', 'IJ', 'wfc3f160w_bulge', 'FUV_obs', 'UJ_bulge', 'sdss_z_bulge', 'sdss_r_bulge', 'musyc_I_bulge', 'halo_id_dust', 'CFHTLS_z_dust', 'RJ_dust', 'musyc_K_dust', 'NUV_obs_dust', 'HawkI_H_dust', 'J125_rest_dust', 'wfc3f105w_dust', 'wfc3f275w_dust', 'FUV_rest_dust', 'H160_rest_dust', 'HawkI_K_dust', 'ctio_U_dust', 'UJ_dust', 'wfc3f160w_dust', 'Y105_rest_dust', 'CFHTLS_u_dust', 'deep_R_dust', 'TWOMASS_K_dust', 'deep_I_dust', 'sdss_z_dust', 'acsf814w_dust', 'sdss_g_dust', 'VJ_dust', 'acsf435w_dust', 'BJ_dust', 'musyc_H_dust', 'acsf606w_dust', 'sdss_u_dust', 'wfc3f336w_dust', 'musyc_z_dust', 'musyc_I_dust', 'musyc_V_dust', 'HawkI_J_dust', 'vmos_U_dust', 'wfc3f125w_dust', 'musyc_R_dust', 'CFHTLS_g_dust', 'musyc_B_dust', 'IJ_dust', 'musyc_u38_dust', 'acsf850lp_dust', 'CFHTLS_r_dust', 'gal_id_dust', 'NUV_rest_dust', 'deep_B_dust', 'musyc_u_dust', 'CFHTLS_i_dust', 'FUV_obs_dust', 'sdss_r_dust', 'sdss_i_dust', 'HawkI_Y_dust', 'acsf775w_dust', 'z_dust', 'TWOMASS_H_dust', 'musyc_J_dust', 'irac_ch4_obs', 'pacs160_obs', 'pacs160', 'scuba850_obs', 'spire500_obs', 'pacs100', 'L_bol', 'pacs70', 'irac_ch1_obs', 'scuba850', 'irac_ch2_obs', 'spire350_obs', 'mips24', 'irac_ch3', 'irac_ch2', 'irac_ch1', 'irac_ch4', 'pacs70_obs', 'pacs100_obs', 'spire250', 'L_dust', 'spire500', 'irac_ch3_obs', 'spire250_obs', 'mips24_obs', 'spire350', 'FUV_dust', 'K_dust', 'I_dust', 'FUV', 'K', 'NUV_dust', 'K_bulge', 'NUV', 'I_bulge', 'NUV_bulge', 'FUV_bulge', 'I', 'beta', 'beta_dust']

    typestr=['i']*2+['f']*41+['l']+['f']*205+['f']*2

    #Since things were getting a little confusing with byte lengths, we're centralizing that as well
    nbytes=[]
    for i in range(len(typestr)):
        if (typestr[i]=='i') or (typestr[i]=='f') or (typestr[i]=='l'):
            nbytes.append(4)
        if (typestr[i]=='q'):
            nbytes.append(8)

    bands=['CFHTLS_g_bulge', 'musyc_u38_bulge', 'acsf850lp', 'NUV_obs_bulge', 'musyc_z_bulge', 'wfc3f125w_bulge', 'deep_I_bulge', 'J125_rest', 'wfc3f275w', 'RJ', 'RJ_bulge', 'musyc_R_bulge', 'sdss_u_bulge', 'musyc_u', 'wfc3f275w_bulge', 'musyc_z', 'HawkI_K_bulge', 'deep_I', 'BJ_bulge', 'musyc_J_bulge', 'acsf775w', 'deep_R', 'H160_rest_bulge', 'musyc_R', 'musyc_V', 'musyc_B', 'HawkI_J_bulge', 'musyc_K', 'musyc_J', 'musyc_I', 'musyc_H', 'sdss_g_bulge', 'acsf814w', 'musyc_H_bulge', 'vmos_U', 'acsf435w', 'wfc3f336w', 'Y105_rest', 'HawkI_Y_bulge', 'TWOMASS_H_bulge', 'Y105_rest_bulge', 'FUV_rest', 'J125_rest_bulge', 'musyc_u_bulge', 'deep_B_bulge', 'acsf850lp_bulge', 'FUV_rest_bulge', 'wfc3f336w_bulge', 'wfc3f105w_bulge', 'ctio_U_bulge', 'acsf435w_bulge', 'TWOMASS_K', 'TWOMASS_H', 'VJ_bulge', 'wfc3f125w', 'UJ', 'NUV_rest_bulge', 'CFHTLS_z', 'sdss_g', 'sdss_i', 'sdss_i_bulge', 'sdss_r', 'sdss_u', 'musyc_V_bulge', 'sdss_z', 'acsf606w_bulge', 'CFHTLS_r', 'CFHTLS_u', 'wfc3f105w', 'acsf606w', 'ctio_U', 'CFHTLS_g', 'CFHTLS_i', 'HawkI_J', 'CFHTLS_u_bulge', 'HawkI_H', 'IJ_bulge', 'CFHTLS_i_bulge', 'wfc3f160w', 'NUV_rest', 'HawkI_Y', 'acsf814w_bulge', 'H160_rest', 'musyc_K_bulge', 'vmos_U_bulge', 'NUV_obs', 'CFHTLS_z_bulge', 'acsf775w_bulge', 'HawkI_K', 'musyc_u38', 'FUV_obs_bulge', 'CFHTLS_r_bulge', 'deep_B', 'musyc_B_bulge', 'HawkI_H_bulge', 'TWOMASS_K_bulge', 'VJ', 'deep_R_bulge', 'IJ', 'wfc3f160w_bulge', 'FUV_obs', 'UJ_bulge', 'sdss_z_bulge', 'sdss_r_bulge', 'musyc_I_bulge', 'CFHTLS_z_dust', 'RJ_dust', 'musyc_K_dust', 'NUV_obs_dust', 'HawkI_H_dust', 'J125_rest_dust', 'wfc3f105w_dust', 'wfc3f275w_dust', 'FUV_rest_dust', 'H160_rest_dust', 'HawkI_K_dust', 'ctio_U_dust', 'UJ_dust', 'wfc3f160w_dust', 'Y105_rest_dust', 'CFHTLS_u_dust', 'deep_R_dust', 'TWOMASS_K_dust', 'deep_I_dust', 'sdss_z_dust', 'acsf814w_dust', 'sdss_g_dust', 'VJ_dust', 'acsf435w_dust', 'BJ_dust', 'musyc_H_dust', 'acsf606w_dust', 'sdss_u_dust', 'wfc3f336w_dust', 'musyc_z_dust', 'musyc_I_dust', 'musyc_V_dust', 'HawkI_J_dust', 'vmos_U_dust', 'wfc3f125w_dust', 'musyc_R_dust', 'CFHTLS_g_dust', 'musyc_B_dust', 'IJ_dust', 'musyc_u38_dust', 'acsf850lp_dust', 'CFHTLS_r_dust', 'NUV_rest_dust', 'deep_B_dust', 'musyc_u_dust', 'CFHTLS_i_dust', 'FUV_obs_dust', 'sdss_r_dust', 'sdss_i_dust', 'HawkI_Y_dust', 'acsf775w_dust', 'TWOMASS_H_dust', 'musyc_J_dust', 'irac_ch4_obs', 'pacs160_obs', 'pacs160', 'scuba850_obs', 'spire500_obs', 'pacs100', 'L_bol', 'pacs70', 'irac_ch1_obs', 'scuba850', 'irac_ch2_obs', 'spire350_obs', 'mips24', 'irac_ch3', 'irac_ch2', 'irac_ch1', 'irac_ch4', 'pacs70_obs', 'pacs100_obs', 'spire250', 'L_dust', 'spire500', 'irac_ch3_obs', 'spire250_obs', 'mips24_obs', 'spire350', 'FUV_dust', 'K_dust', 'I_dust', 'FUV', 'K', 'NUV_dust', 'K_bulge', 'NUV', 'I_bulge', 'NUV_bulge', 'FUV_bulge', 'I']

    r={}
    r['key_order']=key_order
    r['typestr']=typestr
    r['bands']=bands
    r['bytes']=nbytes

    return r


