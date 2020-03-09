#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili. This module saves some common functions,
variables, and data types in the smili module.
'''
import numpy as np

def set_ompnumthreads(numthreads, variable="OMP_NUM_THREADS"):
    '''
    Set the number of threads for Open MP

    Args:
        numthreads (integer, required):
            The number of threads
        variable (str, default:"OMP_NUM_THREADS"):
            The variable name of the number of threads for Open MP.
    '''
    import os

    print("export %s=%d"%(variable,numthreads))
    os.environ[variable] = '%d'%(numthreads)

def set_nproc(nproc):
    '''
    Set the number of threads for Open MP 
    '''
    global __smili_nproc

    print("The number of processes is set to %d."%(nproc))
    __smili_nproc = nproc

def fluxconv(unit1="Jy", unit2="Jy"):
    '''
    convert flux units.
    Availables are [Jy, mJy, uJy, si, cgs]
    '''
    if unit1 == unit2:
        return 1

    # Convert from unit1 to "jy"
    u1low = unit1.lower()
    u2low = unit2.lower()
    if   u1low == "jy":
        conv = 1.
    elif u1low == "mjy":
        conv = 1e-3
    elif u1low == "ujy":
        conv = 1e-6
    elif u1low == "si":
        conv = 1e-26
    elif u1low == "cgs":
        conv = 1e-23
    else:
        print("Error: unit1=%s is not supported" % (unit1))
        return -1

    # Convert from "jy" to u2low
    if u2low == "jy":
        pass
    elif u2low == "mjy":
        conv *= 1e3
    elif u2low == "ujy":
        conv *= 1e6
    elif u2low == "si":
        conv *= 1e26
    elif u2low == "cgs":
        conv *= 1e23
    else:
        print("Error: unit2=%s is not supported" % (unit2))
        return -1
    
    return conv

def angconv(unit1="deg", unit2="deg"):
    '''
    return a conversion factor from unit1 to unit2
    Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
    '''
    if unit1 == unit2:
        return 1

    # Convert from unit1 to "arcsec"
    u1low = unit1.lower()
    u2low = unit2.lower()
    if u1low == "deg":
        conv = 3600.
    elif u1low == "rad":
        conv = 180. * 3600. / np.pi
    elif u1low == "arcmin" or u1low == "amin":
        conv = 60.
    elif u1low == "arcsec" or u1low == "asec":
        conv = 1.
    elif u1low == "mas":
        conv = 1e-3
    elif u1low == "uas":
        conv = 1e-6
    else:
        print("Error: unit1=%s is not supported" % (unit1))
        return -1

    # Convert from "arcsec" to u2low
    if u2low == "deg":
        conv /= 3600.
    elif u2low == "rad":
        conv /= (180. * 3600. / np.pi)
    elif u2low == "arcmin" or u2low == "amin":
        conv /= 60.
    elif u2low == "arcsec" or u2low == "asec":
        pass
    elif u2low == "mas":
        conv *= 1e3
    elif u2low == "uas":
        conv *= 1e6
    else:
        print("Error: unit2=%s is not supported" % (unit2))
        return -1
    return conv

def solidang(x=1.,y=None,angunit="deg",satype="pixel",angunitout=None):
    '''
    Return the solid angle of the pixel or beam

    Args:
        x, y (float):
            if type="pixel", the pixel size for x and y directions.
            if type="beam", the major/minor-axis FWHM size of the Gaussian beam
            if y is not given, y := x.
        angunit (str):
            angular unit of x, y
        satype (str):
            type of the solid angle. Availables are 'pixel' or 'beam'.
        angunitout (str; default=None):
            Angular unit to be used for the solid angle.
            The solid angle of angunitout^2 will be adopted.
            If not specified, angunit will be used.
    Returns:
        solidangle in angunitout^2
    '''

    if y is None:
        y = x
    if angunitout is None:
        angunitout = angunit

    if   satype.lower()[0] == "b":
        beamcorr = np.pi/(4*np.log(2))
    elif satype.lower()[0] == "p":
        beamcorr = 1.
    else:
        raise ValueError("type must be 'beam' or 'pixel'")

    if angunit == angunitout:
        aconv = 1
    else:
        aconv = angconv(angunit, angunitout)

    return np.abs(beamcorr * x * y * aconv**2)

def saconv(x1=1.,y1=None,angunit1="deg",satype1="pixel",
           x2=1.,y2=None,angunit2=None, satype2=None):
    '''
    return a conversion factor between given sizes of the pixel or beam
    '''

    if angunit2 is None:
        angunit2 = angunit1
    if satype2 is None:
        satype2 = satype1
    solidang1 = solidang(x1,y1,angunit1,satype1,angunitout=angunit1)
    solidang2 = solidang(x2,y2,angunit2,satype2,angunitout=angunit1)
    return np.abs(solidang2/solidang1)

def interpolation1d(xd,yd,xi,kind="cubic",bounds_error=False, fill_value=np.nan):
    from scipy.interpolate import interp1d

    f = interp1d(xd,yd,kind=kind,bounds_error=bounds_error, fill_value=fill_value)
    return f(xi)

def average1d(xd,yd,wd,xa,width,minpoint=2):
    idx = np.where(wd>0)
    xd2 = xd[idx]
    yd2 = yd[idx]
    wd2 = wd[idx]
    del idx

    Na = len(xa)
    ya = np.zeros(Na, dtype=yd.dtype)
    wa = np.zeros(Na, dtype=np.float64)
    for i in range(Na):
        idx = np.where(np.abs(xd2-xa[i]) <= width/2.)
        if idx[0].shape[0] == 0:
            continue

        wa[i] = np.sum(wd2[idx])
        ya[i] = np.sum(yd2[idx]*wd2[idx])/wa[i]
        if idx[0].shape[0] < minpoint:
            wa[i] *= -1
    return ya, wa
