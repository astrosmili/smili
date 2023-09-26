#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of sparselab for imaging static images.
'''
__author__ = "Sparselab Developer Team"
# -------------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------------
# standard modules
import os
import copy
import collections
import itertools
import time

# numerical packages
import numpy as np
import pandas as pd
from scipy import interpolate

# astropy
import astropy.time as at

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# internal modules
from .. import util, imdata, fortlib, uvdata
tools = uvdata.uvtable.tools

#-------------------------------------------------------------------------
# Default Parameters
#-------------------------------------------------------------------------
lbfgsbprms = {
    "m": 5,
    "factr": 1e1,
    "pgtol": 0.
}


#-------------------------------------------------------------------------
# Reconstract static imaging
#-------------------------------------------------------------------------
def imaging3d(
        initmovie,
        imregion=None,
        vistable=None,
        amptable=None,
        bstable=None,
        catable=None,
        w_vis=1,
        w_amp=1,
        w_bs=1,
        w_ca=1,
        l1_lambda=-1.,
        l1_prior=None,
        l1_noise=1e-2,
        l1_type=1,
        l1_floor=1e-1,
        sm_lambda=-1,
        sm_maj=50.,
        sm_min=50.,
        sm_phi=0.,
        tv_lambda=-1.,
        tv_prior=None,
        tsv_lambda=-1.,
        tsv_prior=None,
        kl_lambda=-1.,
        kl_prior=None,
        gs_lambda=-1,
        gs_prior=None,
        tfd_lambda=-1,
        tfd_tgterror=0.01,
        lc_lambda = -1,
        lc_array = None,
        lc_tgterror = 0.01,
        lc_normalize=False,
        cen_lambda=-1,
        cen_alpha=3,
        rt_lambda=-1.,
        rt_prior = None,
        ri_lambda=-1.,
        ri_prior = None,
        rs_lambda   =-1,
        rs_prior = None,
        rf_lambda   =-1,
        rf_prior = None,
        niter=1000,
        nonneg=True,
        nprint=500,
        totalflux=None,
        inorm=1.,
        istokes=0, ifreq=0,
        output='list'):
    '''
    FFT 3d imaging (movie) with closure quantities.

    Args:
        initmovie (MOVIE):
            Initial movie model for fft imaging
        imregion (IMRegion, default=None)
            Image region to set image windows.
        vistable (VisTable, default=None):
            Visibility table containing full complex visibilities.
        amptable (VisTable, default=None):
            Amplitude table.
        bstable (BSTabke, default=None):
            Closure phase table.
        catable (CATable, default=None):
            Closure amplitude table.
        l1_lambda (float,default=-1.):
            Regularization parameter for L1 term.
            If negative then, this regularization won't be used.
        l1_prior (IMFITS, default=None):
            Prior image to be used to compute the weighted l1-norm.
            If not specified, the flat prior will be used.
            This prior image will be normalized with the total flux estimator.
        l1_noise (float, default=1e-2):
            Typical noise levels relative to the peak value of l1_prior image.
        l1_type (int, default=1):
            Type of l1 definition.
            1 : EHT paper 4
            2 : test l1 description
        l1_floor (float, default=1e-1):
            Only for l1_type=2 case.
            Flux ratio of the l1 noise floor: totalflux of the floor region is totalflux * l1_floor.
            In this case, l1_noise is a cutoff vaule of the l1_prior.
        sm_lambda (float,default=-1):
            Regularization parameter for second moment.
            If negative then, this regularization won't be used.
        sm_maj and sm_min (float):
            FWHM of major and minor axis for elliptical gaussian
        sm_phi (float):
            position angle of the elliptical gaussian
        tv_lambda (float,default=-1.):
            Regularization parameter for total variation.
            If negative then, this regularization won't be used.
        tv_prior (IMFITS, default=None):
            Prior image to be used to compute the weighted TV term.
            If not specified, the flat prior will be used.
            This prior image will be normalized with the total flux estimator.
        tsv_lambda (float,default=-1.):
            Regularization parameter for total squared variation.
            If negative then, this regularization won't be used.
        tsv_prior (IMFITS, default=None):
            Prior image to be used to compute the weighted TSV term.
            If not specified, the flat prior will be used.
            This prior image will be normalized with the total flux estimator.
        kl_lambda (float,default=-1.):
            Regularization parameter for the KL divergence (relative entropy).
            If negative then, this regularization won't be used.
        kl_prior (IMFITS, default=None):
            Prior image to be used to compute the weighted TSV term.
            If not specified, the flat prior will be used.
            This prior image will be normalized with the total flux estimator.
        gs_lambda (float,default=-1.):
            Regularization parameter for the GS entropy (relative entropy).
            If negative then, this regularization won't be used.
        gs_prior (IMFITS, default=None):
            Prior image to be used to compute the weighted TSV term.
            If not specified, the flat prior will be used.
            This prior image will be normalized with the total flux estimator.
        tfd_lambda (float,default=-1.):
            Regularization parameter for the total flux regularization.
            If negative then, this regularization won't be used.
            The target flux can be specified with "totalflux". If it is not specified,
            then it will be guessed from the maximum visibility amplitudes.
        tfd_tgterror (float, default=0.01):
            The target accracy of the total flux regulariztion. For instance,
            tfd_tgterror = 0.01 and tfd_lambda = 1 will give the cost function of
            unity when the fractional error of the total flux is 0.01.
        lc_lambda (float, default=-1):
            Regularization parameters for the light curve regularization.
            If negative then, this regularization won't be used.
        lc_array (float array, default=None):
            Array of light curve regularizer.
        lc_tgterror (float, default=0.01):
            The target accuracy of the light curve regularization.
            The definition is the same as that of tfd_tgterror.
        lc_normalize (boolean, default=False):
            Method for regularizer normalization with input light curve:
                False     : use constant total flux
                "static"  : static regularizers (l1, tv, tsv) are normalized
                "dynamic" : dynamical regularizers are normalized
                "both"    : both regularizers are normalized

        cen_lambda (float,default=-1.):
            Regularization parameter for the centroid regularization.
            If negative then, this regularization won't be used.
            You should NOT use this regularization if you will use the
            full complex visibilities.
        cen_alpha (float, default=3):
            The power to be used in the centroid regularizaion.
            cen_power = 1 gives the exact center-of-mass regularization, while
            higher value will work as the peak fixing regularization.

        rt_lambda (float, default=-1.):
            Regularization parameter for smoothing varying (Rt) regularizer.
            If negative then, this regularization won't be used.
        rt_prior  (IMFITS, default=None):
            Prior image to be used to compute the weighted Rt term.
            If not specified, the flat prior will be used.
        ri_lambda (float, default=-1.):
            Regularization parameter for averaged image (Ri) regularizer.
            If negative then, this regularization won't be used.
        ri_prior  (IMFITS, default=None):
            Prior image to be used to compute the weighted Ri term.
            If not specified, the flat prior will be used.
        rs_lambda (float, default=-1):
            Regularization parameter for dynamic entropy (Rs) regularizer.
            If negative then, this regularization won't be used.
        rs_prior  (IMFITS, default=None):
            Prior image to be used to compute the weighted Rs term.
            If not specified, the flat prior will be used.
        rf_lambda (float, default=-1.):
            Regularization parameter for totalflux continuous (Rf) regularizer.
            If negative then, this regularization won't be used.
        rf_prior  (IMFITS, default=None):
            Prior image to be used to compute the weighted Rf term.
            If not specified, the flat prior will be used.

        niter (int,defalut=100):
            The number of iterations.
        nonneg (boolean,default=True):
            If nonneg=True, the problem is solved with non-negative constrants.
        totalflux (float, default=None):
            Total flux of the source.
        inorm (float, default=+1):
            If a positive value is specified, all of the input image, amplitudes,
            expected total flux density, and related regularization functions are
            scaled so that the peak intensity of the scaled intensity is the
            specified value. This is essencial for the regularization functions
            to work effectively espicially for faint sources. If your image has
            the intensity lower than 10^{-4} Jy/pixel, then you should use inorm=1.
        nprint (integer, default=200):
            The summary of metrics will be printed with an interval specified
            with this number.
        istokes (int,default=0):
            The ordinal number of stokes parameters.
        ifreq (int,default=0):
            The ordinal number of frequencies.

    Returns:
        movie.MOVIE object
    '''


    # Sanity Check: Data
    if ((vistable is None) and (amptable is None) and
        (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1


    # Sort tables
    if vistable is not None:
        vistable = vistable.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
    if amptable is not None:
        amptable = amptable.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
    if bstable is not None:
        bstable = bstable.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
    if catable is not None:
        catable = catable.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)

    # Sanity Check: Total Flux constraint
    if ((vistable is None) and (amptable is None) and (totalflux is None) and (lc_array is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux / tfd_lambda")
        return -1
    elif ((vistable is None) and (amptable is None) and
          ((totalflux is None) or (tfd_lambda <= 0)) and (lc_array is None)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux must be constrained")
        return -1

    # Guess the Total flux
    elif ((totalflux is None) and (lc_array is not None)):
        totalflux = np.median(lc_array)
    elif totalflux is None:
        totalflux = []
        if vistable is not None:
            totalflux.append(vistable["amp"].max())
        if amptable is not None:
            totalflux.append(amptable["amp"].max())
        totalflux = np.max(totalflux)
    print("Total flux: %g Jy"%(totalflux))


    # Number of frames
    Nt = initmovie.Nt

    # Get initial movie
    Iin = [initmovie.images[i].data[istokes, ifreq] for i in range(Nt)]
    initimage = initmovie.images[0]
    #   Size of images
    Nx = initimage.header["nx"]
    Ny = initimage.header["ny"]
    Nyx = Nx * Ny
    #   Pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    xidx = np.arange(Nx) + 1
    yidx = np.arange(Ny) + 1
    xidx, yidx = np.meshgrid(xidx, yidx)
    Nxref = initimage.header["nxref"]
    Nyref = initimage.header["nyref"]
    dx_rad = np.deg2rad(initimage.header["dx"])
    dy_rad = np.deg2rad(initimage.header["dy"])

    # Get imaging area
    if imregion is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = [Iin[i].reshape(Nyx) for i in range(len(Iin))]
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
        Npix = len(Iin[0])
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        if isinstance(imregion, imdata.IMRegion):
            imagewin = imregion.imagewin(initimage,istokes,ifreq)
        elif isinstance(imregion, imdata.IMFITS):
            imagewin = imregion.data[0,0] > 0.5
        winidx = np.where(imagewin)
        Iin = [Iin[i][winidx] for i in range(len(Iin))]
        x = x[winidx]
        y = y[winidx]
        xidx = xidx[winidx]
        yidx = yidx[winidx]
        Npix = len(Iin[0])

    # Determining intensity-normalization factor
    if inorm < 0:
        inormfactr = 1.0
    else:
        if np.sum(np.concatenate(Iin)) != 0:
            inormfactr = inorm/np.max(np.concatenate(Iin))
        else:
            inormfactr = inorm/(totalflux/Npix)
    print("Intensity Scaling Factor: %g"%(inormfactr))
    totalflux_scaled = totalflux * inormfactr

    # Setup regularizatin functions
    #   l1-norm
    if l1_lambda <= 0:
        l1_wgt  = np.float64(np.asarray([0]))
        l1_nwgt = 1
        l1_l = -1
    else:
        print("  Initialize l1 regularization")
        if l1_prior is None:
            l1_priorarr = copy.deepcopy(Iin[0])
            l1_priorarr[:] = totalflux_scaled/Npix

        else:
            l1_l = l1_lambda

            if l1_type==1:
                if imregion is None:
                    l1_priorarr = l1_prior.data[0,0].reshape(Nyx)
                else:
                    l1_priorarr = l1_prior.data[0,0][winidx]
                l1_priorarr[np.where(np.abs(l1_priorarr)<np.abs(l1_priorarr).max()*l1_noise)] = np.abs(l1_priorarr).max()*l1_noise
                l1_priorarr *= totalflux_scaled/l1_priorarr.sum()
                l1_wgt = fortlib.image.init_l1reg(np.float64(l1_priorarr))
                l1_nwgt = len(l1_wgt)

            elif l1_type==2:
                print("Use preliminary L1 regularizer")
                if imregion is None:
                    l1_priorarr = l1_prior.data[0,0].reshape(Nyx)
                else:
                    l1_priorarr = l1_prior.data[0,0][winidx]

                l1_priorarr *= totalflux_scaled/np.sum(l1_priorarr)
                l1_priorarr[np.where(np.abs(l1_priorarr)<np.abs(l1_priorarr).max()*l1_noise)] = np.abs(l1_priorarr).max()*l1_floor

                # Concatenate each region
                l1_wgt = fortlib.image.init_l1reg(np.float64(l1_priorarr))
                l1_nwgt = len(l1_wgt)
                del l1_priorarr

    #
    # Second momentum regularization functions
    if sm_lambda <= 0:
        sm_l = -1
    else:
        print("Debug: set second moment parameters")
        sm_l = sm_lambda
        # print("sm_l,sm_maj,sm_min,sm_phi = %3.2g, %3.2g, %3.2g, %3.2g"%(sm_l,sm_maj,sm_min,sm_phi))

        dtheta = np.abs(initimage.angconv("deg",initimage.angunit)*initimage.header["dx"])

        # Normalization of the major and minor size and position angle
        sm_maj = (sm_maj/dtheta)**2/(8.*np.log(2.)) # FWHM (angunit) -> lambda_maj (pixel^2)
        sm_min = (sm_min/dtheta)**2/(8.*np.log(2.)) # FWHM (angunit) -> lambda_min (pixel^2)
        sm_phi = sm_phi/180.*np.pi
    #
    #   tv-norm
    if tv_lambda <= 0:
        tv_wgt  = np.float64(np.asarray([0]))
        tv_nwgt = 1
        tv_l = -1
    else:
        print("  Initialize TV regularization")
        if tv_prior is None:
            tv_priorarr = copy.deepcopy(Iin[0])
            tv_priorarr[:] = totalflux_scaled/Npix
            tv_isflat = True
        else:
            if imregion is None:
                tv_priorarr = tv_prior.data[0,0].reshape(Nyx)
            else:
                tv_priorarr = tv_prior.data[0,0][winidx]
            tv_isflat = False
        tv_priorarr *= totalflux_scaled/tv_priorarr.sum()
        tv_wgt = fortlib.image.init_tvreg(
            xidx = np.float32(xidx),
            yidx = np.float32(yidx),
            nx = np.float32(Nx),
            ny = np.float32(Ny),
            tv_isflat=tv_isflat,
            tv_prior=np.float64(tv_priorarr)
        )
        tv_nwgt = len(tv_wgt)
        tv_l = tv_lambda
        del tv_priorarr, tv_isflat
    #
    #   TSV
    if tsv_lambda <= 0:
        tsv_wgt  = np.float64(np.asarray([0]))
        tsv_nwgt = 1
        tsv_l = -1
    else:
        print("  Initialize TSV regularization")
        if tsv_prior is None:
            tsv_priorarr = copy.deepcopy(Iin[0])
            tsv_priorarr[:] = totalflux_scaled/Npix
            tsv_isflat = True
        else:
            if imregion is None:
                tsv_priorarr = tsv_prior.data[0,0].reshape(Nyx)
            else:
                tsv_priorarr = tsv_prior.data[0,0][winidx]
            tsv_isflat = False
        tsv_priorarr *= totalflux_scaled/tsv_priorarr.sum()
        tsv_wgt = fortlib.image.init_tsvreg(
            xidx = np.int32(xidx),
            yidx = np.int32(yidx),
            nx = np.float32(Nx),
            ny = np.float32(Ny),
            tsv_isflat=tsv_isflat,
            tsv_prior=np.float64(tsv_priorarr)
        )
        tsv_nwgt = len(tsv_wgt)
        tsv_l = tsv_lambda
        del tsv_priorarr, tsv_isflat
    #
    #   kl divergence
    if kl_lambda <= 0:
        kl_wgt  = np.float64(np.asarray([0]))
        kl_nwgt = np.int32(1)
        kl_l = -1
    else:
        print("  Initialize the KL Divergence.")
        if kl_prior is None:
            kl_priorarr = copy.deepcopy(Iin[0])
            kl_priorarr[:] = totalflux_scaled/Npix
        else:
            if imregion is None:
                kl_priorarr = kl_prior.data[0,0].reshape(Nyx)
            else:
                kl_priorarr = kl_prior.data[0,0][winidx]
        kl_priorarr *= totalflux_scaled/kl_priorarr.sum()
        kl_l, kl_wgt = fortlib.image.init_klreg(
            kl_l_in=np.float64(kl_lambda),
            kl_prior=np.float64(kl_priorarr)
        )
        kl_nwgt = len(kl_wgt)
        del kl_priorarr
    #
    #   gs divergence
    if gs_lambda <= 0:
        gs_wgt  = np.float64(np.asarray([0]))
        gs_nwgt = np.int32(1)
        gs_l = -1
    else:
        print("  Initialize the GS Entropy.")
        if gs_prior is None:
            gs_priorarr = copy.deepcopy(Iin[0])
            gs_priorarr[:] = totalflux_scaled/Npix
        else:
            if imregion is None:
                gs_priorarr = gs_prior.data[0,0].reshape(Nyx)
            else:
                gs_priorarr = gs_prior.data[0,0][winidx]
        gs_priorarr *= totalflux_scaled/gs_priorarr.sum()
        gs_wgt = fortlib.image.init_gsreg(
            gs_prior=np.float64(gs_priorarr)
        )
        gs_nwgt = len(gs_wgt)
        gs_l = gs_lambda
        del gs_priorarr
    #
    #   total flux regularization divergence
    if tfd_lambda <= 0:
        tfd_l = -1
        tfd_tgtfd = 1
        if len(lc_array) > 0 and (lc_normalize=="static" or lc_normalize=="both"):
            print("Calculate mean total flux for light curve renormalization")
            tfd_tgtfd = np.mean(lc_array) * inormfactr
    else:
        print("  Initialize Total Flux Density regularization")
        tfd_tgtfd = np.float64(totalflux_scaled)
        tfd_l = fortlib.image.init_tfdreg(
            tfd_l_in = np.float64(tfd_lambda),
            tfd_tgtfd = np.float64(tfd_tgtfd),
            tfd_tgter = np.float64(tfd_tgterror))
    #
    #   light curve regularization divergence
    lc_nidx = -1
    if lc_lambda <= 0:
        lc_l = np.zeros(Nt,dtype=np.float64)-1.
        lc_tgtfd = np.zeros(Nt,dtype=np.float64)+1.
    else:
        # check lengths of light curve arrays
        if len(lc_array)!= Nt:
            print("Error: Length of light array is inconsistent with time bin of the movie")
            return -1

        print("  Initialize Light curve regularization")
        lc_array = np.float64(lc_array)
        lc_tgtfd = lc_array * inormfactr
        lc_l = lc_lambda / (lc_tgterror * lc_tgtfd)**2
        #print(lc_tgterror)
        if lc_normalize=="static":
            print("Static regularizers are normalized with a light curve")
            lc_nidx=1
        elif lc_normalize=="dynamic":
            print("Dynamic regularizers are normalized with a light curve")
            lc_nidx=2
        elif lc_normalize=="both":
            print("Static and dynamic regularizers are normalized with a light curve")
            lc_nidx=3

    #   Centroid regularization
    if cen_lambda <= 0:
        cen_l = -1
        cen_alpha = 1
    else:
        cen_l = cen_lambda

    # Rt regularization
    if rt_lambda <= 0:
        rt_wgt  = np.float64(np.asarray([0]))
        rt_nwgt = 1
        rt_l = -1
    else:
        print("  Initialize Rt regularization")
        if rt_prior is None:
            rt_priorarr = copy.deepcopy(Iin[0])
            rt_priorarr[:] = 1./Npix
        else:
            if imregion is None:
                rt_priorarr = rt_prior.data[0,0].reshape(Nyx)
            else:
                rt_priorarr = rt_prior.data[0,0][winidx]

            rt_priorarr *= 1./rt_priorarr.sum()
            zeroeps = np.float64(1.e-5)
            rt_priorarr = np.sqrt(np.float64(rt_priorarr**2 + zeroeps**2))

        #rt_priorarr /= rt_priorarr.sum()
        rt_wgt = 1. / (Nt * Npix * rt_priorarr)
        rt_nwgt = len(rt_wgt)
        rt_l = rt_lambda
        del rt_priorarr

    # Ri regularization
    if ri_lambda <= 0:
        ri_wgt  = np.float64(np.asarray([0]))
        ri_nwgt = 1
        ri_l = -1
    else:
        print("  Initialize Ri regularization")
        if ri_prior is None:
            ri_priorarr = copy.deepcopy(Iin[0])
            ri_priorarr[:] = 1./Npix
        else:
            if imregion is None:
                ri_priorarr = ri_prior.data[0,0].reshape(Nyx)
            else:
                ri_priorarr = ri_prior.data[0,0][winidx]

            ri_priorarr *= 1./ri_priorarr.sum()
            zeroeps = np.float64(1.e-5)
            ri_priorarr = np.sqrt(np.float64(ri_priorarr**2 + zeroeps**2))

        #ri_priorarr /= ri_priorarr.sum()

        ri_wgt = 1. / (Nt * Npix * ri_priorarr)
        ri_nwgt = len(ri_wgt)
        ri_l = ri_lambda
        del ri_priorarr

    # Rs regularization
    if rs_lambda <= 0:
        rs_wgt  = np.float64(np.asarray([0]))
        rs_nwgt = 1
        rs_l = -1
    else:
        print("  Initialize Rs regularization")
        if rs_prior is None:
            rs_priorarr = copy.deepcopy(Iin[0])
            rs_priorarr[:] = totalflux_scaled/Npix
        else:
            if imregion is None:
                rs_priorarr = rs_prior.data[0,0].reshape(Nyx)
            else:
                rs_priorarr = rs_prior.data[0,0][winidx]

            rs_priorarr *= totalflux_scaled/rs_priorarr.sum()
            zeroeps = np.float64(1.e-10)
            rs_priorarr = np.sqrt(np.float64(rs_priorarr**2 + zeroeps))

        rs_priorarr /= rs_priorarr.sum()
        rs_wgt = 1. / (Nt * Npix * rs_priorarr)
        rs_nwgt = len(rs_wgt)
        rs_l = rs_lambda
        del rs_priorarr

    # Rs regularization
    if rf_lambda <= 0:
        rf_wgt  = np.float64(np.asarray([0]))
        rf_nwgt = 1
        rf_l = -1
    else:
        print("  Initialize Rf regularization")
        if rf_prior is None:
            rf_priorarr = copy.deepcopy(Iin[0])
            rf_priorarr[:] = totalflux_scaled/Npix
        else:
            if imregion is None:
                rf_priorarr = rf_prior.data[0,0].reshape(Nyx)
            else:
                rf_priorarr = rf_prior.data[0,0][winidx]
            rf_priorarr *= totalflux_scaled/rf_priorarr.sum()
            zeroeps = np.float64(1.e-10)
            rf_priorarr = np.sqrt(np.float64(rf_priorarr**2 + zeroeps))

        #rf_priorarr /= rf_priorarr.sum()
        rf_wgt = 1. / (Nt * Npix * rf_priorarr)
        rf_nwgt = len(rf_wgt)
        rf_l = rf_lambda
        del rf_priorarr


    dammyreal = np.zeros(1, dtype=np.float64)

    # Full Complex Visibility
    if vistable is None:
        isfcv = False
        vfcvr = dammyreal
        vfcvi = dammyreal
        varfcv = dammyreal
    else:
        isfcv = True
        phase = np.deg2rad(np.array(vistable["phase"], dtype=np.float64))
        amp = np.array(vistable["amp"], dtype=np.float64) * inormfactr
        vfcvr = np.float64(amp*np.cos(phase))
        vfcvi = np.float64(amp*np.sin(phase))
        varfcv = np.square(np.array(vistable["sigma"], dtype=np.float64)) * inormfactr**2
        del phase, amp

    # Visibility Amplitude
    if amptable is None:
        isamp = False
        vamp = dammyreal
        varamp = dammyreal
    else:
        isamp = True
        vamp = np.array(amptable["amp"], dtype=np.float64) * inormfactr
        varamp = np.square(np.array(amptable["sigma"], dtype=np.float64)) * inormfactr**2

    # Closure Phase
    if bstable is None:
        iscp = False
        cp = dammyreal
        varcp = dammyreal
    else:
        iscp = True
        cp = np.deg2rad(np.array(bstable["phase"], dtype=np.float64))
        varcp = np.square(
            np.array(bstable["sigma"] / bstable["amp"], dtype=np.float64))

    # Closure Amplitude
    if catable is None:
        isca = False
        ca = dammyreal
        varca = dammyreal
    else:
        isca = True
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))

    # get uv coordinates and uv indice
    if Nt>1:
        u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = tools.get_uvlist_loop(
            Nt=Nt,fcvconcat=vistable, ampconcat=amptable, bsconcat=bstable, caconcat=catable
        )
    else:
        u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = tools.get_uvlist(
            fcvtable=vistable, amptable=amptable, bstable=bstable, catable=catable
        )
        Nuvs = np.zeros(1,dtype=np.float64)
        Nuvs[0] = len(u)

    # normalize u, v coordinates
    u *= 2*np.pi*dx_rad
    v *= 2*np.pi*dy_rad

    # copy the initimage to the number of frames
    Iin = np.concatenate(Iin)

    # run imaging
    output = fortlib.fftim3d.imaging(
        # Images
        iin=np.float64(Iin*inormfactr),
        xidx=np.int32(xidx),
        yidx=np.int32(yidx),
        nxref=np.float64(Nxref),
        nyref=np.float64(Nyref),
        nx=np.int32(Nx),
        ny=np.int32(Ny),
        # 3D frames
        nz=np.int32(Nt),
        # UV coordinates,
        u=u,
        v=v,
        nuvs=np.int32(Nuvs),
        # Imaging Parameter
        niter=np.int32(niter),
        nonneg=nonneg,
        nprint=np.int32(nprint),
        # Regularization Parameters
        l1_l=np.float64(l1_l),
        l1_wgt=np.float64(l1_wgt),
        l1_nwgt=np.int32(l1_nwgt),
        sm_l=np.float64(sm_l),
        sm_maj=np.float64(sm_maj),
        sm_min=np.float64(sm_min),
        sm_phi=np.float64(sm_phi),
        tv_l=np.float64(tv_l),
        tv_wgt=np.float64(tv_wgt),
        tv_nwgt=np.int32(tv_nwgt),
        tsv_l=np.float64(tsv_l),
        tsv_wgt=np.float64(tsv_wgt),
        tsv_nwgt=np.int32(tsv_nwgt),
        kl_l=np.float64(kl_l),
        kl_wgt=np.float64(kl_wgt),
        kl_nwgt=np.int32(kl_nwgt),
        gs_l=np.float64(gs_l),
        gs_wgt=np.float64(gs_wgt),
        gs_nwgt=np.int32(gs_nwgt),
        tfd_l=np.float64(tfd_l),
        tfd_tgtfd=np.float64(tfd_tgtfd),
        lc_l=np.float64(lc_l),
        lc_tgtfd=np.float64(lc_tgtfd),
        lc_nidx=np.int32(lc_nidx),
        cen_l=np.float64(cen_l),
        cen_alpha=np.float64(cen_alpha),
        # Regularization Parameters for dynamical imaging
        rt_l=np.float64(rt_lambda),
        rt_wgt=np.float64(rt_wgt),
        rt_nwgt=np.float64(rt_nwgt),
        ri_l=np.float64(ri_lambda),
        ri_wgt=np.float64(ri_wgt),
        ri_nwgt=np.float64(ri_nwgt),
        rs_l=np.float64(rs_lambda),
        rs_wgt=np.float64(rs_wgt),
        rs_nwgt=np.float64(rs_nwgt),
        rf_l=np.float64(rf_lambda),
        rf_wgt=np.float64(rf_wgt),
        rf_nwgt=np.float64(rf_nwgt),
        # Full Complex Visibilities
        isfcv=isfcv,
        uvidxfcv=np.int32(uvidxfcv),
        vfcvr=np.float64(vfcvr),
        vfcvi=np.float64(vfcvi),
        varfcv=np.float64(varfcv),
        wfcv=np.float64(w_vis),
        # Visibility Ampltiudes
        isamp=isamp,
        uvidxamp=np.int32(uvidxamp),
        vamp=np.float64(vamp),
        varamp=np.float64(varamp),
        wamp=np.float64(w_amp),
        # Closure Phase
        iscp=iscp,
        uvidxcp=np.int32(uvidxcp),
        cp=np.float64(cp),
        varcp=np.float64(varcp),
        wcp=np.float64(w_bs),
        # Closure Amplituds
        isca=isca,
        uvidxca=np.int32(uvidxca),
        ca=np.float64(ca),
        varca=np.float64(varca),
        wca=np.float64(w_ca),
        # intensity scale factor applied
        inormfactr=np.float64(inormfactr),
        # Following 3 parameters are for L-BFGS-B
        m=np.int32(lbfgsbprms["m"]), factr=np.float64(lbfgsbprms["factr"]),
        pgtol=np.float64(lbfgsbprms["pgtol"]),
        npix=xidx.shape[0]
    )
    Iout = output[0]/inormfactr
    outmovie = copy.deepcopy(initmovie)
    for it in range(Nt):
        for i in range(len(xidx)):
            outmovie.images[it].data[istokes, ifreq, yidx[i]-1, xidx[i]-1] = Iout[i+it*Npix]
        outmovie.images[it].update_fits()

    return outmovie


# ------------------------------------------------------------------------------
# Subfunctions
# ------------------------------------------------------------------------------
def frminp(Iout, Npix, Nt, Ntps):
    '''
    Ntps: the number of interpolated frames in a frame
    '''
    if len(Iout) != Npix*Nt:
        return -1

    totalframe = (Nt-1)*(Ntps+1)+1
    frames = np.linspace(0, Nt-1, totalframe)

    print("\n\n Interpolating %s frames to %s frames" %(Nt, totalframe))
    begin = time.time()

    fstack = []
    for ipix in range(Npix):
        pixinp = interpolate.interp1d(np.arange(Nt), Iout[ipix::Npix])
        fstack.append(pixinp(frames))

    fstack = np.array(fstack)
    Ifrm = fstack.reshape(Npix, totalframe).transpose()
    Ifrm = Ifrm.flatten()

    end = time.time()
    print(' it took %s seconds \n' %(end-begin))
    print("\n Making movie ... \n")

    return Ifrm
