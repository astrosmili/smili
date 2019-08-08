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
        cen_lambda (float,default=-1.):
            Regularization parameter for the centroid regularization.
            If negative then, this regularization won't be used.
            You should NOT use this regularization if you will use the
            full complex visibilities.
        cen_alpha (float, default=3):
            The power to be used in the centroid regularizaion.
            cen_power = 1 gives the exact center-of-mass regularization, while
            higher value will work as the peak fixing regularization.
        niter (int,defalut=100):
            The number of iterations.
        nonneg (boolean,default=True):
            If nonneg=True, the problem is solved with non-negative constrants.
        totalflux (float, default=None):
            Total flux of the source.
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
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux / tfd_lambda")
        return -1
    elif ((vistable is None) and (amptable is None) and
          ((totalflux is None) or (tfd_lambda <= 0))):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux must be constrained")
        return -1
    # Guess the Total flux
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
            l1_priorarr[:] = totalflux/Npix
        else:
            if imregion is None:
                l1_priorarr = l1_prior.data[0,0].reshape(Nyx)
            else:
                l1_priorarr = l1_prior.data[0,0][winidx]
        l1_priorarr *= totalflux/l1_priorarr.sum()
        l1_wgt = fortlib.image.init_l1reg(np.float64(l1_priorarr))
        l1_nwgt = len(l1_wgt)
        l1_l = l1_lambda
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
            tv_priorarr[:] = totalflux/Npix
            tv_isflat = True
        else:
            if imregion is None:
                tv_priorarr = tv_prior.data[0,0].reshape(Nyx)
            else:
                tv_priorarr = tv_prior.data[0,0][winidx]
            tv_isflat = False
        tv_priorarr *= totalflux/tv_priorarr.sum()
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
            tsv_priorarr[:] = totalflux/Npix
            tsv_isflat = True
        else:
            if imregion is None:
                tsv_priorarr = tsv_prior.data[0,0].reshape(Nyx)
            else:
                tsv_priorarr = tsv_prior.data[0,0][winidx]
            tsv_isflat = False
        tsv_priorarr *= totalflux/tsv_priorarr.sum()
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
            kl_priorarr[:] = totalflux/Npix
        else:
            if imregion is None:
                kl_priorarr = kl_prior.data[0,0].reshape(Nyx)
            else:
                kl_priorarr = kl_prior.data[0,0][winidx]
        kl_priorarr *= totalflux/kl_priorarr.sum()
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
            gs_priorarr[:] = totalflux/Npix
        else:
            if imregion is None:
                gs_priorarr = gs_prior.data[0,0].reshape(Nyx)
            else:
                gs_priorarr = gs_prior.data[0,0][winidx]
        gs_priorarr *= totalflux/gs_priorarr.sum()
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
    else:
        print("  Initialize Total Flux Density regularization")
        tfd_tgtfd = np.float64(totalflux)
        tfd_l = fortlib.image.init_tfdreg(
            tfd_l_in = np.float64(tfd_lambda),
            tfd_tgtfd = np.float64(tfd_tgtfd),
            tfd_tgter = np.float64(tfd_tgterror))
    #

    #   Centroid regularization
    if cen_lambda <= 0:
        cen_l = -1
        cen_alpha = 1
    else:
        cen_l = cen_lambda

    # Rt regularization
    #lambrt_sim = rt_lambda / (2 * fluxscale**2 * Nyx * Nt)
    # Ri regularization
    #lambri_sim = ri_lambda / (fluxscale**2 * Nyx * Nt)


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
        amp = np.array(vistable["amp"], dtype=np.float64)
        vfcvr = np.float64(amp*np.cos(phase))
        vfcvi = np.float64(amp*np.sin(phase))
        varfcv = np.square(np.array(vistable["sigma"], dtype=np.float64))
        del phase, amp

    # Visibility Amplitude
    if amptable is None:
        isamp = False
        vamp = dammyreal
        varamp = dammyreal
    else:
        isamp = True
        vamp = np.array(amptable["amp"], dtype=np.float64)
        varamp = np.square(np.array(amptable["sigma"], dtype=np.float64))

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
        iin=np.float64(Iin),
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
        cen_l=np.float64(cen_l),
        cen_alpha=np.float64(cen_alpha),
        # Regularization Parameters for dynamical imaging
        rt_l=np.float64(rt_lambda),
        ri_l=np.float64(ri_lambda),
        rs_l=np.float64(rs_lambda),
        rf_l=np.float64(rf_lambda),
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
        # Following 3 parameters are for L-BFGS-B
        m=np.int32(lbfgsbprms["m"]), factr=np.float64(lbfgsbprms["factr"]),
        pgtol=np.float64(lbfgsbprms["pgtol"])
    )
    Iout = output[0]
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
