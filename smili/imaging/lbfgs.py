#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of sparselab for imaging static images.
'''
__author__ = "Smili Developer Team"
# -------------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------------
# standard modules
import os
import copy
import collections
import itertools

# numerical packages
import numpy as np
import pandas as pd

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
def imaging(
        initimage,
        imregion=None,
        vistable=None,amptable=None, bstable=None, catable=None,
        lambl1=-1.,lambtv=-1.,lambtsv=-1.,lambcom=-1.,normlambda=True,
        reweight=False, dyrange=1e6,
        niter=1000,
        nonneg=True,
        compower=1.,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0):
    '''
    FFT imaging with closure quantities.

    Args:
        initimage (IMFITS):
            Initial model for fft imaging.
        imregion (IMRegion, default=None):
            Image region to set image windows.
        vistable (VisTable, default=None):
            Visibility table containing full complex visiblities.
        amptable (VisTable, default=None):
            Amplitude table.
        bstable (BSTable, default=None):
            Closure phase table.
        catable (CATable, default=None):
            Closure amplitude table.
        lambl1 (float,default=-1.):
            Regularization parameter for L1 term. If lambl1 <= 0,
            then L1 regularizar has no application.
        lambtv (float,default=-1.):
            Regularization parameter for total variation. If lambtv <= 0,
            then total-variation regularizar has no application.
        lambtsv (float,default=-1.):
            Regularization parameter for total squared variation. If lambtsv <= 0,
            then the regularizar of total squared variation has no application.
        lambcom (float,default=-1.):
            Regularization parameter for center of mass weighting. If lambtsv <= 0,
            then the regularizar has no application.
        normlambda (boolean,default=True):
            If normlabda=True, lambl1, lambtv, lambtsv, and lambmem are normalized
            with totalflux and the number of data points.
        reweight (boolean, default=False):
            If true, applying reweighting scheme (experimental)
        dyrange (boolean, default=1e2):
            The target dynamic range of reweighting techniques.
        niter (int,defalut=100):
            The number of iterations.
        nonneg (boolean,default=True):
            If nonneg=True, the problem is solved with non-negative constrants.
        compower (float, default=1.):
            Power of center of mass when lambcom > 0.
        totalflux (float, default=None):
            Total flux of the source.
        fluxconst (boolean,default=False):
            If fluxconst=True, total flux is fixed at the totalflux value.
        istokes (int,default=0):
            The ordinal number of stokes parameters.
        ifreq (int,default=0):
            The ordinal number of frequencies.

    Returns:
        imdata.IMFITS object
    '''
    # Sanity Check: Data
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Sanity Check: Total Flux constraint
    dofluxconst = False
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((vistable is None) and (amptable is None) and
          (totalflux is not None) and (fluxconst is False)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux will be constrained, although you do not set fluxconst=True.")
        dofluxconst = True
    elif fluxconst is True:
        dofluxconst = True

    # Sanity Check: Transform
    transform = None
    transtype = np.int32(0)
    transprm = np.float64(0)

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])

    # size of images
    Nx = initimage.header["nx"]
    Ny = initimage.header["ny"]
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    xidx = np.arange(Nx) + 1
    yidx = np.arange(Ny) + 1
    xidx, yidx = np.meshgrid(xidx, yidx)
    Nxref = initimage.header["nxref"]
    Nyref = initimage.header["nyref"]
    dx_rad = np.deg2rad(initimage.header["dx"])
    dy_rad = np.deg2rad(initimage.header["dy"])

    # apply the imaging area
    if imregion is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = Iin.reshape(Nyx)
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        imagewin = imregion.imagewin(initimage,istokes,ifreq)
        idx = np.where(imagewin)
        Iin = Iin[idx]
        x = x[idx]
        y = y[idx]
        xidx = xidx[idx]
        yidx = yidx[idx]

    # dammy array
    dammyreal = np.zeros(1, dtype=np.float64)

    if totalflux is None:
        totalflux = []
        if vistable is not None:
            totalflux.append(vistable["amp"].max())
        if amptable is not None:
            totalflux.append(amptable["amp"].max())
        totalflux = np.max(totalflux)

    # Full Complex Visibility
    Ndata = 0
    if dofluxconst:
        print("Total Flux Constraint: set to %g" % (totalflux))
        totalfluxdata = {
            'u': [0.],
            'v': [0.],
            'amp': [totalflux],
            'phase': [0.],
            'sigma': [1.]
        }
        totalfluxdata = pd.DataFrame(totalfluxdata)
        fcvtable = pd.concat([totalfluxdata, vistable], ignore_index=True)
    else:
        print("Total Flux Constraint: disabled.")
        if vistable is None:
            fcvtable = None
        else:
            fcvtable = vistable.copy()

    if fcvtable is None:
        isfcv = False
        vfcvr = dammyreal
        vfcvi = dammyreal
        varfcv = dammyreal
    else:
        isfcv = True
        phase = np.deg2rad(np.array(fcvtable["phase"], dtype=np.float64))
        amp = np.array(fcvtable["amp"], dtype=np.float64)
        vfcvr = np.float64(amp*np.cos(phase))
        vfcvi = np.float64(amp*np.sin(phase))
        varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
        Ndata += len(varfcv)
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
        Ndata += len(vamp)

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
        Ndata += len(cp)

    # Closure Amplitude
    if catable is None:
        isca = False
        ca = dammyreal
        varca = dammyreal
    else:
        isca = True
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)

    # Sigma for the total flux
    if dofluxconst:
        varfcv[0] = np.square(fcvtable.loc[0, "amp"] / (Ndata - 1.))

    # Normalize Lambda
    if (normlambda is True) and (reweight is not True):
        fluxscale = np.float64(totalflux)

        # convert Flux Scaling Factor
        fluxscale = np.abs(fluxscale) / Nyx
        #if   transform=="log":   # log correction
        #    fluxscale = np.log(fluxscale+transprm)-np.log(transprm)
        #elif transform=="gamma": # gamma correction
        #    fluxscale = (fluxscale)**transprm

        lambl1_sim = lambl1 / (fluxscale * Nyx)
        lambtv_sim = lambtv / (4 * fluxscale * Nyx)
        lambtsv_sim = lambtsv / (4 *fluxscale**2 * Nyx)
        #lambmem_sim = lambmem / np.abs(fluxscale*np.log(fluxscale) * Nyx)
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv
    lambmem_sim = -1

    # Center of Mass regularization
    lambcom_sim = lambcom # No normalization for COM regularization

    # get uv coordinates and uv indice
    u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = tools.get_uvlist(
        fcvtable=fcvtable, amptable=amptable, bstable=bstable, catable=catable
    )

    # normalize u, v coordinates
    u *= 2*np.pi*dx_rad
    v *= 2*np.pi*dy_rad

    # Reweighting
    if reweight:
        doweight=1
    else:
        doweight=-1

    # run imaging
    Iout = fortlib.fftim2d.imaging(
        # Images
        iin=np.float64(Iin),
        xidx=np.int32(xidx),
        yidx=np.int32(yidx),
        nxref=np.float64(Nxref),
        nyref=np.float64(Nyref),
        nx=np.int32(Nx),
        ny=np.int32(Ny),
        # UV coordinates,
        u=u,
        v=v,
        # Regularization Parameters
        lambl1=np.float64(lambl1_sim),
        lambtv=np.float64(lambtv_sim),
        lambtsv=np.float64(lambtsv_sim),
        lambmem=np.float64(lambmem_sim),
        lambcom=np.float64(lambcom_sim),
        doweight=np.int32(doweight),
        tgtdyrange=np.float64(dyrange),
        # Imaging Parameter
        niter=np.int32(niter),
        nonneg=nonneg,
        transtype=np.int32(transtype),
        transprm=np.float64(transprm),
        pcom=np.float64(compower),
        # Full Complex Visibilities
        isfcv=isfcv,
        uvidxfcv=np.int32(uvidxfcv),
        vfcvr=np.float64(vfcvr),
        vfcvi=np.float64(vfcvi),
        varfcv=np.float64(varfcv),
        # Visibility Ampltiudes
        isamp=isamp,
        uvidxamp=np.int32(uvidxamp),
        vamp=np.float64(vamp),
        varamp=np.float64(varamp),
        # Closure Phase
        iscp=iscp,
        uvidxcp=np.int32(uvidxcp),
        cp=np.float64(cp),
        varcp=np.float64(varcp),
        # Closure Amplituds
        isca=isca,
        uvidxca=np.int32(uvidxca),
        ca=np.float64(ca),
        varca=np.float64(varca),
        # Following 3 parameters are for L-BFGS-B
        m=np.int32(lbfgsbprms["m"]), factr=np.float64(lbfgsbprms["factr"]),
        pgtol=np.float64(lbfgsbprms["pgtol"])
    )

    outimage = copy.deepcopy(initimage)
    outimage.data[istokes, ifreq] = 0.
    for i in np.arange(len(xidx)):
        outimage.data[istokes, ifreq, yidx[i] - 1, xidx[i] - 1] = Iout[i]
    outimage.update_fits()
    return outimage

def statistics(
        initimage,
        imregion=None,
        vistable=None,amptable=None, bstable=None, catable=None,
        lambl1=-1.,lambtv=-1.,lambtsv=-1.,lambmem=-1.,lambcom=-1.,normlambda=True,
        reweight=False, dyrange=1e6,
        niter=1000,
        nonneg=True,
        compower=1.,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0):
    '''
    '''
    # Check Arguments
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # apply the imaging area
    if imregion is not None:
        imagewin = imregion.imagewin(initimage,istokes,ifreq)
    else:
        imagewin = None

    if totalflux is None:
        totalflux = []
        if vistable is not None:
            totalflux.append(vistable["amp"].max())
        if amptable is not None:
            totalflux.append(amptable["amp"].max())
        totalflux = np.max(totalflux)

    # Full Complex Visibility
    Ndata = 0
    if vistable is None:
        isfcv = False
        chisqfcv = 0.
        rchisqfcv = 0.
    else:
        isfcv = True
        chisqfcv, rchisqfcv = vistable.chisq_image(imfits=initimage,
                                                   mask=imagewin,
                                                   amptable=False,
                                                   istokes=istokes,
                                                   ifreq=ifreq)
        Ndata += len(vistable)*2

    # Visibility Amplitude
    if amptable is None:
        isamp = False
        chisqamp = 0.
        rchisqamp = 0.
    else:
        isamp = True
        chisqamp, rchisqamp = amptable.chisq_image(imfits=initimage,
                                                   mask=imagewin,
                                                   amptable=True,
                                                   istokes=istokes,
                                                   ifreq=ifreq)
        Ndata += len(amptable)

    # Closure Phase
    if bstable is None:
        iscp = False
        chisqcp = 0.
        rchisqcp = 0.
    else:
        iscp = True
        chisqcp, rchisqcp = bstable.chisq_image(imfits=initimage,
                                                mask=imagewin,
                                                istokes=istokes,
                                                ifreq=ifreq)
        Ndata += len(bstable)

    # Closure Amplitude
    if catable is None:
        isca = False
        chisqca = 0.
        rchisqca = 0.
    else:
        isca = True
        chisqca, rchisqca = catable.chisq_image(imfits=initimage,
                                                mask=imagewin,
                                                istokes=istokes,
                                                ifreq=ifreq)
        Ndata += len(catable)

    # Normalize Lambda
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    if imagewin is None:
        pixnum = Nyx
    else:
        pixnum = sum(imagewin.reshape(Nyx))

    if normlambda:
        fluxscale = np.float64(totalflux)

        # convert Flux Scaling Factor
        fluxscale = np.abs(fluxscale) / Nyx
        #if   transform=="log":   # log correction
        #    fluxscale = np.log(fluxscale+transprm)-np.log(transprm)
        #elif transform=="gamma": # gamma correction
        #    fluxscale = (fluxscale)**transprm

        lambl1_sim = lambl1 / (fluxscale * Nyx)
        lambtv_sim = lambtv / (4 * fluxscale * Nyx)
        lambtsv_sim = lambtsv / (4 *fluxscale**2 * Nyx)
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    # cost calculation
    l1 = initimage.imagecost(func="l1",out="cost",istokes=istokes,
                             ifreq=ifreq)
    tv = initimage.imagecost(func="tv",out="cost",istokes=istokes,
                             ifreq=ifreq)
    tsv = initimage.imagecost(func="tsv",out="cost",istokes=istokes,
                             ifreq=ifreq)
    if lambl1 > 0:
        l1cost = l1 * lambl1_sim
    else:
        lambl1 = 0.
        lambl1_sim = 0.
        l1cost = 0.

    if lambtv > 0:
        tvcost = tv * lambtv_sim
    else:
        lambtv = 0.
        lambtv_sim = 0.
        tvcost = 0.

    if lambtsv > 0:
        tsvcost = tsv * lambtsv_sim
    else:
        lambtsv = 0.
        lambtsv_sim = 0.
        tsvcost = 0.

    # Cost and Chisquares
    stats = collections.OrderedDict()
    stats["cost"] = l1cost + tvcost + tsvcost
    stats["chisq"] = chisqfcv + chisqamp + chisqcp + chisqca
    stats["rchisq"] = stats["chisq"] / Ndata
    stats["cost"] += stats["rchisq"]
    stats["isfcv"] = isfcv
    stats["isamp"] = isamp
    stats["iscp"] = iscp
    stats["isca"] = isca
    stats["chisqfcv"] = chisqfcv
    stats["chisqamp"] = chisqamp
    stats["chisqcp"] = chisqcp
    stats["chisqca"] = chisqca
    stats["rchisqfcv"] = rchisqfcv
    stats["rchisqamp"] = rchisqamp
    stats["rchisqcp"] = rchisqcp
    stats["rchisqca"] = rchisqca

    # Regularization functions
    stats["lambl1"] = lambl1
    stats["lambl1_sim"] = lambl1_sim
    stats["l1"] = l1
    stats["l1cost"] = l1cost
    stats["lambtv"] = lambtv
    stats["lambtv_sim"] = lambtv_sim
    stats["tv"] = tv
    stats["tvcost"] = tvcost
    stats["lambtsv"] = lambtsv
    stats["lambtsv_sim"] = lambtsv_sim
    stats["tsv"] = tsv
    stats["tsvcost"] = tsvcost

    return stats

def iterative_imaging(initimage, imageprm, Niter=10,
                      dothres=True, threstype="hard", threshold=0.3,
                      doshift=True, shifttype="peak",
                      dowinmod=False, imregion=None,
                      doconv=True, convprm={},
                      save_totalflux=False):
    oldimage = imaging(initimage, **imageprm)
    oldcost = statistics(oldimage, **imageprm)["cost"]
    for i in np.arange(Niter - 1):
        newimage = copy.deepcopy(oldimage)

        if dothres:
            if threstype == "soft":
                newimage = newimage.soft_threshold(threshold=threshold,
                                                   save_totalflux=save_totalflux)
            else:
                newimage = newimage.hard_threshold(threshold=threshold,
                                                   save_totalflux=save_totalflux)
        if doshift:
            if shifttype == "com":
                newimage = newimage.comshift(save_totalflux=save_totalflux)
            else:
                newimage = newimage.peakshift(save_totalflux=save_totalflux)

        # Edit Images
        if dowinmod and imregion is not None:
            newimage = imregion.winmod(newimage,
                                       save_totalflux=save_totalflux)

        if doconv:
            newimage = newimage.gauss_convolve(
                save_totalflux=save_totalflux, **convprm)

        # Imaging Again
        newimage = imaging(newimage, **imageprm)
        newcost = statistics(
            newimage, **imageprm)["cost"]

        if oldcost < newcost:
            print("No improvement in cost fucntions. Don't update image.")
        else:
            oldcost = newcost
            oldimage = newimage
    return oldimage

def plots(outimage, imageprm={}, filename=None, plotargs={'ms': 1., }):
    isinteractive = plt.isinteractive()
    backend = matplotlib.rcParams["backend"]

    if isinteractive:
        plt.ioff()
        matplotlib.use('Agg')

    nullfmt = NullFormatter()

    # Get statistics
    stats = statistics(outimage, **imageprm)

    # Open File
    if filename is not None:
        pdf = PdfPages(filename)

    # Save Image
    if filename is not None:
        util.matplotlibrc(nrows=1, ncols=1, width=600, height=600)
    else:
        matplotlib.rcdefaults()

    plt.figure()
    outimage.imshow()
    if filename is not None:
        pdf.savefig()
        plt.close()

    # fcv
    if stats["isfcv"] == True:
        table = imageprm["vistable"]

        # Get model data
        model = table.eval_image(imfits=outimage,
                                 mask=None,
                                 amptable=False,
                                 istokes=0,
                                 ifreq=0)
        resid = table.residual_image(imfits=outimage,
                                     mask=None,
                                     amptable=False,
                                     istokes=0,
                                     ifreq=0)

        if filename is not None:
            util.matplotlibrc(nrows=3, ncols=1, width=600, height=200)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot(datatype="amp",
                      color="black",
                      **plotargs)
        model.radplot(datatype="amp",
                      color="red",
                      errorbar=False,
                      **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        table.radplot(datatype="phase",
                      color="black",
                      **plotargs)
        model.radplot(datatype="phase",
                      color="red",
                      errorbar=False,
                      **plotargs)
        plt.xlabel("")

        ax = axs[2]
        plt.sca(ax)
        resid.radplot(datatype="real",
                      normerror=True,
                      errorbar=False,
                      color="blue",
                      **plotargs)
        resid.radplot(datatype="imag",
                      normerror=True,
                      errorbar=False,
                      color="red",
                      **plotargs)
        plt.axhline(0, color="black", ls="--")
        plt.ylabel("Normalized Residuals")
        plt.legend(ncol=2)

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        normresidr = resid["amp"]*np.cos(np.deg2rad(resid["phase"])) / resid["sigma"]
        normresidi = resid["amp"]*np.sin(np.deg2rad(resid["phase"])) / resid["sigma"]
        normresid = np.concatenate([normresidr, normresidi])
        N = len(normresid)
        ymin, ymax = ax.get_ylim()
        y = np.linspace(ymin, ymax, 1000)
        x = 1 / np.sqrt(2 * np.pi) * np.exp(-y * y / 2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)),
                 normed=True, orientation='horizontal')
        cax.plot(x, y, color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0, color="black", ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()

    if stats["isamp"] == True:
        table = imageprm["amptable"]

        # Get model data
        model = table.eval_image(imfits=outimage,
                                 mask=None,
                                 amptable=True,
                                 istokes=0,
                                 ifreq=0)
        resid = table.residual_image(imfits=outimage,
                                     mask=None,
                                     amptable=True,
                                     istokes=0,
                                     ifreq=0)

        if filename is not None:
            util.matplotlibrc(nrows=2, ncols=1, width=600, height=300)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot(datatype="amp",
                      color="black",
                      **plotargs)
        model.radplot(datatype="amp",
                      color="red",
                      errorbar=False,
                      **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        resid.radplot(datatype="amp",
                      normerror=True,
                      errorbar=False,
                      color="black",
                      **plotargs)
        plt.axhline(0, color="black", ls="--")
        ymin = np.min(resid["amp"]/resid["sigma"])*1.1
        plt.ylim(ymin,)
        plt.ylabel("Normalized Residuals")

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        normresid = resid["amp"] / resid["sigma"]
        N = len(normresid)
        ymin, ymax = ax.get_ylim()
        y = np.linspace(ymin, ymax, 1000)
        x = 1 / np.sqrt(2 * np.pi) * np.exp(-y * y / 2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)),
                 normed=True, orientation='horizontal')
        cax.plot(x, y, color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0, color="black", ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()

    # Closure Amplitude
    if stats["isca"] == True:
        table = imageprm["catable"]

        # Get model data
        model = table.eval_image(imfits=outimage,
                                 mask=None,
                                 istokes=0,
                                 ifreq=0)
        resid = table.residual_image(imfits=outimage,
                                     mask=None,
                                     istokes=0,
                                     ifreq=0)

        if filename is not None:
            util.matplotlibrc(nrows=2, ncols=1, width=600, height=300)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)


        table.radplot(uvdtype="ave", color="black", log=True,
                      **plotargs)
        model.radplot(uvdtype="ave", color="red", log=True,
                      errorbar=False, **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        resid.radplot(uvdtype="ave",
                      log=True,
                      normerror=True,
                      errorbar=False,
                      color="black",
                      **plotargs)
        plt.axhline(0, color="black", ls="--")
        plt.ylabel("Normalized Residuals")

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        normresid = resid["logamp"] / resid["logsigma"]
        N = len(normresid)
        ymin, ymax = ax.get_ylim()
        xmin = np.min(normresid)
        xmax = np.max(normresid)
        y = np.linspace(ymin, ymax, 1000)
        x = 1 / np.sqrt(2 * np.pi) * np.exp(-y * y / 2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)),
                 normed=True, orientation='horizontal')
        cax.plot(x, y, color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0, color="black", ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()

    # Closure Phase
    if stats["iscp"] == True:
        table = imageprm["bstable"]

        # Get model data
        model = table.eval_image(imfits=outimage,
                                 mask=None,
                                 istokes=0,
                                 ifreq=0)
        resid = table.residual_image(imfits=outimage,
                                     mask=None,
                                     istokes=0,
                                     ifreq=0)

        if filename is not None:
            util.matplotlibrc(nrows=2, ncols=1, width=600, height=300)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot(uvdtype="ave", color="black",
                      **plotargs)
        model.radplot(uvdtype="ave", color="red",
                      errorbar=False, **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        resid.radplot(uvdtype="ave",
                      normerror=True,
                      errorbar=False,
                      color="black",
                      **plotargs)
        plt.axhline(0, color="black", ls="--")
        normresid = resid["phase"] / (np.rad2deg(resid["sigma"] / resid["amp"]))
        N = len(normresid)
        ymin = np.min(normresid)*1.1
        ymax = np.max(normresid)*1.1
        plt.ylim(ymin,ymax)
        plt.ylabel("Normalized Residuals")
        del ymin,ymax
        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
        ymin, ymax = ax.get_ylim()
        y = np.linspace(ymin, ymax, 1000)
        x = 1 / np.sqrt(2 * np.pi) * np.exp(-y * y / 2.)
        cax.hist(normresid, bins=np.int(np.sqrt(N)),
                 normed=True, orientation='horizontal')
        cax.plot(x, y, color="red")
        cax.set_ylim(ax.get_ylim())
        cax.axhline(0, color="black", ls="--")
        cax.yaxis.set_major_formatter(nullfmt)
        cax.xaxis.set_major_formatter(nullfmt)
        if filename is not None:
            pdf.savefig()
            plt.close()

    # Close File
    if filename is not None:
        pdf.close()
    else:
        plt.show()

    # Reset rcsetting
    matplotlib.rcdefaults()
    if isinteractive:
        plt.ion()
        matplotlib.use(backend)


def pipeline(
        initimage,
        imagefunc=iterative_imaging,
        imageprm={},
        imagefargs={},
        lambl1s=[-1.],
        lambtvs=[-1.],
        lambtsvs=[-1.],
        workdir="./",
        skip=False,
        sumtablefile="summary.csv",
        docv=False,
        seed=1,
        nfold=10,
        cvsumtablefile="summary.cv.csv"):
    '''
    A pipeline imaging function using imaging and related fucntions.

    Args:
        initimage (imdata.IMFITS object):
            initial image
        imagefunc (function; default=uvdata.iterative_imaging):
            Function of imageing. It should be defined as
                def imagefunc(initimage, imageprm, **imagefargs)
        imageprm (dict-like; default={}):
            parameter sets for each imaging
        imagefargs (dict-like; default={}):
            parameter sets for imagefunc
        workdir (string; default = "./"):
            The directory where images and summary files will be output.
        sumtablefile (string; default = "summary.csv"):
            The name of the output csv file that summerizes results.
        docv (boolean; default = False):
            Do cross validation
        seed (integer; default = 1):
            Random seed to make CV data sets.
        nfold (integer; default = 10):
            Number of folds in CV.
        cvsumtablefile (string; default = "cvsummary.csv"):
            The name of the output csv file that summerizes results of CV.

    Returns:
        sumtable:
            pd.DataFrame table summerising statistical quantities of each
            parameter set.
        cvsumtable (if docv=True):
            pd.DataFrame table summerising results of cross validation.
    '''
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    cvworkdir = os.path.join(workdir,"cv")
    if docv:
        if not os.path.isdir(cvworkdir):
            os.makedirs(cvworkdir)

    # Lambda Parameters
    lambl1s = -np.sort(-np.asarray(lambl1s))
    lambtvs = -np.sort(-np.asarray(lambtvs))
    lambtsvs = -np.sort(-np.asarray(lambtsvs))
    nl1 = len(lambl1s)
    ntv = len(lambtvs)
    ntsv = len(lambtsvs)

    # Summary Data
    sumtable = pd.DataFrame()
    if docv:
        cvsumtable = pd.DataFrame()
        isvistable = False
        isamptable = False
        isbstable = False
        iscatable = False
        if "vistable" in imageprm.keys():
            if imageprm["vistable"] is not None:
                isvistable = True
                vistables = imageprm["vistable"].gencvtables(nfold=nfold, seed=seed)
        if "amptable" in imageprm.keys():
            if imageprm["amptable"] is not None:
                isamptable = True
                amptables = imageprm["amptable"].gencvtables(nfold=nfold, seed=seed)
        if "bstable" in imageprm.keys():
            if imageprm["bstable"] is not None:
                isbstable = True
                bstables = imageprm["bstable"].gencvtables(nfold=nfold, seed=seed)
        if "catable" in imageprm.keys():
            if imageprm["catable"] is not None:
                iscatable = True
                catables = imageprm["catable"].gencvtables(nfold=nfold, seed=seed)

    # Start Imaging
    for itsv, itv, il1 in itertools.product(
            np.arange(ntsv),
            np.arange(ntv),
            np.arange(nl1)):

        # output
        imageprm["lambl1"] = lambl1s[il1]
        imageprm["lambtv"] = lambtvs[itv]
        imageprm["lambtsv"] = lambtsvs[itsv]

        header = "tsv%02d.tv%02d.l1%02d" % (itsv, itv, il1)
        if imageprm["lambtsv"] <= 0.0:
            place = header.find("tsv")
            header = header[:place] + header[place+6:]
        if imageprm["lambtv"] <= 0.0:
            place = header.find("tv")
            header = header[:place] + header[place+5:]
        if imageprm["lambl1"] <= 0.0:
            place = header.find("l1")
            header = header[:place] + header[place+5:]
        header = header.strip(".")
        if header is "":
            header = "noregularizar"

        # Imaging and Plotting Results
        filename = header + ".fits"
        filename = os.path.join(workdir, filename)
        if (skip is False) or (os.path.isfile(filename) is False):
            newimage = imagefunc(initimage, imageprm=imageprm, **imagefargs)
            newimage.save_fits(filename)
        else:
            newimage = imdata.IMFITS(filename)

        filename = header + ".summary.pdf"
        filename = os.path.join(workdir, filename)
        plots(newimage, imageprm, filename=filename)

        newstats = statistics(newimage, **imageprm)

        # Make Summary
        tmpsum = collections.OrderedDict()
        tmpsum["itsv"] = itsv
        tmpsum["itv"] = itv
        tmpsum["il1"] = il1
        for key in newstats.keys():
            tmpsum[key] = newstats[key]

        # Cross Validation
        if docv:
            # Initialize Summary Table
            #    add keys
            tmpcvsum = pd.DataFrame()
            tmpcvsum["icv"] = np.arange(nfold)
            tmpcvsum["itsv"] = np.zeros(nfold, dtype=np.int32)
            tmpcvsum["itv"] = np.zeros(nfold, dtype=np.int32)
            tmpcvsum["il1"] = np.zeros(nfold, dtype=np.int32)
            tmpcvsum["lambtsv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["lambtv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["lambl1"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["tchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["trchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["tchisqfcv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["tchisqamp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["tchisqcp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["tchisqca"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["trchisqfcv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["trchisqamp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["trchisqcp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["trchisqca"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vrchisq"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vchisqfcv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vchisqamp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vchisqcp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vchisqca"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vrchisqfcv"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vrchisqamp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vrchisqcp"] = np.zeros(nfold, dtype=np.float64)
            tmpcvsum["vrchisqca"] = np.zeros(nfold, dtype=np.float64)

            #    initialize some columns
            tmpcvsum.loc[:, "itsv"] = itsv
            tmpcvsum.loc[:, "itv"] = itv
            tmpcvsum.loc[:, "il1"] = il1
            tmpcvsum.loc[:, "lambtsv"] = lambtsvs[itsv]
            tmpcvsum.loc[:, "lambtv"] = lambtvs[itv]
            tmpcvsum.loc[:, "lambl1"] = lambl1s[il1]

            #   Imaging parameters
            cvimageprm = copy.deepcopy(imageprm)

            #  N-fold CV
            for icv in np.arange(nfold):
                # Header of output files
                cvheader = header+".cv%02d" % (icv)

                # Generate Data sets for imaging
                if isvistable:
                    cvimageprm["vistable"] = vistables["t%d" % (icv)]
                if isamptable:
                    cvimageprm["amptable"] = amptables["t%d" % (icv)]
                if isbstable:
                    cvimageprm["bstable"] = bstables["t%d" % (icv)]
                if iscatable:
                    cvimageprm["catable"] = catables["t%d" % (icv)]

                # Image Training Data
                filename = cvheader + ".t.fits"
                filename = os.path.join(cvworkdir, filename)
                if (skip is False) or (os.path.isfile(filename) is False):
                    cvnewimage = imagefunc(newimage, imageprm=cvimageprm,
                                           **imagefargs)
                    cvnewimage.save_fits(filename)
                else:
                    cvnewimage = imdata.IMFITS(filename)

                # Make Plots
                filename = cvheader + ".t.summary.pdf"
                filename = os.path.join(cvworkdir, filename)
                plots(cvnewimage, cvimageprm, filename=filename)

                # Check Training data
                trainstats = statistics(cvnewimage,
                                              **cvimageprm)

                # Check validating data
                #   Switch to Validating data
                if isvistable:
                    cvimageprm["vistable"] = vistables["v%d" % (icv)]
                if isamptable:
                    cvimageprm["amptable"] = amptables["v%d" % (icv)]
                if isbstable:
                    cvimageprm["bstable"] = bstables["v%d" % (icv)]
                if iscatable:
                    cvimageprm["catable"] = catables["v%d" % (icv)]

                # Make Plots
                filename = cvheader + ".v.summary.pdf"
                filename = os.path.join(cvworkdir, filename)
                plots(cvnewimage, cvimageprm, filename=filename)

                #   Check Statistics
                validstats = statistics(cvnewimage, **cvimageprm)

                #   Save Results
                tmpcvsum.loc[icv, "tchisq"] = trainstats["chisq"]
                tmpcvsum.loc[icv, "trchisq"] = trainstats["rchisq"]
                tmpcvsum.loc[icv, "tchisqfcv"] = trainstats["chisqfcv"]
                tmpcvsum.loc[icv, "tchisqamp"] = trainstats["chisqamp"]
                tmpcvsum.loc[icv, "tchisqcp"] = trainstats["chisqcp"]
                tmpcvsum.loc[icv, "tchisqca"] = trainstats["chisqca"]
                tmpcvsum.loc[icv, "trchisqfcv"] = trainstats["rchisqfcv"]
                tmpcvsum.loc[icv, "trchisqamp"] = trainstats["rchisqamp"]
                tmpcvsum.loc[icv, "trchisqcp"] = trainstats["rchisqcp"]
                tmpcvsum.loc[icv, "trchisqca"] = trainstats["rchisqca"]

                tmpcvsum.loc[icv, "vchisq"] = validstats["chisq"]
                tmpcvsum.loc[icv, "vrchisq"] = validstats["rchisq"]
                tmpcvsum.loc[icv, "vchisqfcv"] = validstats["chisqfcv"]
                tmpcvsum.loc[icv, "vchisqamp"] = validstats["chisqamp"]
                tmpcvsum.loc[icv, "vchisqcp"] = validstats["chisqcp"]
                tmpcvsum.loc[icv, "vchisqca"] = validstats["chisqca"]
                tmpcvsum.loc[icv, "vrchisqfcv"] = validstats["rchisqfcv"]
                tmpcvsum.loc[icv, "vrchisqamp"] = validstats["rchisqamp"]
                tmpcvsum.loc[icv, "vrchisqcp"] = validstats["rchisqcp"]
                tmpcvsum.loc[icv, "vrchisqca"] = validstats["rchisqca"]
            # add current cv summary to the log file.
            cvsumtable = pd.concat([cvsumtable,tmpcvsum], ignore_index=True)
            cvsumtable.to_csv(os.path.join(workdir, cvsumtablefile))

            # Average Varidation Errors and memorized them
            tmpsum["tchisq"] = np.mean(tmpcvsum["tchisq"])
            tmpsum["trchisq"] = np.mean(tmpcvsum["trchisq"])
            tmpsum["tchisqfcv"] = np.mean(tmpcvsum["tchisqfcv"])
            tmpsum["tchisqamp"] = np.mean(tmpcvsum["tchisqamp"])
            tmpsum["tchisqcp"] = np.mean(tmpcvsum["tchisqcp"])
            tmpsum["tchisqca"] = np.mean(tmpcvsum["tchisqca"])
            tmpsum["trchisqfcv"] = np.mean(tmpcvsum["trchisqfcv"])
            tmpsum["trchisqamp"] = np.mean(tmpcvsum["trchisqamp"])
            tmpsum["trchisqcp"] = np.mean(tmpcvsum["trchisqcp"])
            tmpsum["trchisqca"] = np.mean(tmpcvsum["trchisqca"])
            tmpsum["vchisq"] = np.mean(tmpcvsum["vchisq"])
            tmpsum["vrchisq"] = np.mean(tmpcvsum["vrchisq"])
            tmpsum["vchisqfcv"] = np.mean(tmpcvsum["vchisqfcv"])
            tmpsum["vchisqamp"] = np.mean(tmpcvsum["vchisqamp"])
            tmpsum["vchisqcp"] = np.mean(tmpcvsum["vchisqcp"])
            tmpsum["vchisqca"] = np.mean(tmpcvsum["vchisqca"])
            tmpsum["vrchisqfcv"] = np.mean(tmpcvsum["vrchisqfcv"])
            tmpsum["vrchisqamp"] = np.mean(tmpcvsum["vrchisqamp"])
            tmpsum["vrchisqcp"] = np.mean(tmpcvsum["vrchisqcp"])
            tmpsum["vrchisqca"] = np.mean(tmpcvsum["vrchisqca"])

        # Output Summary Table
        tmptable = pd.DataFrame([tmpsum.values()], columns=tmpsum.keys())
        sumtable = pd.concat([sumtable, tmptable], ignore_index=True)
        sumtable.to_csv(os.path.join(workdir, sumtablefile))

    if docv:
        return sumtable, cvsumtable
    else:
        return sumtable
