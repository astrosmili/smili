#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of smili for imaging static images.
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
from .. import util, imdata, fortlib

#-------------------------------------------------------------------------
# Default Parameters
#-------------------------------------------------------------------------
lbfgsbprms = {
    "m": 10,
    "factr": 1e7,
    "pgtol": 0.
}


#-------------------------------------------------------------------------
# Reconstract static imaging
#-------------------------------------------------------------------------
def imaging(
        initimage, imagewin=None,
        vistable=None, amptable=None, bstable=None, catable=None,
        lambl1=-1., lambtv=-1, lambtsv=-1, normlambda=True, nonneg=True,
        logreg=False, niter=1000,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0):
    '''

    '''
    # Check Arguments
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((vistable is None) and (amptable is None) and
          (totalflux is not None) and (fluxconst is False)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux will be constrained, although you do not set fluxconst=True.")
        dofluxconst = True
    elif fluxconst is True:
        dofluxconst = True

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])

    # size of images
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    x = np.float64(x)
    y = np.float64(y)
    xidx = np.int32(np.arange(Nx) + 1)
    yidx = np.int32(np.arange(Ny) + 1)
    xidx, yidx = np.meshgrid(xidx, yidx)

    # apply the imaging area
    if imagewin is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = Iin.reshape(Nyx)
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        idx = np.where(imagewin)
        Iin = Iin[idx]
        x = x[idx]
        y = y[idx]
        xidx = xidx[idx]
        yidx = yidx[idx]

    # dammy array
    dammyreal = np.zeros(1, dtype=np.float64)

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
        vfcvr = amp * np.cos(phase)
        vfcvi = amp * np.sin(phase)
        varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
        Ndata += len(vfcvr)
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
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = []
            if vistable is not None:
                fluxscale.append(vistable["amp"].max())
            if amptable is not None:
                fluxscale.append(amptable["amp"].max())
            fluxscale = np.max(fluxscale)
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        if logreg:
            lambl1_sim = lambl1 / (len(xidx)*np.log(1+fluxscale/len(xidx)))
            lambtv_sim = lambtv / (len(xidx)*np.log(1+fluxscale/len(xidx)))
            lambtsv_sim = lambtsv / (len(xidx)*np.log(1+fluxscale/len(xidx)))**2
        else:
            lambl1_sim = lambl1 / fluxscale
            lambtv_sim = lambtv / fluxscale
            lambtsv_sim = lambtsv / fluxscale**2
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    # get uv coordinates and uv indice
    u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = _get_uvlist(
        fcvtable=fcvtable, amptable=amptable, bstable=bstable, catable=catable
    )

    # run imaging
    Iout = fortlib.dftim2d.imaging(
        # Images
        iin=Iin, x=x, y=y, xidx=xidx, yidx=yidx, nx=Nx, ny=Ny,
        # UV coordinates,
        u=u, v=v,
        # Imaging Parameters
        lambl1=lambl1_sim, lambtv=lambtv_sim, lambtsv=lambtsv_sim,
        nonneg=nonneg, logreg=logreg, niter=niter,
        # Full Complex Visibilities
        isfcv=isfcv, uvidxfcv=uvidxfcv, vfcvr=vfcvr, vfcvi=vfcvi, varfcv=varfcv,
        # Visibility Ampltiudes
        isamp=isamp, uvidxamp=uvidxamp, vamp=vamp, varamp=varamp,
        # Closure Phase
        iscp=iscp, uvidxcp=uvidxcp, cp=cp, varcp=varcp,
        # Closure Amplituds
        isca=isca, uvidxca=uvidxca, ca=ca, varca=varca,
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
        initimage, imagewin=None,
        vistable=None, amptable=None, bstable=None, catable=None,
        lambl1=1., lambtv=-1, lambtsv=1, logreg=False, normlambda=True,
        totalflux=None, fluxconst=False,
        istokes=0, ifreq=0, fulloutput=True, **args):
    '''

    '''
    # Check Arguments
    if ((vistable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((vistable is None) and (amptable is None) and (totalflux is None)):
        print("Error: No absolute amplitude information in the input data.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1
    elif ((vistable is None) and (amptable is None) and
          (totalflux is not None) and (fluxconst is False)):
        print("Warning: No absolute amplitude information in the input data.")
        print("         The total flux will be constrained, although you do not set fluxconst=True.")
        dofluxconst = True
    elif fluxconst is True:
        dofluxconst = True

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])

    # size of images
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    x = np.float64(x)
    y = np.float64(y)
    xidx = np.int32(np.arange(Nx) + 1)
    yidx = np.int32(np.arange(Ny) + 1)
    xidx, yidx = np.meshgrid(xidx, yidx)

    # apply the imaging area
    if imagewin is None:
        print("Imaging Window: Not Specified. We solve the image on all the pixels.")
        Iin = Iin.reshape(Nyx)
        x = x.reshape(Nyx)
        y = y.reshape(Nyx)
        xidx = xidx.reshape(Nyx)
        yidx = yidx.reshape(Nyx)
    else:
        print("Imaging Window: Specified. Images will be solved on specified pixels.")
        idx = np.where(imagewin)
        Iin = Iin[idx]
        x = x[idx]
        y = y[idx]
        xidx = xidx[idx]
        yidx = yidx[idx]

    # dammy array
    dammyreal = np.zeros(1, dtype=np.float64)

    # Full Complex Visibility
    Ndata = 0
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
        vfcvr = amp * np.cos(phase)
        vfcvi = amp * np.sin(phase)
        varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
        Ndata += len(vfcvr)
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

    # Normalize Lambda
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = []
            if vistable is not None:
                fluxscale.append(vistable["amp"].max())
            if amptable is not None:
                fluxscale.append(amptable["amp"].max())
            fluxscale = np.max(fluxscale)
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        if logreg:
            lambl1_sim = lambl1 / (len(xidx)*np.log(1+fluxscale/len(xidx)))
            lambtv_sim = lambtv / (len(xidx)*np.log(1+fluxscale/len(xidx)))
            lambtsv_sim = lambtsv / (len(xidx)*np.log(1+fluxscale/len(xidx)))**2
        else:
            lambl1_sim = lambl1 / fluxscale
            lambtv_sim = lambtv / fluxscale
            lambtsv_sim = lambtsv / fluxscale**2
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    # get uv coordinates and uv indice
    u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = _get_uvlist(
        fcvtable=fcvtable, amptable=amptable, bstable=bstable, catable=catable
    )

    # calculate all
    out = fortlib.dftim2d.statistics(
        # Images
        iin=Iin, x=x, y=y, xidx=xidx, yidx=yidx, nx=Nx, ny=Ny,
        # UV coordinates,
        u=u, v=v,
        # Imaging Parameters
        lambl1=lambl1_sim, lambtv=lambtv_sim, lambtsv=lambtsv_sim,
        logreg=logreg,
        # Full Complex Visibilities
        isfcv=isfcv, uvidxfcv=uvidxfcv, vfcvr=vfcvr, vfcvi=vfcvi, varfcv=varfcv,
        # Visibility Ampltiudes
        isamp=isamp, uvidxamp=uvidxamp, vamp=vamp, varamp=varamp,
        # Closure Phase
        iscp=iscp, uvidxcp=uvidxcp, cp=cp, varcp=varcp,
        # Closure Amplituds
        isca=isca, uvidxca=uvidxca, ca=ca, varca=varca
    )
    stats = collections.OrderedDict()
    # Cost and Chisquares
    stats["cost"] = out[0]
    stats["chisq"] = out[3] + out[4] + out[5] + out[6]
    stats["rchisq"] = stats["chisq"] / Ndata
    stats["isfcv"] = isfcv
    stats["isamp"] = isamp
    stats["iscp"] = iscp
    stats["isca"] = isca
    stats["chisqfcv"] = out[3]
    stats["chisqamp"] = out[4]
    stats["chisqcp"] = out[5]
    stats["chisqca"] = out[6]
    stats["rchisqfcv"] = out[3] / len(vfcvr)
    stats["rchisqamp"] = out[4] / len(vamp)
    stats["rchisqcp"] = out[5] / len(cp)
    stats["rchisqca"] = out[6] / len(ca)

    # Regularization functions
    if lambl1 > 0:
        stats["lambl1"] = lambl1
        stats["lambl1_sim"] = lambl1_sim
        stats["l1"] = out[7]
        stats["l1cost"] = out[7] * lambl1_sim
    else:
        stats["lambl1"] = 0.
        stats["lambl1_sim"] = 0.
        stats["l1"] = out[7]
        stats["l1cost"] = 0.

    if lambtv > 0:
        stats["lambtv"] = lambtv
        stats["lambtv_sim"] = lambtv_sim
        stats["tv"] = out[8]
        stats["tvcost"] = out[8] * lambtv_sim
    else:
        stats["lambtv"] = 0.
        stats["lambtv_sim"] = 0.
        stats["tv"] = out[8]
        stats["tvcost"] = 0.

    if lambtsv > 0:
        stats["lambtsv"] = lambtsv
        stats["lambtsv_sim"] = lambtsv_sim
        stats["tsv"] = out[9]
        stats["tsvcost"] = out[9] * lambtsv_sim
    else:
        stats["lambtsv"] = 0.
        stats["lambtsv_sim"] = 0.
        stats["tsv"] = out[9]
        stats["tsvcost"] = 0.

    if fulloutput:
        # gradcost
        gradcostimage = initimage.data[istokes, ifreq, :, :].copy()
        gradcostimage[:, :] = 0.
        for i in np.arange(len(xidx)):
            gradcostimage[yidx[i] - 1, xidx[i] - 1] = out[1][i]
        stats["gradcost"] = gradcostimage
        del gradcostimage

        if isfcv:
            stats["fcvampmod"] = np.sqrt(out[10] * out[10] + out[11] * out[11])
            stats["fcvphamod"] = np.angle(out[10] + 1j * out[11], deg=True)
            stats["fcvrmod"] = out[10]
            stats["fcvimod"] = out[11]
            stats["fcvres"] = out[12]
        else:
            stats["fcvampmod"] = None
            stats["fcvphamod"] = None
            stats["fcvres"] = None

        if isamp:
            stats["ampmod"] = out[13]
            stats["ampres"] = out[14]
        else:
            stats["ampmod"] = None
            stats["ampres"] = None

        if iscp:
            stats["cpmod"] = np.rad2deg(out[15])
            stats["cpres"] = np.rad2deg(out[16])
        else:
            stats["cpmod"] = None
            stats["cpres"] = None

        if isca:
            stats["camod"] = out[17]
            stats["cares"] = out[18]
        else:
            stats["camod"] = None
            stats["cares"] = None

    return stats


def plots(outimage, imageprm={}, filename=None,
                     angunit="mas", uvunit="ml", plotargs={'ms': 1., }):
    isinteractive = plt.isinteractive()
    backend = matplotlib.rcParams["backend"]

    if isinteractive:
        plt.ioff()
        matplotlib.use('Agg')

    nullfmt = NullFormatter()

    # Label
    if uvunit.lower().find("l") == 0:
        unitlabel = r"$\lambda$"
    elif uvunit.lower().find("kl") == 0:
        unitlabel = r"$10^3 \lambda$"
    elif uvunit.lower().find("ml") == 0:
        unitlabel = r"$10^6 \lambda$"
    elif uvunit.lower().find("gl") == 0:
        unitlabel = r"$10^9 \lambda$"
    elif uvunit.lower().find("m") == 0:
        unitlabel = "m"
    elif uvunit.lower().find("km") == 0:
        unitlabel = "km"
    else:
        print("Error: uvunit=%s is not supported" % (unit2))
        return -1

    # Get model data
    stats = statistics(outimage, fulloutput=True, **imageprm)

    # Open File
    if filename is not None:
        pdf = PdfPages(filename)

    # Save Image
    if filename is not None:
        util.matplotlibrc(nrows=1, ncols=1, width=600, height=600)
    else:
        matplotlib.rcdefaults()

    plt.figure()
    outimage.imshow(angunit=angunit)
    if filename is not None:
        pdf.savefig()
        plt.close()

    # Amplitude
    if stats["isfcv"] == True:
        table = imageprm["vistable"]
        table["comp"] = table["amp"] * np.exp(1j * np.deg2rad(table["phase"]))
        table["real"] = np.real(table["comp"])
        table["imag"] = np.imag(table["comp"])

        normresidr = (stats["fcvrmod"] - table["real"]) / table["sigma"]
        normresidi = (stats["fcvimod"] - table["imag"]) / table["sigma"]
        normresid = np.concatenate([normresidr, normresidi])
        N = len(normresid)

        if filename is not None:
            util.matplotlibrc(nrows=3, ncols=1, width=600, height=200)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot_amp(uvunit=uvunit, color="black", **plotargs)
        table.radplot_amp(uvunit=uvunit, model=stats,
                          modeltype="fcv", color="red", **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        table.radplot_phase(uvunit=uvunit, color="black", **plotargs)
        table.radplot_phase(uvunit=uvunit, model=stats,
                            color="red", **plotargs)
        plt.xlabel("")

        ax = axs[2]
        plt.sca(ax)
        plt.plot(table["uvdist"] * table.uvunitconv("lambda", uvunit),
                 normresidr, ls="none", marker=".", color="blue", label="real", **plotargs)
        plt.plot(table["uvdist"] * table.uvunitconv("lambda", uvunit),
                 normresidi, ls="none", marker=".", color="red", label="imag", **plotargs)
        plt.axhline(0, color="black", ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
        plt.legend(ncol=2)

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
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

    if stats["isamp"] == True:
        table = imageprm["amptable"]
        normresid = stats["ampres"] / table["sigma"]
        N = len(normresid)

        if filename is not None:
            util.matplotlibrc(nrows=2, ncols=1, width=600, height=300)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot_amp(uvunit=uvunit, color="black", **plotargs)
        table.radplot_amp(uvunit=uvunit, model=stats,
                          modeltype="amp", color="red", **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        plt.plot(table["uvdist"] * table.uvunitconv("lambda", uvunit),
                 normresid, ls="none", marker=".", color="black", **plotargs)
        plt.axhline(0, color="black", ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
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

    # Closure Amplitude
    if stats["isca"] == True:
        table = imageprm["catable"]
        # Amplitudes
        normresid = stats["cares"] / table["logsigma"]
        N = len(normresid)

        if filename is not None:
            util.matplotlibrc(nrows=2, ncols=1, width=600, height=300)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot(uvunit=uvunit, uvdtype="ave", color="black", **plotargs)
        table.radplot(uvunit=uvunit, uvdtype="ave",
                      model=stats, color="red", **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        plt.plot(table["uvdistave"] * table.uvunitconv("lambda", uvunit),
                 normresid, ls="none", marker=".", color="black", **plotargs)
        plt.axhline(0, color="black", ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
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
        # Amplitudes
        normresid = stats["cpres"] / np.rad2deg(table["sigma"] / table["amp"])

        if filename is not None:
            util.matplotlibrc(nrows=2, ncols=1, width=600, height=300)
        else:
            matplotlib.rcdefaults()

        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.subplots_adjust(hspace=0)

        ax = axs[0]
        plt.sca(ax)
        table.radplot(uvunit=uvunit, uvdtype="ave", color="black", **plotargs)
        table.radplot(uvunit=uvunit, uvdtype="ave",
                      model=stats, color="red", **plotargs)
        plt.xlabel("")

        ax = axs[1]
        plt.sca(ax)
        plt.plot(table["uvdistave"] * table.uvunitconv("lambda", uvunit),
                 normresid, ls="none", marker=".", color="black", **plotargs)
        plt.axhline(0, color="black", ls="--")
        plt.ylabel("Normalized Residuals")
        plt.xlabel(r"Baseline Length (%s)" % (unitlabel))

        divider = make_axes_locatable(ax)  # Histgram
        cax = divider.append_axes("right", size="10%", pad=0.05)
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

    # Close File
    if filename is not None:
        pdf.close()
    else:
        plt.show()

    if isinteractive:
        plt.ion()
        matplotlib.use(backend)


def iterative_imaging(initimage, imageprm, Niter=10,
                      dothres=True, threstype="hard", threshold=0.3,
                      doshift=True, shifttype="com",
                      dowinmod=False, imagewin=None,
                      doconv=True, convprm={},
                      save_totalflux=False):
    oldimage = imaging(initimage, **imageprm)
    oldcost = statistics(oldimage, fulloutput=False, **imageprm)["cost"]
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
            if shifttype == "peak":
                newimage = newimage.peakshift(save_totalflux=save_totalflux)
            else:
                newimage = newimage.comshift(save_totalflux=save_totalflux)

        # Edit Images
        if dowinmod and imagewin is not None:
            newimage = newimage.winmod(imagewin,
                                       save_totalflux=save_totalflux)

        if doconv:
            newimage = newimage.gauss_convolve(
                save_totalflux=save_totalflux, **convprm)

        # Imaging Again
        newimage = imaging(newimage, **imageprm)
        newcost = statistics(
            newimage, fulloutput=False, **imageprm)["cost"]

        if oldcost < newcost:
            print("No improvement in cost fucntions. Don't update image.")
        else:
            oldcost = newcost
            oldimage = newimage
    return oldimage


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
        cvsumtablefile="summary.cv.csv",
        angunit="uas",
        uvunit="gl"):
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
        angunit (string; default = None):
            Angular units for plotting results.
        uvunit (string; default = None):
            Units of baseline lengths for plotting results.

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
    for itsv, itv, il1 in itertools.product(np.arange(ntsv),
                                            np.arange(ntv),
                                            np.arange(nl1)):
        header = "tsv%02d.tv%02d.l1%02d" % (itsv, itv, il1)

        # output
        imageprm["lambl1"] = lambl1s[il1]
        imageprm["lambtv"] = lambtvs[itv]
        imageprm["lambtsv"] = lambtsvs[itsv]

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
        plots(newimage, imageprm, filename=filename,
                         angunit=angunit, uvunit=uvunit)

        newstats = statistics(newimage, fulloutput=False, **imageprm)

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
                plots(cvnewimage, cvimageprm, filename=filename,
                                 angunit=angunit, uvunit=uvunit)

                # Check Training data
                trainstats = statistics(cvnewimage, fulloutput=False,
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
                plots(cvnewimage, cvimageprm, filename=filename,
                                 angunit=angunit, uvunit=uvunit)

                #   Check Statistics
                validstats = statistics(cvnewimage, fulloutput=False,
                                              **cvimageprm)

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


# ------------------------------------------------------------------------------
# Subfunctions
# ------------------------------------------------------------------------------
def _get_uvlist(fcvtable=None, amptable=None, bstable=None, catable=None, thres=1e-2):
    '''

    '''
    if ((fcvtable is None) and (amptable is None) and
            (bstable is None) and (catable is None)):
        print("Error: No data are input.")
        return -1

    # Stack uv coordinates
    ustack = None
    vstack = None
    if fcvtable is not None:
        ustack = np.array(fcvtable["u"], dtype=np.float64)
        vstack = np.array(fcvtable["v"], dtype=np.float64)
        Nfcv = len(ustack)
    else:
        Nfcv = 0

    if amptable is not None:
        utmp = np.array(amptable["u"], dtype=np.float64)
        vtmp = np.array(amptable["v"], dtype=np.float64)
        Namp = len(utmp)
        if ustack is None:
            ustack = utmp
            vstack = vtmp
        else:
            ustack = np.concatenate((ustack, utmp))
            vstack = np.concatenate((vstack, vtmp))
    else:
        Namp = 0

    if bstable is not None:
        utmp1 = np.array(bstable["u12"], dtype=np.float64)
        vtmp1 = np.array(bstable["v12"], dtype=np.float64)
        utmp2 = np.array(bstable["u23"], dtype=np.float64)
        vtmp2 = np.array(bstable["v23"], dtype=np.float64)
        utmp3 = np.array(bstable["u31"], dtype=np.float64)
        vtmp3 = np.array(bstable["v31"], dtype=np.float64)
        Ncp = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3))
    else:
        Ncp = 0

    if catable is not None:
        utmp1 = np.array(catable["u1"], dtype=np.float64)
        vtmp1 = np.array(catable["v1"], dtype=np.float64)
        utmp2 = np.array(catable["u2"], dtype=np.float64)
        vtmp2 = np.array(catable["v2"], dtype=np.float64)
        utmp3 = np.array(catable["u3"], dtype=np.float64)
        vtmp3 = np.array(catable["v3"], dtype=np.float64)
        utmp4 = np.array(catable["u4"], dtype=np.float64)
        vtmp4 = np.array(catable["v4"], dtype=np.float64)
        Nca = len(utmp1)
        if ustack is None:
            ustack = np.concatenate((utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vtmp1, vtmp2, vtmp3, vtmp4))
        else:
            ustack = np.concatenate((ustack, utmp1, utmp2, utmp3, utmp4))
            vstack = np.concatenate((vstack, vtmp1, vtmp2, vtmp3, vtmp4))
    else:
        Nca = 0

    # make non-redundant u,v lists and index arrays for uv coordinates.
    Nstack = Nfcv + Namp + 3 * Ncp + 4 * Nca
    uvidx = np.zeros(Nstack, dtype=np.int32)
    maxidx = 1
    u = []
    v = []
    uvstack = np.sqrt(np.square(ustack) + np.square(vstack))
    uvthres = np.max(uvstack) * thres
    for i in np.arange(Nstack):
        if uvidx[i] == 0:
            dist1 = np.sqrt(
                np.square(ustack - ustack[i]) + np.square(vstack - vstack[i]))
            dist2 = np.sqrt(
                np.square(ustack + ustack[i]) + np.square(vstack + vstack[i]))
            #uvdist = np.sqrt(np.square(ustack[i])+np.square(vstack[i]))

            #t = np.where(dist1<uvthres)
            t = np.where(dist1 < thres * (uvstack[i] + 1))
            uvidx[t] = maxidx
            #t = np.where(dist2<uvthres)
            t = np.where(dist2 < thres * (uvstack[i] + 1))
            uvidx[t] = -maxidx
            u.append(ustack[i])
            v.append(vstack[i])
            maxidx += 1
    u = np.asarray(u)  # Non redundant u coordinates
    v = np.asarray(v)  # Non redundant v coordinates

    # distribute index information into each data
    if fcvtable is None:
        uvidxfcv = np.zeros(1, dtype=np.int32)
    else:
        uvidxfcv = uvidx[0:Nfcv]

    if amptable is None:
        uvidxamp = np.zeros(1, dtype=np.int32)
    else:
        uvidxamp = uvidx[Nfcv:Nfcv + Namp]

    if bstable is None:
        uvidxcp = np.zeros([3, 1], dtype=np.int32, order="F")
    else:
        uvidxcp = uvidx[Nfcv + Namp:Nfcv + Namp + 3 *
                        Ncp].reshape([Ncp, 3], order="F").transpose()

    if catable is None:
        uvidxca = np.zeros([4, 1], dtype=np.int32, order="F")
    else:
        uvidxca = uvidx[Nfcv + Namp + 3 * Ncp:Nfcv + Namp + 3 *
                        Ncp + 4 * Nca].reshape([Nca, 4], order="F").transpose()
    return (u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca)
