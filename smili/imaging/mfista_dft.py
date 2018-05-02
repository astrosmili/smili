#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of smili. This module is a wrapper of C library of
MFISTA in src/mfista
'''
# -------------------------------------------------------------------------
# Modules
# -------------------------------------------------------------------------
# standard modules
import ctypes
import os
import copy
import collections
import itertools

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# numerical packages
import numpy as np
import pandas as pd

# internal LoadLibrary
from .. import uvdata, util, imdata

#-------------------------------------------------------------------------
# Default Parameters
#-------------------------------------------------------------------------
mfistaprm = {}
mfistaprm["dftsign"]=1
mfistaprm["cinit"]=10000.0

#-------------------------------------------------------------------------
# CLASS
#-------------------------------------------------------------------------
class _MFISTA_RESULT(ctypes.Structure):
    '''
    This class is for loading structured variables for results
    output from MFISTA.
    '''
    _fields_ = [
        ("M",ctypes.c_int),
        ("N",ctypes.c_int),
        ("NX",ctypes.c_int),
        ("NY",ctypes.c_int),
        ("N_active",ctypes.c_int),
        ("maxiter",ctypes.c_int),
        ("ITER",ctypes.c_int),
        ("nonneg",ctypes.c_int),
        ("lambda_l1",ctypes.c_double),
        ("lambda_tv",ctypes.c_double),
        ("lambda_tsv",ctypes.c_double),
        ("sq_error",ctypes.c_double),
        ("mean_sq_error",ctypes.c_double),
        ("l1cost",ctypes.c_double),
        ("tvcost",ctypes.c_double),
        ("tsvcost",ctypes.c_double),
        ("looe_m",ctypes.c_double),
        ("looe_std",ctypes.c_double),
        ("Hessian_positive",ctypes.c_double),
        ("finalcost",ctypes.c_double),
        ("comp_time",ctypes.c_double),
        ("model",ctypes.POINTER(ctypes.c_double)),
        ("residual",ctypes.POINTER(ctypes.c_double))
    ]

    def __init__(self,M,N):
        c_double_p = ctypes.POINTER(ctypes.c_double)

        self.M = ctypes.c_int(M)
        self.N = ctypes.c_int(N)
        self.NX = ctypes.c_int(0)
        self.NY = ctypes.c_int(0)
        self.N_active = ctypes.c_int(0)
        self.maxiter = ctypes.c_int(0)
        self.ITER = ctypes.c_int(0)
        self.nonneg = ctypes.c_int(0)
        self.lambda_l1 = ctypes.c_double(0)
        self.lambda_tv = ctypes.c_double(0)
        self.lambda_tsv = ctypes.c_double(0)
        self.sq_error = ctypes.c_double(0.0)
        self.mean_sq_error = ctypes.c_double(0.0)
        self.l1cost = ctypes.c_double(0.0)
        self.tvcost = ctypes.c_double(0.0)
        self.tsvcost = ctypes.c_double(0.0)
        self.looe_m = ctypes.c_double(0.0)
        self.looe_std = ctypes.c_double(0.0)
        self.Hessian_positive = ctypes.c_double(0.0)
        self.finalcost = ctypes.c_double(0.0)
        self.residarr = np.zeros(M)
        self.residual = self.residarr.ctypes.data_as(c_double_p)
        self.modelarr = np.zeros(M)
        self.model = self.modelarr.ctypes.data_as(c_double_p)

#-------------------------------------------------------------------------
# Wrapping Function
#-------------------------------------------------------------------------
def imaging(
    initimage, vistable,
    lambl1=-1., lambtv=-1, lambtsv=-1,
    normlambda=True, nonneg=True, looe=True,
    totalflux=None, fluxconst=False,
    istokes=0, ifreq=0):

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1

    # LOOE flags
    if looe:
        looe_flag=1
    else:
        looe_flag=0

    # Nonneg condition
    if nonneg:
        nonneg_flag=1
    else:
        nonneg_flag=0

    # get initial images
    Iin = np.float64(initimage.data[istokes, ifreq])
    Iout = copy.deepcopy(Iin)

    # size of images
    Nx = np.int32(initimage.header["nx"])
    Ny = np.int32(initimage.header["ny"])
    Nyx = Nx * Ny

    # pixel coordinates
    x, y = initimage.get_xygrid(twodim=True, angunit="rad")
    x = np.float64(x)
    y = np.float64(y)

    # reshape image and coordinates
    Iin = Iin.reshape(Nyx)
    x = x.reshape(Nyx)
    y = y.reshape(Nyx)

    # Add zero-uv point for total flux constraints.
    if dofluxconst:
        print("Total Flux Constraint: set to %g" % (totalflux))
        totalfluxdata = {
            'u': [0.],
            'v': [0.],
            'amp': [totalflux],
            'phase': [0.],
            'sigma': [1.]
        }
        totalfluxdata = uvdata.VisTable(totalfluxdata)
        fcvtable = pd.concat([totalfluxdata, vistable], ignore_index=True)
    else:
        print("Total Flux Constraint: disabled.")
        fcvtable = vistable.copy()

    # Pick up data sets
    u = np.asarray(fcvtable["u"], dtype=np.float64)
    v = np.asarray(fcvtable["v"], dtype=np.float64)
    Vamp = np.asarray(fcvtable["amp"], dtype=np.float64)
    Vpha = np.deg2rad(np.asarray(fcvtable["phase"], dtype=np.float64))
    Verr = np.asarray(fcvtable["sigma"], dtype=np.float64)
    Vfcv = np.concatenate([Vamp*np.cos(Vpha)/Verr, Vamp*np.sin(Vpha)/Verr])
    M = Vfcv.size
    Vfcv *= 1/np.sqrt(M/2.)
    del Vamp, Vpha

    # scale lambda
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = vistable["amp"].max()
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 / fluxscale
        lambtv_sim = lambtv / fluxscale / 4.
        lambtsv_sim = lambtsv / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv
    #print(lambl1_sim,lambtv_sim,lambtsv_sim)
    if lambl1_sim < 0: lambl1_sim = 0.
    if lambtv_sim < 0: lambtv_sim = 0.
    if lambtsv_sim < 0: lambtsv_sim = 0.
    #print(lambl1_sim,lambtv_sim,lambtsv_sim)

    # make an MFISTA_result object
    mfista_result = _MFISTA_RESULT(M,Nyx)
    mfista_result.lambda_l1 = lambl1_sim
    mfista_result.lambda_tv = lambtv_sim
    mfista_result.lambda_tsv = lambtsv_sim

    # get pointor to variables
    c_double_p = ctypes.POINTER(ctypes.c_double)
    Iin_p = Iin.ctypes.data_as(c_double_p)
    Iout_p = Iout.ctypes.data_as(c_double_p)
    x_p = x.ctypes.data_as(c_double_p)
    y_p = y.ctypes.data_as(c_double_p)
    u_p = u.ctypes.data_as(c_double_p)
    v_p = v.ctypes.data_as(c_double_p)
    Vfcv_p = Vfcv.ctypes.data_as(c_double_p)
    Verr_p = Verr.ctypes.data_as(c_double_p)
    mfista_result_p = ctypes.byref(mfista_result)

    # Load libmfista.so
    libmfistapath = os.path.dirname(os.path.abspath(__file__))
    libmfistapath = os.path.join(libmfistapath,"libmfista_dft.so")
    libmfista = ctypes.cdll.LoadLibrary(libmfistapath)
    libmfista.mfista_imaging(
        #Full Complex Visibility
        Vfcv_p,
        # Array Size
        ctypes.c_int(M), ctypes.c_int(Nx), ctypes.c_int(Ny),
        # UV coordinates and Errors
        ctypes.c_int(mfistaprm["dftsign"]),
        u_p, v_p, x_p, y_p, Verr_p,
        # Imaging Parameters
        ctypes.c_double(lambl1_sim), ctypes.c_double(lambtv_sim),
        ctypes.c_double(lambtsv_sim),
        ctypes.c_double(mfistaprm["cinit"]), Iin_p, Iout_p,
        ctypes.c_int(nonneg_flag), ctypes.c_int(looe_flag),
        # Results
        mfista_result_p)

    # Get Results
    outimage = copy.deepcopy(initimage)
    outimage.data[istokes, ifreq] = Iout.reshape(Ny, Nx)
    outimage.update_fits()

    return outimage

def statistics(
    initimage, vistable,
    lambl1=-1., lambtv=-1, lambtsv=-1,
    normlambda=True, nonneg=True, looe=True,
    totalflux=None, fluxconst=False,
    istokes=0, ifreq=0, fulloutput=False):

    # Total Flux constraint: Sanity Check
    dofluxconst = False
    if ((totalflux is None) and (fluxconst is True)):
        print("Error: No total flux is specified, although you set fluxconst=True.")
        print("       You need to set the total flux constraint by totalflux.")
        return -1

    # LOOE flags
    if looe:
        looe_flag=1
    else:
        looe_flag=0

    # Nonneg condition
    if nonneg:
        nonneg_flag=1
    else:
        nonneg_flag=0

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

    # reshape image and coordinates
    Iin = Iin.reshape(Nyx)
    x = x.reshape(Nyx)
    y = y.reshape(Nyx)

    # Add zero-uv point for total flux constraints.
    '''
    if dofluxconst:
        print("Total Flux Constraint: set to %g" % (totalflux))
        totalfluxdata = {
            'u': [0.],
            'v': [0.],
            'amp': [totalflux],
            'phase': [0.],
            'sigma': [1.]
        }
        totalfluxdata = uvdata.VisTable(totalfluxdata)
        fcvtable = pd.concat([totalfluxdata, vistable], ignore_index=True)
    else:
        print("Total Flux Constraint: disabled.")
        fcvtable = vistable.copy()
    '''
    fcvtable = vistable.copy()

    # Pick up data sets
    u = np.asarray(fcvtable["u"], dtype=np.float64)
    v = np.asarray(fcvtable["v"], dtype=np.float64)
    Vamp = np.asarray(fcvtable["amp"], dtype=np.float64)
    Vpha = np.deg2rad(np.asarray(fcvtable["phase"], dtype=np.float64))
    Verr = np.asarray(fcvtable["sigma"], dtype=np.float64)
    Vfcv = np.concatenate([Vamp*np.cos(Vpha)/Verr, Vamp*np.sin(Vpha)/Verr])
    M = Vfcv.size
    Ndata = M//2
    Vfcv *= 1/np.sqrt(M/2)
    del Vamp, Vpha

    # scale lambda
    if normlambda:
        # Guess Total Flux
        if totalflux is None:
            fluxscale = vistable["amp"].max()
            print("Flux Scaling Factor for lambda: The expected total flux is not given.")
            print("                                The scaling factor will be %g" % (fluxscale))
        else:
            fluxscale = np.float64(totalflux)
            print("Flux Scaling Factor for lambda: The scaling factor will be %g" % (fluxscale))
        lambl1_sim = lambl1 / fluxscale
        lambtv_sim = lambtv / fluxscale / 4.
        lambtsv_sim = lambtsv / fluxscale**2 / 4.
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv
    if lambl1_sim < 0: lambl1_sim = 0.
    if lambtv_sim < 0: lambtv_sim = 0.
    if lambtsv_sim < 0: lambtsv_sim = 0.

    # make an MFISTA_result object
    mfista_result = _MFISTA_RESULT(M,Nyx)
    mfista_result.lambda_l1 = lambl1_sim
    mfista_result.lambda_tv = lambtv_sim
    mfista_result.lambda_tsv = lambtsv_sim

    # get pointor to variables
    c_double_p = ctypes.POINTER(ctypes.c_double)
    Iin_p = Iin.ctypes.data_as(c_double_p)
    x_p = x.ctypes.data_as(c_double_p)
    y_p = y.ctypes.data_as(c_double_p)
    u_p = u.ctypes.data_as(c_double_p)
    v_p = v.ctypes.data_as(c_double_p)
    Vfcv_p = Vfcv.ctypes.data_as(c_double_p)
    Verr_p = Verr.ctypes.data_as(c_double_p)
    mfista_result_p = ctypes.byref(mfista_result)

    # Load libmfista.so
    libmfistapath = os.path.dirname(os.path.abspath(__file__))
    libmfistapath = os.path.join(libmfistapath,"libmfista_dft.so")
    libmfista = ctypes.cdll.LoadLibrary(libmfistapath)
    libmfista.mfista_imaging_results(
        #Full Complex Visibility
        Vfcv_p,
        # Array Size
        ctypes.c_int(M), ctypes.c_int(Nx), ctypes.c_int(Ny),
        # UV coordinates and Errors
        ctypes.c_int(mfistaprm["dftsign"]),
        u_p, v_p, x_p, y_p, Verr_p,
        # Imaging Parameters
        ctypes.c_double(lambl1_sim), ctypes.c_double(lambtv_sim),
        ctypes.c_double(lambtsv_sim),
        ctypes.c_double(mfistaprm["cinit"]), Iin_p,
        ctypes.c_int(nonneg_flag), ctypes.c_int(looe_flag),
        # Results
        mfista_result_p)

    stats = collections.OrderedDict()

    # Cost and Chisquares
    stats["cost"] = mfista_result.finalcost
    stats["chisq"] = mfista_result.sq_error * M / 2
    stats["rchisq"] = mfista_result.sq_error / 2
    stats["looe_m"] = mfista_result.looe_m
    stats["looe_std"] = mfista_result.looe_std
    stats["isfcv"] = True
    stats["isamp"] = False
    stats["iscp"] = False
    stats["isca"] = False
    stats["chisqfcv"] = mfista_result.sq_error * M / 2
    stats["chisqamp"] = 0
    stats["chisqcp"] = 0
    stats["chisqca"] = 0
    stats["rchisqfcv"] = mfista_result.sq_error / 2
    stats["rchisqamp"] = 0
    stats["rchisqcp"] = 0
    stats["rchisqca"] = 0

    # Regularization functions
    if lambl1 > 0:
        stats["lambl1"] = lambl1
        stats["lambl1_sim"] = lambl1_sim
        stats["l1"] = mfista_result.l1cost
        stats["l1cost"] = mfista_result.l1cost*lambl1_sim
    else:
        stats["lambl1"] = 0.
        stats["lambl1_sim"] = 0.
        stats["l1"] = 0.
        stats["l1cost"] = 0.

    if lambtv > 0:
        stats["lambtv"] = lambtv
        stats["lambtv_sim"] = lambtv_sim
        stats["tv"] = mfista_result.tvcost
        stats["tvcost"] = mfista_result.tvcost*lambtv_sim
    else:
        stats["lambtv"] = 0.
        stats["lambtv_sim"] = 0.
        stats["tv"] = 0.
        stats["tvcost"] = 0.

    if lambtsv > 0:
        stats["lambtsv"] = lambtsv
        stats["lambtsv_sim"] = lambtsv_sim
        stats["tsv"] = mfista_result.tsvcost
        stats["tsvcost"] = mfista_result.tsvcost*lambtsv_sim
    else:
        stats["lambtsv"] = 0.
        stats["lambtsv_sim"] = 0.
        stats["tsv"] = 0.
        stats["tsvcost"] = 0.

    if fulloutput:
        # gradcost
        stats["gradcost"] = None
        # full complex visibilities
        model = mfista_result.modelarr
        resid = mfista_result.residarr
        rmod = model[0:Ndata] * np.sqrt(Ndata) * Verr
        imod = model[Ndata:2*Ndata] * np.sqrt(Ndata) * Verr
        rred = resid[0:Ndata] * np.sqrt(Ndata)
        ired = resid[Ndata:2*Ndata] * np.sqrt(Ndata)
        stats["fcvampmod"] = np.sqrt(rmod*rmod + imod*imod)
        stats["fcvphamod"] = np.angle(rmod + 1j * imod, deg=True)
        stats["fcvrmod"] = rmod
        stats["fcvimod"] = imod
        stats["fcvres"] = np.sqrt(rred*rred + ired*ired)

        # Others
        stats["ampmod"] = None
        stats["ampres"] = None
        stats["cpmod"] = None
        stats["cpres"] = None
        stats["camod"] = None
        stats["cares"] = None

    return stats

def plots(outimage, imageprm={}, filename=None,
                 angunit="mas", uvunit="gl", plotargs={'ms': 1., }):
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

    # Close File
    if filename is not None:
        pdf.close()
    else:
        plt.show()

    if isinteractive:
        plt.ion()
        matplotlib.use(backend)


def pipeline(
        initimage,
        imageprm={},
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
    A pipeline imaging function using static_dft_imaging and related fucntions.

    Args:
        initimage (imdata.IMFITS object):
            initial image
        imageprm (dict-like; default={}):
            parameter sets for each imaging
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
        vistables = imageprm["vistable"].gencvtables(nfold=nfold, seed=seed)

    # Start Imaging
    previmage=initimage
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
            newimage = imaging(previmage, **imageprm)
            newimage.save_fits(filename)
        else:
            newimage = imdata.IMFITS(filename)
        previmage = newimage

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
                cvimageprm["vistable"] = vistables["t%d" % (icv)]

                # Image Training Data
                filename = cvheader + ".t.fits"
                filename = os.path.join(cvworkdir, filename)
                if (skip is False) or (os.path.isfile(filename) is False):
                    cvnewimage = imaging(newimage, **imageprm)
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
                cvimageprm["vistable"] = vistables["v%d" % (icv)]

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
