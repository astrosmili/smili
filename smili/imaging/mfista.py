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
mfistaprm["eps"]=1.0e-2
##mfistaprm["fftw_measure"]=1
mfistaprm["cinit"]=1.0e+8

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
        ("residual",ctypes.POINTER(ctypes.c_double)),
        ("Lip_const",ctypes.c_double)
    ]

    def __init__(self,M,N):
        self.residarr = np.zeros(M, dtype=np.float64)

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
        self.comp_time = ctypes.c_double(0.0)
        self.residual = self.residarr.ctypes.data_as(c_double_p)
        self.Lip_const = ctypes.c_double(0.0)

#-------------------------------------------------------------------------
# Wrapping Function
#-------------------------------------------------------------------------
def imaging(
    initimage,
    vistable,
    imregion=None,
    lambl1=-1., lambtv=-1, lambtsv=-1, normlambda=True,
    niter=5000,
    nonneg=True,
    totalflux=None,
    istokes=0, ifreq=0):

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
    dx_rad = np.deg2rad(initimage.header["dx"])
    dy_rad = np.deg2rad(initimage.header["dy"])

    # image region
    if imregion is not None:
        box_flag = 1
        mask = np.float64(imregion.maskimage(initimage).data[0,0].reshape(Nyx))
    else:
        box_flag = 0
        mask = np.zeros(Nyx, dtype=np.float64)

    if totalflux is None:
        totalflux = vistable["amp"].max()

    # nomarlization of lambda
    if normlambda:
        fluxscale = np.float64(totalflux)
        # convert Flux Scaling Factor
        fluxscale = np.abs(fluxscale) / Nyx
        lambl1_sim = lambl1 / (fluxscale * Nyx)
        lambtv_sim = lambtv / (4 * fluxscale * Nyx)
        lambtsv_sim = lambtsv / (4 *fluxscale**2 * Nyx)
    else:
        lambl1_sim = lambl1
        lambtv_sim = lambtv
        lambtsv_sim = lambtsv

    # reshape image and coordinates
    Iin = Iin.reshape(Nyx)
    Iout = Iout.reshape(Nyx)

    u = np.asarray(vistable.u.values, dtype=np.float64)
    v = np.asarray(vistable.v.values, dtype=np.float64)
    u *= 2*np.pi*dx_rad
    v *= 2*np.pi*dy_rad
    Vcomp = vistable.amp.values*np.exp(1j*np.deg2rad(vistable.phase.values))

    phase = np.deg2rad(np.array(vistable["phase"], dtype=np.float64))
    amp = np.array(vistable["amp"], dtype=np.float64)
    Vreal = np.float64(amp*np.cos(phase))
    Vimag = np.float64(amp*np.sin(phase))
    Verr = np.abs(np.asarray(vistable.sigma.values, dtype=np.float64))

    M = len(Verr)
    Verr *= np.sqrt(M)
    del Vcomp

    # Lambda
    if lambl1_sim < 0: lambl1_sim = 0.
    if lambtv_sim < 0: lambtv_sim = 0.
    if lambtsv_sim < 0: lambtsv_sim = 0.
    print("lambl1_sim:",lambl1_sim)
    print("lambtsv_sim:",lambtsv_sim)
    print("lambtv_sim:",lambtv_sim)


    # make an MFISTA_result object
    mfista_result = _MFISTA_RESULT(M,Nyx)
    mfista_result.lambda_l1 = lambl1_sim
    mfista_result.lambda_tv = lambtv_sim
    mfista_result.lambda_tsv = lambtsv_sim

    # get pointor to variables
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    u_p = u.ctypes.data_as(c_double_p)
    v_p = v.ctypes.data_as(c_double_p)
    Vreal_p = Vreal.ctypes.data_as(c_double_p)
    Vimag_p = Vimag.ctypes.data_as(c_double_p)
    Verr_p = Verr.ctypes.data_as(c_double_p)
    Iin_p = Iin.ctypes.data_as(c_double_p)
    Iout_p = Iout.ctypes.data_as(c_double_p)
    mask_p = mask.ctypes.data_as(c_double_p)
    mfista_result_p = ctypes.byref(mfista_result)

    # Load libmfista.so
    libmfistapath = os.path.dirname(os.path.abspath(__file__))
    libmfistapath = os.path.join(libmfistapath,"libmfista_nufft.so")
    libmfista = ctypes.cdll.LoadLibrary(libmfistapath)
    libmfista.mfista_imaging_core_nufft(
        # uv coordinates (u_idx, v_idx), MFISTA handles u is column, v is row. so no need to flip
        u_p, v_p,
        # full complex Visibilities (y_r, y_i, noise_stdev)
        Vreal_p, Vimag_p, Verr_p,
        # Array Size (M, Nx, Ny, maxiter, eps), Since the input image is column order, Nx, Ny are flipped
        ctypes.c_int(M), ctypes.c_int(Ny), ctypes.c_int(Nx),
        ctypes.c_int(niter), ctypes.c_double(mfistaprm["eps"]),
        # Imaging Parameters (lambdas, cinit)
        ctypes.c_double(lambl1_sim),
        ctypes.c_double(lambtv_sim),
        ctypes.c_double(lambtsv_sim),
        ctypes.c_double(mfistaprm["cinit"]),
        # Input and Output Images (xinit, xout)
        Iin_p, Iout_p,
        # Flags (nonneg_flag, fftw_measure)
        ctypes.c_int(nonneg_flag),
        # clean box (box_flag, cl_box, mfista_result)
        ctypes.c_int(box_flag),
        mask_p,
        mfista_result_p)


    # Get Results
    outimage = copy.deepcopy(initimage)
    outimage.data[istokes, ifreq] = Iout.reshape(Ny, Nx)
    outimage.update_fits()
    return outimage
