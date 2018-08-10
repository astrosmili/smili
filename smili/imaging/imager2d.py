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
class Imager2D(object):

    

    def _set_data(self):
        '''
        Set uv data sets for imaging.

        Args:
            UV Tables:
                vistable (uvdata.VisTable, default=None):
                    Visibility table containing full complex visiblities.
                amptable (uvdata.VisTable, default=None):
                    Amplitude table.
                bstable (uvdata.BSTable, default=None):
                    Closure phase table.
                catable (uvdata.CATable, default=None):
                    Closure amplitude table.

            Data weights:
                visweight, ampweight, bsweight, caweight (float, default=1):
                    relative weights on chisquares in imaging.
                    These weights are normalized so that the sum of them will be
                    one.
                dataweight (float, default=1):
                    The absolute weight on the total chisquare in imaging.
        '''
        #-----------------------------------------------------------------------
        # set uvdata sets
        #-----------------------------------------------------------------------
        self.vistable = vistable
        self.visweight = visweight
        self.amptable = bstable
        self.ampweight = bsweight
        self.bstable = bstable
        self.bsweight = bsweight
        self.catable = catable
        self.caweight = caweight
        self.dataweight = dataweight
        self.check_uvdata()

    def _check_uvdata(self):
        '''this function will be called by set_uvdata for sanity checks
        '''
        if len(self.vistable) == 0:
            self.vistable = None
            print("Warining: vistable is empty.")
        if self.vistable is None
            self.visweight = 0
        if self.visweight <0:
            raise ValueError("visweight must be non-negative.")

        if len(self.amptable) == 0:
            self.amptable = None
            print("Warining: amptable is empty.")
        if self.amptable is None
            self.ampweight = 0
        if self.ampweight <0:
            raise ValueError("ampweight must be non-negative.")

        if len(self.bstable) == 0:
            self.bstable = None
            print("Warining: bstable is empty.")
        if self.bstable is None
            self.bsweight = 0
        if self.bsweight < 0:
            raise ValueError("bsweight must be non-negative.")

        if len(self.catable) == 0:
            self.catable = None
            print("Warining: catable is empty.")
        if self.catable is None
            self.caweight = 0
        if self.caweight < 0:
            raise ValueError("caweight must be non-negative.")

        if self.dataweight < 0:
            raise ValueError("dataweight must be non-negative.")

        weightsum = self.visweight + self.ampweight + self.bsweight + self.caweight
        if weightsum == 0:
            raise ValueError("No data sets are specified.")

        print("Scaling factor for relative data weights: %g"%(weightsum))
        self.visweight /= weightsum
        self.ampweight /= weightsum
        self.bsweight /= weightsum
        self.caweight /= weightsum

    def _set_regfunc(self,):
        '''
        Set regularization functions

        Args:
            l1 regularization (see, Honma et al. 2014, PASJ)
                l1_lambda (float; default=-1):
                    Regularization parameter. If negative, this regularization
                    will not be used.
                l1_prior (imdata.IMFITS object; default=None)
                    If specified, it will be used to reweight l1-norm.
                    (see Candes et al. 2007)

            TV regularization (see, Akiyama et al. 2017ab, ApJ/AJ)
                tv_lambda (float; default=-1):
                    Regularization parameter. If negative, this regularization
                    will not be used.
                tv_prior (imdata.IMFITS object; default=None)
                    If specified, it will be used to reweight TV.
                    (see Candes et al. 2007)

            TSV regularization (see, Kuramochi et al. 2018, ApJ)
                tsv_lambda (float; default=-1):
                    Regularization parameter. If negative, this regularization
                    will not be used.
                tsv_prior (imdata.IMFITS object; default=None)
                    If specified, it will be used to reweight TSV.

            Shannon Information Entropy
                shent_lambda (float; default=-1):
                    Regularization parameter. If negative, this regularization
                    will not be used.
                shent_prior (imdata.IMFITS object; default=None)
                    Prior image for the entropy function.

            Gull & Skilling Entropy
                gsent_lambda (float; default=-1):
                    Regularization parameter. If negative, this regularization
                    will not be used.
                gsent_prior (imdata.IMFITS object; default=None)
                    Prior image for the entropy function.

            Centor of the Mass Regularization
                This will add an regularization function defined as the
                squared distance of the Center of Mass of the (powered) image
                from its tracking center in the unit of pixel^2.

                com_lambda (float; default=-1):
                    Regularization parameter. If negative, this regularization
                    will not be used.
                com_power (float; default=1):
                    The absolute of the image will be powered by this value
                    before computing the center-of-mass (com) of the image.
                    com_power=1 will give the exact com location, while
                    higher com_power will give the location closer to the peak
                    of the image. So if you fix the peak location, a higher value
                    (say 3-5) can be adopted.

            Optimizer:
                You can select optimization methods.
                auto:
                    Automatically select the imaging function based on your
                    imaging parameter.
                lbfgs:
                    Use L-BFGS-B. This is a general option.
                mfista:
                    Use MFISTA (see Akiyama et al. 2017, AJ). This can be
                    applied if your imaging parameters are
                        A) using only full complex visibility data sets
                        B) using only one of L1, L1+TV, L1+TSV regularizations
                        C) regardless of using total flux regularization
                        D) regardless of using non-negative constraint.
                    MFISTA is much faster and more accurate than L-BFGS-B.
        '''
        self.l1_lambda = l1_lambda
        self.l1_prior = l1_prior

        self.tv_lambda = tv_lambda
        self.tv_prior = tv_prior

        self.tsv_lambda = tsv_lambda
        self.tsv_prior = tsv_prior

        self.shent_lambda = shent_lambda
        self.shent_prior = shent_prior

        self.gsent_lambda = gsent_lambda
        self.gsent_prior = gsent_prior

        self.com_lambda = com_lambda
        self.com_power = com_power

    def _set_image(self, image):
        '''
        '''
        # set image
        self.image = image
        # reference pixels
        self.Nyref = image.header["nxref"]
        self.Nxref = image.header["nyref"]
        # number of pixels
        self.Nx = image.header["nx"]
        self.Ny = image.header["ny"]
        self.Nyx = Nx * Ny
        # pixel size in rad
        self.dx_rad = np.deg2rad(image.header["dx"])
        self.dy_rad = np.deg2rad(image.header["dy"])
        # set vectors
        xidx = np.arange(self.Nx) + 1
        yidx = np.arange(self.Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)
        # set imvec
        self.image_vec = self.image.data[0,0].reshape(self.Nyx)
        self.xidx = xidx.reshape(self.Nyx)
        self.yidx = yidx.reshape(self.Nyx)
        # set vectors for prios
        if self.l1_prior is None:
            self.l1_prior_vec = None
        else:
            self.l1_prior_vec = self.l1_prior.data[0,0].reshape(self.Nyx)

        if self.tv_prior is None:
            self.tv_prior_vec = None
        else:
            self.tv_prior_vec = self.tv_prior.data[0,0].reshape(self.Nyx)

        if self.tsv_prior is None:
            self.tsv_prior_vec = None
        else:
            self.tsv_prior_vec = self.tsv_prior.data[0,0].reshape(self.Nyx)

        if self.shent_prior is None:
            self.shent_prior_vec = None
        else:
            self.shent_prior_vec = self.shent_prior.data[0,0].reshape(self.Nyx)

        if self.gsent_prior is None:
            self.gsent_prior_vec = None
        else:
            self.gsent_prior_vec = self.gsent_prior.data[0,0].reshape(self.Nyx)

    def init_image_vectors(self):
        '''
        '''
        xidx = np.arange(self.Nx) + 1
        yidx = np.arange(self.Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)

        if self.maskimage is None:

            self.Npix = self.Nyx
            if self.l1_prior is None:
                self.l1_prior_vec = None
            else:
                self.l1_prior_vec = self.l1_prior.data[0,0].reshape(self.Nyx)

            if self.tv_prior is None:
                self.tv_prior_vec = None
            else:
                self.tv_prior_vec = self.tv_prior.data[0,0].reshape(self.Nyx)

            if self.tsv_prior is None:
                self.tsv_prior_vec = None
            else:
                self.tsv_prior_vec = self.tsv_prior.data[0,0].reshape(self.Nyx)

            if self.shent_prior is None:
                self.shent_prior_vec = None
            else:
                self.shent_prior_vec = self.shent_prior.data[0,0].reshape(self.Nyx)

            if self.gsent_prior is None:
                self.gsent_prior_vec = None
            else:
                self.gsent_prior_vec = self.gsent_prior.data[0,0].reshape(self.Nyx)
        else:
            self.maskimage_vec = self.maskimage.data[0,0].reshape(self.Nyx)
            idx = np.where(self.maskimage.data[0,0] > 0.5)

            self.image_vec = self.initimage.data[0,0][idx]
            self.xidx = xidx[idx]
            self.yidx = yidx[idx]
            self.Npix = len(self.xidx)

            if self.l1_prior is None:
                self.l1_prior_vec = None
            else:
                self.l1_prior_vec = self.l1_prior.data[0,0][idx]

            if self.tv_prior is None:
                self.tv_prior_vec = None
            else:
                self.tv_prior_vec = self.tv_prior.data[0,0][idx]

            if self.tsv_prior is None:
                self.tsv_prior_vec = None
            else:
                self.tsv_prior_vec = self.tsv_prior.data[0,0][idx]

            if self.shent_prior is None:
                self.shent_prior_vec = None
            else:
                self.shent_prior_vec = self.shent_prior.data[0,0][idx]

            if self.gsent_prior is None:
                self.gsent_prior_vec = None
            else:
                self.gsent_prior_vec = self.gsent_prior.data[0,0][idx]

    def _imaging_lbfgsb(niter=1000):

        self.set_image(initimage)
        self.init_image_vectors()

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


def imaging(
      initimage=None,
      maskimage=None,
      vistable=None,
      amptable=None,
      bstable=None,
      catable=None,
      lambl1=-1.,
      lambtv=-1.,
      lambtsv=-1.,
      lambcom=-1.,
      normlambda=True,
      reweight=False,
      dyrange=1e6,
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
        if isinstance(imregion, imdata.IMRegion):
            imagewin = imregion.imagewin(initimage,istokes,ifreq)
        elif isinstance(imregion, imdata.IMFITS):
            imagewin = imregion.data[0,0] > 0.5
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
        l1_lamb=np.float64(lambl1_sim),
        l1_prior=
        tv_lamb=np.float64(lambtv_sim),
        tsv_lamb=np.float64(lambtsv_sim),
        =np.float64(lambmem_sim),
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
