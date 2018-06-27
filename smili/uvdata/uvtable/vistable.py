#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes uv data table for full complex visibilities.
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import copy
import itertools
import tqdm

# numerical packages
import numpy as np
import pandas as pd
import scipy.special as ss
from scipy import optimize
from scipy import linalg
import theano.tensor as T

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import astropy.time as at

# internal
from .uvtable   import UVTable, UVSeries
from .gvistable import GVisTable, GVisSeries
from .catable   import CATable, CASeries
from .bstable   import BSTable, BSSeries
from .tools import get_uvlist, get_uvlist_loop
from ... import imdata, fortlib
# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class VisTable(UVTable):
    '''
    This class is for handling two dimensional tables of full complex visibilities
    and amplitudes. The class inherits pandas.DataFrame class, so you can use this
    class like pandas.DataFrame. The class also has additional methods to edit,
    visualize and convert data.
    '''
    @property
    def _constructor(self):
        return VisTable

    @property
    def _constructor_sliced(self):
        return VisSeries

    def set_uvunit(self, uvunit=None):
        # Check uvunits
        if uvunit is None:
            uvmax = np.max(self.uvdist.values)
            if uvmax < 1e3:
                self.uvunit = "lambda"
            elif uvmax < 1e6:
                self.uvunit = "klambda"
            elif uvmax < 1e9:
                self.uvunit = "mlambda"
            else:
                self.uvunit = "glambda"
        else:
            self.uvunit = uvunit

    def uvsort(self):
        '''
        Sort uvdata. First, it will check station IDs of each visibility
        and switch its order if "st1" > "st2". Then, data will be TB-sorted.
        '''
        outdata = self.copy()

        # search where st1 > st2
        t = outdata["st1"] > outdata["st2"]
        dammy = outdata.loc[t, "st2"]
        outdata.loc[t, "st2"] = outdata.loc[t, "st1"]
        outdata.loc[t, "st1"] = dammy
        outdata.loc[t, "phase"] *= -1

        # sort with time, and stations
        outdata = outdata.sort_values(
            by=["utc", "st1", "st2", "stokesid", "ch"])

        return outdata

    def recalc_uvdist(self):
        '''
        Re-calculate the baseline length from self["u"] and self["v"].
        '''
        self["uvdist"] = np.sqrt(self["u"] * self["u"] + self["v"] * self["v"])

    def rotate(self, dPA, deg=True):
        '''
        '''
        outdata = self.copy()
        if deg:
            theta = np.deg2rad(dPA)
        else:
            theta = dPA
        cost = np.cos(theta)
        sint = np.sin(theta)

        outdata["v"] = self["v"] * cost - self["u"] * sint
        outdata["u"] = self["v"] * sint + self["u"] * sint

        return outdata

    def deblurr(self, thetamaj=1.309, thetamin=0.64, alpha=2.0, pa=78.0):
        '''
        This is a special function for Sgr A* data, which deblurrs
        visibility amplitudes and removes a dominant effect of scattering effects.

        This method calculates a scattering kernel based on specified parameters
        of the lambda-square scattering law, and devide visibility amplitudes
        and their errors by corresponding kernel amplitudes.

        (see Fish et al. 2014, ApJL for a reference.)

        Args: Default values are from Bower et al. 2006, ApJL
            thetamaj=1.309, thetamin=0.64 (float):
                Factors for the scattering power law in mas/cm^(alpha)
            alpha=2.0 (float):
                Index of the scattering power law
            pa=78.0 (float)
                Position angle of the scattering kernel in degree
        Returns:
            De-blurred visibility data in uvdata.VisTable object
        '''
        # create a table to be output
        outtable = copy.deepcopy(self)

        # calculate scattering kernel (in visibility domain)
        kernel = geomodel.geomodel.vis_scatt(
            u=self["u"].values,
            v=self["v"].values,
            nu=self["freq"].values,
            thetamaj=thetamaj,
            thetamin=thetamin,
            alpha=alpha, pa=pa)

        # devide amplitudes and their errors by kernel amplitudes
        outtable.loc[:,"amp"] /= kernel
        outtable.loc[:,"sigma"] /= kernel
        outtable.loc[:,"weight"] = np.sqrt(1/outtable["sigma"])

        return outtable


    def fit_beam(self, angunit="mas", errweight=0., ftsign=+1):
        '''
        This method estimates the synthesized beam size at natural weighting.

        Args:
          angunit (string):
            Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
          errweight (float; experimental):
            index for errer weighting
          ftsign (integer):
            a sign for fourier matrix
        Returns:
          beam parameters in a dictionary.
        '''
        # infer the parameters of clean beam
        parm0 = calc_bparms(self)

        # generate a small image 4 times larger than the expected major axis
        # size of the beam
        fitsdata = imdata.IMFITS(fov=[parm0[0], -parm0[0], -parm0[0], parm0[0]],
                                 nx=20, ny=20, angunit="deg")

        # create output fits
        dbfitsdata, dbflux = calc_dbeam(
            fitsdata, self, errweight=errweight, ftsign=ftsign)

        X, Y = fitsdata.get_xygrid(angunit="deg", twodim=True)
        dbeam = dbfitsdata.data[0, 0]
        dbeam /= np.max(dbeam)

        parms = optimize.leastsq(fit_chisq, parm0, args=(X, Y, dbeam))

        (maja, mina, PA) = parms[0]
        maja = np.abs(maja)
        mina = np.abs(mina)

        # adjust these parameters
        if maja < mina:
            maja, mina = mina, maja
            PA += 90
        while np.abs(PA) > 90:
            if PA > 90:
                PA -= 90
            elif PA < -90:
                PA += 90

        # return as parameters of gauss_convolve
        factor = fitsdata.angconv("deg", angunit)
        cb_parms = ({'majsize': maja * factor, 'minsize': mina *
                     factor, 'angunit': angunit, 'pa': PA})
        return cb_parms


    def snrcutoff(self, threshold=5):
        '''
        Thesholding data with SNR (amp/sigma)

        Args:
            threshold (float; default=5): snrcutoff
        Returns:
            uvdata.VisTable object
        '''
        outtable = copy.deepcopy(self)
        outtable = outtable.loc[outtable["amp"]/outtable["sigma"]>threshold, :].reset_index(drop=True)
        return outtable


    def eval_image(self, imfits, mask=None, amptable=False, istokes=0, ifreq=0):
        #uvdata.VisTable object (storing model full complex visibility
        model = self._call_fftlib(imfits=imfits,mask=mask,amptable=amptable,
                                  istokes=istokes, ifreq=ifreq)

        if not amptable:
            modelr = model[0][2]
            modeli = model[0][3]
            fcv = modelr + 1j*modeli
            amp = np.abs(fcv)
            phase = np.angle(fcv,deg=True)
            fcvmodel = self.copy()
            fcvmodel["amp"] = amp
            fcvmodel["phase"] = phase
            return fcvmodel
        else:
            Ndata = model[1]
            model = model[0][2]
            amptable = self.copy()
            amptable["amp"] = model
            amptable["phase"] = np.zeros(Ndata)
            return amptable

    def eval_image3d(self, movie, mask=None, amptable=False, istokes=0, ifreq=0):
        #uvdata.VisTable object (storing model full complex visibility
        model = self._call_fftlib3d(movie=movie,mask=mask,amptable=amptable,
                                  istokes=istokes, ifreq=ifreq)

        if not amptable:
            modelr = model[0][2]
            modeli = model[0][3]
            fcv = modelr + 1j*modeli
            amp = np.abs(fcv)
            phase = np.angle(fcv,deg=True)
            fcvmodel = self.copy()
            fcvmodel["amp"] = amp
            fcvmodel["phase"] = phase
            return fcvmodel
        else:
            Ndata = model[1]
            model = model[0][2]
            amptable = self.copy()
            amptable["amp"] = model
            amptable["phase"] = np.zeros(Ndata)
            return amptable


    def residual_image(self, imfits, mask=None, amptable=False, istokes=0, ifreq=0):
        #uvdata VisTable object (storing residual full complex visibility)
        model = self._call_fftlib(imfits=imfits,mask=mask,amptable=amptable,
                                  istokes=istokes, ifreq=ifreq)

        if not amptable:
            residr = model[0][4]
            residi = model[0][5]
            resid = residr + 1j*residi
            resida = np.abs(resid)
            residp = np.angle(resid,deg=True)
            residtable = self.copy()
            residtable["amp"] = resida
            residtable["phase"] = residp

        else:
            Ndata = model[1]
            resida = model[0][3]
            residtable = self.copy()
            residtable["amp"] = resida
            residtable["phase"] = np.zeros(Ndata)

        return residtable

    def residual_image3d(self, movie, mask=None, amptable=False, istokes=0, ifreq=0):
        #uvdata VisTable object (storing residual full complex visibility)
        model = self._call_fftlib3d(movie=movie,mask=mask,amptable=amptable,
                                  istokes=istokes, ifreq=ifreq)

        if not amptable:
            residr = model[0][4]
            residi = model[0][5]
            resid = residr + 1j*residi
            resida = np.abs(resid)
            residp = np.angle(resid,deg=True)
            residtable = self.copy()
            residtable["amp"] = resida
            residtable["phase"] = residp

        else:
            Ndata = model[1]
            resida = model[0][3]
            residtable = self.copy()
            residtable["amp"] = resida
            residtable["phase"] = np.zeros(Ndata)

        return residtable


    def chisq_image(self, imfits, mask=None, amptable=False, istokes=0, ifreq=0):
        # calcurate chisqared and reduced chisqred.
        model = self._call_fftlib(imfits=imfits,mask=mask,amptable=amptable,
                                  istokes=istokes, ifreq=ifreq)
        chisq = model[0][0]
        Ndata = model[1]

        if not amptable:
            rchisq = chisq/(Ndata*2)
        else:
            rchisq = chisq/Ndata

        return chisq,rchisq

    def _call_fftlib(self, imfits, mask, amptable, istokes=0, ifreq=0):
        # get initial images
        istokes = istokes
        ifreq = ifreq

        # size of images
        Iin = np.float64(imfits.data[istokes, ifreq])
        Nx = imfits.header["nx"]
        Ny = imfits.header["ny"]
        Nyx = Nx * Ny

        # pixel coordinates
        x, y = imfits.get_xygrid(twodim=True, angunit="rad")
        xidx = np.arange(Nx) + 1
        yidx = np.arange(Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)
        Nxref = imfits.header["nxref"]
        Nyref = imfits.header["nyref"]
        dx_rad = np.deg2rad(imfits.header["dx"])
        dy_rad = np.deg2rad(imfits.header["dy"])

        # apply the imaging area
        if mask is None:
            print("Imaging Window: Not Specified. We calcurate the image on all the pixels.")
            Iin = Iin.reshape(Nyx)
            x = x.reshape(Nyx)
            y = y.reshape(Nyx)
            xidx = xidx.reshape(Nyx)
            yidx = yidx.reshape(Nyx)
        else:
            print("Imaging Window: Specified. Images will be calcurated on specified pixels.")
            idx = np.where(mask)
            Iin = Iin[idx]
            x = x[idx]
            y = y[idx]
            xidx = xidx[idx]
            yidx = yidx[idx]

        # Full Complex Visibility
        if not amptable:
            Ndata = 0
            fcvtable = self.copy()
            phase = np.deg2rad(np.array(fcvtable["phase"], dtype=np.float64))
            amp = np.array(fcvtable["amp"], dtype=np.float64)
            vfcvr = np.float64(amp*np.cos(phase))
            vfcvi = np.float64(amp*np.sin(phase))
            varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
            Ndata += len(varfcv)
            del phase, amp

            # get uv coordinates and uv indice
            u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
                    fcvtable=fcvtable, amptable=None, bstable=None, catable=None
                    )

            # normalize u, v coordinates
            u *= 2*np.pi*dx_rad
            v *= 2*np.pi*dy_rad

            # run model_fcv
            model = fortlib.fftlib.model_fcv(
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
                    # Full Complex Visibilities
                    uvidxfcv=np.int32(uvidxfcv),
                    vfcvr=np.float64(vfcvr),
                    vfcvi=np.float64(vfcvi),
                    varfcv=np.float64(varfcv)
                    )

            return model,Ndata

        else:
            Ndata = 0
            amptable = self.copy()
            dammyreal = np.zeros(1, dtype=np.float64)
            vfcvr = dammyreal
            vfcvi = dammyreal
            varfcv = dammyreal
            vamp = np.array(amptable["amp"], dtype=np.float64)
            varamp = np.square(np.array(amptable["sigma"], dtype=np.float64))
            Ndata += len(vamp)

            # get uv coordinates and uv indice
            u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
                    fcvtable=None, amptable=amptable, bstable=None, catable=None
                    )

            # normalize u, v coordinates
            u *= 2*np.pi*dx_rad
            v *= 2*np.pi*dy_rad

            # run model_fcv
            model = fortlib.fftlib.model_amp(
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
                    # Full Complex Visibilities
                    uvidxamp=np.int32(uvidxamp),
                    vamp=np.float64(vamp),
                    varamp=np.float64(varamp)
                    )
            return model,Ndata




    def chisq_image3d(self, movie, mask=None, amptable=False, istokes=0, ifreq=0):
        # calcurate chisqared and reduced chisqred.
        model = self._call_fftlib3d(movie=movie,mask=mask,amptable=amptable,
                                  istokes=istokes, ifreq=ifreq)
        chisq = model[0][0]
        Ndata = model[1]
        if not amptable:
            rchisq = chisq/(Ndata*2)
        else:
            rchisq = chisq/Ndata

        return chisq,rchisq

    def _call_fftlib3d(self, movie, mask, amptable, istokes=0, ifreq=0):
        # get initial images
        istokes = istokes
        ifreq  = ifreq
        Nt     = movie.Nt
        # size of images
        Iin = []
        for im in movie.images:
            Iin.append(np.float64(im.data[istokes, ifreq]))
        imfits = movie.images[0]

        Nx = imfits.header["nx"]
        Ny = imfits.header["ny"]
        Nyx = Nx * Ny

        # pixel coordinates
        x, y = imfits.get_xygrid(twodim=True, angunit="rad")
        xidx = np.arange(Nx) + 1
        yidx = np.arange(Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)
        Nxref = imfits.header["nxref"]
        Nyref = imfits.header["nyref"]
        dx_rad = np.deg2rad(imfits.header["dx"])
        dy_rad = np.deg2rad(imfits.header["dy"])

        # apply the imaging area
        if mask is None:
            print("Imaging Window: Not Specified. We calcurate the image on all the pixels.")
            for i in xrange(len(Iin)):
                Iin[i] = Iin[i].reshape(Nyx)
            x = x.reshape(Nyx)
            y = y.reshape(Nyx)
            xidx = xidx.reshape(Nyx)
            yidx = yidx.reshape(Nyx)
        else:
            print("Imaging Window: Specified. Images will be calcurated on specified pixels.")
            idx = np.where(mask)
            for i in xrange(len(Iin)):
                Iin[i] = Iin[i][idx]
            x = x[idx]
            y = y[idx]
            xidx = xidx[idx]
            yidx = yidx[idx]

        # Full Complex Visibility
        if not amptable:
            Ndata = 0
            fcvtable = self.copy()
            phase = np.deg2rad(np.array(fcvtable["phase"], dtype=np.float64))
            amp = np.array(fcvtable["amp"], dtype=np.float64)
            vfcvr = np.float64(amp*np.cos(phase))
            vfcvi = np.float64(amp*np.sin(phase))
            varfcv = np.square(np.array(fcvtable["sigma"], dtype=np.float64))
            Ndata += len(varfcv)
            del phase, amp

            # get uv coordinates and uv indice
            u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = get_uvlist_loop(Nt=Nt,
                fcvconcat=fcvtable, ampconcat=None, bsconcat=None, caconcat=None
            )

            # normalize u, v coordinates
            u *= 2*np.pi*dx_rad
            v *= 2*np.pi*dy_rad

            # concatenate the initimages
            Iin = np.concatenate(Iin)

            # run model_fcv
            model = fortlib.fftlib3d.model_fcv(
                    # Images
                    iin=np.float64(Iin),
                    xidx=np.int32(xidx),
                    yidx=np.int32(yidx),
                    nxref=np.float64(Nxref),
                    nyref=np.float64(Nyref),
                    nx=np.int32(Nx),
                    ny=np.int32(Ny),
                    nz=np.int32(Nt),
                    # UV coordinates,
                    u=u,
                    v=v,
                    nuvs=np.int32(Nuvs),
                    # Full Complex Visibilities
                    uvidxfcv=np.int32(uvidxfcv),
                    vfcvr=np.float64(vfcvr),
                    vfcvi=np.float64(vfcvi),
                    varfcv=np.float64(varfcv)
                    )

            return model,Ndata

        else:
            Ndata = 0
            amptable = self.copy()
            dammyreal = np.zeros(1, dtype=np.float64)
            vfcvr = dammyreal
            vfcvi = dammyreal
            varfcv = dammyreal
            vamp = np.array(amptable["amp"], dtype=np.float64)
            varamp = np.square(np.array(amptable["sigma"], dtype=np.float64))
            Ndata += len(vamp)

            # get uv coordinates and uv indice
            u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = get_uvlist_loop(Nt=Nt,
                fcvconcat=None, ampconcat=amptable, bsconcat=None, caconcat=None
            )

            # normalize u, v coordinates
            u *= 2*np.pi*dx_rad
            v *= 2*np.pi*dy_rad

            # concatenate the initimages
            Iin = np.concatenate(Iin)

            # run model_fcv
            model = fortlib.fftlib3d.model_amp(
                    # Images
                    iin=np.float64(Iin),
                    xidx=np.int32(xidx),
                    yidx=np.int32(yidx),
                    nxref=np.float64(Nxref),
                    nyref=np.float64(Nyref),
                    nx=np.int32(Nx),
                    ny=np.int32(Ny),
                    nz=np.int32(Nt),
                    # UV coordinates,
                    u=u,
                    v=v,
                    nuvs=np.int32(Nuvs),
                    # Full Complex Visibilities
                    uvidxamp=np.int32(uvidxamp),
                    vamp=np.float64(vamp),
                    varamp=np.float64(varamp)
                    )

            return model,Ndata


    def eval_geomodel(self, geomodel, evalargs={}):
        '''
        Evaluate model values and output them to a new table

        Args:
            geomodel (geomodel.geomodel.GeoModel) object
        Returns:
            uvdata.VisTable object
        '''
        # create a table to be output
        outtable = copy.deepcopy(self)

        # u,v coordinates
        u = outtable.u.values
        v = outtable.v.values
        outtable["amp"] = geomodel.Vamp(u,v).eval(**evalargs)
        outtable["phase"] = geomodel.Vphase(u,v).eval(**evalargs) * 180./np.pi

        return outtable

    def residual_geomodel(self, geomodel, normed=True, amp=False, doeval=False, evalargs={}):
        '''
        Evaluate Geometric Model

        Args:
            geomodel (geomodel.geomodel.GeoModel object):
                input model
            normed (boolean, default=True):
                if True, residuals will be normalized by 1 sigma error
            amp (boolean, default=False):
                if True, residuals will be calculated for amplitudes.
                Otherwise, residuals will be calculate for full complex visibilities
            eval (boolean, default=False):
                if True, actual residual values will be calculated.
                Otherwise, resduals will be given as a theano graph.
        Returns:
            ndarray (if doeval=True) or theano object (otherwise)
        '''
        # u,v coordinates
        u = self.u.values
        v = self.v.values
        Vamp = self.amp.values
        Vpha = self.phase.values * np.pi / 180.
        sigma = self.sigma.values

        if amp is False:
            modVre = geomodel.Vreal(u,v)
            modVim = geomodel.Vimag(u,v)
            Vre = Vamp * T.cos(Vpha)
            Vim = Vamp * T.sin(Vpha)
            resid_re = Vre - modVre
            resid_im = Vim - modVim
            if normed:
                resid_re /= sigma
                resid_im /= sigma
            residual = T.concatanate([resid_re, resid_im])
        else:
            modVamp = geomodel.Vamp(u,v)
            residual = Vamp - modVamp
            if normed:
                residual /= sigma

        if doeval:
            return residual.eval(**evalargs)
        else:
            return residual

    def make_bstable(self, redundant=None):
        '''
        Form bi-spectra from complex visibilities.

        Args:
            redandant (list of sets of redundant station IDs; default=None):
                If this is specified, non-redandant and non-trivial bispectra will be formed.
                This is useful for EHT-like array that have redandant stations in the same site.
                For example, if stations 1,2,3 and 4,5 are on the same sites, respectively, then
                you can specify redundant=[[1,2,3],[4,5]].
        Returns:
            uvdata.BSTable object
        '''
        # Number of Stations
        Ndata = len(self["ch"])

        # make dictionary of stations
        st1table = self.drop_duplicates(subset='st1')
        st2table = self.drop_duplicates(subset='st2')
        stdict = dict(zip(st1table["st1"], st1table["st1name"]))
        stdict.update(dict(zip(st2table["st2"], st2table["st2name"])))

        # Check redundant
        if redundant is not None:
            for i in xrange(len(redundant)):
                redundant[i] = sorted(set(redundant[i]))

        print("(1/5) Sort data")
        vistable = self.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
        vistable["bl"] = vistable["st1"] * 256 + vistable["st2"]

        print("(2/5) Tagging data")
        vistable["tag"] = np.zeros(Ndata, dtype=np.int64)
        for idata in tqdm.tqdm(xrange(1,Ndata)):
            flag = vistable.loc[idata, "utc"] == vistable.loc[idata-1, "utc"]
            flag&= vistable.loc[idata, "stokesid"] == vistable.loc[idata-1, "stokesid"]
            flag&= vistable.loc[idata, "ch"] == vistable.loc[idata-1, "ch"]
            if flag:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]
            else:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]+1
        Ntag = vistable["tag"].max() + 1
        print("  Number of Tags: %d"%(Ntag))

        print("(3/5) Checking Baseline Combinations")
        blsets = [] # baseline coverages
        cblsets = [] # combination to make closure amplitudes
        cstsets = [] # combination to make closure amplitudes
        blsetid=0
        bltype = np.zeros(Ntag, dtype=np.int64) # this is an array storing an ID
                                                # number of cl sets corresponding timetag.
        for itag in tqdm.tqdm(xrange(Ntag)):
            tmptab = vistable.loc[vistable["tag"]==itag, :].reset_index(drop=True)

            # Check number of baselines
            blset = sorted(set(tmptab["bl"]))
            Nbl = len(blset)
            if Nbl < 3:
                bltype[itag] = -1
                #print("Tag %d: skip Nbl=%d<4"%(itag,Nbl))
                continue

            # Check number of stations
            stset = sorted(set(tmptab["st1"].tolist()+tmptab["st2"].tolist()))
            Nst = len(stset)
            if Nst < 3:
                bltype[itag] = -1
                #print("Tag %d: skip Nst=%d<4"%(itag,Nst))
                continue

            # Check this baseline combinations are already detected
            try:
                iblset = blsets.index(blset)
                bltype[itag] = iblset
                #print("Tag %d: the same combination was already detected %d"%(itag, iblset))
                continue
            except ValueError:
                pass

            # Number of baseline combinations
            Nblmax = Nst * (Nst - 1) // 2
            Nbsmax = (Nst-1) * (Nst - 2) // 2

            # Check combinations
            rank = 0
            matrix = None
            rank = 0
            cblset = []
            cstset = []
            for stid1, stid2, stid3 in itertools.combinations(range(Nst), 3):
                Ncomb=0

                # Stations
                st1 = stset[stid1]
                st2 = stset[stid2]
                st3 = stset[stid3]

                # baselines
                bl12 = st1*256 + st2
                bl23 = st2*256 + st3
                bl13 = st1*256 + st3

                # baseline ID
                blid12 = getblid(stid1, stid2, Nst)
                blid23 = getblid(stid2, stid3, Nst)
                blid13 = getblid(stid1, stid3, Nst)

                if rank>=Nbsmax:
                    break
                isnontrivial = check_nontrivial([[st1,st2], [st1,st3], [st2,st3]],redundant)
                isbaselines = check_baselines([bl12,bl23,bl13], blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_bs(matrix, blid12, blid23, blid13, Nblmax)
                    if newrank > rank:
                        cblset.append([bl12,bl23,bl13])
                        cstset.append([st1,st2,st3])
                        rank = newrank
                        matrix = newmatrix
                        Ncomb +=1
            if rank == 0:
                cblset=None
                cstset=None
            #print(itag, blset, cblset, cstset)
            blsets.append(blset) # baseline coverages
            cblsets.append(cblset) # combination to make closure amplitudes
            cstsets.append(cstset) # combination to make closure amplitudes
            bltype[itag] = len(cblsets) - 1
        print("  Detect %d combinations for Closure Phases"%(len(cblsets)))

        print("(4/5) Forming Closure Phases")
        keys = "utc,gsthour,freq,stokesid,ifid,chid,ch,"
        keys+= "st1,st2,st3,"
        keys+= "st1name,st2name,st3name,"
        keys+= "u12,v12,w12,u23,v23,w23,u31,v31,w31,"
        keys+= "amp,phase,sigma"
        #keys+="frmidx"
        keys = keys.split(",")
        outtab = {}
        for key in keys:
            outtab[key]=[]
        for itag in tqdm.tqdm(xrange(Ntag)):
            # Check ID number for Baseline combinations
            blsetid = bltype[itag]
            if blsetid == -1:
                continue
            # Get combination
            cblset = cblsets[blsetid]
            cstset = cstsets[blsetid]
            if cblset is None or cstset is None:
                continue
            tmptab = vistable.loc[vistable["tag"]==itag, :].reset_index(drop=True)
            utc = tmptab.loc[0, "utc"]
            gsthour = tmptab.loc[0, "gsthour"]
            freq = tmptab.loc[0, "freq"]
            stokesid = tmptab.loc[0, "stokesid"]
            ifid = tmptab.loc[0, "ifid"]
            chid = tmptab.loc[0, "chid"]
            ch = tmptab.loc[0, "ch"]
            for iset in xrange(len(cblset)):
                bl1tab = tmptab.loc[tmptab["bl"]==cblset[iset][0],:].reset_index(drop=True)
                bl2tab = tmptab.loc[tmptab["bl"]==cblset[iset][1],:].reset_index(drop=True)
                bl3tab = tmptab.loc[tmptab["bl"]==cblset[iset][2],:].reset_index(drop=True)
                #
                ratio_1 = bl1tab.loc[0,"sigma"] / bl1tab.loc[0,"amp"]
                ratio_2 = bl2tab.loc[0,"sigma"] / bl2tab.loc[0,"amp"]
                ratio_3 = bl3tab.loc[0,"sigma"] / bl3tab.loc[0,"amp"]
                amp = bl1tab.loc[0,"amp"] * bl2tab.loc[0,"amp"] * bl3tab.loc[0,"amp"]
                phase = bl1tab.loc[0,"phase"] + bl2tab.loc[0,"phase"] - bl3tab.loc[0,"phase"]
                sigma = amp * np.sqrt((ratio_1)**2 + (ratio_2)**2 + (ratio_3)**2)

                #
                outtab["utc"].append(utc)
                outtab["gsthour"].append(gsthour)
                outtab["freq"].append(freq)
                outtab["stokesid"].append(stokesid)
                outtab["ifid"].append(ifid)
                outtab["chid"].append(chid)
                outtab["ch"].append(ch)
                outtab["st1"].append(cstset[iset][0])
                outtab["st2"].append(cstset[iset][1])
                outtab["st3"].append(cstset[iset][2])
                outtab["st1name"].append(stdict[cstset[iset][0]])
                outtab["st2name"].append(stdict[cstset[iset][1]])
                outtab["st3name"].append(stdict[cstset[iset][2]])
                outtab["u12"].append(bl1tab.loc[0,"u"])
                outtab["v12"].append(bl1tab.loc[0,"v"])
                outtab["w12"].append(bl1tab.loc[0,"w"])
                outtab["u23"].append(bl2tab.loc[0,"u"])
                outtab["v23"].append(bl2tab.loc[0,"v"])
                outtab["w23"].append(bl2tab.loc[0,"w"])
                outtab["u31"].append(-bl3tab.loc[0,"u"])
                outtab["v31"].append(-bl3tab.loc[0,"v"])
                outtab["w31"].append(-bl3tab.loc[0,"w"])
                outtab["amp"].append(amp)
                outtab["phase"].append(phase)
                outtab["sigma"].append(sigma)

                #if(movie is None):
                #    frmidx = 0
                #    outtab["frmidx"].append(frmidx)

        print("(5/5) Creating BSTable object")
        # Calculate UV Distance
        outtab = pd.DataFrame(outtab)
        outtab["uvdist12"] = np.sqrt(np.square(outtab["u12"])+np.square(outtab["v12"]))
        outtab["uvdist23"] = np.sqrt(np.square(outtab["u23"])+np.square(outtab["v23"]))
        outtab["uvdist31"] = np.sqrt(np.square(outtab["u31"])+np.square(outtab["v31"]))
        uvdists = np.asarray([outtab["uvdist12"],outtab["uvdist23"],outtab["uvdist31"]])
        outtab["uvdistave"] = np.mean(uvdists, axis=0)
        outtab["uvdistmax"] = np.max(uvdists, axis=0)
        outtab["uvdistmin"] = np.min(uvdists, axis=0)
        outtab["phase"] = np.deg2rad(outtab["phase"])
        outtab["phase"] = np.rad2deg(np.arctan2(np.sin(outtab["phase"]),np.cos(outtab["phase"])))
        # generate CATable object
        outtab = BSTable(outtab)[BSTable.bstable_columns].reset_index(drop=True)
        for i in xrange(len(BSTable.bstable_columns)):
            column = BSTable.bstable_columns[i]
            outtab[column] = BSTable.bstable_types[i](outtab[column])

            #if (movie is not None):
            #    outtab = movie.set_frmidx(outtab)
        return outtab

    def make_catable(self, redundant=None, debias=True):
        '''
        Form closure amplitudes from complex visibilities.

        Args:
            redandant (list of sets of redundant station IDs; default=None):
                If this is specified, non-redandant and non-trivial closure amplitudes will be formed.
                This is useful for EHT-like array that have redandant stations in the same site.
                For example, if stations 1,2,3 and 4,5 are on the same sites, respectively, then
                you can specify redundant=[[1,2,3],[4,5]].
            debias (boolean, default=True):
                if debias==True, then closure amplitudes and log closure amplitudes will be debiased using
                a formula for high-SNR limits.
        Returns:
            uvdata.CATable object
        '''
        from scipy.special import expi
        # Number of Stations
        Ndata = len(self["ch"])

        # make dictionary of stations
        st1table = self.drop_duplicates(subset='st1')
        st2table = self.drop_duplicates(subset='st2')
        stdict = dict(zip(st1table["st1"], st1table["st1name"]))
        stdict.update(dict(zip(st2table["st2"], st2table["st2name"])))

        # Check redundant
        if redundant is not None:
            for i in xrange(len(redundant)):
                redundant[i] = sorted(set(redundant[i]))

        print("(1/5) Sort data")
        vistable = self.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
        vistable["bl"] = vistable["st1"] * 256 + vistable["st2"]

        print("(2/5) Tagging data")
        vistable["tag"] = np.zeros(Ndata, dtype=np.int64)
        for idata in tqdm.tqdm(xrange(1,Ndata)):
            flag = vistable.loc[idata, "utc"] == vistable.loc[idata-1, "utc"]
            flag&= vistable.loc[idata, "stokesid"] == vistable.loc[idata-1, "stokesid"]
            flag&= vistable.loc[idata, "ch"] == vistable.loc[idata-1, "ch"]
            if flag:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]
            else:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]+1
        Ntag = vistable["tag"].max() + 1
        print("  Number of Tags: %d"%(Ntag))

        print("(3/5) Checking Baseline Combinations")
        blsets = [] # baseline coverages
        cblsets = [] # combination to make closure amplitudes
        cstsets = [] # combination to make closure amplitudes
        blsetid=0
        bltype = np.zeros(Ntag, dtype=np.int64) # this is an array storing an ID
                                                # number of cl sets corresponding timetag.
        for itag in tqdm.tqdm(xrange(Ntag)):
            tmptab = vistable.loc[vistable["tag"]==itag, :].reset_index(drop=True)

            # Check number of baselines
            blset = sorted(set(tmptab["bl"]))
            Nbl = len(blset)
            if Nbl < 4:
                bltype[itag] = -1
                #print("Tag %d: skip Nbl=%d<4"%(itag,Nbl))
                continue

            # Check number of stations
            stset = sorted(set(tmptab["st1"].tolist()+tmptab["st2"].tolist()))
            Nst = len(stset)
            if Nst < 4:
                bltype[itag] = -1
                #print("Tag %d: skip Nst=%d<4"%(itag,Nst))
                continue

            # Check this baseline combinations are already detected
            try:
                iblset = blsets.index(blset)
                bltype[itag] = iblset
                #print("Tag %d: the same combination was already detected %d"%(itag, iblset))
                continue
            except ValueError:
                pass

            # Number of baseline combinations
            Nblmax = Nst * (Nst - 1) // 2
            Ncamax = Nst * (Nst - 3) // 2

            # Check combinations
            rank = 0
            matrix = None
            rank = 0
            cblset = []
            cstset = []
            for stid1, stid2, stid3, stid4 in itertools.combinations(range(Nst), 4):
                Ncomb=0

                # Stations
                st1 = stset[stid1]
                st2 = stset[stid2]
                st3 = stset[stid3]
                st4 = stset[stid4]

                # baselines
                bl12 = st1*256 + st2
                bl13 = st1*256 + st3
                bl14 = st1*256 + st4
                bl23 = st2*256 + st3
                bl24 = st2*256 + st4
                bl34 = st3*256 + st4

                # baseline ID
                blid12 = getblid(stid1, stid2, Nst)
                blid13 = getblid(stid1, stid3, Nst)
                blid14 = getblid(stid1, stid4, Nst)
                blid23 = getblid(stid2, stid3, Nst)
                blid24 = getblid(stid2, stid4, Nst)
                blid34 = getblid(stid3, stid4, Nst)

                # Number of combinations
                Ncomb = 0

                # Get number
                # Combination 1: (V12 V34) / (V13 V24)
                #   This conmbination becomes trivial if
                #   site1 == site4 or site2 == site3.
                if rank>=Ncamax:
                    break
                isnontrivial = check_nontrivial([[st1,st4], [st2,st3]],redundant)
                isbaselines = check_baselines([bl12,bl34,bl13,bl24], blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_ca(matrix, blid12, blid34, blid13, blid24, Nblmax)
                    if newrank > rank:
                        cblset.append([bl12,bl34,bl13,bl24])
                        cstset.append([st1,st2,st3,st4])
                        rank = newrank
                        matrix = newmatrix
                        Ncomb +=1

                # Combination 2: (V13 V24) / (V14 V23)
                #   This conmbination becomes trivial if
                #   site1 == site2 or site3 == site4.
                if rank>=Ncamax:
                    break
                isnontrivial = check_nontrivial([[st1,st2],[st3,st4]],redundant)
                isbaselines = check_baselines([bl13,bl24,bl14,bl23],blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_ca(matrix, blid13, blid24, blid14, blid23, Nblmax)
                    if newrank > rank:
                        cblset.append([bl13,bl24,bl14,bl23])
                        cstset.append([st1,st3,st4,st2])
                        rank = newrank
                        matrix = newmatrix
                        Ncomb +=1

                # Combination 3: (V12 V34) / (V14 V23)
                #   This conmbination becomes trivial if
                #   site1 == site3 or site2 == site4.
                if Ncomb>1:
                    continue
                if rank>=Ncamax:
                    break
                isnontrivial = check_nontrivial([[st1,st3],[st2,st4]],redundant)
                isbaselines = check_baselines([bl12,bl34,bl14,bl23],blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_ca(matrix, blid12, blid34, blid14, blid23, Nblmax)
                    if newrank > rank:
                        cblset.append([bl12,bl34,bl14,bl23])
                        cstset.append([st1,st2,st4,st3])
                        rank = newrank
                        matrix = newmatrix
            if rank == 0:
                cblset=None
                cstset=None
            #print(itag, blset, cblset, cstset)
            blsets.append(blset) # baseline coverages
            cblsets.append(cblset) # combination to make closure amplitudes
            cstsets.append(cstset) # combination to make closure amplitudes
            bltype[itag] = len(cblsets) - 1
        print("  Detect %d combinations for Closure Amplitudes"%(len(cblsets)))

        print("(4/5) Forming Closure Amplitudes")
        keys = "utc,gsthour,freq,stokesid,ifid,chid,ch,"
        keys+= "st1,st2,st3,st4,"
        keys+= "st1name,st2name,st3name,st4name,"
        keys+= "u1,v1,w1,u2,v2,w2,u3,v3,w3,u4,v4,w4,"
        keys+= "amp,sigma,logamp,logsigma"
        keys = keys.split(",")
        outtab = {}
        for key in keys:
            outtab[key]=[]
        for itag in tqdm.tqdm(xrange(Ntag)):
            # Check ID number for Baseline combinations
            blsetid = bltype[itag]
            if blsetid == -1:
                continue
            # Get combination
            cblset = cblsets[blsetid]
            cstset = cstsets[blsetid]
            if cblset is None or cstset is None:
                continue
            tmptab = vistable.loc[vistable["tag"]==itag, :].reset_index(drop=True)
            utc = tmptab.loc[0, "utc"]
            gsthour = tmptab.loc[0, "gsthour"]
            freq = tmptab.loc[0, "freq"]
            stokesid = tmptab.loc[0, "stokesid"]
            ifid = tmptab.loc[0, "ifid"]
            chid = tmptab.loc[0, "chid"]
            ch = tmptab.loc[0, "ch"]
            for iset in xrange(len(cblset)):
                bl1tab = tmptab.loc[tmptab["bl"]==cblset[iset][0],:].reset_index(drop=True)
                bl2tab = tmptab.loc[tmptab["bl"]==cblset[iset][1],:].reset_index(drop=True)
                bl3tab = tmptab.loc[tmptab["bl"]==cblset[iset][2],:].reset_index(drop=True)
                bl4tab = tmptab.loc[tmptab["bl"]==cblset[iset][3],:].reset_index(drop=True)
                #
                # 1 sigma debiasing of snr
                rhosq_1 = np.nanmax([1,np.square(bl1tab.loc[0,"amp"] / bl1tab.loc[0,"sigma"])-1])
                rhosq_2 = np.nanmax([1,np.square(bl2tab.loc[0,"amp"] / bl2tab.loc[0,"sigma"])-1])
                rhosq_3 = np.nanmax([1,np.square(bl3tab.loc[0,"amp"] / bl3tab.loc[0,"sigma"])-1])
                rhosq_4 = np.nanmax([1,np.square(bl4tab.loc[0,"amp"] / bl4tab.loc[0,"sigma"])-1])

                # closure amplitudes
                amp = bl1tab.loc[0,"amp"] * bl2tab.loc[0,"amp"] / \
                      bl3tab.loc[0,"amp"] / bl4tab.loc[0,"amp"]
                logamp = np.log(amp)
                logsigma = np.sqrt(rhosq_1**-1 + rhosq_2**-1 +
                                   rhosq_3**-1 + rhosq_4**-1)
                sigma = amp * logsigma

                if debias:
                    # bias estimator
                    bias = expi(-rhosq_1/2.)
                    bias+= expi(-rhosq_2/2.)
                    bias-= expi(-rhosq_3/2.)
                    bias-= expi(-rhosq_4/2.)
                    bias*= 0.5

                    logamp += bias
                    amp *= np.exp(bias)
                #
                outtab["utc"].append(utc)
                outtab["gsthour"].append(gsthour)
                outtab["freq"].append(freq)
                outtab["stokesid"].append(stokesid)
                outtab["ifid"].append(ifid)
                outtab["chid"].append(chid)
                outtab["ch"].append(ch)
                outtab["st1"].append(cstset[iset][0])
                outtab["st2"].append(cstset[iset][1])
                outtab["st3"].append(cstset[iset][2])
                outtab["st4"].append(cstset[iset][3])
                outtab["st1name"].append(stdict[cstset[iset][0]])
                outtab["st2name"].append(stdict[cstset[iset][1]])
                outtab["st3name"].append(stdict[cstset[iset][2]])
                outtab["st4name"].append(stdict[cstset[iset][3]])
                outtab["u1"].append(bl1tab.loc[0,"u"])
                outtab["v1"].append(bl1tab.loc[0,"v"])
                outtab["w1"].append(bl1tab.loc[0,"w"])
                outtab["u2"].append(bl2tab.loc[0,"u"])
                outtab["v2"].append(bl2tab.loc[0,"v"])
                outtab["w2"].append(bl2tab.loc[0,"w"])
                outtab["u3"].append(bl3tab.loc[0,"u"])
                outtab["v3"].append(bl3tab.loc[0,"v"])
                outtab["w3"].append(bl3tab.loc[0,"w"])
                outtab["u4"].append(bl4tab.loc[0,"u"])
                outtab["v4"].append(bl4tab.loc[0,"v"])
                outtab["w4"].append(bl4tab.loc[0,"w"])
                outtab["amp"].append(amp)
                outtab["sigma"].append(sigma)
                outtab["logamp"].append(logamp)
                outtab["logsigma"].append(logsigma)

        print("(5/5) Creating CATable object")
        # Calculate UV Distance
        outtab = pd.DataFrame(outtab)
        outtab["uvdist1"] = np.sqrt(np.square(outtab["u1"])+np.square(outtab["v1"]))
        outtab["uvdist2"] = np.sqrt(np.square(outtab["u2"])+np.square(outtab["v2"]))
        outtab["uvdist3"] = np.sqrt(np.square(outtab["u3"])+np.square(outtab["v3"]))
        outtab["uvdist4"] = np.sqrt(np.square(outtab["u4"])+np.square(outtab["v4"]))
        uvdists = np.asarray([outtab["uvdist1"],outtab["uvdist2"],outtab["uvdist3"],outtab["uvdist4"]])
        outtab["uvdistave"] = np.mean(uvdists, axis=0)
        outtab["uvdistmax"] = np.max(uvdists, axis=0)
        outtab["uvdistmin"] = np.min(uvdists, axis=0)

        # generate CATable object
        outtab = CATable(outtab)[CATable.catable_columns].reset_index(drop=True)
        for i in xrange(len(CATable.catable_columns)):
            column = CATable.catable_columns[i]
            outtab[column] = CATable.catable_types[i](outtab[column])
        return outtab


    def gridding(self, fitsdata, conj=False, j=3., beta=2.34):
        '''
        Perform uv-girdding based on image pixel information of
        the input fitsdata using the 0-th order Keiser-Bessel Function.

        Args:
          vistable (pandas.Dataframe object):
            input visibility table

          fitsdata (imdata.IMFITS object):
            input imdata.IMFITS object

          conj (boolean, default=False):
            If true, output also conjugate components

          j (int; default = 3):
            The number of grids (j x j pixels) giving the size of the
            convolution kernel. Default is 3x3 pixels.

          beta (float; default = 2.34):
            The spread of the Kaiser Bessel function. The default value
            is based on work presented in Fessler & Sutton (2003).

        Returns:
          uvdata.GVisTable object
        '''
        # Copy vistable for edit
        vistable = copy.deepcopy(self)
        # Flip uv cordinates and phase, where u < 0 for avoiding redundant
        vistable.loc[vistable["u"] < 0, ("u", "v", "phase")] *= -1
        # sort data with u,v coordinates
        vistable.sort_values(by=["u","v"])

        # get images
        dx = np.abs(fitsdata.header["dx"])
        dy = np.abs(fitsdata.header["dy"])
        nx = fitsdata.header["nx"]
        ny = fitsdata.header["ny"]
        nxref = fitsdata.header["nxref"]
        nyref = fitsdata.header["nyref"]
        Lx = np.deg2rad(dx*nx)
        Ly = np.deg2rad(dy*ny)

        # Calculate du and dv
        du = 1 / Lx
        dv = 1 / Ly

        # uv index
        u = vistable.u.values
        v = vistable.v.values
        ugidxs = np.int32(np.around(u / du))
        vgidxs = np.int32(np.around(v / dv))

        # Create column of full-comp visibility
        Vcomps = vistable.amp.values * \
                 np.exp(1j * np.deg2rad(vistable.phase.values))
        sigma = vistable.sigma.values
        weight = 1 / sigma**2
        Ntable = len(vistable)

        # Calculate index of uv for gridding
        # Flag for skipping already averaged data
        skip = np.asarray([False for i in xrange(Ntable)])

        # Create new list for gridded data
        outlist = {
            "ugidx": [],
            "vgidx": [],
            "u": [],
            "v": [],
            "uvdist": [],
            "amp": [],
            "phase": [],
            "weight": [],
            "sigma": []
        }

        # Convolutional gridding
        for itable in xrange(Ntable):
            if skip[itable]:
                continue

            # Get the grid index for the current data
            ugidx = ugidxs[itable]
            vgidx = vgidxs[itable]
            ugrid = ugidx * du
            vgrid = vgidx * dv

            # Data index for visibilities on the same grid
            gidxs = np.where((ugidxs == ugidx) & (vgidxs == vgidx))
            # Flip flags
            skip[gidxs] = True
            del gidxs

            # Gridding Kernel
            gidxs = np.abs(ugidxs - ugidx)<=j//2
            gidxs&= np.abs(vgidxs - vgidx)<=j//2
            gidxs = np.where(gidxs)

            # Calculate Keiser-Bessel Function
            unorm = beta*j*np.sqrt(1-np.square(2*(ugidxs[gidxs]-ugidx)/3.))
            vnorm = beta*j*np.sqrt(1-np.square(2*(vgidxs[gidxs]-vgidx)/3.))
            ukern = ss.iv(0, unorm)
            vkern = ss.iv(0, vnorm)
            norm = np.sum(ukern * vkern)

            # Convolutional gridding
            Vcomp_ave = np.sum(Vcomps[gidxs] * ukern * vkern) / norm
            weight_ave = np.sum(weight[gidxs] * ukern * vkern) / norm
            sigma_ave = 1 / np.sqrt(weight_ave)

            # Save gridded data on the grid
            outlist["ugidx"].append(ugidx)
            outlist["vgidx"].append(vgidx)
            outlist["u"].append(ugrid)
            outlist["v"].append(vgrid)
            outlist["uvdist"].append(np.sqrt(ugrid**2 + vgrid**2))
            outlist["amp"].append(np.abs(Vcomp_ave))
            outlist["phase"].append(np.angle(Vcomp_ave, deg=True))
            outlist["weight"].append(weight_ave)
            outlist["sigma"].append(sigma_ave)

        # Output as pandas.DataFrame
        outtable = pd.DataFrame(outlist,
            columns=["ugidx", "vgidx", "u", "v", "uvdist",
                     "amp", "phase", "weight", "sigma"])
        if conj==True:
            outtable_c = copy.deepcopy(outtable)
            outtable_c.loc[:,["u","v","ugidx","vgidx","phase"]] *= -1
            outtable = pd.concat([outtable, outtable_c], ignore_index=True)
        outtable = GVisTable(outtable)
        outtable.nx = nx
        outtable.ny = ny
        outtable.nxref = nxref
        outtable.nyref = nyref
        outtable.du = du
        outtable.dv = dv
        return outtable


    #-------------------------------------------------------------------------
    # Plot Functions
    #-------------------------------------------------------------------------
    def uvplot(self, uvunit=None, conj=True,
               ls="none", marker=".", **plotargs):
        '''
        Plot uv-plot on the current axes.
        This method uses matplotlib.pyplot.plot().

        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.

          conj (boolean, default = True):
            if conj=True, it will plot complex conjugate components (i.e. (-u, -v)).

          **plotargs:
            You can set parameters of matplotlib.pyplot.plot.
            Defaults are {'ls': "none", 'marker': "."}
        '''
        # Set Unit
        if uvunit is None:
            self.set_uvunit()
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # Label
        unitlabel = self.get_unitlabel(uvunit)

        # plotting
        plt.plot(self["u"] * conv, self["v"] * conv,
                 ls=ls, marker=marker, **plotargs)
        if conj:
            plotargs2 = copy.deepcopy(plotargs)
            plotargs2["label"] = ""
            plt.plot(-self["u"] * conv, -self["v"] * conv,
                     ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))

    def radplot(self, uvunit=None, datatype="amp", normerror=False, errorbar=True,
                ls="none", marker=".", **plotargs):
        '''
        Plot visibility amplitudes as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbar().

        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.

          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot().

            If you plot model closure phases (i.e. model is not None),
            it will plot without errobars regardless of this parameter.

          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            self.set_uvunit()
            uvunit = self.uvunit

        # Copy data
        vistable = copy.deepcopy(self)

        # add real and imaginary part of full-comp. visibilities
        if datatype=="real" or datatype=="imag" or datatype=="real&imag":
            amp = np.float64(vistable["amp"])
            phase = np.radians(np.float64(vistable["phase"]))
            #
            vistable["real"] = amp * np.cos(phase)
            vistable["imag"] = amp * np.sin(phase)

        # Normalized by error
        if normerror:
            if datatype=="amp" or datatype=="amp&phase":
                vistable["amp"] /= vistable["sigma"]
            if datatype=="phase" or datatype=="amp&phase":
                pherr = np.rad2deg(vistable["sigma"] / vistable["amp"])
                vistable["phase"] /= pherr
            if datatype=="real" or datatype=="real&imag":
                vistable["real"] /= vistable["sigma"]
            if datatype=="imag" or datatype=="real&imag":
                vistable["imag"] /= vistable["sigma"]
            errorbar = False

        #Plotting data
        if datatype=="amp":
            _radplot_amp(vistable, uvunit, errorbar, ls, marker, **plotargs)
        if datatype=="phase":
            _radplot_phase(vistable, uvunit, errorbar, ls, marker, **plotargs)
        if datatype=="amp&phase":
            _radplot_ampph(vistable, uvunit, errorbar, ls, marker, **plotargs)
        if datatype=="real":
            _radplot_real(vistable, uvunit, errorbar, ls, marker, **plotargs)
        if datatype=="imag":
            _radplot_imag(vistable, uvunit, errorbar, ls, marker, **plotargs)
        if datatype=="real&imag":
            _radplot_fcv(vistable, uvunit, errorbar, ls, marker, **plotargs)

    def vplot(self, station=None, datatype="amp&phase", timescale="utc", normerror=False,
              errorbar=True, ls="none", marker=".", **plotargs):
        '''
        Plot visibility amplitudes as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbar().

        Args:
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.

          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot().

            If you plot model closure phases (i.e. model is not None),
            it will plot without errobars regardless of this parameter.

          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # make dictionary of stations
        st1table = self.drop_duplicates(subset='st1')
        st2table = self.drop_duplicates(subset='st2')
        stdict = dict(zip(st1table["st1"], st1table["st1name"]))
        stdict.update(dict(zip(st2table["st2"], st2table["st2name"])))

        if station is None:
            st1 = 1
            st1name = stdict[1]
        else:
            st1 = int(station)
            st1name = stdict[st1]

        # edit timescale
        if timescale=="gsthour" or timescale=="gst":
            timescale = "gsthour"
            #
            ttable = self.drop_duplicates(subset=timescale)
            time = np.array(ttable[timescale])
            tmin = time[0]
            tmax = time[-1]
            #
            if tmin > tmax:
                self.loc[self.gsthour<=tmax, "gsthour"] += 24.

        # setting limits of min and max
        ttable = self.drop_duplicates(subset=timescale)
        if timescale=="utc":
            ttable[timescale] = pd.to_datetime(ttable[timescale])
        if timescale=="gsthour":
            ttable[timescale] = pd.to_datetime(ttable[timescale], unit="h")
        #
        time = np.array(ttable[timescale])
        tmin = time[0]
        tmax = time[-1]

        # setting indices
        tmptable = self.set_index(timescale)

        # search data of baseline
        plttable = tmptable.query("st1 == @st1")

        # add real and imaginary part of full-comp. visibilities
        if datatype=="real" or datatype=="imag" or datatype=="real&imag":
            amp = np.float64(plttable["amp"])
            phase = np.radians(np.float64(plttable["phase"]))
            #
            plttable["real"] = amp * np.cos(phase)
            plttable["imag"] = amp * np.sin(phase)

        # normalized by error
        if normerror:
            if datatype=="amp" or datatype=="amp&phase":
                plttable["amp"] /= plttable["sigma"]
            if datatype=="phase" or datatype=="amp&phase":
                pherr = np.rad2deg(plttable["sigma"] / plttable["amp"])
                plttable["phase"] /= pherr
            if datatype=="real" or datatype=="real&imag":
                plttable["real"] /= plttable["sigma"]
            if datatype=="imag" or datatype=="real&imag":
                plttable["imag"] /= plttable["sigma"]
            errorbar = False

        # plotting data
        if datatype=="amp":
            _vplot_amp(plttable, st1name, stdict, timescale, tmin, tmax,
                       errorbar, ls, marker, **plotargs)
        if datatype=="phase":
            _vplot_phase(plttable, st1name, stdict, timescale, tmin, tmax,
                         errorbar, ls, marker, **plotargs)
        if datatype=="amp&phase":
            _vplot_ampph(plttable, st1name, stdict, timescale, tmin, tmax,
                         errorbar, ls, marker, **plotargs)
        if datatype=="real":
            _vplot_real(plttable, st1name, stdict, timescale, tmin, tmax,
                        errorbar, ls, marker, **plotargs)
        if datatype=="imag":
            _vplot_imag(plttable, st1name, stdict, timescale, tmin, tmax,
                        errorbar, ls, marker, **plotargs)
        if datatype=="real&imag":
            _vplot_fcv(plttable, st1name, stdict, timescale, tmin, tmax,
                       errorbar, ls, marker, **plotargs)


class VisSeries(UVSeries):

    @property
    def _constructor(self):
        return VisSeries

    @property
    def _constructor_expanddim(self):
        return VisTable


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def read_vistable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.VisTable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None):
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]",
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.VisTable object
    '''
    table = VisTable(pd.read_csv(filename, **args))
    if "utc" in table.columns:
        table["utc"] = at.Time(table["utc"].values.tolist()).datetime
    table.set_uvunit()
    return table

#-------------------------------------------------------------------------
# Subfunctions for VisTable
#-------------------------------------------------------------------------
def getblid(st1, st2, Nst):
    '''
    This function is a subfunction for uvdata.VisTable.
    It calculates an id number of the baseline from a given set of
    station numbers and the total number of stations.

    Arguments:
      st1 (int): the first station ID number
      st2 (int): the second station ID number
      Nst (int): the total number of stations

    Return (int): the baseline ID number
    '''
    stmin = np.min([st1, st2])
    stmax = np.max([st1, st2])

    return stmin * Nst - stmin * (stmin + 1) // 2 + stmax - stmin - 1


def check_nontrivial(baselines, redundant):
    if redundant is None:
        return True

    flag = False
    for baseline in baselines:
        flag |= baseline in redundant
    return not flag


def check_baselines(baselines, blset):
    flag = True
    for baseline in baselines:
        flag &= baseline in blset
    return flag


def calc_matrix_ca(matrix, blid1, blid2, blid3, blid4, Nbl):
    # New row
    row = np.zeros(Nbl)
    row[blid1] = 1
    row[blid2] = 1
    row[blid3] = -1
    row[blid4] = -1

    # add New row
    if matrix is None:
        newmatrix = np.asarray([row])
    else:
        nrow,ncol = matrix.shape
        newmatrix = np.append(matrix, row).reshape(nrow+1, Nbl)

    # calc rank of the new matrix
    newrank = np.linalg.matrix_rank(newmatrix)
    return newrank, newmatrix


def calc_matrix_bs(matrix, blid1, blid2, blid3, Nbl):
    # New row
    row = np.zeros(Nbl)
    row[blid1] = 1
    row[blid2] = 1
    row[blid3] = -1

    # add New row
    if matrix is None:
        newmatrix = np.asarray([row])
    else:
        nrow,ncol = matrix.shape
        newmatrix = np.append(matrix, row).reshape(nrow+1, Nbl)

    # calc rank of the new matrix
    newrank = np.linalg.matrix_rank(newmatrix)
    return newrank, newmatrix


def calc_dbeam(fitsdata, vistable, errweight=0, ftsign=+1):
    '''
    Calculate an array and total flux of dirty beam from the input visibility data

    Args:
      fitsdata:
        input imdata.IMFITS object
      vistable:
        input visibility data
      errweight (float):
        index for errer weighting
      ftsign (integer):
        a sign for fourier matrix
    '''
    # create output fits
    outfitsdata = copy.deepcopy(fitsdata)

    # read uv information
    M = len(vistable)
    U = np.float64(vistable["u"])
    V = np.float64(vistable["v"])

    # create visibilies and error weighting
    Vis_point = np.ones(len(vistable), dtype=np.complex128)
    if errweight != 0:
        sigma = np.float64(vistable["sigma"])
        weight = np.power(sigma, errweight)
        Vis_point *= weight / np.sum(weight)

    # create matrix of X and Y
    Npix = outfitsdata.header["nx"] * outfitsdata.header["ny"]
    X, Y = outfitsdata.get_xygrid(angunit="deg", twodim=True)
    X = np.radians(X)
    Y = np.radians(Y)
    X = X.reshape(Npix)
    Y = Y.reshape(Npix)

    # create matrix of A
    if ftsign > 0:
        factor = 2 * np.pi
    elif ftsign < 0:
        factor = -2 * np.pi
    A = linalg.blas.dger(factor, X, U)
    A += linalg.blas.dger(factor, Y, V)
    A = np.exp(1j * A) / M

    # calculate synthesized beam
    dbeam = np.real(A.dot(Vis_point))
    dbtotalflux = np.sum(dbeam)
    dbeam /= dbtotalflux

    # save as fitsdata
    dbeam = dbeam.reshape((outfitsdata.header["ny"], outfitsdata.header["nx"]))
    for idxs in xrange(outfitsdata.header["ns"]):
        for idxf in xrange(outfitsdata.header["nf"]):
            outfitsdata.data[idxs, idxf] = dbeam[:]

    outfitsdata.update_fits()
    return outfitsdata, dbtotalflux


def calc_bparms(vistable):
    '''
    Infer beam parameters (major size, minor size, position angle)

    Args:
      vistable: input visibility data
    '''
    # read uv information
    U = np.float64(vistable["u"])
    V = np.float64(vistable["v"])

    # calculate minor size of the beam
    uvdist = np.sqrt(U * U + V * V)
    maxuvdist = np.max(uvdist)
    mina = np.rad2deg(1 / maxuvdist) * 0.6

    # calculate PA
    index = np.argmax(uvdist)
    angle = np.rad2deg(np.arctan2(U[index], V[index]))

    # rotate uv coverage for calculating major size
    PA = angle + 90
    cosPA = np.cos(np.radians(PA))
    sinPA = np.sin(np.radians(PA))
    newU = U * cosPA - V * sinPA
    newV = U * sinPA + V * cosPA

    # calculate major size of the beam
    maxV = np.max(np.abs(newV))
    maja = np.rad2deg(1 / maxV) * 0.6

    return maja, mina, PA


def gauss_func(X, Y, maja, mina, PA, x0=0., y0=0., scale=1.):
    '''
    Calculate 2-D gauss function

    Args:
      X: 2-D array of x-axis
      Y: 2-D array of y-axis
      maja (float): major size of the gauss
      mina (float): minor size
      PA (float): position angle
      x0 (float): value of x-position at the center of the gauss
      y0 (float): value of y-position at the center of the gauss
      scale (float): scaling factor
    '''
    # scaling
    maja *= scale
    mina *= scale

    # calculate gauss function
    cosPA = np.cos(np.radians(PA))
    sinPA = np.sin(np.radians(PA))
    L = ((X * sinPA + Y * cosPA)**2) / (maja**2) + \
        ((X * cosPA - Y * sinPA)**2) / (mina**2)
    return np.exp(-L * 4 * np.log(2))


def fit_chisq(parms, X, Y, dbeam):
    '''
    Calculate residuals of two 2-D array

    Args:
      parms: information of clean beam
      X: 2-D array of x-axis
      Y: 2-D array of y-axis
      dbeam: an array of dirty beam
    '''
    # get parameters of clean beam
    (maja, mina, angle) = parms

    # calculate clean beam and residuals
    cbeam = gauss_func(X, Y, maja, mina, angle)
    cbeam /= np.max(cbeam)
    if cbeam.size == dbeam.size:
        return (dbeam - cbeam).reshape(dbeam.size)
    else:
        print("not equal the size of two beam array")


def _radplot_amp(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_unitlabel(uvunit)

    # Plotting data
    if errorbar:
        plt.errorbar(vistable["uvdist"] * conv, vistable["amp"], vistable["sigma"],
                     ls=ls, marker=marker, **plotargs)
    else:
        plt.plot(vistable["uvdist"] * conv, vistable["amp"],
                     ls=ls, marker=marker, **plotargs)
    # Label (Plot)
    plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
    plt.ylabel(r"Visibility Amplitude (Jy)")
    plt.xlim(0.,)
    plt.ylim(0.,)


def _radplot_phase(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_unitlabel(uvunit)

    # Plotting data
    if errorbar:
        pherr = vistable["sigma"] / vistable["sigma"]
        plt.errorbar(vistable["uvdist"] * conv, vistable["phase"], pherr,
                     ls=ls, marker=marker, **plotargs)
    else:
        plt.plot(vistable["uvdist"] * conv, vistable["phase"],
                     ls=ls, marker=marker, **plotargs)
    # Label (Plot)
    plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
    plt.ylabel(r"Visibility Phase ($^\circ$)")
    plt.xlim(0.,)
    plt.ylim(-180., 180.)


def _radplot_ampph(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_unitlabel(uvunit)

    # Plotting data
    if errorbar:
        plt.errorbar(vistable["uvdist"] * conv, vistable["amp"], vistable["sigma"],
                     ls=ls, marker=marker, **plotargs)
    else:
        plt.plot(vistable["uvdist"] * conv, vistable["amp"],
                     ls=ls, marker=marker, **plotargs)
    # Label (Plot)
    plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
    plt.ylabel(r"Visibility Amplitude (Jy)")
    plt.xlim(0.,)
    plt.ylim(0.,)


def _radplot_real(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_unitlabel(uvunit)

    data  = np.float64(vistable["real"])
    ymin = np.min(data)
    ymax = np.max(data)

    # Plotting data
    if errorbar:
        plt.errorbar(vistable["uvdist"] * conv, vistable["real"], vistable["sigma"],
                     ls=ls, marker=marker, **plotargs)
    else:
        plt.plot(vistable["uvdist"] * conv, vistable["real"],
                     ls=ls, marker=marker, **plotargs)
    plt.xlim(0.,)
    ymin = np.min(vistable["real"])
    if ymin>=0.:
        plt.ylim(0.,)

    # Label (Plot)
    plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
    plt.ylabel(r"Real Part of Visibilities (Jy)")


def _radplot_imag(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_unitlabel(uvunit)

    data  = np.float64(vistable["imag"])
    ymin = np.min(data)
    ymax = np.max(data)

    # Plotting data
    if errorbar:
        plt.errorbar(vistable["uvdist"] * conv, vistable["imag"], vistable["sigma"],
                     ls=ls, marker=marker, **plotargs)
    else:
        plt.plot(vistable["uvdist"] * conv, vistable["imag"],
                     ls=ls, marker=marker, **plotargs)
    #
    plt.xlim(0.,)
    ymin = np.min(vistable["imag"])
    if ymin>=0.:
        plt.ylim(0.,)
    # Label (Plot)
    plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
    plt.ylabel(r"Real Part of Visibilities (Jy)")


def _radplot_fcv(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_unitlabel(uvunit)

    data  = np.float64(vistable["real"])
    ymin = np.min(data)
    ymax = np.max(data)

    # Plotting data
    if errorbar:
        plt.errorbar(vistable["uvdist"] * conv, vistable["real"], vistable["sigma"],
                     ls=ls, marker=marker, **plotargs)
    else:
        plt.plot(vistable["uvdist"] * conv, vistable["real"],
                     ls=ls, marker=marker, **plotargs)
    #
    plt.xlim(0.,)
    ymin = np.min(vistable["real"])
    if ymin>=0.:
        plt.ylim(0.,)
    # Label (Plot)
    plt.xlabel(r"Baseline Length (%s)" % (unitlabel))
    plt.ylabel(r"Real Part of Visibilities (Jy)")


def _vplot_amp(vistable, st1name, stdict, timescale, tmin, tmax, errorbar,
               ls, marker, **plotargs):
    '''
    Plot visibility amplitudes as a function of baseline lengths
    on the current axes. This method uses matplotlib.pyplot.plot() or
    matplotlib.pyplot.errorbar().

    Args:
      uvunit (str, default = None):
    The unit of the baseline length. if uvunit is None, it will use
        self.uvunit.

      errorbar (boolean, default = True):
        If errorbar is True, it will plot data with errorbars using
        matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
        errorbars using matplotlib.pyplot.plot().

        If you plot model closure phases (i.e. model is not None),
        it will plot without errobars regardless of this parameter.

      **plotargs:
        You can set parameters of matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbars().
        Defaults are {'ls': "none", 'marker': "."}.
    '''
    # get antenna
    st2 = np.sort(np.int32((vistable.drop_duplicates(subset='st2'))['st2']))
    Nant = len(st2)

    # convert timescale to pd.to_datetime
    if timescale=="utc":
        vistable.index = pd.to_datetime(vistable.index)
    if timescale=="gsthour":
        vistable.index = pd.to_datetime(vistable.index, unit="h")

    # plotting data
    fig, axs = plt.subplots(nrows=Nant, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    for iant in range(Nant):
        ax = axs[iant]
        plt.sca(ax)

        plttable = vistable.query("st2 == @st2[@iant]")
        if errorbar:
            plt.errorbar(plttable.index, plttable["amp"], plttable["sigma"],
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(plttable.index, plttable["amp"],
                     ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip(), horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        plt.ylim(0.,)
        #
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)

    # major ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    #
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    # minor ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 2)))
    #
    if timescale=="utc":
        plt.xlabel(r"Universal Time (UTC)")
    elif timescale=="gsthour":
        plt.xlabel(r"Greenwich Sidereal Time (GST)")


def _vplot_phase(vistable, st1name, stdict, timescale, tmin, tmax, errorbar,
                 ls, marker, **plotargs):
    '''
    Plot visibility amplitudes as a function of baseline lengths
    on the current axes. This method uses matplotlib.pyplot.plot() or
    matplotlib.pyplot.errorbar().

    Args:
      uvunit (str, default = None):
        The unit of the baseline length. if uvunit is None, it will use
        self.uvunit.

      errorbar (boolean, default = True):
        If errorbar is True, it will plot data with errorbars using
        matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
        errorbars using matplotlib.pyplot.plot().

        If you plot model closure phases (i.e. model is not None),
        it will plot without errobars regardless of this parameter.

      **plotargs:
        You can set parameters of matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbars().
        Defaults are {'ls': "none", 'marker': "."}.
    '''
    # get antenna
    st2 = np.sort(np.int32((vistable.drop_duplicates(subset='st2'))['st2']))
    Nant = len(st2)

    # convert timescale to pd.to_datetime
    if timescale=="utc":
        vistable.index = pd.to_datetime(vistable.index)
    if timescale=="gsthour":
        vistable.index = pd.to_datetime(vistable.index, unit="h")

    # plotting data
    fig, axs = plt.subplots(nrows=Nant, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    for iant in range(Nant):
        ax = axs[iant]
        plt.sca(ax)

        plttable = vistable.query("st2 == @st2[@iant]")
        if errorbar:
            pherr = np.rad2deg(plttable["sigma"] / plttable["amp"])
            plt.errorbar(plttable.index, plttable["phase"], pherr, ls=ls, marker=marker,
                         **plotargs)
        else:
            plt.plot(plttable.index, plttable["phase"], ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip(), horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        # major ticks
        ax.yaxis.set_major_locator(ticker.FixedLocator([-90, 0, 90]))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        plt.ylim(-180., 180.)

    # major ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    #
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    # minor ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 2)))
    #
    if timescale=="utc":
        plt.xlabel(r"Universal Time (UTC)")
    elif timescale=="gsthour":
        plt.xlabel(r"Greenwich Sidereal Time (GST)")


def _vplot_ampph(vistable, st1name, stdict, timescale, tmin, tmax, errorbar,
                 ls, marker, **plotargs):
    '''
    Plot visibility amplitudes as a function of baseline lengths
    on the current axes. This method uses matplotlib.pyplot.plot() or
    matplotlib.pyplot.errorbar().

    Args:
      uvunit (str, default = None):
        The unit of the baseline length. if uvunit is None, it will use
        self.uvunit.

      errorbar (boolean, default = True):
        If errorbar is True, it will plot data with errorbars using
        matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
        errorbars using matplotlib.pyplot.plot().

        If you plot model closure phases (i.e. model is not None),
        it will plot without errobars regardless of this parameter.

      **plotargs:
        You can set parameters of matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbars().
        Defaults are {'ls': "none", 'marker': "."}.
    '''
    # get antenna
    st2 = np.sort(np.int32((vistable.drop_duplicates(subset='st2'))['st2']))
    Nant = len(st2)

    # convert timescale to pd.to_datetime
    if timescale=="utc":
        vistable.index = pd.to_datetime(vistable.index)
    if timescale=="gsthour":
        vistable.index = pd.to_datetime(vistable.index, unit="h")

    # plotting data
    fig, axs = plt.subplots(nrows=Nant * 2, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    for iant in range(Nant):
        # amp
        ax = axs[iant*2]
        plt.sca(ax)

        plttable = vistable.query("st2 == @st2[@iant]")
        if errorbar:
            plt.errorbar(plttable.index, plttable["amp"], plttable["sigma"],
                         ls=ls, marker=marker, **plotargs)
        else:
            plt.plot(plttable.index, plttable["amp"],
                     ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip(), horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        plt.ylim(0.,)
        #
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)

        # phase
        ax = axs[iant*2+1]
        plt.sca(ax)

        if errorbar:
            pherr = np.rad2deg(plttable["sigma"] / plttable["amp"])
            plt.errorbar(plttable.index, plttable["phase"], pherr, ls=ls, marker=marker,
                         **plotargs)
        else:
            plt.plot(plttable.index, plttable["phase"], ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, "phase", horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        # major ticks
        ax.yaxis.set_major_locator(ticker.FixedLocator([-90, 0, 90]))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        plt.ylim(-180., 180.)

    # major ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    #
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    # minor ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 2)))
    #
    if timescale=="utc":
        plt.xlabel(r"Universal Time (UTC)")
    elif timescale=="gsthour":
        plt.xlabel(r"Greenwich Sidereal Time (GST)")


def _vplot_real(vistable, st1name, stdict, timescale, tmin, tmax, errorbar,
                ls, marker, **plotargs):
    '''
    Plot visibility amplitudes as a function of baseline lengths
    on the current axes. This method uses matplotlib.pyplot.plot() or
    matplotlib.pyplot.errorbar().

    Args:
      uvunit (str, default = None):
        The unit of the baseline length. if uvunit is None, it will use
        self.uvunit.

      errorbar (boolean, default = True):
        If errorbar is True, it will plot data with errorbars using
        matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
        errorbars using matplotlib.pyplot.plot().

        If you plot model closure phases (i.e. model is not None),
        it will plot without errobars regardless of this parameter.

      **plotargs:
        You can set parameters of matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbars().
        Defaults are {'ls': "none", 'marker': "."}.
    '''
    # add real part of full-comp. visibilities
    amp = np.float64(vistable["amp"])
    phase = np.radians(np.float64(vistable["phase"]))
    #
    vistable["real"] = amp * np.cos(phase)

    # get antenna
    st2 = np.sort(np.int32((vistable.drop_duplicates(subset='st2'))['st2']))
    Nant = len(st2)

    # convert timescale to pd.to_datetime
    if timescale=="utc":
        vistable.index = pd.to_datetime(vistable.index)
    if timescale=="gsthour":
        vistable.index = pd.to_datetime(vistable.index, unit="h")

    # plotting data
    fig, axs = plt.subplots(nrows=Nant, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    for iant in range(Nant):
        ax = axs[iant]
        plt.sca(ax)

        plttable = vistable.query("st2 == @st2[@iant]")
        #
        if errorbar:
            plt.errorbar(plttable.index, plttable["real"], plttable["sigma"], ls=ls,
                         marker=marker, **plotargs)
        else:
            plt.plot(plttable.index, plttable["real"], ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip(), horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        #
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        ymin = np.min(plttable["real"])
        ymax = np.max(plttable["real"])
        if ymin>=0.:
            plt.ylim(0.,)
        if ymax<=0.:
            plt.ylim(ymax=0.)

    # major ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    #
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    # minor ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 2)))
    #
    if timescale=="utc":
        plt.xlabel(r"Universal Time (UTC)")
    elif timescale=="gsthour":
        plt.xlabel(r"Greenwich Sidereal Time (GST)")


def _vplot_imag(vistable, st1name, stdict, timescale, tmin, tmax, errorbar,
                ls, marker, **plotargs):
    '''
    Plot visibility amplitudes as a function of baseline lengths
    on the current axes. This method uses matplotlib.pyplot.plot() or
    matplotlib.pyplot.errorbar().

    Args:
      uvunit (str, default = None):
        The unit of the baseline length. if uvunit is None, it will use
        self.uvunit.

      errorbar (boolean, default = True):
        If errorbar is True, it will plot data with errorbars using
        matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
        errorbars using matplotlib.pyplot.plot().

        If you plot model closure phases (i.e. model is not None),
        it will plot without errobars regardless of this parameter.

      **plotargs:
        You can set parameters of matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbars().
        Defaults are {'ls': "none", 'marker': "."}.
    '''
    # add real part of full-comp. visibilities
    amp = np.float64(vistable["amp"])
    phase = np.radians(np.float64(vistable["phase"]))
    #
    vistable["imag"] = amp * np.sin(phase)

    # get antenna
    st2 = np.sort(np.int32((vistable.drop_duplicates(subset='st2'))['st2']))
    Nant = len(st2)

    # convert timescale to pd.to_datetime
    if timescale=="utc":
        vistable.index = pd.to_datetime(vistable.index)
    if timescale=="gsthour":
        vistable.index = pd.to_datetime(vistable.index, unit="h")

    # plotting data
    fig, axs = plt.subplots(nrows=Nant, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    for iant in range(Nant):
        ax = axs[iant]
        plt.sca(ax)

        plttable = vistable.query("st2 == @st2[@iant]")
        #
        if errorbar:
            plt.errorbar(plttable.index, plttable["imag"], plttable["sigma"], ls=ls,
                         marker=marker, **plotargs)
        else:
            plt.plot(plttable.index, plttable["imag"], ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip(), horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        #
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        ymin = np.min(plttable["imag"])
        ymax = np.max(plttable["imag"])
        if ymin>=0.:
            plt.ylim(0.,)
        if ymax<=0.:
            plt.ylim(ymax=0.)

    # major ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    #
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    # minor ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 2)))
    #
    if timescale=="utc":
        plt.xlabel(r"Universal Time (UTC)")
    elif timescale=="gsthour":
        plt.xlabel(r"Greenwich Sidereal Time (GST)")


def _vplot_fcv(vistable, st1name, stdict, timescale, tmin, tmax, errorbar,
               ls, marker, **plotargs):
    '''
    Plot visibility amplitudes as a function of baseline lengths
    on the current axes. This method uses matplotlib.pyplot.plot() or
    matplotlib.pyplot.errorbar().

    Args:
      uvunit (str, default = None):
        The unit of the baseline length. if uvunit is None, it will use
        self.uvunit.

      errorbar (boolean, default = True):
        If errorbar is True, it will plot data with errorbars using
        matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
        errorbars using matplotlib.pyplot.plot().

        If you plot model closure phases (i.e. model is not None),
        it will plot without errobars regardless of this parameter.

      **plotargs:
        You can set parameters of matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbars().
        Defaults are {'ls': "none", 'marker': "."}.
    '''
    # add real part of full-comp. visibilities
    amp = np.float64(vistable["amp"])
    phase = np.radians(np.float64(vistable["phase"]))
    #
    vistable["real"] = amp * np.cos(phase)
    vistable["imag"] = amp * np.sin(phase)

    # get antenna
    st2 = np.sort(np.int32((vistable.drop_duplicates(subset='st2'))['st2']))
    Nant = len(st2)

    # convert timescale to pd.to_datetime
    if timescale=="utc":
        vistable.index = pd.to_datetime(vistable.index)
    if timescale=="gsthour":
        vistable.index = pd.to_datetime(vistable.index, unit="h")

    # plotting data
    fig, axs = plt.subplots(nrows=Nant*2, ncols=1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    for iant in range(Nant):
        # real part
        ax = axs[iant*2]
        plt.sca(ax)

        plttable = vistable.query("st2 == @st2[@iant]")
        #
        if errorbar:
            plt.errorbar(plttable.index, plttable["real"], plttable["sigma"], ls=ls,
                         marker=marker, **plotargs)
        else:
            plt.plot(plttable.index, plttable["real"], ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip(), horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        #
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        ymin = np.min(plttable["real"])
        ymax = np.max(plttable["real"])
        if ymin>=0.:
            plt.ylim(0.,)
        if ymax<=0.:
            plt.ylim(ymax=0.)

        # imaginary part
        ax = axs[iant*2+1]
        plt.sca(ax)

        if errorbar:
            plt.errorbar(plttable.index, plttable["imag"], plttable["sigma"], ls=ls,
                         marker=marker, **plotargs)
        else:
            plt.plot(plttable.index, plttable["imag"], ls=ls, marker=marker, **plotargs)
        #
        plt.text(0.97, 0.9, "imag part", horizontalalignment='right',
                 verticalalignment='top', transform=ax.transAxes, fontsize=8,
                 color="black")
        plt.xlim(tmin, tmax)
        #
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        ymin = np.min(plttable["imag"])
        ymax = np.max(plttable["imag"])
        if ymin>=0.:
            plt.ylim(0.,)
        if ymax<=0.:
            plt.ylim(ymax=0.)

    # major ticks
    ax.xaxis.set_major_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    #
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    # minor ticks
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 2)))
    #
    if timescale=="utc":
        plt.xlabel(r"Universal Time (UTC)")
    elif timescale=="gsthour":
        plt.xlabel(r"Greenwich Sidereal Time (GST)")
