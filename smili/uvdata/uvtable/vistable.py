#!/usr/bin/env python
# -*- coding: utf-8 -*-




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
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter

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
from ... import imdata, fortlib,util

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
    debiased=False

    @property
    def _constructor(self):
        return VisTable

    @property
    def _constructor_sliced(self):
        return VisSeries

    def set_uvunit(self, uvunit=None):
        '''
        Set or guess uvunit.

        Args:
            uvunit (str, default=None)
                The unit for spacial frequencies.
                Availables are lambda, klambda, mlambda and glambda.
                If not specified, uvunit will be guessed.
        '''
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

    def debias_amp(self):
        '''
        De-biasing amplitudes.
        '''
        if self.debiased:
            print("[WARNING] Amplitudes in this table were already debiased.")

        outdata = self.copy()
        outdata.debiased = True

        ratio = np.square(outdata.amp.values/outdata.sigma.values)
        idx1 = ratio>=2
        idx2 = ratio<2
        outdata.loc[idx1, "amp"] = np.sqrt(ratio[np.where(idx1)]-1)
        outdata.loc[idx2, "amp"] = np.sqrt(ratio[np.where(idx2)])
        outdata.loc[:, "amp"] *= outdata.sigma
        return outdata

    def recalc_uvdist(self):
        '''
        Re-calculate the baseline length from self["u"] and self["v"].
        '''
        self["uvdist"] = np.sqrt(self["u"] * self["u"] + self["v"] * self["v"])

    def rotate(self, dPA, deg=True):
        '''
        Rotate uv-coordinates.

        Args:
            dPA (float):
                Rotation angle.
            deg (boolean, default=True):
                The unit of dPA. If True, it will be degree. Otherwise,
                it will be radian
        Returns:
            uvdata.VisTable object
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

    def add_error(self, error, quadrature=True):
        '''
        Increase errors by a specified value

        Args:
            error (float or array like):
                errors to be added.
            quadrature (boolean; default=True):
                if True, specified errors will be added to sigma in quadrature.
                Otherwise, specified errors will be simply added.
        '''
        outtable = copy.deepcopy(self)
        if quadrature:
            outtable["sigma"] = np.sqrt(outtable["sigma"]**2 + error**2)
        else:
            outtable["sigma"] += error
        return outtable

    def station_list(self, id=False):
        '''
        Return list of stations. If id=True, return list of station IDs.
        '''
        if id:
            return np.unique([self["st1"],self["st2"]]).tolist()
        else:
            return np.unique([self["st1name"],self["st2name"]]).tolist()

    def station_dic(self, id2name=True):
        '''
        Return dictionary of stations. If id2name=True, return a dictionary
        whose key is the station ID number and value is the station name.
        Otherwise return a dictionary whose key is the name and value is ID.
        '''
        st1table = self.drop_duplicates(subset='st1')
        st2table = self.drop_duplicates(subset='st2')
        if id2name:
            outdict = dict(list(zip(st1table.st1.values, st1table.st1name.values)))
            outdict.update(dict(list(zip(st2table.st2.values, st2table.st2name.values))))
        else:
            outdict = dict(list(zip(st1table.st1name.values,st1table.st1.values)))
            outdict.update(dict(list(zip(st2table.st2name.values,st2table.st2.values))))
        return outdict

    def baseline_list(self, id=False):
        '''
        Return the list of baselines. If id=False, then the names of stations
        will be returned. Otherwise, the ID numbers of stations will be returned.
        '''
        if id:
            table = self.drop_duplicates(subset=['st1','st2'])
            return list(zip(table.st1.values,table.st2.values))
        else:
            table = self.drop_duplicates(subset=['st1name','st2name'])
            return list(zip(table.st1name.values,table.st2name.values))

    def comp(self):
        '''
        Return full complex visibilities
        '''
        return self["amp"]*np.exp(1j*np.deg2rad(self["phase"]))

    def real(self):
        '''
        Return the real part of full complex visibilities
        '''
        return self["amp"]*np.cos(np.deg2rad(self["phase"]))

    def imag(self):
        '''
        Return the imag part of full complex visibilities
        '''
        return self["amp"]*np.sin(np.deg2rad(self["phase"]))

    def sigma_phase(self, deg=True):
        '''
        Return the phase error estimator using a high SNR limit formula

        Args:
            deg (boolean; default=True):
                IF true, returns values in degree. Otherwide, values will be
                returned in radian
        '''
        if deg:
            return np.rad2deg(self["sigma"]/self["amp"])
        else:
            return self["sigma"]/self["amp"]

    def snr(self):
        '''
        Return the SNR estimator
        '''
        return self["amp"]/self["sigma"]

    def eval_image(self, imfits, mask=None, amptable=False, istokes=0, ifreq=0):
        #uvdata.VisTable object (storing model full complex visibility
        if(isinstance(imfits,imdata.IMFITS) or isinstance(imfits,imdata.MOVIE)):
            model = self._call_fftlib(imfits=imfits,mask=mask,amptable=amptable,
                                    istokes=istokes, ifreq=ifreq)
        else:
            print("[Error] imfits is not IMFITS nor MOVIE object")
            return -1

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
        if(isinstance(imfits,imdata.IMFITS) or isinstance(imfits,imdata.MOVIE)):
            model = self._call_fftlib(imfits=imfits,mask=mask,amptable=amptable,
                                    istokes=istokes, ifreq=ifreq)
        else:
            print("[Error] imfits is not IMFITS nor MOVIE object")
            return -1

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
        # calculate chisqared and reduced chisqred.
        if(isinstance(imfits,imdata.IMFITS) or isinstance(imfits,imdata.MOVIE)):
            model = self._call_fftlib(imfits=imfits,mask=mask,amptable=amptable,
                                    istokes=istokes, ifreq=ifreq)
        else:
            print("[Error] imfits is not IMFITS nor MOVIE object")
            return -1

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
        if(isinstance(imfits,imdata.IMFITS)):
            Iin = np.float64(imfits.data[istokes, ifreq])
            image = imfits
        elif(isinstance(imfits,imdata.MOVIE)):
            movie=imfits
            Nt    = movie.Nt
            # size of images
            Iin = []
            for im in movie.images:
                Iin.append(np.float64(im.data[istokes, ifreq]))
                image = movie.images[0]
        else:
            print(("[Error] imfits=%s is not IMFITS nor MOVIE object" % (imfits)))
            return -1

        Nx = image.header["nx"]
        Ny = image.header["ny"]
        Nyx = Nx * Ny

        # pixel coordinates
        x, y = image.get_xygrid(twodim=True, angunit="rad")
        xidx = np.arange(Nx) + 1
        yidx = np.arange(Ny) + 1
        xidx, yidx = np.meshgrid(xidx, yidx)
        Nxref = image.header["nxref"]
        Nyref = image.header["nyref"]
        dx_rad = np.deg2rad(image.header["dx"])
        dy_rad = np.deg2rad(image.header["dy"])

        # apply the imaging area
        if mask is None:
            print("Imaging Window: Not Specified. We calculate the image on all the pixels.")
            if(isinstance(imfits,imdata.IMFITS)):
                Iin = Iin.reshape(Nyx)
            else:
                for i in range(len(Iin)):
                    Iin[i] = Iin[i].reshape(Nyx)

            x = x.reshape(Nyx)
            y = y.reshape(Nyx)
            xidx = xidx.reshape(Nyx)
            yidx = yidx.reshape(Nyx)
        else:
            print("Imaging Window: Specified. Images will be calculated on specified pixels.")
            idx = np.where(mask)
            if(isinstance(imfits,imdata.IMFITS)):
                Iin = Iin[idx]
            else:
                for i in range(len(Iin)):
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
            if(isinstance(imfits,imdata.IMFITS)):
                u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
                        fcvtable=fcvtable, amptable=None, bstable=None, catable=None
                )

            else:
                u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = get_uvlist_loop(Nt=Nt,
                    fcvconcat=fcvtable, ampconcat=None, bsconcat=None, caconcat=None
                )

            # normalize u, v coordinates
            u *= 2*np.pi*dx_rad
            v *= 2*np.pi*dy_rad

            # run model_fcv
            if(isinstance(imfits,imdata.IMFITS)):
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
            else:
                # concatenate the initimages
                Iin = np.concatenate(Iin)
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
            if(isinstance(imfits,imdata.IMFITS)):
                u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
                    fcvtable=None, amptable=amptable, bstable=None, catable=None
                )
            else:
                u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = get_uvlist_loop(Nt=Nt,
                    fcvconcat=None, ampconcat=amptable, bsconcat=None, caconcat=None
                )

            # normalize u, v coordinates
            u *= 2*np.pi*dx_rad
            v *= 2*np.pi*dy_rad

            # run model_fcv
            if(isinstance(imfits,imdata.IMFITS)):
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
                        #
                        uvidxamp=np.int32(uvidxamp),
                        vamp=np.float64(vamp),
                        varamp=np.float64(varamp)
                        )
            else:
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
            modVre = geomodel.real(u,v)
            modVim = geomodel.imag(u,v)
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

    def make_bstable(self, redundant=None, dependent=False):
        '''
        Form bi-spectra from complex visibilities.

        Args:
            redandant (list of sets of redundant station IDs or names; default=None):
                If this is specified, non-trivial bispectra will be formed.
                This is useful for EHT-like array that have redandant stations in the same site.
                For example, if stations 1,2,3 and 4,5 are on the same sites, respectively, then
                you can specify redundant=[[1,2,3],[4,5]].
            dependent (boolean; default=False):
                If False, only independent dependent closure amplitudes will be formed.
                Otherwise, dependent closure amplitudes also will be formed as well.
        Returns:
            uvdata.BSTable object
        '''
        # Number of Stations
        Ndata = len(self["ch"])

        # make dictionary of stations
        stdict = self.station_dic(id2name=True)

        # Check redundant
        if redundant is not None:
            stdict2 = self.station_dic(id2name=False)
            for i in range(len(redundant)):
                stationids = []
                for stid in redundant[i]:
                    if isinstance(stid,str):
                        if stid in list(stdict2.keys()):
                            stationids.append(stdict2[stid])
                    else:
                        stationids.append(stid)
                redundant[i] = sorted(set(stationids))
            del stid, stdict2

        print("(1/5) Sort data")
        vistable = self.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
        vistable["bl"] = vistable["st1"] * 256 + vistable["st2"]

        print("(2/5) Tagging data")
        vistable["tag"] = np.zeros(Ndata, dtype=np.int64)
        for idata in tqdm.tqdm(list(range(1,Ndata))):
            flag = vistable.loc[idata, "utc"] == vistable.loc[idata-1, "utc"]
            flag&= vistable.loc[idata, "stokesid"] == vistable.loc[idata-1, "stokesid"]
            flag&= vistable.loc[idata, "ch"] == vistable.loc[idata-1, "ch"]
            if flag:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]
            else:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]+1
        Ntag = vistable["tag"].max() + 1
        print(("  Number of Tags: %d"%(Ntag)))

        print("(3/5) Checking Baseline Combinations")
        blsets = [] # baseline coverages
        cblsets = [] # combination to make closure amplitudes
        cstsets = [] # combination to make closure amplitudes
        blsetid=0
        bltype = np.zeros(Ntag, dtype=np.int64) # this is an array storing an ID
                                                # number of cl sets corresponding timetag.
        for itag in tqdm.tqdm(list(range(Ntag))):
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
            for stid1, stid2, stid3 in itertools.combinations(list(range(Nst)), 3):
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

                if rank>=Nbsmax and (not dependent):
                    break
                isnontrivial = check_nontrivial([[st1,st2], [st1,st3], [st2,st3]],redundant)
                isbaselines = check_baselines([bl12,bl23,bl13], blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_bs(matrix, blid12, blid23, blid13, Nblmax, dependent)
                    if newrank > rank or dependent:
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
        print(("  Detect %d combinations for Closure Phases"%(len(cblsets))))

        print("(4/5) Forming Closure Phases")
        keys = "utc,gsthour,freq,stokesid,ifid,chid,ch,"
        keys+= "st1,st2,st3,"
        keys+= "st1name,st2name,st3name,"
        keys+= "u12,v12,w12,u23,v23,w23,u31,v31,w31,"
        keys+= "amp,phase,sigma"
        keys = keys.split(",")
        outtab = {}
        for key in keys:
            outtab[key]=[]
        for itag in tqdm.tqdm(list(range(Ntag))):
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
            for iset in range(len(cblset)):
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
        for i in range(len(BSTable.bstable_columns)):
            column = BSTable.bstable_columns[i]
            outtab[column] = BSTable.bstable_types[i](outtab[column])
        return outtab

    def make_catable(self, redundant=None, dependent=False, debias=True):
        '''
        Form closure amplitudes from complex visibilities.

        Args:
            redandant (list of sets of redundant station IDs or names; default=None):
                If this is specified, non-trivial closure amplitudes will be formed.
                This is useful for EHT-like array that have redundant stations in the same site.
                For example, if stations 1,2,3 and 4,5 are on the same sites, respectively, then
                you can specify redundant=[[1,2,3],[4,5]].
            dependent (boolean; default=False):
                If False, only independent dependent closure amplitudes will be formed.
                Otherwise, dependent closure amplitudes also will be formed as well.
            debias (boolean; default=True):
                If True, visibility amplitudes will be debiased before closing
                closure amplitudes.
        Returns:
            uvdata.CATable object
        '''
        from scipy.special import expi
        # Number of Stations
        Ndata = len(self["ch"])

        # make dictionary of stations
        stdict = self.station_dic(id2name=True)

        # Check redundant
        if redundant is not None:
            stdict2 = self.station_dic(id2name=False)
            for i in range(len(redundant)):
                stationids = []
                for stid in range(len(redundant[i])):
                    if isinstance(stid,str):
                        stationids.append(stdict2[stid])
                    else:
                        stationids.append(stid)
                redundant[i] = sorted(set(stationids))
            del stid, stdict2

        print("(1/5) Sort data")
        vistable = self.sort_values(by=["utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
        vistable["bl"] = vistable["st1"] * 256 + vistable["st2"]

        print("(2/5) Tagging data")
        vistable["tag"] = np.zeros(Ndata, dtype=np.int64)
        for idata in tqdm.tqdm(list(range(1,Ndata))):
            flag = vistable.loc[idata, "utc"] == vistable.loc[idata-1, "utc"]
            flag&= vistable.loc[idata, "stokesid"] == vistable.loc[idata-1, "stokesid"]
            flag&= vistable.loc[idata, "ch"] == vistable.loc[idata-1, "ch"]
            if flag:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]
            else:
                vistable.loc[idata,"tag"] = vistable.loc[idata-1,"tag"]+1
        Ntag = vistable["tag"].max() + 1
        print(("  Number of Tags: %d"%(Ntag)))

        print("(3/5) Checking Baseline Combinations")
        blsets = [] # baseline coverages
        cblsets = [] # combination to make closure amplitudes
        cstsets = [] # combination to make closure amplitudes
        blsetid=0
        bltype = np.zeros(Ntag, dtype=np.int64) # this is an array storing an ID
                                                # number of cl sets corresponding timetag.
        for itag in tqdm.tqdm(list(range(Ntag))):
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
            for stid1, stid2, stid3, stid4 in itertools.combinations(list(range(Nst)), 4):
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
                if rank>=Ncamax and (not dependent):
                    break
                isnontrivial = check_nontrivial([[st1,st4], [st2,st3]],redundant)
                isbaselines = check_baselines([bl12,bl34,bl13,bl24], blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_ca(matrix, blid12, blid34, blid13, blid24, Nblmax, dependent)
                    if newrank > rank or dependent:
                        cblset.append([bl12,bl34,bl13,bl24])
                        cstset.append([st1,st2,st3,st4])
                        rank = newrank
                        matrix = newmatrix
                        Ncomb +=1

                # Combination 2: (V13 V24) / (V14 V23)
                #   This conmbination becomes trivial if
                #   site1 == site2 or site3 == site4.
                if rank>=Ncamax and (not dependent):
                    break
                isnontrivial = check_nontrivial([[st1,st2],[st3,st4]],redundant)
                isbaselines = check_baselines([bl13,bl24,bl14,bl23],blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_ca(matrix, blid13, blid24, blid14, blid23, Nblmax, dependent)
                    if newrank > rank or dependent:
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
                if rank>=Ncamax and (not dependent):
                    break
                isnontrivial = check_nontrivial([[st1,st3],[st2,st4]],redundant)
                isbaselines = check_baselines([bl12,bl34,bl14,bl23],blset)
                if isnontrivial and isbaselines:
                    newrank, newmatrix = calc_matrix_ca(matrix, blid12, blid34, blid14, blid23, Nblmax, dependent)
                    if newrank > rank or dependent:
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
        print(("  Detect %d combinations for Closure Amplitudes"%(len(cblsets))))

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
        for itag in tqdm.tqdm(list(range(Ntag))):
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
            for iset in range(len(cblset)):
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
        for i in range(len(CATable.catable_columns)):
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
        skip = np.asarray([False for i in range(Ntable)])

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
        for itable in range(Ntable):
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
        unitlabel = self.get_uvunitlabel(uvunit)

        # plotting
        plt.plot(self["u"] * conv, self["v"] * conv,
                 ls=ls, marker=marker, **plotargs)
        if conj:
            plotargs2 = copy.deepcopy(plotargs)
            plotargs2["label"] = ""
            plt.plot(-self["u"] * conv, -self["v"] * conv,
                     ls=ls, marker=marker, **plotargs2)
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

    def vplot(self,
            axis1="utc",
            axis2="amp",
            baseline=None,
            normerror1=False,
            normerror2=None,
            errorbar=True,
            gst_continuous=True,
            gst_wraphour=0.,
            time_maj_loc=mdates.HourLocator(),
            time_min_loc=mdates.MinuteLocator(byminute=np.arange(0,60,10)),
            time_maj_fmt='%H:%M',
            uvunit=None,
            ls="none",
            marker=".",
            label=None,
            **plotargs):
        '''
        Plot various type of data. Available types of data are
        ["utc","gst","amp","phase","sigma","real","imag","u","v","uvd","snr"]

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
        self = self.sort_values(by="utc").reset_index(drop=True)

        # Check if baseline is specified
        if baseline is None:
            pltdata = self
        else:
            stndict = self.station_dic(id2name=True)
            stidict = self.station_dic(id2name=False)
            # make dictionary of stations
            if isinstance(baseline[0], str):
                st1 = stidict[baseline[0]]
            else:
                st1 = int(baseline[0])
            if isinstance(baseline[1], str):
                st2 = stidict[baseline[1]]
            else:
                st2 = int(baseline[1])
            st1, st2 = sorted([st1,st2])
            st1name = stndict[st1]
            st2name = stndict[st2]
            pltdata = self.query("st1==@st1 & st2==@st2").reset_index(drop=True)
            del stndict, stidict
            if len(pltdata["st1"])==0:
                print("No data can be plotted.")
                return

        # Check label
        if label is None:
            if baseline is None:
                label=""
            else:
                label="%s - %s"%(st1name,st2name)

        # check
        if normerror2 is None:
            normerror2 = normerror1

        # Set Unit
        if uvunit is None:
            self.set_uvunit()
            uvunit = self.uvunit
        # Conversion Factor
        uvunitconv = self.uvunitconv(unit1="lambda", unit2=uvunit)
        # Label
        uvunitlabel = self.get_uvunitlabel(uvunit)

        # get data to be plotted
        axises = [axis1.lower(),axis2.lower()]
        normerrors = [normerror1, normerror2]
        useerrorbar=False
        errors=[]
        pltarrays = []
        axislabels = []
        deflims = []
        for i in range(2):
            axis = axises[i]
            normerror = normerrors[i]
            if   "utc" in axis:
                pltarrays.append(pltdata.utc.values)
                axislabels.append("Universal Time")
                deflims.append((None,None))
                errors.append(None)
            elif "gst" in axis:
                pltarrays.append(pltdata.get_gst_datetime(continuous=gst_continuous, wraphour=gst_wraphour))
                axislabels.append("Greenwich Sidereal Time")
                deflims.append((None,None))
                errors.append(None)
            elif "real" in axis:
                if not normerror:
                    pltarrays.append(pltdata.real())
                    axislabels.append("Real Part of Visibility (Jy)")
                    errors.append(pltdata.sigma.values)
                    if errorbar and (not useerrorbar):
                        useerrorbar=True
                else:
                    pltarrays.append(pltdata.real()/pltdata.sigma.values)
                    axislabels.append("Error-normalized Real Part")
                    errors.append(None)
                deflims.append((None,None))
            elif "imag" in axis:
                if not normerror:
                    pltarrays.append(pltdata.imag())
                    axislabels.append("Imag Part of Visibility (Jy)")
                    errors.append(pltdata.sigma.values)
                    if errorbar and (not useerrorbar):
                        useerrorbar=True
                else:
                    pltarrays.append(pltdata.imag()/pltdata.sigma.values)
                    axislabels.append("Error-normalized Imag Part")
                    errors.append(None)
                deflims.append((None,None))
            elif "amp" in axis:
                if not normerror:
                    pltarrays.append(pltdata.amp.values)
                    axislabels.append("Visibility Amplitude (Jy)")
                    errors.append(pltdata.sigma.values)
                    if errorbar and (not useerrorbar):
                        useerrorbar=True
                else:
                    pltarrays.append(pltdata.amp.values/pltdata.sigma.values)
                    axislabels.append("Error-normalized Amplitudes")
                    errors.append(None)
                deflims.append((0,None))
            elif "phase" in axis:
                if not normerror:
                    pltarrays.append(pltdata.phase.values)
                    axislabels.append("Visibility Phase (deg)")
                    deflims.append((-180,180))
                    errors.append(pltdata.sigma_phase().values)
                    if errorbar and (not useerrorbar):
                        useerrorbar=True
                else:
                    pltarrays.append(pltdata.phase.values/pltdata.sigma_phase().values)
                    axislabels.append("Error-normalized Phases")
                    deflims.append((None,None))
                    errors.append(None)
            elif "sigma" in axis:
                pltarrays.append(pltdata.sigma.values)
                axislabels.append("Error (Jy)")
                deflims.append((0,None))
                errors.append(None)
            elif "snr" in axis:
                pltarrays.append(pltdata.snr().values)
                axislabels.append("SNR")
                deflims.append((0,None))
                errors.append(None)
            elif axis=="u":
                pltarrays.append(pltdata.u.values*uvunitconv)
                axislabels.append("$u$ (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif axis=="v":
                pltarrays.append(pltdata.u.values*uvunitconv)
                axislabels.append("$v$ (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif "uvd" in axis:
                pltarrays.append(pltdata.u.values*uvunitconv)
                axislabels.append("Baseline Length (%s)"%(uvunitlabel))
                deflims.append((0,None))
                errors.append(None)
            else:
                raise ValueError("Invalid axis type: %s"%(axis))

        # plot
        ax = plt.gca()

        if useerrorbar:
            plt.errorbar(
                pltarrays[0],
                pltarrays[1],
                yerr=errors[1],
                xerr=errors[0],
                label=label,
                marker=marker,
                ls=ls,
                **plotargs
            )
        else:
            plt.plot(
                pltarrays[0],
                pltarrays[1],
                label=label,
                marker=marker,
                ls=ls,
                **plotargs
            )

        # set formatter if utc or gst will be plotted
        for i in range(2):
            axis = axises[i]
            # Set time
            if i==0:
                axaxis = ax.xaxis
            else:
                axaxis = ax.yaxis
            if "gst" in axis or "utc" in axis:
                axaxis.set_major_locator(time_maj_loc)
                axaxis.set_minor_locator(time_min_loc)
                axaxis.set_major_formatter(mdates.DateFormatter(time_maj_fmt))
            del axaxis

            # Set labels and limits
            if i==0:
                plt.xlabel(axislabels[0])
                plt.xlim(deflims[0])
            else:
                plt.ylabel(axislabels[1])
                plt.ylim(deflims[1])

    def tplot(self,
            axis1="utc",
            gst_continuous=True,
            gst_wraphour=0.,
            time_maj_loc=mdates.HourLocator(),
            time_min_loc=mdates.MinuteLocator(byminute=np.arange(0,60,10)),
            time_maj_fmt='%H:%M',
            ls="none",
            marker=".",
            label=None,
            **plotargs):
        '''
        Plot time-coverage of data. Available types for the xaxis is
        ["utc","gst"]

        Args:
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        plttable = copy.deepcopy(self)

        # Get GST
        if "gst" in axis1.lower():
            plttable = plttable.sort_values(by="utc").reset_index(drop=True)
            plttable["gst"] = plttable.get_gst_datetime(continuous=gst_continuous, wraphour=gst_wraphour)

        # Station List
        stations = plttable.station_list()[::-1]
        Nst = len(stations)

        # Plotting
        ax = plt.gca()
        for stname in stations:
            pltdata = plttable.query("st1name == @stname | st2name == @stname")
            Ndata = len(pltdata)

            # y value
            antid = np.ones(Ndata)
            antid[:] = stations.index(stname)

            # x value
            if "gst" in axis1.lower():
                axis1data = pltdata.gst.values
            else:
                axis1data = pltdata.utc.values

            plt.plot(axis1data, antid, ls=ls, marker=marker, label=label, **plotargs)

        # y-tickes
        ax.set_yticks(np.arange(Nst))
        ax.set_yticklabels(stations)
        plt.ylabel("Station")

        # x-tickes
        ax.xaxis.set_major_locator(time_maj_loc)
        ax.xaxis.set_minor_locator(time_min_loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_maj_fmt))
        if "gst" in axis1.lower():
            plt.xlabel("Greenwich Sidereal Time")
        else:
            plt.xlabel("Universal Time")

    def uvdensity(self, image, npix=2):
        '''
        Compute the density of uvpoints by convolving the uv-sample function by
        a circular Gaussian of which FWHM is npix/fov, where fov is given by
        fov = 2 * max(ra**2+dec**2) of the image.

        Args:
            image (imdata.IMFITS object): required to compute a fov
            npix (integer, default=2): FWHM size of the kernel in the unit of 1/fov.
        Returns:
            1-D numpy array contains density functions
        '''
        # compute a grid size of the image in uv-domain.
        x,y = image.get_xygrid(angunit="rad", twodim=True)
        r = np.sqrt(x*x + y*y).max()*2
        du = 1/r

        # density
        Ndata = len(self.amp.values)
        u = self.u.values/du
        v = self.v.values/du

        '''
        def _count(u, v, u0, v0, fwhm):
            # compute sigma
            sigma = fwhm/np.sqrt(8*np.log(2))
            gamma = -1/2./sigma**2

            # compute Gaussian
            sum = np.sum(np.exp(gamma*((u-u0)**2+(v-v0)**2)))
            sum+= np.sum(np.exp(gamma*((u+u0)**2+(v+v0)**2)))
            return sum
        '''
        def _count(u, v, u0, v0, npix):
            # compute sigma
            cnt = np.zeros(u.shape)
            cnt[np.where((u-u0)**2+(v-v0)**2<npix**2)] = 1
            cnt[np.where((u+u0)**2+(v+v0)**2<npix**2)] = 1
            #cnt[np.where(np.max(np.abs([u-u0,v-v0]))<=npix)] = 1
            #cnt[np.where(np.max(np.abs([u+u0,v+v0]))<=npix)] = 1

            # compute Gaussian
            return cnt.sum()

        npoints = [_count(u,v,u[i],v[i],npix) for i in range(Ndata)]
        return np.asarray(npoints)

    def uvweight(self, uniform=True, image=None, werror=0, npix=2):
        '''
        Compute relative weights on each data points. This function computes two types of weights.
        (A) density weightings (uniform or natural):
            weight \propto rho^wuv where rho is the density of uv-points computed with an effecitve bin size
            of npix/fov where fov is computed from the input image (see self.uvdensity for details).

            wuv=1 gives uniform weighting, while wuv=0 gives natural weighting.

        (B) error weightings:
            weight \propto sigma^werror where sigma is the thermal error estimate of each visibility.

            werror = 0: no error, werror = -1: popular weighting in DIFMAP, werror = -2: statistically-correct errors used in imaging

        Compute the density of uvpoints by convolving the uv-sample function by a circular Gaussian of which FWHM is npix/fov,
        where fov is given by fov = 2 * max(ra**2+dec**2) of the image.

        Args:
            uniform (boolean, default=True):
                use uniform weighting (True) or natrual weighting (False)
            image (imdata.IMFITS object):
                required to compute a fov for uniform weighting
            npix (integer, default=2):
                FWHM size of the kernel in the unit of 1/fov, which will be used to compute uv density for uniform weighting
            werror (integer, default=0):
                index for the error weighting. -2 gives the statistically-correct weights used in imaging for imaging.
            normalize (boolean, default=True):
                If True, rescale the total flux such that its peak becomes the unity.
        Returns:
            1-D numpy array contains weighting functions
        '''
        # number of data
        Ndata = len(self.amp.values)

        # Density Weight
        uvweight = np.zeros(Ndata)
        if not uniform:
            print("use natural weighting")
            uvweight[:] = 1
        else:
            print("use uniform weighting")
            if image is None:
                raise ValueError
            uvweight[:] = 1./self.uvdensity(image=image, npix=npix)
            uvweight[:] /= uvweight.max()

        # Error Weight
        errweight = np.zeros(Ndata)
        if werror == 0:
            errweight[:] = 1
        else:
            errweight[:] = np.power(self["sigma"].values, werror)
            errweight[:] /= errweight.max()

        # Total Weight
        totweight = uvweight * errweight
        totweight/= totweight.sum()

        return totweight

    def fit_beam(self,image,uniform=True,werror=0,npix=2,angunit=None,
                 difmap_fudge=False,istokes=0,ifreq=0):
        '''
        This method calculates the synthesized beam. It computes the second momentum of
        the uv-coverage by computing the eigenvalues and eigenvectors of its
        co-variance matrix weighted by specified uv/error-weights.

        In default, the FWHM of a Gaussian on the image domain whose variances
        on uv-domain are given by these eigenvalues will be computed as the beam size.
        This will use a fudge factor of sqrt(2*ln(2))/pi ~ 0.37. However, one can
        use another fudge factor of ~0.35 adopted in DIFMAP, if one specify difmap_fudge=True.

        This function computes the beam using two types of weights.
        (A) density weightings (uniform or natural):
            weight \propto rho^wuv where rho is the density of uv-points computed with an effecitve bin size
            of npix/fov where fov is computed from the input image (see self.uvdensity for details).

            wuv=1 gives uniform weighting, while wuv=0 gives natural weighting.

        (B) error weightings:
            weight \propto sigma^werror where sigma is the thermal error estimate of each visibility.

            werror = 0: no error, werror = -1: popular weighting in DIFMAP, werror = -2: statistically-correct errors used in imaging

        Compute the density of uvpoints by convolving the uv-sample function by a circular Gaussian of which FWHM is npix/fov,
        where fov is given by fov = 2 * max(ra**2+dec**2) of the image.

        Args:
            image (imdata.IMFITS object):
                The image where the synthesized beam will be mapped.
            uniform (boolean, default=True):
                use uniform weighting (True) or natrual weighting (False)
            npix (integer, default=2):
                FWHM size of the kernel in the unit of 1/fov, which will be used to compute uv density for uniform weighting
            werror (integer, default=0):
                index for the error weighting. -2 gives the statistically-correct weights used in imaging for imaging.
            angunit (str, default=None):
                Angular unit of the output beam. If not specified,
                it will use the unit of the input image.
            difmap_fudge (boolean, default=False):
                If True, it will use a fudge factor adopted in DIFMAP.
                Note that for uniform-weighting or error-weighting, this option
                does necesarily give the exact same beam size, since SMILI
                does not use uv-gridding and adopts a slightly different way
                to compute the density of the uv coverage.
        Returns:
            imdata.IMFITS object of synthesized beam map
        '''
        # angular unit
        if angunit is None:
            angunit = image.angunit

        # Fudge factor adopted in Difmap
        w = self.uvweight(uniform=uniform, image=image, werror=werror, npix=npix)
        w/= w.sum()

        # extract uv coordinates
        u = self.u.values
        v = self.v.values

        # take weighted mean of u^2, v^2, uv
        muu = np.sum(u*u*w)
        mvv = np.sum(v*v*w)
        muv = np.sum(u*v*w)

        # This is deriving the second moment of uv coverages
        ftmp = np.sqrt(4.0*muv*muv+(muu-mvv)**2)
        bpa = -0.5*np.arctan2(2.0*muv, muu - mvv) * 180. / np.pi # Direction of the Eigen Vector
        varuv_maj = (muu+mvv)/2. + ftmp/2.  # Eigen values
        varuv_min = (muu+mvv)/2. - ftmp/2.  # Eigen values

        # Choise of the fudge factor
        if difmap_fudge:
            # This is a fudge factor adopted in Difmap with no reasons
            fudge = 0.7/2.
        else:
            # This is a fudge factor of ~0.37, giving the equivalent image-domain FWHM
            #  of the Gaussian corresponding
            fudge = np.sqrt(8*np.log(2))/(2*np.pi)

        # compute the beam size
        bmin = fudge/np.sqrt(varuv_maj) * util.angconv("rad", angunit)
        bmaj = fudge/np.sqrt(varuv_min) * util.angconv("rad", angunit)

        # return as parameters of gauss_convolve
        cb_parms = ({'majsize': bmaj, 'minsize': bmin, 'angunit': angunit, 'pa': bpa})
        return cb_parms

    def map_beam(self,image,uniform=True,werror=0,npix=2,normalize=True,istokes=0,ifreq=0):
        '''
        This method calculates the synthesized beam
        Compute the synthesized beam. This function computes the beam using two types of weights.
        (A) density weightings (uniform or natural):
            weight \propto rho^wuv where rho is the density of uv-points computed with an effecitve bin size
            of npix/fov where fov is computed from the input image (see self.uvdensity for details).

            wuv=1 gives uniform weighting, while wuv=0 gives natural weighting.

        (B) error weightings:
            weight \propto sigma^werror where sigma is the thermal error estimate of each visibility.

            werror = 0: no error, werror = -1: popular weighting in DIFMAP, werror = -2: statistically-correct errors used in imaging

        Compute the density of uvpoints by convolving the uv-sample function by a circular Gaussian of which FWHM is npix/fov,
        where fov is given by fov = 2 * max(ra**2+dec**2) of the image.

        Args:
            image (imdata.IMFITS object):
                The image where the synthesized beam will be mapped.
            uniform (boolean, default=True):
                use uniform weighting (True) or natrual weighting (False)
            npix (integer, default=2):
                FWHM size of the kernel in the unit of 1/fov, which will be used to compute uv density for uniform weighting
            werror (integer, default=0):
                index for the error weighting. -2 gives the statistically-correct weights used in imaging for imaging.

        Returns:
            imdata.IMFITS object of synthesized beam map
        '''
        Nx = image.header["nx"]
        Ny = image.header["ny"]
        Nxref = image.header["nxref"]
        Nyref = image.header["nyref"]
        dx_rad = np.deg2rad(image.header["dx"])
        dy_rad = np.deg2rad(image.header["dy"])

        # u-v coverage
        u = np.copy(self["u"].values)
        v = np.copy(self["v"].values)
        u *= 2*np.pi*dx_rad * -1
        v *= 2*np.pi*dy_rad * -1
        ut = np.concatenate([u,-u])
        vt = np.concatenate([v,-v])
        Nuv = len(ut)

        # weights
        w = self.uvweight(uniform=uniform, image=image, werror=werror, npix=npix)
        wt = np.concatenate([w,w])
        wt /= wt.sum()

        # u-v coverage
        dix = Nx/2. + 1 - Nxref
        diy = Ny/2. + 1 - Nyref
        Vsynsc = np.exp(1j * (ut*dix + vt*diy))
        Vsynsr = np.real(Vsynsc)
        Vsynsi = np.imag(Vsynsc)

        # compute weighted visibilities
        Vinreal = Vsynsr*wt
        Vinimag = Vsynsi*wt

        # synthesized beam
        Isyns=fortlib.fftlib.nufft_adj_real1d(ut,vt,Vinreal,Vinimag,Nx,Ny,Nuv)
        Isyns=Isyns.reshape([Ny,Nx])
        if normalize:
            Isyns /= Isyns.max()

        imageout = copy.deepcopy(image)
        imageout.data[istokes,ifreq]=Isyns
        imageout.update_fits()
        return imageout

    def map_residual(self,image,uniform=False,werror=-2,npix=2,restore=False,istokes=0,ifreq=0):
        '''
        This method calculates the synthesized beam
        Compute the residual beam. This function computes the beam using two types of weights.
        (A) density weightings (uniform or natural):
            weight \propto rho^wuv where rho is the density of uv-points computed with an effecitve bin size
            of npix/fov where fov is computed from the input image (see self.uvdensity for details).

            wuv=1 gives uniform weighting, while wuv=0 gives natural weighting.

        (B) error weightings:
            weight \propto sigma^werror where sigma is the thermal error estimate of each visibility.

            werror = 0: no error, werror = -1: popular weighting in DIFMAP, werror = -2: statistically-correct errors used in imaging

        Compute the density of uvpoints by convolving the uv-sample function by a circular Gaussian of which FWHM is npix/fov,
        where fov is given by fov = 2 * max(ra**2+dec**2) of the image.

        Args:
            image (imdata.IMFITS object):
                The image where the synthesized beam will be mapped.
            uniform (boolean, default=False):
                use uniform weighting (True) or natrual weighting (False)
            npix (integer, default=2):
                FWHM size of the kernel in the unit of 1/fov, which will be used to compute uv density for uniform weighting
            werror (integer, default=-2):
                index for the error weighting. -2 gives the statistically-correct weights used in imaging for imaging.

        Returns:
            imdata.IMFITS object of the residual map
        '''
        residvis = self.residual_image(image)

        Nx = image.header["nx"]
        Ny = image.header["ny"]
        Nxref = image.header["nxref"]
        Nyref = image.header["nyref"]
        dx_rad = np.deg2rad(image.header["dx"])
        dy_rad = np.deg2rad(image.header["dy"])

        # u-v coverage
        u = np.copy(residvis["u"].values)
        v = np.copy(residvis["v"].values)
        u *= 2*np.pi*dx_rad * -1
        v *= 2*np.pi*dy_rad * -1
        ut = np.concatenate([u,-u])
        vt = np.concatenate([v,-v])
        Nuv = len(ut)
        del u,v

        # weights
        w = self.uvweight(uniform=uniform, image=image, werror=werror, npix=npix)
        wt = np.concatenate([w,w])
        wt /= wt.sum()

        # visibility data
        Vcomp_data = residvis.comp()
        Vcomp_data = np.concatenate([Vcomp_data, np.conj(Vcomp_data)])

        # shift the tracking center to the center of the image
        dix = Nx/2. + 1 - Nxref
        diy = Ny/2. + 1 - Nyref
        Vcomp_data *= np.exp(1j * (ut*dix + vt*diy))
        Vfcvrt = np.real(Vcomp_data)
        Vfcvit = np.imag(Vcomp_data)

        # take weighted residual between data and model visibilities
        Vfcvrt *= wt
        Vfcvit *= wt

        # compute residual image
        residual=fortlib.fftlib.nufft_adj_real1d(ut,vt,Vfcvrt,Vfcvit,Nx,Ny,Nuv)
        residual = residual.reshape([Ny,Nx])

        imageout = copy.deepcopy(image)
        imageout.data[istokes,ifreq]=residual
        imageout.update_fits()
        return imageout

    def map_clean(self,image,restore=False,errorweight=-2,istokes=0,ifreq=0):
        '''
        This method calculates the residual map by using visibility and model image

        Args:
            images (imdata.IMFITS object): model image
            errorweight: index of weight
        Returns:
            imdata.IMFITS object of residual+model image
        '''
        if restore:
            beamprm = image.get_beam()
            if beamprm["majsize"]==0 or beamprm["minsize"]==0:
                raise ValueError("please set beam parameter with image.set_beam()")
            restored = image.convolve_gauss(**beamprm)
        else:
            restored = copy.deepcopy(image)
        residual = self.map_residual(image,errorweight=errorweight)
        imageout = copy.deepcopy(image)
        imageout.data[istokes,ifreq]=residual.data[istokes,ifreq]+restored.data[istokes,ifreq]
        imageout.update_fits()
        return imageout

    def plot_model_fcv(self, outimage, filename=None, plotargs={'ms': 1., }):
        '''
        Make summary pdf figures and csv file of checking model, residual
        and chisquare of real and imaginary parts of visibility for each baseline
        Args:
            outimage (imdata.IMFITS object):
                model image to construct a model visibility
            filename:
              str, pathlib.Path, py._path.local.LocalPath or any object with a read()
              method (such as a file handle or StringIO)
            **plotargs:
              You can set parameters of matplotlib.pyplot.plot.
              Defaults are {'ms': 1., }
        Returns:
            summary pdf file (default = "model.pdf"):
                Output pdf file that summerizes results.
            cvsumtablefile (default = "model.csv"):
                Output csv file that summerizes results.
        '''

        if filename is not None:
            pdf = PdfPages(filename)
        else:
            filename = "model"
            pdf = PdfPages(filename)

        # model,residual,chisq
        nullfmt = NullFormatter()
        model = self.eval_image(imfits=outimage,mask=None)
        resid = self.residual_image(imfits=outimage,mask=None)
        chisq,rchisq = self.chisq_image(imfits=outimage,mask=None)

        # set figure size
        util.matplotlibrc(ncols=2, nrows=2, width=500, height=300)

        # First figure: All data
        fig, axs = plt.subplots(nrows=2, ncols=2)
        plt.suptitle(r"$\chi ^2$"+"=%04f"%(chisq)+", "+r"$\chi ^2 _{\nu}$"+"=%04f"%(rchisq),fontsize=18)
        plt.subplots_adjust(hspace=0.4)

        # 1. Radplot of visibilities
        ax = axs[0,0]
        plt.sca(ax)
        plt.title("Radplot of Visibilities")
        plotargs1=copy.deepcopy(plotargs)
        plotargs1["label"]="Real"
        self.radplot(datatype="real",errorbar=False, **plotargs1)
        plotargs1["label"]="Imag"
        self.radplot(datatype="imag",errorbar=False, **plotargs1)
        plotargs1["label"]="Real model"
        model.radplot(datatype="real",  errorbar=False, **plotargs1)
        plotargs1["label"]="Imag model"
        model.radplot(datatype="imag", errorbar=False, **plotargs1)

        # set xyrange
        plt.autoscale()
        plt.ylabel("Visibilities (Jy)")
        plt.autoscale()
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.5)
        plt.locator_params(axis='x',nbins=6)
        plt.locator_params(axis='y',nbins=6)
        plt.legend(loc='best',markerscale=2.,ncol=4,handlelength=0.1,mode="expand")

        # 2. Radplot of normalized residuals
        ax = axs[1,0]
        plt.sca(ax)
        plt.title("Radplot of normalized residuals")
        plotargs1["label"]="Real"
        resid.radplot(datatype="real",normerror=True,errorbar=False,color="red",**plotargs1)
        plotargs1["label"]="Imag"
        resid.radplot(datatype="imag",normerror=True,errorbar=False,color="blue",**plotargs1)

        # set xyrange
        plt.autoscale()
        plt.ylabel("Normalized Residuals")
        plt.autoscale()
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.5)
        plt.locator_params(axis='x',nbins=6)
        plt.locator_params(axis='y',nbins=6)
        plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.1)


        # 3. Histogram of residuals
        ax = axs[0,1]
        plt.sca(ax)
        plt.title("Histogram of residuals")
        N = len(2*resid["amp"])
        plt.hist(resid["amp"]*np.cos(resid["phase"]), bins=np.int(np.sqrt(N)),
                 normed=True, alpha=0.3, histtype='stepfilled', color='red',orientation='vertical',label="Real")
        plt.hist(resid["amp"]*np.sin(resid["phase"]), bins=np.int(np.sqrt(N)),
                 normed=True, alpha=0.3, histtype='stepfilled', color='blue',orientation='vertical',label="Imag")

        # set xyrange
        plt.autoscale()
        plt.xlabel("Residual Visibilities")
        xmin,xmax = plt.xlim()
        xmax = max(xmax,abs(xmin))
        plt.xlim(-xmax,xmax)
        plt.locator_params(axis='x',nbins=6)
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.3)
        plt.locator_params(axis='y',nbins=6)
        plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.5)


        # 4. Histogram of normalized residuals
        ax = axs[1,1]
        plt.sca(ax)
        plt.title("Histogram of normalized residuals")
        normresid_real = resid["amp"]*np.cos(resid["phase"]) / resid["sigma"]
        normresid_imag = resid["amp"]*np.sin(resid["phase"]) / resid["sigma"]
        N = len(normresid_real)
        plt.hist(normresid_real, bins=np.int(np.sqrt(N)),
                 normed=True, alpha=0.3, histtype='stepfilled', color="red",orientation='vertical',label="Real")
        plt.hist(normresid_imag, bins=np.int(np.sqrt(N)),
                 normed=True, alpha=0.3, histtype='stepfilled', color="blue",orientation='vertical',label="Imag")

        # model line
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        y = 1 / np.sqrt(2 * np.pi) * np.exp(-x * x / 2.)
        plt.plot(x, y, color="black")

        # set xyrange
        plt.autoscale()
        plt.xlabel("Normalized Residuals")
        xmin,xmax = plt.xlim()
        xmax = max(xmax,abs(xmin))
        plt.xlim(-xmax,xmax)
        plt.locator_params(axis='x',nbins=6)
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.3)
        plt.locator_params(axis='y',nbins=6)
        plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.5)


        # save file
        if filename is not None:
            pdf.savefig()
            plt.close()

        matplotlib.rcdefaults()

        # tplot and residual of baselines
        baselines= list(self.baseline_list())
        Nbsl = len(baselines)
        stconcat = []
        chiconcat = []
        rchiconcat = []
        tchiconcat = []
        Ndataconcat = []
        baselines= self.baseline_list()
        Nqua = len(baselines)
        tNdata = len(self["amp"])

        for ibsl in range(Nbsl):
            st1 = baselines[ibsl][0]
            st2 = baselines[ibsl][1]

            frmid =  self["st1name"] == st1
            frmid &= self["st2name"] == st2
            idx = np.where(frmid == True)
            single = self.loc[idx[0], :]

            chisq,rchisq = single.chisq_image(imfits=outimage,mask=None)

            nullfmt = NullFormatter()
            model        = single.eval_image(imfits=outimage,mask=None)
            resid        = single.residual_image(imfits=outimage,mask=None)
            chisq,rchisq = single.chisq_image(imfits=outimage,mask=None)

            Ndata = len(single)
            Ndataconcat.append(Ndata)
            stconcat.append(st1+"-"+st2)
            chiconcat.append(chisq)
            rchiconcat.append(rchisq)
            tchiconcat.append(chisq/tNdata)

            util.matplotlibrc(ncols=2, nrows=2, width=500, height=300)
            fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False)
            plt.suptitle(st1+"-"+st2+": "+r"$\chi ^2$"+"=%04f"%(chisq)+", "+r"$\chi ^2 _{\nu}$"+"=%04f"%(rchisq) ,fontsize=18)
            plt.subplots_adjust(hspace=0.4)

            # 1. Time plot of visibilities
            ax = axs[0,0]
            plt.sca(ax)
            plt.title("Time plot of visibilities")
            single.vplot("utc", "real",errorbar=False,label="Real")
            single.vplot("utc", "imag",errorbar=False,label="Imag")
            model.vplot("utc", "real",errorbar=False,label="Real model")
            model.vplot("utc", "imag",errorbar=False,label="Imag model")

            # set xyrange
            plt.autoscale()
            plt.ylabel("Visibilities (Jy)")
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.5)
            plt.locator_params(axis='x',nbins=6)
            plt.locator_params(axis='y',nbins=6)
            plt.legend(loc='best',markerscale=2.,ncol=4,handlelength=0.1,mode="expand")

            # 2. Time plot of normalized residuals
            ax = axs[1,0]
            plt.sca(ax)
            plt.title("Time plot of normalized residuals")
            resid.vplot("utc", "real",normerror1=True,errorbar=False,label="Real",color="blue")
            resid.vplot("utc", "imag",normerror1=True,errorbar=False,label="Imag",color="red")
            plt.ylabel("Normalized Residuals")

            # set xyrange
            plt.autoscale()
            plt.ylabel("Normalized residuals")
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.5)
            plt.locator_params(axis='x',nbins=6)
            plt.locator_params(axis='y',nbins=6)
            plt.legend(loc='upper left',markerscale=2.,ncol=2,handlelength=0.1)

            # 3. Histogram of residuals
            ax = axs[0,1]
            plt.sca(ax)
            plt.title("Histogram of residuals")
            N = len(2*resid["amp"])
            plt.hist(resid["amp"]*np.cos(resid["phase"]), bins=np.int(np.sqrt(N)),
                     normed=True, alpha=0.3, histtype='stepfilled',color="blue", orientation='vertical',label="Real")
            plt.hist(resid["amp"]*np.sin(resid["phase"]), bins=np.int(np.sqrt(N)),
                     normed=True, alpha=0.3, histtype='stepfilled',color="red", orientation='vertical',label="Imag")

            # set xyrange
            plt.autoscale()
            plt.xlabel("Residual Visibilities")
            xmin,xmax = plt.xlim()
            xmax = max(xmax,abs(xmin))
            plt.xlim(-xmax,xmax)
            plt.locator_params(axis='x',nbins=6)
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.3)
            plt.locator_params(axis='y',nbins=6)
            plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.5)


            # 4. Histogram of normalized residuals
            ax = axs[1,1]
            plt.sca(ax)
            plt.title("Histogram of normalized residuals")
            normresid_real = resid["amp"]*np.cos(resid["phase"]) / resid["sigma"]
            normresid_imag = resid["amp"]*np.sin(resid["phase"]) / resid["sigma"]
            N = len(normresid_real+normresid_imag)
            plt.hist(normresid_real, bins=np.int(np.sqrt(N)),
                     normed=True, alpha=0.3, histtype='stepfilled',color="blue", orientation='vertical',label="Real")
            plt.hist(normresid_imag, bins=np.int(np.sqrt(N)),
                     normed=True, alpha=0.3, histtype='stepfilled',color="red", orientation='vertical',label="Imag")
            plt.xlabel("Normalized Residuals")
            # model line
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 1000)
            y = 1 / np.sqrt(2 * np.pi) * np.exp(-x * x / 2.)

            plt.plot(x, y, color="black")

            # set xyrange
            plt.xlabel("Normalized Residuals")
            xmin,xmax = plt.xlim()
            xmax = max(xmax,abs(xmin))
            plt.xlim(-xmax,xmax)
            plt.locator_params(axis='x',nbins=6)
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.3)
            plt.locator_params(axis='y',nbins=6)
            plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.5)


            matplotlib.rcdefaults()
            if filename is not None:
                pdf.savefig()
                plt.close()

            del single, model, resid, normresid_real, normresid_imag

        matplotlib.rcdefaults()


        # plot residual of baselines
        util.matplotlibrc(ncols=1, nrows=3, width=700, height=400)
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False)
        plt.subplots_adjust(hspace=0.55)

        ax = axs[0]
        plt.sca(ax)
        plt.title(r"$\chi ^2$"+" for each baseline")
        plt.plot(stconcat,chiconcat,"o")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.ylabel(r"$\chi ^2$")
        plt.locator_params(axis='y',nbins=6)

        ax = axs[1]
        plt.sca(ax)
        plt.title(r"$\chi ^2 _{\nu}$"+" for each baseline")
        plt.plot(stconcat,rchiconcat,"o")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.ylabel(r"$\chi ^2 _{\nu}$")
        plt.locator_params(axis='y',nbins=6)

        ax = axs[2]
        plt.sca(ax)
        plt.title(r"$\chi ^2 _{\rm total}$"+" for each baseline")
        plt.plot(stconcat,tchiconcat,"o")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.ylabel(r"$\chi ^2_{\rm total}$")
        plt.locator_params(axis='y',nbins=6)

        # close pdf file
        if filename is not None:
            pdf.savefig()
            plt.close()
            pdf.close()

        matplotlib.rcdefaults()

        # make csv table
        stconcat.insert(0,"total")
        Ndataconcat.insert(0,np.sum(Ndataconcat))
        chiconcat.insert(0,np.sum(chiconcat))
        rchiconcat.insert(0,np.sum(rchiconcat))
        tchiconcat.insert(0,np.sum(tchiconcat))
        table = pd.DataFrame()
        table["baseline"] = stconcat
        table["Ndata"] = Ndataconcat
        table["chisq"] = chiconcat
        table["rchisq_bl"] = rchiconcat
        table["rchisq_total"] = tchiconcat
        table.to_csv(filename+".csv")

        del table

    def plot_model_amp(self, outimage, filename=None, plotargs={'ms': 1., }):
        '''
        Make summary pdf figures and csv file of checking model, residual
        and chisquare of visibility amplitudes for each baseline
        Args:
            outimage (imdata.IMFITS object):
                model image to construct a model visibility
            filename:
              str, pathlib.Path, py._path.local.LocalPath or any object with a read()
              method (such as a file handle or StringIO)
            **plotargs:
              You can set parameters of matplotlib.pyplot.plot.
              Defaults are {'ms': 1., }
        Returns:
            summary pdf file (default = "model.pdf"):
                Output pdf file that summerizes results.
            cvsumtablefile (default = "model.csv"):
                Output csv file that summerizes results.
        '''

        if filename is not None:
            pdf = PdfPages(filename)
        else:
            filename = "model"
            pdf = PdfPages(filename)

        # model,residual,chisq
        nullfmt = NullFormatter()
        model = self.eval_image(imfits=outimage,mask=None,amptable=True,istokes=0,ifreq=0)
        resid = self.residual_image(imfits=outimage,mask=None,amptable=True,istokes=0,ifreq=0)
        chisq,rchisq = self.chisq_image(imfits=outimage,mask=None,amptable=True,istokes=0,ifreq=0)

        # set figure size
        util.matplotlibrc(ncols=2, nrows=2, width=500, height=300)

        # First figure: All data
        fig, axs = plt.subplots(nrows=2, ncols=2)
        plt.suptitle(r"$\chi ^2$"+"=%04f"%(chisq)+", "+r"$\chi ^2 _{\nu}$"+"=%04f"%(rchisq),fontsize=18)
        plt.subplots_adjust(hspace=0.4)

        # 1. Radplot of visibility amplitudes
        ax = axs[0,0]
        plt.sca(ax)
        plt.title("Radplot of amplitudes")
        plotargs1=copy.deepcopy(plotargs)
        plotargs1["label"]="Observation"
        self.radplot(datatype="amp", color="black",errorbar=False, **plotargs1)
        plotargs1["label"]="Model"
        model.radplot(datatype="amp", color="red",  errorbar=False, **plotargs1)

        # set xyrange
        plt.autoscale()
        plt.ylabel("Visibility amplitudes (Jy)")
        plt.autoscale()
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.5)
        plt.locator_params(axis='x',nbins=6)
        plt.locator_params(axis='y',nbins=6)
        plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.1)

        # 2. Radplot of normalized residuals
        ax = axs[1,0]
        plt.sca(ax)
        plt.title("Radplot of normalized residuals")
        resid.radplot(datatype="amp",normerror=True,errorbar=False,color="black",**plotargs)
        plt.ylabel("Normalized Residuals")

        # set xyrange
        plt.autoscale()
        plt.ylabel("Normalized residuals")
        ymin,ymax = plt.ylim()
        plt.ylim(ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1)
        plt.locator_params(axis='x',nbins=6)
        plt.locator_params(axis='y',nbins=6)

        # 3. Histogram of residuals
        ax = axs[0,1]
        plt.sca(ax)
        plt.title("Histogram of residuals")
        N = len(resid["amp"])
        plt.hist(resid["amp"], bins=np.int(np.sqrt(N)),
                 normed=True, orientation='vertical')

        # set xyrange
        plt.autoscale()
        plt.xlabel("Residual Amplitudes")
        xmin,xmax = plt.xlim()
        xmax = max(xmax,abs(xmin))
        plt.xlim(-xmax,xmax)
        plt.locator_params(axis='x',nbins=6)
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.2)
        plt.locator_params(axis='y',nbins=6)

        # 4. Histogram of normalized residuals
        ax = axs[1,1]
        plt.sca(ax)
        plt.title("Histogram of normalized residuals")
        normresid = resid["amp"] / resid["sigma"]
        N = len(normresid)
        plt.hist(normresid, bins=np.int(np.sqrt(N)),
                 normed=True, orientation='vertical')

        # model line
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        y = 1 / np.sqrt(2 * np.pi) * np.exp(-x * x / 2.)
        plt.plot(x, y, color="red")

        # set xyrange
        plt.autoscale()
        plt.xlabel("Normalized Residuals")
        xmin,xmax = plt.xlim()
        xmax = max(xmax,abs(xmin))
        plt.xlim(-xmax,xmax)
        plt.locator_params(axis='x',nbins=6)
        ymin,ymax = plt.ylim()
        plt.ylim(ymin,ymax+(ymax-ymin)*0.2)
        plt.locator_params(axis='y',nbins=6)

        # save file
        if filename is not None:
            pdf.savefig()
            plt.close()

        # tplot  and residual of each baseline
        baselines= self.baseline_list()
        Nbsl = len(baselines)
        stconcat = []
        chiconcat = []
        rchiconcat = []
        tchiconcat = []
        Ndataconcat = []
        tNdata = len(self["amp"])

        for ibsl in range(Nbsl):


            st1 = baselines[ibsl][0]
            st2 = baselines[ibsl][1]

            frmid =  self["st1name"] == st1
            frmid &= self["st2name"] == st2
            idx = np.where(frmid == True)
            single = self.loc[idx[0], :]

            nullfmt = NullFormatter()
            model        = single.eval_image(imfits=outimage,mask=None,amptable=True,istokes=0,ifreq=0)
            resid        = single.residual_image(imfits=outimage,mask=None,amptable=True,istokes=0,ifreq=0)
            chisq,rchisq = single.chisq_image(imfits=outimage,mask=None,amptable=True,istokes=0,ifreq=0)

            Ndata = len(single)
            Ndataconcat.append(Ndata)
            stconcat.append(st1+"-"+st2)
            chiconcat.append(chisq)
            rchiconcat.append(rchisq)
            tchiconcat.append(chisq/tNdata)

            util.matplotlibrc(ncols=2, nrows=2, width=500, height=300)
            fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False)
            plt.suptitle(st1+"-"+st2+": "+r"$\chi ^2$"+"=%04f"%(chisq)+", "+r"$\chi ^2 _{\nu}$"+"=%04f"%(rchisq) ,fontsize=18)
            plt.subplots_adjust(hspace=0.4)

            # 1. Time plot of closure phases
            ax = axs[0,0]
            plt.sca(ax)
            plt.title("Time plot of visibility amplitudes")
            single.vplot("utc", "amp",errorbar=False,label="Observation")
            model.vplot("utc", "amp",errorbar=False,label="Model")

            # set xyrange
            plt.autoscale()
            plt.ylabel("Visibility amplitudes (Jy)")
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.5)
            plt.locator_params(axis='x',nbins=6)
            plt.locator_params(axis='y',nbins=6)
            plt.legend(loc='upper left',markerscale=2.,ncol=4,handlelength=0.1)

            # 2. Time plot of normalized residuals
            ax = axs[1,0]
            plt.sca(ax)
            plt.title("Time plot of normalized residuals")
            resid.vplot("utc", "amp",normerror1=True,errorbar=False)
            plt.ylabel("Normalized Residuals")
            plt.autoscale()

            # set xyrange
            plt.autoscale()
            plt.ylabel("Normalized residuals")
            plt.locator_params(axis='x',nbins=6)
            plt.locator_params(axis='y',nbins=6)

            # 3. Histogram of residuals
            ax = axs[0,1]
            plt.sca(ax)
            plt.title("Histogram of residuals")
            N = len(resid["amp"])
            plt.hist(resid["amp"], bins=np.int(np.sqrt(N)),
                     normed=True, orientation='vertical')
            # set xyrange
            plt.autoscale()
            plt.xlabel("Residual Amplitudes")
            xmin,xmax = plt.xlim()
            xmax = max(xmax,abs(xmin))
            plt.xlim(-xmax,xmax)
            plt.locator_params(axis='x',nbins=6)
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.2)
            plt.locator_params(axis='y',nbins=6)

            # 4. Histogram of normalized residuals
            ax = axs[1,1]
            plt.sca(ax)
            plt.title("Histogram of normalized residuals")
            normresid = resid["amp"] / resid["sigma"]
            N = len(normresid)
            plt.hist(normresid, bins=np.int(np.sqrt(N)),
                     normed=True, orientation='vertical')
            plt.xlabel("Normalized Residuals")
            # model line
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 1000)
            y = 1 / np.sqrt(2 * np.pi) * np.exp(-x * x / 2.)

            plt.plot(x, y, color="red")

            # set xyrange
            plt.xlabel("Normalized Residuals")
            xmin,xmax = plt.xlim()
            xmax = max(xmax,abs(xmin))
            plt.xlim(-xmax,xmax)
            plt.locator_params(axis='x',nbins=6)
            ymin,ymax = plt.ylim()
            plt.ylim(ymin,ymax+(ymax-ymin)*0.2)
            plt.locator_params(axis='y',nbins=6)

            matplotlib.rcdefaults()
            if filename is not None:
                pdf.savefig()
                plt.close()

            del single, model, resid, normresid

        matplotlib.rcdefaults()

        # plot residual of baselines
        util.matplotlibrc(ncols=1, nrows=3, width=700, height=400)
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False)
        plt.subplots_adjust(hspace=0.55)

        ax = axs[0]
        plt.sca(ax)
        plt.title(r"$\chi ^2$"+" for each baseline")
        plt.plot(stconcat,chiconcat,"o")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.ylabel(r"$\chi ^2$")
        plt.locator_params(axis='y',nbins=6)

        ax = axs[1]
        plt.sca(ax)
        plt.title(r"$\chi ^2 _{\nu}$"+" for each baseline")
        plt.plot(stconcat,rchiconcat,"o")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.ylabel(r"$\chi ^2 _{\nu}$")
        plt.locator_params(axis='y',nbins=6)

        ax = axs[2]
        plt.sca(ax)
        plt.title(r"$\chi ^2 _{\rm total}$"+" for each baseline")
        plt.plot(stconcat,tchiconcat,"o")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.ylabel(r"$\chi ^2_{\rm total}$")
        plt.locator_params(axis='y',nbins=6)

        # close pdf file
        if filename is not None:
            pdf.savefig()
            plt.close()
            pdf.close()

        matplotlib.rcdefaults()


        stconcat.insert(0,"total")
        Ndataconcat.insert(0,np.sum(Ndataconcat))
        chiconcat.insert(0,np.sum(chiconcat))
        rchiconcat.insert(0,np.sum(rchiconcat))
        tchiconcat.insert(0,np.sum(tchiconcat))

        # make csv table
        table = pd.DataFrame()
        table["baseline"] = stconcat
        table["Ndata"] = np.zeros(Nbsl+1)
        table["chisq"] = chiconcat
        table["rchisq_bl"] = rchiconcat
        table["rchisq_total"] = tchiconcat
        table.to_csv(filename+".csv")
        del table


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
    for baseline, red in itertools.product(baselines, redundant):
        flag |= (baseline[0] in red) and (baseline[1] in red)
        if flag: break
    return not flag

def check_baselines(baselines, blset):
    flag = True
    for baseline in baselines:
        flag &= baseline in blset
    return flag


def calc_matrix_ca(matrix, blid1, blid2, blid3, blid4, Nbl, dependent):
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
    if dependent:
        newrank = 1
    else:
        newrank = np.linalg.matrix_rank(newmatrix)
    return newrank, newmatrix


def calc_matrix_bs(matrix, blid1, blid2, blid3, Nbl, dependent):
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
    if dependent:
        newrank = 1
    else:
        newrank = np.linalg.matrix_rank(newmatrix)
    return newrank, newmatrix

def _radplot_amp(vistable, uvunit, errorbar, ls ,marker, **plotargs):
    # Conversion Factor
    conv = vistable.uvunitconv(unit1="lambda", unit2=uvunit)

    # Label
    unitlabel = vistable.get_uvunitlabel(uvunit)

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
    unitlabel = vistable.get_uvunitlabel(uvunit)

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
    unitlabel = vistable.get_uvunitlabel(uvunit)

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
    unitlabel = vistable.get_uvunitlabel(uvunit)

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
    unitlabel = vistable.get_uvunitlabel(uvunit)

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
    unitlabel = vistable.get_uvunitlabel(uvunit)

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
