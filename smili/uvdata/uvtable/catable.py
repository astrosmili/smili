#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes uv data table for closure amplitudes.
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import copy
import itertools

# numerical packages
import numpy as np
import pandas as pd
import theano.tensor as T

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import astropy.time as at

# internal
from .uvtable import UVTable, UVSeries
from .tools import get_uvlist,get_uvlist_loop
from ... import fortlib


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class CATable(UVTable):
    catable_columns = ["utc", "gsthour",
                       "freq", "stokesid", "ifid", "chid", "ch",
                       "u1", "v1", "w1", "uvdist1",
                       "u2", "v2", "w2", "uvdist2",
                       "u3", "v3", "w3", "uvdist3",
                       "u4", "v4", "w4", "uvdist4",
                       "uvdistmin", "uvdistmax", "uvdistave",
                       "st1", "st2", "st3", "st4",
                       "st1name", "st2name", "st3name", "st4name",
                       "amp", "sigma", "logamp", "logsigma"]
    catable_types = [np.asarray, np.float64,
                     np.float64, np.int32, np.int32, np.int32, np.int32,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,
                     np.int32, np.int32, np.int32, np.int32,
                     np.asarray, np.asarray, np.asarray, np.asarray,
                     np.float64, np.float64, np.float64, np.float64]

    @property
    def _constructor(self):
        return CATable

    @property
    def _constructor_sliced(self):
        return CASeries

    def set_uvunit(self, uvunit=None):
        # Check uvunits
        if uvunit is None:
            uvmax = np.max(self.uvdistmax.values)
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


    def station_list(self, id=False):
        '''
        Return list of stations. If id=True, return list of station IDs.
        '''
        if id:
            return np.unique([self["st1"],self["st2"],self["st3"],self["st4"]]).tolist()
        else:
            return np.unique([self["st1name"],self["st2name"],self["st3name"],self["st4name"]]).tolist()

    def station_dic(self, id2name=True):
        '''
        Return dictionary of stations. If id2name=True, return a dictionary
        whose key is the station ID number and value is the station name.
        Otherwise return a dictionary whose key is the name and value is ID.
        '''
        st1table = self.drop_duplicates(subset='st1')
        st2table = self.drop_duplicates(subset='st2')
        st3table = self.drop_duplicates(subset='st3')
        st4table = self.drop_duplicates(subset='st4')
        if id2name:
            outdict = dict(zip(st1table.st1.values, st1table.st1name.values))
            outdict.update(dict(zip(st2table.st2.values, st2table.st2name.values)))
            outdict.update(dict(zip(st3table.st3.values, st3table.st3name.values)))
            outdict.update(dict(zip(st4table.st4.values, st4table.st4name.values)))
        else:
            outdict = dict(zip(st1table.st1name.values,st1table.st1.values))
            outdict.update(dict(zip(st2table.st2name.values,st2table.st2.values)))
            outdict.update(dict(zip(st3table.st3name.values,st3table.st3.values)))
            outdict.update(dict(zip(st4table.st4name.values,st4table.st4.values)))
        return outdict

    def quadrature_list(self, id=False):
        '''
        Return the list of baselines. If id=False, then the names of stations
        will be returned. Otherwise, the ID numbers of stations will be returned.
        '''
        if id:
            table = self.drop_duplicates(subset=['st1','st2','st3','st4'])
            return zip(table.st1.values,table.st2.values,table.st3.values,table.st4.values)
        else:
            table = self.drop_duplicates(subset=['st1name','st2name','st3name','st4name'])
            return zip(table.st1name.values,table.st2name.values,table.st3name.values,table.st4name.values)

    def snr(self):
        '''
        Return the SNR estimator
        '''
        return self["amp"]/self["sigma"]

    def eval_image(self, imfits, mask=None, istokes=0, ifreq=0):
        #uvdata.CATable object (storing model closure phase)
        model = self._call_fftlib(imfits=imfits,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        Ndata = model[1]
        camodel = model[0][2]
        catable = self.copy()
        catable["phase"] = np.zeros(Ndata)
        catable["logamp"] = camodel
        return catable

    def eval_image3d(self, movie, mask=None, istokes=0, ifreq=0):
        #uvdata.CATable object (storing model closure phase)
        model = self._call_fftlib3d(movie=movie,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        Ndata = model[1]
        camodel = model[0][2]
        catable = self.copy()
        catable["phase"] = np.zeros(Ndata)
        catable["logamp"] = camodel
        return catable

    def residual_image(self, imfits, mask=None, istokes=0, ifreq=0):
        #uvdata CATable object (storing residual closure phase)
        model = self._call_fftlib(imfits=imfits,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        Ndata = model[1]
        resida = model[0][3]
        residtable = self.copy()
        residtable["logamp"] = resida
        residtable["phase"] = np.zeros(Ndata)
        return residtable

    def residual_image3d(self, movie, mask=None, istokes=0, ifreq=0):
        #uvdata CATable object (storing residual closure phase)
        model = self._call_fftlib3d(movie=movie,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        Ndata = model[1]
        resida = model[0][3]
        residtable = self.copy()
        residtable["logamp"] = resida
        residtable["phase"] = np.zeros(Ndata)
        return residtable


    def chisq_image(self, imfits, mask=None, istokes=0, ifreq=0):
        # calcurate chisqared and reduced chisqred.
        model = self._call_fftlib(imfits=imfits,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        chisq = model[0][0]
        Ndata = model[1]
        rchisq = chisq/Ndata

        return chisq,rchisq

    def chisq_image3d(self, movie, mask=None, istokes=0, ifreq=0):
        # calcurate chisqared and reduced chisqred.
        model = self._call_fftlib3d(movie=movie,mask=mask,
                                  istokes=istokes, ifreq=ifreq)
        chisq = model[0][0]
        Ndata = model[1]
        rchisq = chisq/Ndata

        return chisq,rchisq


    def _call_fftlib(self, imfits, mask, istokes=0, ifreq=0):
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

        # Closure Phase
        Ndata = 0
        catable = self.copy()
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)

        # get uv coordinates and uv indice
        u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca = get_uvlist(
                fcvtable=None, amptable=None, bstable=None, catable=catable
                )

        # normalize u, v coordinates
        u *= 2*np.pi*dx_rad
        v *= 2*np.pi*dy_rad

        # run model_cp
        model = fortlib.fftlib.model_ca(
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
                # Closure Phase
                uvidxca=np.int32(uvidxca),
                ca=np.float64(ca),
                varca=np.float64(varca)
                )

        return model,Ndata

    def _call_fftlib3d(self, movie, mask, istokes=0, ifreq=0):
        # get initial images
        istokes = istokes
        ifreq = ifreq
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

        # Closure Phase
        Ndata = 0
        catable = self.copy()
        ca = np.array(catable["logamp"], dtype=np.float64)
        varca = np.square(np.array(catable["logsigma"], dtype=np.float64))
        Ndata += len(ca)

        # get uv coordinates and uv indice
        u, v, uvidxfcv, uvidxamp, uvidxcp, uvidxca, Nuvs = get_uvlist_loop(Nt=Nt,
            fcvconcat=None, ampconcat=None, bsconcat=None, caconcat=catable
        )

        # normalize u, v coordinates
        u *= 2*np.pi*dx_rad
        v *= 2*np.pi*dy_rad

        # concatenate the initimages
        Iin = np.concatenate(Iin)

        # run model_cp
        model = fortlib.fftlib3d.model_ca(
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
                nuvs=np.int32(Nuvs),
                # Closure Phase
                uvidxca=np.int32(uvidxca),
                ca=np.float64(ca),
                varca=np.float64(varca)
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
        u1 = outtable.u1.values
        v1 = outtable.v1.values
        u2 = outtable.u2.values
        v2 = outtable.v2.values
        u3 = outtable.u3.values
        v3 = outtable.v3.values
        u4 = outtable.u4.values
        v4 = outtable.v4.values
        outtable["amp"] = geomodel.Camp(u1,v1,u2,v2,u3,v3,u4,v4).eval(**evalargs)
        outtable["logamp"] = geomodel.logCamp(u1,v1,u2,v2,u3,v3,u4,v4).eval(**evalargs)
        return outtable


    def residual_geomodel(self, geomodel, normed=True, doeval=False, evalargs={}):
        '''
        Calculate residuals of log closure amplitudes
        for an input geometric model

        Args:
            geomodel (geomodel.geomodel.GeoModel object):
                input model
            normed (boolean, default=True):
                if True, residuals will be normalized by 1 sigma error
            eval (boolean, default=False):
                if True, actual residual values will be calculated.
                Otherwise, resduals will be given as a theano graph.
        Returns:
            ndarray (if doeval=True) or theano object (otherwise)
        '''
        # u,v coordinates
        u1 = self.u1.values
        v1 = self.v1.values
        u2 = self.u2.values
        v2 = self.v2.values
        u3 = self.u3.values
        v3 = self.v3.values
        u4 = self.u4.values
        v4 = self.v4.values
        logCA = self.logamp.values
        logsigma = self.logsigma.values

        modlogCA = geomodel.logCamp(u1,v1,u2,v2,u3,v3,u4,v4)
        residual = logCA - modlogCA
        if normed:
            residual /= logsigma

        if doeval:
            return residual.eval(**evalargs)
        else:
            return residual


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

        plotargs2 = copy.deepcopy(plotargs)
        plotargs2["label"] = ""

        # plotting
        plt.plot(self["u1"] * conv, self["v1"] * conv,
                 ls=ls, marker=marker, **plotargs)
        plt.plot(self["u2"] * conv, self["v2"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u3"] * conv, self["v3"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        plt.plot(self["u4"] * conv, self["v4"] * conv,
                 ls=ls, marker=marker, **plotargs2)
        if conj:
            plt.plot(-self["u1"] * conv, -self["v1"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u2"] * conv, -self["v2"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u3"] * conv, -self["v3"] * conv,
                     ls=ls, marker=marker, **plotargs2)
            plt.plot(-self["u4"] * conv, -self["v4"] * conv,
                     ls=ls, marker=marker, **plotargs2)
        plt.xlabel(r"Baseline Length $u$ (%s)" % (unitlabel))
        plt.ylabel(r"Baseline Length $v$ (%s)" % (unitlabel))

        ax = plt.gca()
        ax.set_aspect("equal")
        xlim = np.asarray(ax.get_xlim())
        ylim = np.asarray(ax.get_ylim())
        ax.set_xlim(-np.sort(-xlim))
        ax.set_ylim(np.sort(ylim))

    def radplot(self, uvdtype="ave", uvunit=None, normerror=False, log=True,
                errorbar=True, ls="none", marker=".", **plotargs):
        '''
        Plot log(closure amplitudes) as a function of baseline lengths
        on the current axes. This method uses matplotlib.pyplot.plot() or
        matplotlib.pyplot.errorbar().

        Args:
          uvdtype (str, default = "ave"):
            The type of the baseline length plotted along the horizontal axis.
              "max": maximum of four baselines (=self["uvdistmax"])
              "min": minimum of four baselines (=self["uvdistmin"])
              "ave": average of four baselines (=self["uvdistave"])
          uvunit (str, default = None):
            The unit of the baseline length. if uvunit is None, it will use
            self.uvunit.
          errorbar (boolean, default = True):
            If errorbar is True, it will plot data with errorbars using
            matplotlib.pyplot.errorbar(). Otherwise, it will plot data without
            errorbars using matplotlib.pyplot.plot().

            If you plot model closure phases (i.e. model is not None),
            it will plot without errobars regardless of this parameter.
          model (dict-like such as pd.DataFrame, pd.Series, default is None):
            Model data sets. Model closure amplitudes must be given by model["camod"].
            Otherwise, it will plot closure amplitudes in the table (i.e. self["logamp"]).
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        # Set Unit
        if uvunit is None:
            self.set_uvunit()
            uvunit = self.uvunit

        # Conversion Factor
        conv = self.uvunitconv(unit1="lambda", unit2=uvunit)

        # uvdistance
        if uvdtype.lower().find("ave") * uvdtype.lower().find("mean") == 0:
            uvdist = self["uvdistave"] * conv
            head = "Mean"
        elif uvdtype.lower().find("min") == 0:
            uvdist = self["uvdistmin"] * conv
            head = "Minimum"
        elif uvdtype.lower().find("max") == 0:
            uvdist = self["uvdistmax"] * conv
            head = "Maximum"
        else:
            print("[Error] uvdtype=%s is not available." % (uvdtype))
            return -1

        # Label
        unitlabel = self.get_uvunitlabel(uvunit)

        # Copy data
        vistable = copy.deepcopy(self)

        # normalized by error
        if normerror:
            if log:
                vistable["logamp"] /= vistable["logsigma"]
            else:
                vistable["amp"] /= vistable["sigma"]
            errorbar = False

        # plotting data
        if errorbar:
            if log:
                plt.errorbar(uvdist, vistable["logamp"], vistable["logsigma"],
                             ls=ls, marker=marker, **plotargs)
            else:
                plt.errorbar(uvdist, vistable["amp"], vistable["sigma"],
                             ls=ls, marker=marker, **plotargs)
        else:
            if log:
                plt.plot(uvdist, vistable["logamp"], ls=ls, marker=marker, **plotargs)
            else:
                plt.plot(uvdist, vistable["amp"], ls=ls, marker=marker, **plotargs)
        plt.xlabel(r"%s Baseline Length (%s)" % (head,unitlabel))
        if log:
            plt.ylabel(r"Log Closure Amplitude")
        else:
            plt.ylabel(r"Closure Amplitude")
        plt.xlim(0,)


    def vplot(self,
            axis1="utc",
            axis2="phase",
            quadrature=None,
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
        ["utc","gst","amp","sigma","logamp","logsigma","snr",
         "uvd(ist)mean","uvd(ist)min","uvd(ist)max",
         "uvd(ist)1","uvd(ist)2","uvd(ist)3","uvd(ist)4"]

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
        # Check if quadrature is specified
        if quadrature is None:
            pltdata = self
        else:
            stndict = self.station_dic(id2name=True)
            stidict = self.station_dic(id2name=False)
            # make dictionary of stations
            if isinstance(quadrature[0], basestring):
                st1 = stidict[quadrature[0]]
            else:
                st1 = int(quadrature[0])
            if isinstance(quadrature[1], basestring):
                st2 = stidict[quadrature[1]]
            else:
                st2 = int(quadrature[1])
            if isinstance(quadrature[2], basestring):
                st3 = stidict[quadrature[2]]
            else:
                st3 = int(quadrature[2])
            if isinstance(quadrature[3], basestring):
                st4 = stidict[quadrature[3]]
            else:
                st4 = int(quadrature[3])
            st1name = stndict[st1]
            st2name = stndict[st2]
            st3name = stndict[st3]
            st4name = stndict[st4]
            pltdata = self.query("st1==@st1 & st2==@st2 & st3==@st3 & st4==@st4").reset_index(drop=True)
            del stndict, stidict
            if len(pltdata["st1"])==0:
                print("No data can be plotted.")
                return

        # Check label
        if label is None:
            if quadrature is None:
                label=""
            else:
                label="%s - %s - %s - %s"%(st1name,st2name,st3name,st4name)

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
        for i in xrange(2):
            axis = axises[i]
            normerror = normerrors[i]
            if   "utc" in axis:
                pltarrays.append(pltdata.utc.values)
                axislabels.append("Universal Time")
                deflims.append((None,None))
                errors.append(None)
            elif "gst" in axis:
                pltarrays.append(pltdata.gst_datetime(continuous=gst_continuous, wraphour=gst_wraphour))
                axislabels.append("Greenwich Sidereal Time")
                deflims.append((None,None))
                errors.append(None)
            elif "logamp" in axis:
                if not normerror:
                    pltarrays.append(pltdata.logamp.values)
                    axislabels.append("Log Closure Amplitude")
                    errors.append(pltdata.sigma.values)
                    if errorbar and (not useerrorbar):
                        useerrorbar=True
                else:
                    pltarrays.append(pltdata.logamp.values/pltdata.logsigma.values)
                    axislabels.append("Error-normalized Log Closure Amplitudes")
                    errors.append(None)
                deflims.append((None,None))
            elif "logsigma" in axis:
                pltarrays.append(pltdata.sigma.values)
                axislabels.append("Log Closure Amplitude Error")
                deflims.append((0,None))
                errors.append(None)
            elif "amp" in axis:
                if not normerror:
                    pltarrays.append(pltdata.amp.values)
                    axislabels.append("Closure Amplitude")
                    errors.append(pltdata.sigma.values)
                    if errorbar and (not useerrorbar):
                        useerrorbar=True
                else:
                    pltarrays.append(pltdata.amp.values/pltdata.sigma.values)
                    axislabels.append("Error-normalized Closure Amplitudes")
                    errors.append(None)
                deflims.append((0,None))
            elif "sigma" in axis:
                pltarrays.append(pltdata.sigma.values)
                axislabels.append("Closure Amplitude Error")
                deflims.append((0,None))
                errors.append(None)
            elif "snr" in axis:
                pltarrays.append(pltdata.snr().values)
                axislabels.append("SNR")
                deflims.append((0,None))
                errors.append(None)
            elif ("uvd" in axis) and ("1" in axis):
                pltarrays.append(pltdata.uvdist1.values*uvunitconv)
                axislabels.append("Baseline Length of the 1st Baseline (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif ("uvd" in axis) and ("2" in axis):
                pltarrays.append(pltdata.uvdist2.values*uvunitconv)
                axislabels.append("Baseline Length of the 2nd Baseline (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif ("uvd" in axis) and ("3" in axis):
                pltarrays.append(pltdata.uvdist3.values*uvunitconv)
                axislabels.append("Baseline Length of the 3rd Baseline (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif ("uvd" in axis) and ("4" in axis):
                pltarrays.append(pltdata.uvdist4.values*uvunitconv)
                axislabels.append("Baseline Length of the 4th Baseline (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif ("uvd" in axis) and ("max" in axis):
                pltarrays.append(pltdata.uvdistmax.values*uvunitconv)
                axislabels.append("Maximum Baseline Length (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif ("uvd" in axis) and ("min" in axis):
                pltarrays.append(pltdata.uvdistmin.values*uvunitconv)
                axislabels.append("Minimum Baseline Length (%s)"%(uvunitlabel))
                deflims.append((None,None))
                errors.append(None)
            elif ("uvd" in axis):
                pltarrays.append(pltdata.uvdistave.values*uvunitconv)
                axislabels.append("Mean Baseline Length (%s)"%(uvunitlabel))
                deflims.append((None,None))
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
        for i in xrange(2):
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


class CASeries(UVSeries):

    @property
    def _constructor(self):
        return CASeries

    @property
    def _constructor_expanddim(self):
        return CATable


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def read_catable(filename, uvunit=None, **args):
    '''
    This fuction loads uvdata.CATable from an input csv file using pd.read_csv().

    Args:
      filename:
        str, pathlib.Path, py._path.local.LocalPath or any object with a read()
        method (such as a file handle or StringIO)
      uvunit (str, default is None):
        units of uvdistance for plotting. If uvunit is None, uvunit will be
        inferred from the maximum baseline length. Availables are ["l[ambda]",
        "kl[ambda]", "ml[ambda]", "gl[ambda]", "m", "km"].

    Returns:
      uvdata.CATable object
    '''
    table = CATable(pd.read_csv(filename, **args))
    if "utc" in table.columns:
        table["utc"] = at.Time(table["utc"].values.tolist()).datetime
    table.set_uvunit()
    return table
