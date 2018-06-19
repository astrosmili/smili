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
from .tools import get_uvlist
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

    def chisq_image(self, imfits, mask=None, istokes=0, ifreq=0):
        # calcurate chisqared and reduced chisqred.
        model = self._call_fftlib(imfits=imfits,mask=mask,
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
        unitlabel = self.get_unitlabel(uvunit)

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
        unitlabel = self.get_unitlabel(uvunit)

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

    def vplot(self, station=1, timescale="utc", normerror=False, errorbar=True,
              log=True, ls="none", marker=".", **plotargs):
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
        st3table = self.drop_duplicates(subset='st3')
        st4table = self.drop_duplicates(subset='st4')
        stdict = dict(zip(st1table["st1"], st1table["st1name"]))
        stdict.update(dict(zip(st2table["st2"], st2table["st2name"])))
        stdict.update(dict(zip(st3table["st3"], st3table["st3name"])))
        stdict.update(dict(zip(st4table["st4"], st4table["st4name"])))

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
        tmptable = tmptable.query("st1 == @st1")

        # normalized by error
        if normerror:
            if log:
                tmptable["logamp"] /= tmptable["logsigma"]
            else:
                tmptable["amp"] /= tmptable["sigma"]
            errorbar = False

        # convert timescale to pd.to_datetime
        if timescale=="utc":
            tmptable.index = pd.to_datetime(tmptable.index)
        if timescale=="gsthour":
            tmptable.index = pd.to_datetime(tmptable.index, unit="h")

        # get antenna
        st2 = np.int32((tmptable.drop_duplicates(subset=['st2', 'st3', 'st4']))['st2'])
        st3 = np.int32((tmptable.drop_duplicates(subset=['st2', 'st3', 'st4']))['st3'])
        st4 = np.int32((tmptable.drop_duplicates(subset=['st2', 'st3', 'st4']))['st4'])
        Nant = len(st2)

        # plotting data
        fig, axs = plt.subplots(nrows=Nant, ncols=1, sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0.)
        for iant in range(Nant):
            ax = axs[iant]
            plt.sca(ax)

            plttable = tmptable.query("st2 == @st2[@iant] & st3 == @st3[@iant] & st4 == @st4[@iant]")
            if errorbar:
                if log:
                    plt.errorbar(plttable.index, plttable["logamp"], plttable["logsigma"],
                                 ls=ls, marker=marker, **plotargs)
                else:
                    plt.errorbar(plttable.index, plttable["amp"], plttable["sigma"],
                                 ls=ls, marker=marker, **plotargs)
            else:
                if log:
                    plt.plot(plttable[timescale], plttable["logamp"], ls=ls, marker=marker, **plotargs)
                else:
                    plt.plot(plttable[timescale], plttable["amp"], ls=ls, marker=marker, **plotargs)
            #
            plt.text(0.97, 0.9, st1name.strip()+"-"+stdict[st2[iant]].strip()+"-"+stdict[st3[iant]].strip()+"-"+stdict[st4[iant]].strip(),
                     horizontalalignment='right', verticalalignment='top',
                     transform=ax.transAxes, fontsize=8, color="black")
            plt.xlim(tmin, tmax)
            if log:
                ymin = np.min(plttable["logamp"])
                ymax = np.max(plttable["logamp"])
            else:
                ymin = np.min(plttable["amp"])
                ymax = np.max(plttable["amp"])
            #
            if ymin>=0.:
                plt.ylim(0.,)
            if ymax<=0.:
                plt.ylim(ymax=0.)
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
