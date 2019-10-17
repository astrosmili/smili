#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a submodule of smili handling various types of Lightcurve data sets.
'''

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# standard modules
import copy
import itertools
import collections
import datetime
import tqdm

# numerical packages
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import astropy.constants as ac
import astropy.coordinates as acd
import astropy.time as at
import astropy.io.fits as pf

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# internal
from .. import imdata, fortlib, util
from ..util import prt, average1d, interpolation1d

indent = "  "

# ------------------------------------------------------------------------------
# Classes for Light curve
# ------------------------------------------------------------------------------

class Lightcurve(pd.DataFrame):

    '''
    This is a class describing lightcurves
    '''

    @property
    def _constructor(self):
        return Lightcurve

    @property
    def _constructor_sliced(self):
        return LightcurveSeries

    #lightcurve_columns = ["utc","gsthour","flux","sigma"]
    #lightcurve_types   = [np.asarray, np.float64, np.float64, np.float64]

    def get_utc(self):
        '''
        get utc in astropy.time.Time object
        '''
        return at.Time(np.datetime_as_string(self["utc"]), scale="utc")


    def get_gst_datetime(self, continuous=False, wraphour=0):
        '''
        get GST in datetime
        '''
        Ndata = len(self.utc)

        utc = self.get_utc()
        gsthour = self.gsthour.values

        if continuous:
            dgsthour = -np.diff(gsthour)
            dgsthour[np.where(dgsthour<1e-6)]=0
            dgsthour[np.where(dgsthour>=1e-6)]=24
            dgsthour = np.add.accumulate(dgsthour)
            gsthour[1:] += dgsthour[:]
            origin = utc.min().datetime.strftime("%Y-%m-%d")
        else:
            gsthour = self.gsthour.values
            gsthour[np.where(gsthour<wraphour)]+=24
            origin = dt.datetime(2000,1,1,0,0,0)

        return pd.to_datetime(gsthour, unit="h", origin=origin)


    def cplightcurve(self, old_lc, kind="cubic"):
        '''
        Returns interpolated light curves based on Lightcurve object
        Args:
            old_lc: previous light curve
        Return:
            Lightcurve object
        '''

        told = old_lc.get_utc().cxcsec
        tref = self.get_utc().cxcsec
        tzero = told[0]
        told -= tzero
        tref -= tzero

        oldflux = old_lc["flux"]

        newflux = util.interpolation1d(
            told, oldflux, tref, kind=kind, fill_value="extrapolate",
        )

        lc = copy.deepcopy(self)
        lc["flux"] = newflux

        return lc

    def plot(self,
        axis1="utc",
        fluxunit="jy",
        gst_continuous=False,
        gst_wraphour=0.,
        time_maj_loc=mdates.HourLocator(),
        time_min_loc=mdates.MinuteLocator(byminute=np.arange(0,60,10)),
        time_maj_fmt='%H:%M',
        ls="none",
        marker=".",
        label=None,
        **plotargs):

        '''
        '''

        self["flux"] = copy.deepcopy(self["flux"] * util.fluxconv("jy", fluxunit))
        if "gst" in axis1.lower():
            self["gst"] = self.get_gst_datetime(continuous=gst_continuous, wraphour=gst_wraphour)
            self.sort_values(by="gst", inplace=True)
            axis1data = self.gst.values
        else:
            self.sort_values(by="utc", inplace=True)
            axis1data = self.utc.values

        # Plotting
        ax = plt.gca()
        plt.plot(axis1data, self.flux.values, ls=ls, marker=marker, label=label, **plotargs)

        # Yaxis
        plt.ylabel("Total flux density (%s)"%(imdata.IMFITS().get_fluxunitlabel(fluxunit)))

        # x-tickes
        ax.xaxis.set_major_locator(time_maj_loc)
        ax.xaxis.set_minor_locator(time_min_loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_maj_fmt))
        if "gst" in axis1.lower():
            plt.xlabel("Greenwich Sidereal Time")
        else:
            plt.xlabel("Universal Time")



    def smooth(self, solint=-1):
        '''
        Time smoothing function
        '''

    def read_lightcurve(filename, uvunit=None, **args):
        '''
        This fuction loads lightcurve.Lightcurve from an input csv file using pd.read_csv().

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
        table = Lightcurve(pd.read_csv(filename, **args))
        if "utc" in table.columns:
            table["utc"] = at.Time(table["utc"].values.tolist()).datetime

        return table


class LightcurveSeries(pd.Series):
    @property
    def _constructor(self):
        return LightcurveSeries

    @property
    def _constructor_expanddim(self):
        return Lightcurve
