#!/usr/bin/env python
# -*- coding: utf-8 -*-




'''
This module describes gain calibration table for full complex visibilities.
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
from scipy import optimize
import matplotlib.pyplot as plt
import astropy.time as at
import matplotlib.dates as mdates

# internal
#from ..uvtable   import UVTable, UVSeries
#from ..gvistable import GVisTable, GVisSeries
#from ..catable   import CATable, CASeries
#from ..bstable   import BSTable, BSSeries
#from ... import imdata
from ...util import gp_interp

# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class CLTable(object):
    '''
    This is a class describing gain calibrations.
    '''

    def __init__(self, uvfits):
        '''
        '''
        self.gaintabs = {} #空dictionaryの作成

        # make gain table for each subarray
        subarrids = list(uvfits.subarrays.keys())
        for subarrid in subarrids:
            # make empty dictionary
            self.gaintabs[subarrid] = {}

            # get UTC
            utc = np.datetime_as_string(uvfits.visdata.coord["utc"])
            utc = sorted(set(utc))

            self.gaintabs[subarrid]["utc"]=utc

            # get the number of data along each dimension
            Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp=uvfits.visdata.data.shape
            Ntime = len(utc)

            # get the number of antennas
            arraydata = uvfits.subarrays[subarrid]
            Nant = arraydata.antable["name"].shape[0]

            # gain for first two stokes parameters (RR,LL)
            if Nstokes == 1:
                pass
            else:
                Nstokes = 2

            # make gain matrix
            gain = np.zeros([Ntime,Nif,Nch,Nstokes,Nant,3])

            # initialize gain
            gain[:,:,:,:,:,0]=1.0    # real part of gain
            gain[:,:,:,:,:,2]=1.0    # flagging (<0 for flagged data)
            self.gaintabs[subarrid]["gain"]=gain

    def clear_phase(self):
        '''
        This method makes a new CLTable with gain phase = 0.
        '''
        out = copy.deepcopy(self)

        # make gain table for each subarray
        subarrids = list(self.gaintabs.keys())
        for subarrid in subarrids:
            # get gain amplitude
            greal = self.gaintabs[subarrid]["gain"][:,:,:,:,:,0]
            gimag = self.gaintabs[subarrid]["gain"][:,:,:,:,:,1]
            gainamp = np.sqrt(greal*greal + gimag*gimag)

            # make zero-phase gain
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,0] = gainamp
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,1] = 0.0

        return out

    def clear_amp(self):
        '''
        This method makes a new CLTable with gain amplitude = 1.
        '''
        out = copy.deepcopy(self)

        # make gain table for each subarray
        subarrids = list(self.gaintabs.keys())
        for subarrid in subarrids:
            # get gain amplitude
            greal = self.gaintabs[subarrid]["gain"][:,:,:,:,:,0]
            gimag = self.gaintabs[subarrid]["gain"][:,:,:,:,:,1]
            gainamp = np.sqrt(greal*greal + gimag*gimag)

            # make normalized gain
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,0] = greal/gainamp
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,1] = gimag/gainamp

        return out


    def smoothing_gp(self, timescale=None, amp=True, phase=True, n_restarts_optimizer=1):
        '''
        Smooth gains with the Gaussian Process Regression

        Args:
            timescale (optional; float):
                The length scale of the RBF kernel in second.
                If not specified, the minimal interval of data will be used.
            amp, phase (optional, boolean):
                If True, amp or phase gain solutions will be smoothed
            n_restarts_optimizer (optional, integer):
                The number of run for the Gaussian Process Regression.
                A larger number will reduce the risk of finding a local minima
                in the Gaussian Process Regression, but will also increase the
                computational time.
        Return:
            cltable object: new table with smoothed gains
        '''
        newcltable = copy.deepcopy(self)
        subarrids = self.gaintabs.keys()

        if (not amp) and (not phase):
            raise ValueError("Either of Amplitudes or Phases must be smoothed.")

        # Parameters for the gaussian processing reguressor
        gprargs = dict(
            length_scale=timescale,
            n_restarts_optimizer=n_restarts_optimizer
        )

        for subarrid in subarrids:
            print("Smoothing subarray %d"%(subarrid))
            # get gains
            gainall = self.gaintabs[subarrid]["gain"]
            Ndata,Nif,Nch,Npol,Nant,dummy = gainall.shape
            # get utc
            utcall = at.Time(self.gaintabs[1]["utc"], format="isot", scale="utc")

            # get all iterations
            Nit = Nif*Nch*Npol*Nant
            if Nit == 0:
                continue

            for iit in tqdm.tqdm(range(Nit)):
                iif,ich,ipol,iant = np.unravel_index(iit, [Nif,Nch,Npol,Nant])

                # get flag
                gsg = gainall[:,iif,ich,ipol,iant,2]
                idx = np.where(gsg>0)


                # get utc and other data
                utctmp = utcall[idx]
                if len(utctmp) == 0:
                    continue
                sectmp = utctmp.cxcsec
                sectmp0= sectmp.min()
                sectmp = sectmp - sectmp0

                # get gains
                gre = gainall[:,iif,ich,ipol,iant,0][idx]
                gim = gainall[:,iif,ich,ipol,iant,1][idx]
                gamp = np.sqrt(gre**2+gim**2)

                if amp:
                    gamp_pred = gp_interp(sectmp, gamp, **gprargs)
                else:
                    gamp_pred = gamp.copy()

                if phase:
                    gre_pred = gp_interp(sectmp, gre/gamp, **gprargs)
                    gim_pred = gp_interp(sectmp, gim/gamp, **gprargs)
                    ampratio = gamp_pred/np.sqrt(gre_pred**2+gim_pred**2)
                    gre_pred *= ampratio
                    gim_pred *= ampratio
                else:
                    gre_pred = gre/gamp*gamp_pred
                    gim_pred = gim/gamp*gamp_pred

                newcltable.gaintabs[subarrid]["gain"][idx,iif,ich,ipol,iant,0] = gre_pred
                newcltable.gaintabs[subarrid]["gain"][idx,iif,ich,ipol,iant,1] = gim_pred
        return newcltable


    def get_gaintable(self,uvfits):
        '''
        This method make a gain table with pandas.DataFrame.
        '''
        out = pd.DataFrame()

        # make gain table for each subarray
        subarrids = list(self.gaintabs.keys())
        for subarrid in subarrids:

            # get the number of data along each dimension
            Ntime,Nif,Nch,Nstokes,Nant,Ncomp = self.gaintabs[subarrid]["gain"].shape
            for iif,ich,istokes in itertools.product(list(range(Nif)),list(range(Nch)),list(range(Nstokes))):

                # get complex gain
                greal = self.gaintabs[subarrid]["gain"][:,iif,ich,istokes,:,0]
                gimag = self.gaintabs[subarrid]["gain"][:,iif,ich,istokes,:,1]
                gain = greal + 1j*gimag
                gain = pd.DataFrame(gain)
                gain.columns = list(range(1,Nant+1,1))

                # get the number of time stamps
                Ndata = len(self.gaintabs[subarrid]["utc"])

                # make gain table
                table = pd.DataFrame()
                table["utc"] = self.gaintabs[subarrid]["utc"]
                table["if"] = np.ones(Ndata,dtype=np.int)*(iif+1)
                table["ch"] = np.ones(Ndata,dtype=np.int)*(ich+1)
                table["stokes"] = np.ones(Ndata,dtype=np.int)*(istokes+1)
                table["subarray"] = np.ones(Ndata,dtype=np.int)*(subarrid)
                table = pd.concat([table,gain],axis=1)
                out = out.append(table)

        out = out.reset_index(drop=True)
        return out

    def rescale(self, mean=False):
        '''
        This method re-scales gain amplitude with median or mean value.
        If mean=False, it will take the median. Otherwise, it will take mean.
        '''
        out = copy.deepcopy(self)

        # make gain table for each subarray
        subarrids = list(self.gaintabs.keys())
        for subarrid in subarrids:
            # get mean amplitude
            greal = self.gaintabs[subarrid]["gain"][:,:,:,:,:,0]
            gimag = self.gaintabs[subarrid]["gain"][:,:,:,:,:,1]
            if mean:
                meanamp = np.average(np.sqrt(greal*greal + gimag*gimag))
            else:
                meanamp = np.median(np.sqrt(greal*greal + gimag*gimag))
            print(meanamp)

            # make rescaled gain
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,0] = greal/meanamp
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,1] = gimag/meanamp

        return out

    def gainplot(self,
            uvfits,
            station=None,
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
        Plot gains. Available types for the xaxis is
        ["utc","gst"]

        Args:
          station:
            station name or number to plot.
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        plttable = self.get_gaintable(uvfits)

        # Get GST
        if "gst" in axis1.lower():
            plttable = plttable.sort_values(by="utc").reset_index(drop=True)
            plttable["gst"] = self.get_gst_datetime(uvfits,continuous=gst_continuous, wraphour=gst_wraphour)

        gaintable = pd.DataFrame()
        stnameall = np.asarray([])

        # Antenna Name for Each Subarray
        subarrids = list(self.gaintabs.keys())
        for subarrid in subarrids:
            table = plttable[plttable["subarray"]==subarrid]
            namedic = uvfits.get_ant()
            subid = np.where(uvfits.visdata.coord.subarray.values == subarrid)
            st1 = set(np.int32(uvfits.visdata.coord.ant1.values[subid[0]]))
            st2 = set(np.int32(uvfits.visdata.coord.ant2.values[subid[0]]))
            st = np.asarray(sorted(st1 | st2))
            del st1, st2

            stname = np.asarray([namedic[(subarrid,i)] for i in st])
            stnameall = np.r_[stnameall,stname]
            stnameall = np.asarray(sorted(set(stnameall)))

            table = table.rename(columns=dict(list(zip(st, stname))))
            gaintable = pd.concat([gaintable,table],ignore_index=True)
            gaintable = gaintable.fillna(0)

        # Check if station are specified
        if station is None:
            stnum = st[0]
            stname = stnameall[0]
            print(type(stnum),type(stname))
        else:
            if isinstance(station, str):
                stname = np.string_(station)
                stnum = np.int32(st[stnameall==stname])
            else:
                stnum = np.int32(station)
                stname = np.string_(stnameall[st==stnum][0])

        # Plotting
        ax = plt.gca()

        # y value
        gain = gaintable[stname].values

        # x value
        if "gst" in axis1.lower():
            axis1data = gaintable.gst.values
        else:
            axis1data = gaintable.utc.values

        plt.subplots_adjust(hspace=0.)

        ax1 = plt.subplot(2,1,1)
        plt.title("%s"%(stname))
        print(stname,stnum)
        plt.plot(axis1data, abs(gain), ls=ls, marker=marker, label=label, **plotargs)
        plt.ylabel("gain amplitude [Jy]")
        plt.ylim(0,)
        ax1.set_xticklabels([])

        plt.subplot(2,1,2)
        plt.plot(axis1data, np.rad2deg(np.angle(gain)), ls=ls, marker=marker, label=label, **plotargs)
        plt.ylabel("gain phase [deg]")
        plt.ylim(-180,180)

        # x-tickes
        ax.xaxis.set_major_locator(time_maj_loc)
        ax.xaxis.set_minor_locator(time_min_loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_maj_fmt))
        if "gst" in axis1.lower():
            plt.xlabel("Greenwich Sidereal Time")
        else:
            plt.xlabel("Universal Time")

    def get_gst_datetime(self, uvfits, continuous=True, wraphour=0):
        '''
        get GST in datetime
        '''

        plttable = self.get_gaintable(uvfits)

        utc = at.Time(np.datetime_as_string(pd.to_datetime(plttable.utc.values)))
        gsthour = np.float64(utc.sidereal_time('apparent', 'greenwich').hour)

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
