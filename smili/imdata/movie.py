#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a sub-module of sparselab handling dynamical imaging.
'''
__author__ = "Sparselab Developer Team"
#-------------------------------------------------------------------------
# Modules
#-------------------------------------------------------------------------
# standard modules
import os
import copy
import datetime as dt

# numerical packages
import numpy as np
import pandas as pd
import scipy.ndimage as sn
import astropy.time as at
import astropy.coordinates as coord
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft

# matplotlib
import matplotlib.pyplot as plt

# for to_movie
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation

# for lightcurve
import itertools
# internal
from .. import fortlib, util
from . import imdata
#-------------------------------------------------------------------------
# IMAGEFITS (Manupulating FITS FILES)
#-------------------------------------------------------------------------
class MOVIE(object):
    def __init__(self, tstart=None, tend=None, Nt=None, tint=None, tunit="sec", **imgprm):
        '''
        Args:
            tstart (datetime):
                start time
            tend (datetime):
                end time
            tint (float):
                constant time span of each frame (sec)
            tunit (string):
                unit of time difference (sec, min, hrs, day)
            Nt (integer):
                number of frames
            init2dim (imdata.IMFITS object):
                initial image
        Returns:
            imdata.MOVIE object
        '''

        if (tint is not None):
            if tunit == "sec":
                self.tint = at.TimeDelta(tint, format='sec')
            elif tunit == "min":
                self.tint = at.TimeDelta(tint*60, format='sec')
            elif tunit == "hrs":
                self.tint = at.TimeDelta(tint*3600, format='sec')
            elif tunit == "day":
                self.tint = at.TimeDelta(tint*3600*24, format='sec')
        # assigning the input Nt

        # Check tstart
        if tstart is None:
            raise ValueError("tstart is not specified.")
        else:
            self.tstart = at.Time(tstart)
        # Check tend
        if (tend is not None) and (Nt is not None) and (tint is not None):
            raise ValueError("All of tend, Nt and tint are specified. We need only two of them.")
        elif (tend is None) and (Nt is not None) and (tint is not None):
            self.Nt   = Nt
            self.tint = self.tint
        elif (tend is not None) and (Nt is None) and (tint is not None):
            self.tint = self.tint
            # Guess Nt
            tdif = (at.Time(tend) - self.tstart)
            self.Nt = int(tdif.sec/self.tint.value) + 1
        elif (tend is not None) and (Nt is not None) and (tint is None):
            self.Nt = Nt
            # Guess tint
            tend    = at.Time(tend)
            tdif = (at.Time(tend) - self.tstart)
            tint = (tdif.sec)/Nt
            self.tint = at.TimeDelta(tint, format='sec')
            # tintがインプットされていない場合、tunitは意味をなさないことに注意
        else:
            raise ValueError("Two of tend, Nt and tint must be specifed.")

        tmtable = pd.DataFrame()
        tmtable["utc"] = np.zeros(self.Nt)
        for it in xrange(self.Nt):
            tmtable.loc[it, "utc"] = it*self.tint+self.tstart
        self.timetable = tmtable

        # Initialize images
        self.images = [imdata.IMFITS(**imgprm) for i in xrange(self.Nt)]

        # Test images
        #beamprm={}
        #u'minsize': 13.109819596064836, u'pa': 35.655022685947685, u'majsize': 13.109819596064836, u'angunit': 'uas'}
        #self.images = [imdata.IMFITS(**imgprm).add_gauss(x0=0., totalflux=1., **beamprm) for i in xrange(self.Nt)]

    def initmovie(self,image):
        '''
        image: imdata.IMFITS
        '''
        outmovie = copy.deepcopy(self)
        for it in xrange(self.Nt):
            outmovie.images[it]=image
            outmovie.images[it].update_fits()
        return outmovie

    def add_gauss(self,**beamprm):
        '''
        add gayssian model to the initial movie
        '''
        outmovie=copy.deepcopy(self)
        for it in xrange(self.Nt):
            outmovie.images[it]=self.images[it].add_gauss(**beamprm)
            outmovie.images[it].update_fits()
        return outmovie


    def winmod(self,imregion,save_totalflux=False):
        '''
        clear brightness distribution outside regions
        '''
        outmovie=copy.deepcopy(self)
        for it in xrange(self.Nt):
            outmovie.images[it]=self.images[it].winmod(imregion,save_totalflux)
            outmovie.images[it].update_fits()
        return outmovie


    def timetable2(self): # ilje型
        tstart = self.tstart # at.TImeで変換済み
        Ntr    = self.Nt

        tmtable = pd.DataFrame()
        tmtable["frame"] = np.zeros(self.Nt, dtype='int32')
        tmtable["utc"] = np.zeros(self.Nt)
        tmtable["gsthour"] = np.zeros(self.Nt)
        tmtable["tint(sec)"] = np.zeros(self.Nt)
        for i in np.arange(Ntr):
            tmtable.loc[i, "frame"] = int(i)
            centime = tstart + self.tint*i
            utctime = centime.datetime
            gsthour = centime.sidereal_time("apparent", "greenwich").hour
            tmtable.loc[i, "utc"] = utctime
            tmtable.loc[i, "gsthour"] = gsthour
            tmtable.loc[i, "tint(sec)"] = self.tint
        return tmtable


    def tabconcat(self,dtable):
        '''
        concatenate table

        Args
            dtable:
                vistable, amptable, bstable, catable
        '''
        if (dtable is None) or (dtable is [None]):
            print("DataFrame table is not given.")
            return -1
        # concatenate the multiple tables in a list
        if type(dtable) == list:
            tablist = dtable
        else:
            tablist = list([dtable])
        frmtable = None
        for tab in tablist:
            if frmtable is not None:
                frmtable = pd.concat((frmtable, tab), ignore_index=True)
            else:
                frmtable = tab
        return frmtable



    def set_fridx(self,dtable):
        '''
        add the frame index to the concatenated table
        Args
            dtable:
                vistable, amptable, bstable, catable

        '''

        frmtable = self.tabconcat(dtable)

        # time of input table which DataFrame
        attime = np.asarray(frmtable["utc"], np.str)
        attime = at.Time(attime)
        utctime = attime.datetime
        # call timetable
        tmtable = self.timetable2()
        idx = tmtable["frame"].values
        tmframe = np.asarray(tmtable["utc"], np.str)
        tmframe = at.Time(tmframe)
        tmframe = tmframe.datetime
        # assigning the frame index
        frmtable["frmidx"] = np.zeros(len(frmtable), dtype='int32')
        for i in range(len(utctime)):
            for j in range(len(idx)-1):
                if (utctime[i] >= tmframe[j]) and (utctime[i] < tmframe[j+1]):
                    frmtable.loc[i, "frmidx"] = idx[j]

#            if utctime[i] >= tmframe[-1]:
#                frmtable.loc[i, "frmidx"] = idx[-1]

        frmtable = frmtable.sort_values(by=["frmidx", "utc", "stokesid", "ch", "st1", "st2"]).reset_index(drop=True)
        return frmtable



    def to_movie(self, filename, vmin=0, vmax=None,vmax_type=None, **imshowprm):
        '''
        Args:
            movie_list list(Nt*(IMFITS))?:
            logscale
            xregion, yregion
            filename
        Returns:
            movie.mp4
        '''

        FFMpegWriter = manimation.writers['ffmpeg']
        #metadata = dict(title='Movie', artist='Matplotlib',
        #                comment='Movie support!')
        metadata = dict(title='Movie')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        if vmax is None:
            vmax = self.to_3darray().max()
        fig = plt.figure()
#        plt.xlim(-xregion, xregion) # xrange
#        plt.ylim(-yregion, yregion) # yrange

        with writer.saving(fig, filename, 100):  # 最後の数字が解像度?
            for it in xrange(self.Nt):  # 回数
                if(vmax_type is "eachtime"):
                    vmax =lightcurve()[it]
                    self.images[it].imshow(vmin=vmin, vmax=vmax, **imshowprm)
                else:
                    self.images[it].imshow(vmin=vmin, vmax=vmax, **imshowprm)
                writer.grab_frame()


    #========================
    def to_fits(self,filename="snapshot"):
        '''
        Args:
            filename
        Returns:
            filename+"%03d.fits"%(it)
        '''
        for it in xrange(self.Nt):
            self.images[it].save_fits(filename+"%03d.fits"%(it))

    def average(self):
        '''
        Args:
            movie_image (IMFITS):
            filename
        Returns:
            ave_movie (np.array)
        '''
        outimage = copy.deepcopy(self.images[0])
        aveimage = self.images[0].data[0,0,:,:]
        for it in xrange(self.Nt):
            aveimage += self.images[it].data[0,0,:,:]
        aveimage /= self.Nt
        outimage.data[0,0] = aveimage
        outimage.update_fits()
        return outimage

    def to_3darray(self):
        '''
        Output 3-dimensional array for movie. Its dimension will be [Nt, Ny, Nx].

        Returns:
            3d ndarray object contains 3-dimensional images
        '''
        return np.asarray([self.images[i].data[0,0] for i in xrange(self.Nt)])

    def lightcurve(self):
        '''
        Args:
            movie_list
        Returns:
        '''
        movie = self.to_3darray()
        lightcurve = movie.sum(axis=2)
        lightcurve = lightcurve.sum(axis=1)
        return lightcurve

    def plot_lc(self):
        Nt = self.Nt
        lightcurve=self.lightcurve()

        time = np.zeros(Nt)
        for it in xrange(Nt):
            time[it]=it*self.tint.value
        plt.plot(time,lightcurve)


    def imshow(self,it,**imshowprm):
        image=self.images[it]
        image.imshow(**imshowprm)


    def initimlist(self):
        pass
        #mul2dim = list([self.init2dim])*Nt
        #return mul2dim
