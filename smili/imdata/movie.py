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
    def __init__(self, tstart=None, tend=None, Nt=None, format=None, **imgprm):
        '''
        This will create a blank movie on specified time frames.

        Args:
            tstart (format readable with astropy.time.Time):
                start time
            tend (datetime):
                end time
            Nt (integer):
                number of frames
            format (string):
                The format for tstart and tend.
                See documentations of astropy.time.Time for available
                data formats
            **imgprm:
                Parameters for the blank image at each frame.
                See documentations for imdata.IMFITS for parameters.
        Returns:
            imdata.MOVIE object
        '''
        if tstart is None:
            raise ValueError("tstart is not specified.")
        else:
            self.tstart = at.Time(tstart, format=format)

        if tend is None:
            raise ValueError("tend is not specified.")
        else:
            self.tend = at.Time(tend, format=format)

        if Nt is None:
            raise ValueError("tstart is not specified.")
        else:
            self.Nt = np.int64(Nt)

        tstsec = self.tstart.cxcsec
        tedsec = self.tend.cxcsec
        utcarr = np.linspace(tstsec,tedsec,self.Nt)
        utcarr = at.Time(utcarr, format="cxcsec")

        tmtable = pd.DataFrame()
        tmtable["utc"] = utcarr.datetime
        tmtable["gsthour"] = utcarr.sidereal_time("apparent", "greenwich").hour
        self.timetable = tmtable

        # Initialize images
        self.images = [imdata.IMFITS(**imgprm) for i in xrange(self.Nt)]

    def initmovie(self,image):
        '''
        image: imdata.IMFITS
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [copy.deepcopy(image) for i in xrange(self.Nt)]
        return outmovie


    def add_gauss(self,**gaussprm):
        '''
        add gayssian model to the initial movie
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].add_gauss(**gaussprm) for i in xrange(self.Nt)]
        return outmovie


    def winmod(self,imregion,save_totalflux=False):
        '''
        clear brightness distribution outside regions
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].winmod(imregion,save_totalflux) for i in xrange(self.Nt)]
        return outmovie

    def set_frmidx(self,uvtable):
        '''
        This method will put a frame index to the input uvtable, based on
        the time table of this movie object. If list of uvtables are given,
        they will be concatenated into a single uvtable.

        Args:
            uvtable or list of uvtables:
                uvtable could be vistable, amptable, bstable, catable
        Returns:
            concatenated uvtable
        '''
        # create a table to be output
        outtable = copy.deepcopy(uvtable)

        # concat tables if list is given.
        if type(outtable) == list:
            outtable = pd.concat(outtable, ignore_index=True)

        # time sort and re-index table
        outtable.sort_values(by="utc", inplace=True)  # Sorting
        outtable.reset_index(drop=True, inplace=True) # reindexing

        # get an array and non-redundant array of utc time stamps in uvtable
        dutcarr = at.Time(np.datetime_as_string(outtable.utc.values))
        dutcset = at.Time(np.datetime_as_string(outtable.utc.unique()))

        # get an array of utc time stamps of this movie
        mutcarr = at.Time(np.datetime_as_string(self.timetable.utc.values))

        outtable["frmidx"] = np.zeros(len(outtable.utc), dtype='int32')
        for i in xrange(len(dutcset)):
            deltat = mutcarr - dutcset[i]
            deltat.format = "sec"
            deltat = np.abs(deltat.value)
            outtable.loc[dutcarr==dutcset[i], "frmidx"] = np.argmin(deltat)

        outtable.sort_values(by=["frmidx", "utc", "stokesid", "ch", "st1", "st2"], inplace=True)
        outtable.reset_index(drop=True, inplace=True)
        return outtable

    def to_movie(self, filename, vmin=0, vmax=None, vmax_type=None, **imshowprm):
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
