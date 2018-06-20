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

    def hard_threshold(self, threshold=0.01, relative=True, save_totalflux=False):
        '''
        hard_thresholding
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].hard_threshold(
                                threshold=threshold,
                                relative=relative,
                                save_totalflux=save_totalflux)
                           for i in xrange(self.Nt)]
        return outmovie

    def soft_threshold(self, threshold=0.01, relative=True, save_totalflux=False):
        '''
        soft_thresholding
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].soft_threshold(
                                threshold=threshold,
                                relative=relative,
                                save_totalflux=save_totalflux)
                           for i in xrange(self.Nt)]
        return outmovie

    def refshift(self,x0=0.,y0=0.,angunit=None,save_totalflux=False):
        '''
        Shift the reference position to the specified coordinate.
        The shift will be applied to all frames.

        Args:
          x0, y0 (float, default=0):
            RA, Dec coordinate of the reference position
          angunit (string, optional):
            The angular unit of the coordinate. If not specified,
            self.images[0].angunit will be used.
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.
        Returns:
          imdata.Movie object
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].refshift(
                                x0=x0, y0=y0, angunit=angunit,
                                save_totalflux=save_totalflux)
                           for i in xrange(self.Nt)]
        return outmovie

    def comshift(self, alpha=1., save_totalflux=False):
        '''
        Shift all images so that the center-of-mass position of the averaged
        image will coincides with the reference pixel.

        Args:
          alpha (float):
            if alpha != 0, then the averaged image is powered by alpha prior
            to compute the center of the mass.
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.Movie object
        '''
        pos = self.average().compos(alpha=alpha)
        return self.refshift(save_totalflux=save_totalflux, **pos)

    def peakshift(self, save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift all images so that the peak position of the averaged
        image will coincides with the reference pixel.

        Arg:
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.Movie object
        '''
        pos = self.average().peakpos(alpha=alpha)
        return self.refshift(save_totalflux=save_totalflux, **pos)

    def gauss_convolve(self, majsize, minsize=None, x0=None, y0=None, pa=0.0, scale=1.0, angunit=None, pos='rel', save_totalflux=False):
        '''

        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].gauss_convolve(
                                majsize=majsize,
                                minsize=minsize,
                                x0=x0, y0=y0,
                                pa=pa, scale=scale,
                                angunit=angunit, pos=pos,
                                save_totalflux=save_totalflux)
                           for i in xrange(self.Nt)]
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

    def to_movie(self, filename, vmin=0, vmax=None, vmax_type=None, fps=15, dpi=100, **imshowprm):
        '''
        Args:
            filename (string; mandatory):
                output filename
            vmin, vmax:
            logscale
            xregion, yregion
            filename
        Returns:
            movie.mp4
        '''

        FFMpegWriter = manimation.writers['ffmpeg']
        #metadata = dict(title='Movie', artist='Matplotlib',
        #                comment='Movie support!')
        writer = FFMpegWriter(fps=fps)

        if vmax is None:
            vmax = self.to_3darray().max()
        fig = plt.figure()
        with writer.saving(fig, filename, dpi):
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
