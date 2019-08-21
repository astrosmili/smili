#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from tqdm import tqdm

# numerical packages
import numpy as np
import pandas as pd
import scipy.ndimage as sn

# astropy
import astropy.time as at
import astropy.coordinates as coord
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

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
    def __init__(self, timetable=None, tcen=None, tint=None, **imgprm):
        '''
        This will create a blank movie on specified time frames.

        Args:
            tcen (format readable with astropy.time.Time):
                the central time of each time frame
            tint (float):
                the integration time of each time frame in sec
            **imgprm:
                Parameters for the blank image at each frame.
                See documentations for imdata.IMFITS for parameters.
        Returns:
            imdata.MOVIE object
        '''
        if timetable is not None:
            tmtable = pd.read_csv(timetable)
            if "utc" in tmtable.columns:
                tmtable["utc"] = at.Time(tmtable["utc"].values.tolist()).datetime
        else:
            if tcen is None:
                raise ValueError("tcen is not specified.")
            else:
                utc = at.Time(tcen)

            tmtable = pd.DataFrame()
            tmtable["utc"]     = utc.datetime
            tmtable["gsthour"] = utc.sidereal_time("apparent", "greenwich").hour
            self.Nt = len(utc)

            if len(utc) < 2:
                print("Warning: You have only one frame!")

            if len(utc) == 0:
                raise ValueError("No time frame was input.")

            if hasattr(tint, "__iter__"):
                if len(utc) != len(tint):
                    raise ValueError("len(utc) != len(tint)")
                else:
                    tmtable["tint"] = tint
            else:
                tmtable["tint"] = [tint for i in range(self.Nt)]
        self.timetable = tmtable

        # Initialize images
        self.images = [imdata.IMFITS(**imgprm) for i in range(self.Nt)]

    #---------------------------------------------------------------------------
    # Get some information of movie data
    #---------------------------------------------------------------------------
    def get_utc(self):
        '''
        get utc in astropy.time.Time object
        '''
        return at.Time(np.datetime_as_string(self.timetable["utc"]), scale="utc")

    def get_gst_datetime(self, continuous=False, wraphour=0):
        '''
        get GST in datetime
        '''
        Ndata = len(self.utc)

        utc = self.get_utc()
        gsthour = self.timetable.gsthour.values

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

    def average(self):
        '''
        Get an averaged image.

        Args:
            movie_image (IMFITS):
            filename
        Returns:
            ave_movie (np.array)
        '''
        outimage = copy.deepcopy(self.images[0])
        outimage.data[0,0] = self.get_3darray().mean(axis=0)
        outimage.update_fits()
        return outimage

    def get_3darray(self):
        '''
        Output 3-dimensional array for movie. Its dimension will be [Nt, Ny, Nx].

        Returns:
            3d ndarray object contains 3-dimensional images
        '''
        return np.asarray([self.images[i].data[0,0] for i in range(self.Nt)])

    def get_lightcurve(self):
        '''
        Args:
            movie_list
        Returns:
        '''
        movie = self.get_3darray()
        lightcurve = movie.sum(axis=2)
        lightcurve = lightcurve.sum(axis=1)
        return lightcurve

    #---------------------------------------------------------------------------
    # Edit Movies
    #---------------------------------------------------------------------------
    def set_image(self, image):

        '''
        This will make the initial movie by superposing input images

        Args:
            image (imdata.IMFITS)

        Returns:
            imdata.MOVIE ovbject
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [copy.deepcopy(image) for i in range(self.Nt)]
        return outmovie


    def add_gauss(self,**gaussprm):

        '''
        add a Gaussian model to the initial movie

        Args:
            gaussprm

        Returns:
            imdata.MOVIE object including gaussian model
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].add_gauss(**gaussprm) for i in range(self.Nt)]
        return outmovie

    def winmod(self,imregion,save_totalflux=False):

        '''
        clear brightness distribution outside regions

        Args:
            region (imdata.ImRegTable object):
                region data
            save_totalflux (boolean; default=False):
                if True, keep Totalflux
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].winmod(imregion,save_totalflux) for i in range(self.Nt)]
        return outmovie

    def min_threshold(self,  threshold=0.0, replace=0.0,
                      relative=True, save_totalflux=False):
        '''
        This is thresholding with the mininum value. This is slightly different
        from hard thresholding, since this function resets all of pixels where
        their brightness is smaller than a given threshold. On the other hand,
        hard thresholding resets all of pixels where the absolute of their
        brightness is smaller than the threshold.

        Args:
          threshold (float): threshold
          replace (float): the brightness to be replaced for thresholded pixels.
          relative (boolean): If true, theshold & Replace value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''


        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].min_threshold(
                                threshold=threshold,
                                relative=relative,
                                replace=replace,
                                save_totalflux=save_totalflux)
                           for i in range(self.Nt)]
        return outmovie

    def hard_threshold(self, threshold=0.01, relative=True, save_totalflux=False):

        '''
        Do hard-threshold the input image

        Args:
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''


        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].hard_threshold(
                                threshold=threshold,
                                relative=relative,
                                save_totalflux=save_totalflux)
                           for i in range(self.Nt)]
        return outmovie

    def soft_threshold(self, threshold=0.01, relative=True, save_totalflux=False):

        '''
        Do soft-threshold the input image

        Args:
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].soft_threshold(
                                threshold=threshold,
                                relative=relative,
                                save_totalflux=save_totalflux)
                           for i in range(self.Nt)]
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
                           for i in range(self.Nt)]
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
        pos = self.average().peakpos(ifreq=ifreq, istokes=istokes)
        return self.refshift(save_totalflux=save_totalflux, **pos)

    def convolve_gauss(self, majsize, minsize=None, x0=None, y0=None, pa=0.0,
                             scale=1.0, angunit=None, save_totalflux=False):
        '''
        '''
        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].convolve_gauss(
                                majsize=majsize,
                                minsize=minsize,
                                x0=x0, y0=y0,
                                pa=pa, scale=scale,
                                angunit=angunit,
                                save_totalflux=save_totalflux)
                           for i in range(self.Nt)]
        return outmovie


    #---------------------------------------------------------------------------
    # Edit other objects
    #---------------------------------------------------------------------------
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
        dutcarr = outtable.get_utc().cxcsec
        dutcset = outtable.drop_duplicates(subset="utc").get_utc().cxcsec

        # get an array of utc time stamps of this movie
        mutcarr = self.get_utc().cxcsec
        mutcint = self.timetable.tint

        outtable["frmidx"] = np.zeros(len(outtable.utc), dtype='int32')
        outtable.loc[:, "frmidx"] = -1
        for i in range(len(dutcset)):
            idx = np.abs(mutcarr - dutcset[i]) < mutcint/2
            if True in idx:
                outtable.loc[dutcarr==dutcset[i], "frmidx"] = np.where(idx)[0][0]

        if -1 in outtable.frmidx.values:
            print("WARNING: there are data sets not on movie frames. Please check time coverages of data and movie")
        outtable.sort_values(by=["frmidx", "utc", "stokesid", "ch", "st1", "st2"], inplace=True)
        outtable.reset_index(drop=True, inplace=True)
        return outtable

    def split_uvfits(self, uvfitslist):
        '''
        Args:

        Returns:
        '''
        # Number of uvfitslist components
        Nuvfits = len(uvfitslist)

        # uvfits list and iuvfits list for each frame
        uvfits_framelist_list =[]
        uvfits_idlist_list=[]

        utc = self.get_utc()
        dutc = utc[1:] - utc[:-1]
        utcbound = utc[:-1]+dutc/2

        print("Split uvfits objects to each time frame.")
        for ifrm in tqdm(list(range(self.Nt))):
            # list of uvframe components and iuvfits in a frame
            uvfits_framelist=[]
            uvfits_idlist=[]

            for iuvfits in range(Nuvfits):
                uvfits_frm = copy.deepcopy(uvfitslist[iuvfits])
                Ndata = len(uvfits_frm.visdata.coord["utc"])
                utc_uvfits = uvfits_frm.get_utc()

                idx = np.array([True for i in range(Ndata)])
                if ifrm > 1:
                    idx &= utc_uvfits > utcbound[ifrm-1]
                if ifrm < self.Nt-1:
                    idx &= utc_uvfits < utcbound[ifrm]

                if True in idx:
                    uvfits_frm.visdata.data  = uvfits_frm.visdata.data[np.where(idx)]
                    uvfits_frm.visdata.coord = uvfits_frm.visdata.coord.loc[idx, :].reset_index(drop=True)
                    uvfits_framelist.append(uvfits_frm)
                    uvfits_idlist.append(iuvfits)

            if len(uvfits_framelist)==0:
                uvfits_framelist = None
                uvfits_idlist = None
            uvfits_framelist_list.append(uvfits_framelist)
            uvfits_idlist_list.append(uvfits_idlist)
        return uvfits_framelist_list,uvfits_idlist_list

    def selfcal(self,uvfitslist,std_amp=100,std_pha=100):
        '''
        This perform a self calibration to concatenated uvfits (uvfits_frame_cal)
        and make list of cltable

        Args:

        Returns:
        '''
        Nuvfits = len(uvfitslist)
        uvfits_framelist_list,uvfits_idlist_list=self.split_uvfits(uvfitslist)

        cltable_list_list = []
        print("Start selfcal")
        for it in tqdm(list(range(self.Nt))):
            if uvfits_framelist_list[it] is None:
                cltable_list_list.append(None)
                continue
            cltable_list = [uvfits.selfcal(self.images[it],std_amp=std_amp,std_pha=std_pha) for uvfits in uvfits_framelist_list[it]]
            cltable_list_list.append(cltable_list)
        return uvfits_framelist_list,uvfits_idlist_list,cltable_list_list

    #---------------------------------------------------------------------------
    # Load data
    #---------------------------------------------------------------------------
    def read_fits(self, filenames, check=True, **imgprm):
        '''
        load images from list of filenames

        Args:
            filenames (list of str):
                list of filenames
            check (boolean; default=True)
                check if number of files are consistent with self.Nt
            **imgprm: other parameters of imdata.IMFITS
        '''
        if check and self.Nt != len(filenames):
            raise ValueError("The number of files are not consistent with the number of time frames.")

        outmovie=copy.deepcopy(self)
        outmovie.images = [imdata.IMFITS(filename) for filename in filenames]
        return outmovie

    def cpmovie(self, oldmovie, save_totalflux=False, kind="cubic"):

        '''
        Copy the brightness ditribution of the input MOVIE object
        into the image grid of this movie data.

        Args:
            oldmovie (imdata.MOVIE object):
                The movie will be copied into the movie grid of this data.
            save_totalflux (boolean):
                If true, the total flux of each image is conserved.
            kind (string): kind of interpolation whose definition is the same as scypy functions
                                (e.g, 'linear', 'nearest', 'quadratic', 'cubic', ...")

        Returns:
            imdata.MOVIE object: the copied movie

        '''
        told=oldmovie.get_utc().cxcsec
        tref=self.get_utc().cxcsec
        tzero = told[0]
        told -= tzero
        tref -= tzero

        Nt=oldmovie.Nt
        Ny,Nx=oldmovie.images[0].data[0,0].shape
        Ntref=self.Nt

        oldimage = copy.deepcopy(oldmovie.images[0])
        refimage = copy.deepcopy(self.images[0])
        imold = np.zeros((Nt, Ny, Nx), dtype=np.float64)
        imref = np.zeros((Ntref, Ny, Nx),dtype=np.float64)

        # Set array of the old movie
        for it in range(Nt):
            imold[it,:,:] = oldmovie.images[it].data[0,0,:,:]

        # Interpolation to the time direction
        iterator=itertools.product(range(Ny), range(Nx))
        for iy,ix in iterator:
            imref[:,iy,ix]=util.interpolation1d(
                told, imold[:,iy,ix], tref, kind=kind, fill_value="extrapolate",
            ) # Ntref, Nx, Ny

        # Interpolation to spatial directions
        newmovie = copy.deepcopy(self)
        for itref in range(Ntref):
            oldimage.data[0,0,:,:] = imref[itref, :, :]
            newmovie.images[itref]=refimage.cpimage(oldimage,save_totalflux=save_totalflux)
        return newmovie

    def copy(self):

        '''
        Copy the movie object
        '''

        outmovie=copy.deepcopy(self)
        outmovie.images = [copy.deepcopy(self.images[it]) for it in range(self.Nt)]

        return outmovie

    #---------------------------------------------------------------------------
    # Output
    #---------------------------------------------------------------------------
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
            vmax = self.get_3darray().max()
        fig = plt.figure()
        with writer.saving(fig, filename, dpi):
            for it in range(self.Nt):  # 回数
                if(vmax_type is "eachtime"):
                    vmax =lightcurve()[it]
                    self.images[it].imshow(vmin=vmin, vmax=vmax, **imshowprm)
                else:
                    self.images[it].imshow(vmin=vmin, vmax=vmax, **imshowprm)
                writer.grab_frame()

    def to_csv(self, filename):
        '''
        Save the time table to a csv file
        '''
        self.timetable.to_csv(filename)

    def to_fits(self,header="movie",ext="fits"):
        '''
        Save movies into series of image FITS files.
        The filename would be header + ".%03d." + ext.
        (e.g. movie.001.fits)



        Args:
            header (str; default="movie"): header of the filename
            ext (str; default="fits"): extention of the file
        Returns:
            filename+"%03d.fits"%(it)
        '''
        for it in range(self.Nt):
            self.images[it].to_fits(header+".%03d."%(it)+ext)

    #---------------------------------------------------------------------------
    # Plotting
    #---------------------------------------------------------------------------
    def plot_lightcurve(self,
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
        Args:
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        timetable = copy.deepcopy(self.timetable)
        timetable["totalflux"] = self.get_lightcurve() * util.fluxconv("jy", fluxunit)
        if "gst" in axis1.lower():
            timetable["gst"] = self.get_gst_datetime(continuous=gst_continuous, wraphour=gst_wraphour)
            timetable.sort_values(by="gst", inplace=True)
            axis1data = timetable.gst.values
        else:
            timetable.sort_values(by="utc", inplace=True)
            axis1data = timetable.utc.values

        # Plotting
        ax = plt.gca()
        plt.plot(axis1data, timetable.totalflux.values, ls=ls, marker=marker, label=label, **plotargs)

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

    def plot_tspan(self,
            time_maj_loc=mdates.HourLocator(),
            time_min_loc=mdates.MinuteLocator(byminute=np.arange(0,60,10)),
            time_maj_fmt='%H:%M',
            colors=None,
            alpha=0.2,
            **plotargs):
        '''
        Args:
          **plotargs:
            You can set parameters of matplotlib.pyplot.plot() or
            matplotlib.pyplot.errorbars().
            Defaults are {'ls': "none", 'marker': "."}.
        '''
        if colors is None:
            colors = cm.jet(np.linspace(0,1,self.Nt))

        utccen = self.get_utc()
        tint = at.TimeDelta(self.timetable.tint.values, format="sec")/2
        utcst = utccen - tint
        utced = utccen + tint


        # Plotting
        ax = plt.gca()
        for i in range(self.Nt):
            ax.axvspan(xmin=utcst[i].datetime, xmax=utced[i].datetime, alpha=alpha, color=colors[i])

    def imshow(self, it,**imshowprm):
        image=self.images[it]
        image.imshow(**imshowprm)

def concat_uvfits(uvfits_framelist_list,uvfits_idlist_list):

    '''
    This Concatenate the components of uvfits_frame list,
    sort by utc time, and tag index of uvfits components.

    Args:

    Returns:
    '''

    Nt      = len(uvfits_idlist_list)
    uvfits_ids = uvfits_idlist_list.copy()
    while None in uvfits_ids:
        uvfits_ids.remove(None)
    Nuvfits = np.max(uvfits_ids)+1
    uvfits_con_list = [None for iuvfits in range(Nuvfits)]

    print("concatenate uvfits files")
    for it in tqdm(list(range(Nt))):
        uvfits_idlist = uvfits_idlist_list[it]
        uvfits_framelist = uvfits_framelist_list[it]

        if uvfits_idlist is None:
            continue
        elif len(uvfits_idlist)==0:
            continue

        Nuvfits_split = len(uvfits_idlist)
        for iuvfits in range(Nuvfits_split):
            id = uvfits_idlist[iuvfits]
            uvfits = uvfits_framelist[iuvfits]
            if uvfits_con_list[id] is None:
                uvfits_con_list[id] = copy.deepcopy(uvfits)
            else:
                uvfits_con_list[id].visdata.data = np.concatenate(
                    [uvfits_con_list[id].visdata.data,uvfits.visdata.data]
                )
                uvfits_con_list[id].visdata.coord = pd.concat(
                    [uvfits_con_list[id].visdata.coord,uvfits.visdata.coord],
                    ignore_index=True
                )

    while None in uvfits_con_list:
        uvfits_con_list.remove(None)

    print("sort uvfits files")
    for iuvfits in tqdm(list(range(len(uvfits_con_list)))):
        uvfits_con_list[iuvfits].visdata.sort()

    return uvfits_con_list

def apply_cltable(uvfits_framelist_list,uvfits_idlist_list,cltable_list_list):

    '''
    This makes a list of uvfits objects by performing a self calibration for all frames

    Args:

    Returns:
    '''
    Nt = len(uvfits_framelist_list)

    uvfitscal_framelist_list=[]

    for it in tqdm(list(range(Nt))):
        if uvfits_framelist_list[it] is None:
            uvfitscal_framelist_list.append(None)
            continue

        Nuvfits = len(uvfits_framelist_list[it])
        uvfits_framelist = uvfits_framelist_list[it]
        caltable_list = cltable_list_list[it]

        uvfitscal_framelist = [uvfits_framelist[iuvfits].apply_cltable(caltable_list[iuvfits]) for iuvfits in range(Nuvfits)]
        uvfitscal_framelist_list.append(uvfitscal_framelist)

    uvfits_totlist=concat_uvfits(uvfitscal_framelist_list,uvfits_idlist_list)
    return uvfits_totlist
