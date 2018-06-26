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
        tmtable["utc"]     = utcarr.datetime
        tmtable["gsthour"] = utcarr.sidereal_time("apparent", "greenwich").hour
        self.timetable = tmtable

        # Initialize images
        self.images = [imdata.IMFITS(**imgprm) for i in xrange(self.Nt)]

    def initmovie(self,image):

        '''
        This will make the initial movie by superposing input images

        Args:
            image (imdata.IMFITS)

        Returns:
            imdata.MOVIE ovbject
        '''

        outmovie=copy.deepcopy(self)
        outmovie.images = [copy.deepcopy(image) for i in xrange(self.Nt)]
        return outmovie


    def add_gauss(self,**gaussprm):

        '''
        add gayssian model to the initial movie

        Args:
            gaussprm

        Returns:
            imdata.MOVIE object including gaussian model
        '''

        outmovie=copy.deepcopy(self)
        outmovie.images = [self.images[i].add_gauss(**gaussprm) for i in xrange(self.Nt)]
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
        outmovie.images = [self.images[i].winmod(imregion,save_totalflux) for i in xrange(self.Nt)]
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
                           for i in xrange(self.Nt)]
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


    def save_fits(self,filename=None):
        if filename is None:
            filename = "%0"+"%d"%(np.int64(np.log10(self.Nt)+1))+"d.fits"
        for i in xrange(self.Nt):
            self.images[i].save_fits(filename%(i))

    # load関数の原型
    def load_fits(self,filename=None):
        loadmovie=copy.deepcopy(self)
        if filename is None:
            filename = "%0"+"%d"%(np.int64(np.log10(self.Nt)+1))+"d.fits"
        for i in xrange(self.Nt):
            loadmovie.images[i]=imdata.IMFITS(filename%(i))
            loadmovie.images[i].update_fits()
        return loadmovie

    def split_uvfits(self,iframe,uvfitslist):

        '''
        This extracts uvfits components in a frame denoted by iframe

        Args:
            iframe (int):
            frame id
            uvfitslist (list of uvfits objects):

        Returns:
            uvfits_framelist (list of uvfits object):
            list of uvfits including in iframe
        '''

        uvfits_framelist=[]
        # Number of uvfitslist components
        Nuvfits = len(uvfitslist)

        # extract uvfits in a frame
        for iuvfits in xrange(Nuvfits):
            uvfits= copy.deepcopy(uvfitslist[iuvfits])
            Ndata=len(uvfits.visdata.coord["utc"])
            # initialize extracted uvfits
            uvfits_frame = copy.deepcopy(uvfits)
            uvfits_frame.visdata.coord= pd.DataFrame([])
            istart=0
            idatalist=[]

            # extract uvfits in a frame denoted by iframe
            for idata in xrange(Ndata):
                tt        = uvfits.visdata.coord["utc"][idata].value
                deltatmin = self.timetable["utc"].max().value#-self.timetable["utc"].min().value
                deltatmax = deltatmin
                tfrm      = self.timetable["utc"][iframe].value
                deltat    = np.abs(tfrm-tt)

                if(iframe-1>=0):
                    tmin      = self.timetable["utc"][iframe-1].value
                    deltatmin = np.abs(tmin-tt)

                if(iframe+1<self.Nt):
                    tmax      = self.timetable["utc"][iframe+1].value
                    deltatmax = np.abs(tmax-tt)

                if(deltat<deltatmin and deltat<deltatmax):
                    idatalist = idatalist+[idata]

            idatalist = np.array(idatalist)
            print("uvfits%d/%d: Number of extracted data=%d"%(iuvfits,Nuvfits,len(idatalist)))
            if(len(idatalist)>0):
                idmin     = idatalist.min()
                idmax     = idatalist.max()

            print("lower and upper limit of data number=%d %d"%(idmin,idmax))

            # make uvfitslist of extracted uvfits components
            uvfits_frame.visdata.data  = uvfits.visdata.data[idmin:idmax+1]
            uvfits_frame.visdata.coord = uvfits.visdata.coord.loc[idmin:idmax]
            uvfits_framelist=uvfits_framelist+[uvfits_frame]

        return uvfits_framelist

    def concat_uvfits(self,uvfits_framelist):

        '''
        This Concatenate the components of uvfits_frame list,
        sort by utc time, and tag index of uvfits components.

        Args:
            uvfits_framelist (list of uvfits constructed in split_uvfits)

        Returns:
            uvfits_frame_con (uvfits):
            Concatenated and sorted uvfits object
        '''

        # initialize concatenated uvfits (uvfits_frame_con)
        uvfits_frame_con = copy.deepcopy(uvfits_framelist[0])
        uvfits_frame_con.visdata.coord= pd.DataFrame([])
        # initialize index of each component uvfits_frame list
        uv_idx = pd.DataFrame([])
        Nuvfits = len(uvfits_framelist)
        istart = 0
        # Concatenate each component of uvfits_frame list
        for iuvfits in xrange(Nuvfits):
            uvfits = copy.deepcopy(uvfits_framelist[iuvfits])
            if(istart==0):
                uvfits_frame_con.visdata.data=uvfits.visdata.data
                istart=1
            else:
                uvfits_frame_con.visdata.data = np.concatenate((uvfits_frame_con.visdata.data,uvfits.visdata.data))

            uvfits_frame_con.visdata.coord     = uvfits_frame_con.visdata.coord.append(uvfits.visdata.coord)

            # tag index of uvfits (uvfits_idx)
            Ndata       = uvfits.visdata.coord.shape[0]
            uv_idxp = np.full(int(Ndata),iuvfits,dtype=np.int32)
            uv_idx      = np.int64(np.append(uv_idx,uv_idxp))

        # add uvfits_idx in uvfits_frame_con
        uvfits_frame_con.visdata.coord["uvfits_id"]=uv_idx # float?
        uvfits_frame_con.visdata.sort()
        return uvfits_frame_con

    def selfcal(self,uvfitslist,image,std_amp=1,std_pha=100):
        '''
        This perform a self calibration to concatenated uvfits (uvfits_frame_cal)
        and make list of cltable

        Args:
            uvfitslist (list of input uvfits list):
            image (imdata):
            model image for a self calibration
            std_amp:
            standard amplitude for a self calibration
            std_pha:
            standard phase for a self calibration

        Returns:
            cltable_list (list of cltable):

        '''

        print(type(uvfitslist))
        Nuvfits = len(uvfitslist)
        uvfits_totlist=[]
        cltable_list=[]
        for iuvfits in xrange(Nuvfits):
            uvfits_tot = copy.deepcopy(uvfitslist[iuvfits])
            uvfits_tot.visdata.coord=pd.DataFrame([])
            uvfits_totlist=uvfits_totlist+[uvfits_tot]
        istart=np.zeros(Nuvfits)
        for it in xrange(self.Nt):
            print("Frame number=%d/%d:"%(it,self.Nt-1))

            print("STEP1: Extract uvfits components in a frame")
            uvfits_framelist=self.split_uvfits(it,uvfitslist)

            print("STEP2: Concatenate the components of uvfits")
            uvfits_frame_con = self.concat_uvfits(uvfits_framelist)

            print("STEP3: Perform a self calibration")

            # Perform a self calibration to concatenated uvfits
            cltable      = uvfits_frame_con.selfcal(image,std_amp,std_pha)
            cltable_list = cltable_list+[cltable]

        return cltable_list

    def cltable_uvfits(self,cltable,uvfits_frame_con,uvfitslist):

        '''
        This make uvfitslist for cltable of a frame

        Args:
            cltable:
            component of cltable_list of selfcal
            uvfits_frame_con (uvfits):
            Concatenated and sorted uvfits object (see concat_uvfits)
            uvfitslist (list of uvfits objects):

        Returns:
            uvfits_newlist (list of uvfits):
            uvfitslist of a frame

        '''
        Nuvfits = len(uvfitslist)
        # Perform a self calibration to concatenated uvfits
        uvfits_frame_cal = uvfits_frame_con.apply_cltable(cltable)
        #uvfits_frame_cal.visdata.coord.shape

        Ndata = len(uvfits_frame_cal.visdata.coord["utc"])

        # make list of uvfits_frame_cal
        uvfits_newlist=[]
        for iuvfits,idata in itertools.product(xrange(Nuvfits),xrange(Ndata)):
            if(idata==0):
                uvfits=copy.deepcopy(uvfits_frame_con)
                uvfits.visdata.coord= pd.DataFrame([])
                istart = 0
            if(uvfits_frame_cal.visdata.coord["uvfits_id"][idata]== iuvfits):
                if(istart==0):
                    # 型を変えないように、[idata:idata+1]としている
                    uvfits.visdata.data=uvfits_frame_cal.visdata.data[idata:idata+1]
                    istart=1
                else:
                    uvfits.visdata.data = np.concatenate((uvfits.visdata.data,uvfits_frame_cal.visdata.data[idata:idata+1]))
                uvfits.visdata.coord = uvfits.visdata.coord.append(uvfits_frame_cal.visdata.coord.loc[idata:idata])
            if(idata==Ndata-1):
                uvfits.visdata.coord = uvfits.visdata.coord.drop("uvfits_id",axis=1)
                uvfits_newlist =uvfits_newlist+[uvfits]
        return uvfits_newlist

    def apply_cltable(self,cltable_list,uvfitslist,std_amp=1,std_pha=100):
        '''
        This makes a list of uvfits objects by performing a self calibration for all frames

        Args:
            cltable_list (list of cltable):
            uvfitslist (list of uvfits objects):
            std_amp:
            standard amplitude for a self calibration
            std_pha:
            standard phase for a self calibration

        Returns:
            uvfits_totlist ():
            list of uvfits after a self calibration
        '''

        Nuvfits = len(uvfitslist)
        uvfits_totlist=[]
        for iuvfits in xrange(Nuvfits):
            uvfits_tot = copy.deepcopy(uvfitslist[iuvfits])
            uvfits_tot.visdata.coord=pd.DataFrame([])
            uvfits_totlist=uvfits_totlist+[uvfits_tot]
        istart=np.zeros(Nuvfits)
        Nt = len(cltable_list)
        for it in xrange(Nt):
            cltable=cltable_list[it]
            print("Frame number=%d/%d:"%(it,Nt-1))

            print("STEP1: Extract uvfits components in a frame")
            uvfits_framelist=self.split_uvfits(it,uvfitslist)

            print("STEP2: Concatenate the components of uvfits")
            uvfits_frame_con = self.concat_uvfits(uvfits_framelist)

            print("STEP3: Perform a self calibration")
            uvfits_newlist=self.cltable_uvfits(cltable,uvfits_frame_con,uvfitslist)

            # Concatenate uvfits objects of each frame by using uvfits_id
            print("STEP4: Concatenate uvfits objects of each frame")
            for iuvfits in xrange(Nuvfits):
                if(istart[iuvfits]==0):
                    uvfits_totlist[iuvfits].visdata.data  = uvfits_newlist[iuvfits].visdata.data
                    uvfits_totlist[iuvfits].visdata.coord = uvfits_newlist[iuvfits].visdata.coord
                    istart[iuvfits]=1
                else:
                    uvfits_totlist[iuvfits].visdata.data  = np.concatenate((uvfits_totlist[iuvfits].visdata.data,uvfits_newlist[iuvfits].visdata.data))
                    uvfits_totlist[iuvfits].visdata.coord = uvfits_totlist[iuvfits].visdata.coord.append(uvfits_newlist[iuvfits].visdata.coord)
                uvfits_totlist[iuvfits].visdata.sort()
            print("============================")
            print("it=%d/%d:"%(it,self.Nt-1))
            for iuvfits in xrange(Nuvfits):
                print(uvfits_totlist[iuvfits].visdata.data.shape)
                print(uvfits_totlist[iuvfits].visdata.coord.shape)
            print("============================")

        return uvfits_totlist




    def initimlist(self):
        pass
        #mul2dim = list([self.init2dim])*Nt
        #return mul2dim
