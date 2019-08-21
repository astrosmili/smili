#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a sub-module of smili handling image fits data.
'''
__author__ = "Smili Developer Team"
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
import astropy.coordinates as coord
import astropy.io.fits as pyfits
import astropy.time as at
from astropy.convolution import convolve_fft

# ds9
import pyds9

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# com_position, peak_position,shift_position
import itertools

# internal
from .. import fortlib, util
from . import imregion as imr

#-------------------------------------------------------------------------
# IMAGEFITS (Manupulating FITS FILES)
#-------------------------------------------------------------------------
class IMFITS(object):
    # Initialization
    def __init__(self,
            imfits=None, imfitstype="standard",
            uvfits=None,
            source=None,
            instrument=None,
            observer=None,
            #dateobs=None,
            dx=2., dy=None, angunit="uas",
            nx=100, ny=None, nxref=None, nyref=None,
            **args):
        '''
        This is a class to handle image data, in particular, a standard image
        FITS data sets.

        The order of priority for duplicated parameters is
            1 uvfits (strongest)
            2 source, instrument, observer
            3 imfits
            4 dx, dy, nx, ny, nxref, nyref
            5 other parameters (weakest).

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified image fits file
                or specified HDUlist object.
            imfitstype (string, optional, default="standard"):
                Type of FITS images to be load. Currently we have three choices:
                    standard: assuming a format close to the AIPS IMAGE FITS format.
                              It can read images from AIPS, DIFMAP and CASA.
                    aipscc: same to standard but loading brightness informations
                            from the associated AIPS CC table.
                    ehtim: assuming a format close to that adopted in EHT imaging Library
                           which does not have a lot of header information.
            uvfits (string, optional):
                If specified, date, source, frequency information will be loaded
                from header information of the input uvfits file
            source (string, optional):
               The source of the image RA and Dec will be obtained from CDS.
            instrument (string, optional):
               The name of instrument, telescope.
            observer (string, optional):
               The name of instrument, telescope.
            dx (float, optional, default=2.):
               The pixel size along the RA axis. If dx > 0, the sign of dx
               will be switched.
            dy (float, optional, default=abs(dx)):
               The pixel size along the Dec axis.
            nx (integer, optional, default=100):
                the number of pixels in the RA axis.
            ny (integer, optional, default=ny):
                the number of pixels in the Dec axis.
                Default value is same to nx.
            nxref (float, optional, default=(nx+1)/2.):
                The reference pixel in RA direction.
                "1" will be the left-most pixel.
            nyref (float, optional, default=(ny+1)/2.):
                the reference pixel of the DEC axis.
                "1" will be the bottom pixel.
            angunit (string, optional, default="uas"):
                angular unit for fov, x, y, dx and dy.

            **args: you can also specify other header information.
                x (float):
                    The source RA at the reference pixel.
                y (float):
                    The source DEC at the reference pixel.
                dy (float):
                    the grid size in the DEC axis.
                    MUST BE POSITIVE for the astronomical image.
                f (float):
                    the reference frequency in Hz
                nf (integer):
                    the number of pixels in the Frequency axis
                nfref (float):
                    the reference pixel of the Frequency axis
                df (float):
                    the grid size of the Frequency axis
                s (float):
                    the reference Stokes parameter
                ns (integer):
                    the number of pixels in the Stokes axis
                nsref (float):
                    the reference pixel of the Stokes axis
                ds (float):
                    the grid size of the Stokes axis
                observer (string)
                telescope (string)
                instrument (string)
                object (string)
                dateobs (string)

        Returns:
            imdata.IMFITS object
        '''
        # get conversion factor for angular scale
        angconv = self.angconv(angunit, "deg")

        # set the default angular unit
        self.angunit = angunit

        # get keys of Args
        argkeys = list(args.keys())

        # Set header and data
        self.init_header()
        self.data = None

        # set pixel size
        if dy is None:
            dy = np.abs(dx)
        self.header["dx"] = -np.abs(dx)
        self.header["dy"] = dy

        # set pixel size
        if ny is None:
            ny = nx
        self.header["nx"] = nx
        self.header["ny"] = ny

        # ref pixel
        if nxref is None:
            nxref = (nx+1.)/2
        if nyref is None:
            nyref = (ny+1.)/2
        self.header["nxref"] = nxref
        self.header["nyref"] = nyref

        # read header from Args
        for argkey in argkeys:
            headerkeys = list(self.header.keys())
            if argkey in headerkeys:
                self.header[argkey] = self.header_dtype[argkey](args[argkey])

        self.header["x"] *= angconv
        self.header["y"] *= angconv
        self.header["dx"] *= angconv
        self.header["dy"] *= angconv
        self.data = np.zeros([self.header["ns"], self.header["nf"],
                              self.header["ny"], self.header["nx"]])

        # Initialize from the image fits file.
        if imfits is not None:
            if  imfitstype=="standard":
                self.read_fits_standard(imfits)
            elif imfitstype=="ehtim":
                self.read_fits_ehtim(imfits)
            elif imfitstype=="aipscc":
                self.read_fits_aipscc(imfits)
            else:
                raise ValueError("imfitstype must be standard, ehtim or aipscc")

        # Set source, observer, instrument
        if instrument is not None:
            self.set_instrument(instrument)

        if observer is not None:
            self.set_observer(observer)

        if source is not None:
            self.set_source(source)

        # copy headers from uvfits file
        if uvfits is not None:
            self.read_uvfitsheader(uvfits)

        # initialize fitsdata
        self.update_fits()

    # Definition of Headers and their datatypes
    def init_header(self):
        header = {}
        header_dtype = {}

        # Information
        header["object"] = "NONE"
        header_dtype["object"] = str
        header["telescope"] = "NONE"
        header_dtype["telescope"] = str
        header["instrument"] = "NONE"
        header_dtype["instrument"] = str
        header["observer"] = "NONE"
        header_dtype["observer"] = str
        header["dateobs"] = "NONE"
        header_dtype["dateobs"] = str

        # RA information
        header["x"] = np.float64(0.)
        header_dtype["x"] = np.float64
        header["dx"] = np.float64(-1.)
        header_dtype["dx"] = np.float64
        header["nx"] = np.int64(1)
        header_dtype["nx"] = np.int64
        header["nxref"] = np.float64(1.)
        header_dtype["nxref"] = np.float64

        # Dec information
        header["y"] = np.float64(0.)
        header_dtype["y"] = np.float64
        header["dy"] = np.float64(1.)
        header_dtype["dy"] = np.float64
        header["ny"] = np.int64(1)
        header_dtype["ny"] = np.int64
        header["nyref"] = np.float64(1.)
        header_dtype["nyref"] = np.float64

        # Third Axis Information
        header["f"] = np.float64(229.345e9)
        header_dtype["f"] = np.float64
        header["df"] = np.float64(4e9)
        header_dtype["df"] = np.float64
        header["nf"] = np.int64(1)
        header_dtype["nf"] = np.int64
        header["nfref"] = np.float64(1.)
        header_dtype["nfref"] = np.float64

        # Stokes Information
        header["s"] = np.int64(1)
        header_dtype["s"] = np.int64
        header["ds"] = np.int64(1)
        header_dtype["ds"] = np.int64
        header["ns"] = np.int64(1)
        header_dtype["ns"] = np.int64
        header["nsref"] = np.int64(1)
        header_dtype["nsref"] = np.int64

        # Beam information
        header["bmaj"] = np.float64(0)
        header_dtype["bmaj"] = np.float64
        header["bmin"] = np.float64(0)
        header_dtype["bmin"] = np.float64
        header["bpa"] = np.float64(0)
        header_dtype["bpa"] = np.float64

        self.header = header
        self.header_dtype = header_dtype

    # set source name and source coordinates
    def set_source(self, source="SgrA*", srccoord=None):
        '''
        Set the source name and the source coordinate to the header.
        If source coordinate is not given, it will be taken from the CDS.

        Args:
            source (str; default="SgrA*"):
                Source Name
            srccoord (astropy.coordinates.Skycoord object; default=None):
                Source position. If not specified, it is automatically pulled
                from the CDS
        '''
        # get source coordinates if it is not given.
        if srccoord is None:
            srccoord = coord.SkyCoord.from_name(source)
        elif not isinstance(srccoord, coord.sky_coordinate.SkyCoord):
            raise ValueError("The source coordinate must be astropy.coordinates.sky_coordinate.SkyCoord obejct")

        # Information
        self.header["object"] = source
        self.header["x"] = srccoord.ra.deg
        self.header["y"] = srccoord.dec.deg
        self.update_fits()

    def set_instrument(self, instrument):
        '''
        Update headers for instrument and telescope with a
        specified name of the instrument.
        '''
        for key in "instrument,telescope".split(","):
            self.header[key]=self.header_dtype[key](instrument)

    def set_observer(self, observer):
        '''
        Update headers for instrument, telescope and observer with a
        specified name of the instrument.
        '''
        self.header["observer"]=self.header_dtype["observer"](observer)

    def set_beam(self, majsize=0., minsize=0., pa=0., scale=1., angunit=None):
        '''
        Set beam parameters into headers.

        Args:
            majsize, minsize (float, default=0):
                major/minor-axis FWHM size
            scale (float, default=1):
                scaling factor that will be multiplied to maj/min size.
            pa (float, default=0):
                position angle in deg

        '''
        if angunit is None:
            angunit = self.angunit
        angconv = util.angconv(angunit, "deg")
        self.header["bmaj"] = majsize * angconv * scale
        self.header["bmin"] = minsize * angconv * scale
        self.header["bpa"] = pa

    def set_frequency(self, freq):
        '''
        Set the reference frequency into headers.

        Args:
            freq (float, default=0): the reference frequency in Hz.
        '''
        self.header["f"] = freq

    # Read data from an image fits file
    def read_fits_standard(self, imfits):
        '''
        Read data from the image FITS file.

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified image fits file
                or specified HDUlist object.
        '''
        if isinstance(imfits, pyfits.hdu.hdulist.HDUList):
            hdulist = copy.deepcopy(imfits)
        else:
            hdulist = pyfits.open(imfits)
        self.hdulist = hdulist

        isx = False
        isy = False
        isf = False
        iss = False
        naxis = hdulist[0].header.get("NAXIS")
        for i in range(naxis):
            ctype = hdulist[0].header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            if ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1
            elif ctype[0:6] == "STOKES":
                iss = i + 1

        if isx != False:
            self.header["nx"]    = hdulist[0].header.get("NAXIS%d" % (isx))
            self.header["x"]     = hdulist[0].header.get("CRVAL%d" % (isx))
            self.header["dx"]    = hdulist[0].header.get("CDELT%d" % (isx))
            self.header["nxref"] = hdulist[0].header.get("CRPIX%d" % (isx))
            for key in "nx,x,dx,nxref".split(","):
                self.header[key] = self.header_dtype[key](self.header[key])
        else:
            print("Warning: No image data along RA axis.")

        if isy != False:
            self.header["ny"]    = hdulist[0].header.get("NAXIS%d" % (isy))
            self.header["y"]     = hdulist[0].header.get("CRVAL%d" % (isy))
            self.header["dy"]    = hdulist[0].header.get("CDELT%d" % (isy))
            self.header["nyref"] = hdulist[0].header.get("CRPIX%d" % (isy))
            for key in "ny,y,dy,nyref".split(","):
                self.header[key] = self.header_dtype[key](self.header[key])
        else:
            print("Warning: No image data along DEC axis.")

        if isf != False:
            self.header["nf"]    = hdulist[0].header.get("NAXIS%d" % (isf))
            self.header["f"]     = hdulist[0].header.get("CRVAL%d" % (isf))
            self.header["df"]    = hdulist[0].header.get("CDELT%d" % (isf))
            self.header["nfref"] = hdulist[0].header.get("CRPIX%d" % (isf))
            for key in "nf,f,df,nfref".split(","):
                self.header[key] = self.header_dtype[key](self.header[key])
        else:
            print("Warning: No image data along FREQ axis.")

        if iss != False:
            self.header["ns"]    = hdulist[0].header.get("NAXIS%d" % (iss))
            self.header["s"]     = hdulist[0].header.get("CRVAL%d" % (iss))
            self.header["ds"]    = hdulist[0].header.get("CDELT%d" % (iss))
            self.header["nsref"] = hdulist[0].header.get("CRPIX%d" % (iss))
            for key in "ns,s,ds,nsref".split(","):
                self.header[key] = self.header_dtype[key](self.header[key])
        else:
            print("Warning: No image data along STOKES axis.")

        keys = "object,telescope,instrument,observer,dateobs".split(",")
        for key in keys:
            keyname = key.upper()[0:8]
            try:
                self.header[key] = hdulist[0].header.get(keyname)
                self.header_dtype[key](self.header[key])
            except:
                print("warning: FITS file doesn't have a header info of '%s'"%(keyname))

        # load data
        self.data = hdulist[0].data.reshape([self.header["ns"],
                                             self.header["nf"],
                                             self.header["ny"],
                                             self.header["nx"]])

        # get beam information
        try:
            bunit = hdulist[0].header.get("BUNIT").lower()
        except:
            bunit = "jy/pixel"

        if bunit=="jy/beam":
            keys = "bmaj,bmin,bpa".split(",")
            for key in keys:
                keyname = key.upper()
                try:
                    self.header[key] = hdulist[0].header.get(keyname)
                    self.header_dtype[key](self.header[key])
                except:
                    print("warning: FITS file doesn't have a header info of '%s'"%(keyname))
            self.data *= util.saconv(
                x1=self.header["bmaj"],
                y1=self.header["bmin"],
                angunit1="deg",
                satype1="beam",
                x2=self.header["dx"],
                y2=self.header["dy"],
                angunit2="deg",
                satype2="pixel",
            )

        self.update_fits()

    def read_fits_aipscc(self, imfits):
        '''
        Read data from the image FITS file. For the brightness distribution,
        this function loads the AIPS CC table rather than data of the primary HDU.

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified image fits file
                or specified HDUlist object.
        '''
        # Load FITS File
        self.read_fits_standard(imfits)
        self.header["nf"] = 1
        self.header["ns"] = 1
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        self.data = np.zeros([1,1,Ny,Nx])

        # Get AIPS CC Table
        aipscc = pyfits.open(imfits)["AIPS CC"]
        flux = aipscc.data["FLUX"]
        deltax = aipscc.data["DELTAX"]
        deltay = aipscc.data["DELTAY"]
        checkmtype = np.abs(np.unique(aipscc.data["TYPE OBJ"]))<1.0
        if False in checkmtype.tolist():
            raise ValueError("Input FITS file has non point-source CC components, which are not currently supported.")
        ix = np.int64(np.round(deltax/self.header["dx"] + self.header["nxref"] - 1))
        iy = np.int64(np.round(deltay/self.header["dy"] + self.header["nyref"] - 1))
        print("There are %d clean components in the AIPS CC Table."%(len(flux)))

        # Add the brightness distribution to the image
        count = 0
        for i in range(len(flux)):
            try:
                self.data[0,0,iy[i],ix[i]] += flux[i]
            except:
                count += 1
        if count > 0:
            print("%d components are ignore since they are outside of the image FoV."%(count))
        self.update_fits()

    def read_fits_ehtim(self, imfits):
        '''
        Read data from the image FITS file geneated from the eht-imaging library

        Args:
            imfits (string or hdulist, optional):
                If specified, image will be loaded from the specified image fits file
                or specified HDUlist object.
        '''
        import ehtim as eh
        im = eh.image.load_fits(imfits)
        obsdate = at.Time(im.mjd, format="mjd")
        obsdate = "%04d-%02d-%02d"%(obsdate.datetime.year,obsdate.datetime.month,obsdate.datetime.day)
        self.header["object"] = im.source
        self.header["x"] = im.ra * 12
        self.header["y"] = im.dec
        self.header["dx"] = -np.abs(im.psize * util.angconv("rad","deg"))
        self.header["dy"] = im.psize * util.angconv("rad","deg")
        self.header["nx"] = im.xdim
        self.header["ny"] = im.ydim
        self.header["nxref"] = im.xdim/2.+1
        self.header["nyref"] = im.ydim/2.+1
        self.header["f"] = im.rf
        self.header["dateobs"]=obsdate
        self.data = np.flipud(im.imvec.reshape([self.header["ny"],self.header["nx"]]))
        self.data = self.data.reshape([1,1,self.header["ny"],self.header["nx"]])
        self.update_fits()

    def read_uvfitsheader(self, uvfits):
        '''
        Read header information from uvfits file

        Args:
          uvfits (string or hdulist):
            input uv-fits file specified by its filename or HDUList object.
        '''
        if isinstance(uvfits, pyfits.hdu.hdulist.HDUList):
            hdulist = copy.deepcopy(uvfits)
        else:
            hdulist = pyfits.open(uvfits)
        self.hdulist = hdulist
        hduinfos = hdulist.info(output=False)

        for hduinfo in hduinfos:
            idx = hduinfo[0]
            if hduinfo[1] == "PRIMARY":
                grouphdu = hdulist[idx]

        if not 'grouphdu' in locals():
            print("[Error] the input file does not contain the Primary HDU")

        keys = "object,telescope,instrument,observer,dateobs".split(",")
        for key in keys:
            keyname = key.upper()[0:8]
            try:
                self.header[key] = grouphdu.header.get(keyname)
                self.header_dtype[key](self.header[key])
            except:
                print("warning: UVFITS file doesn't have a header info of '%s'"%(keyname))

        naxis = grouphdu.header.get("NAXIS")
        for i in range(naxis):
            ctype = grouphdu.header.get("CTYPE%d" % (i + 1))
            if ctype is None:
                continue
            elif ctype[0:2] == "RA":
                isx = i + 1
            elif ctype[0:3] == "DEC":
                isy = i + 1
            elif ctype[0:4] == "FREQ":
                isf = i + 1

        if isx != False:
            self.header["x"] = self.header_dtype["x"](
                hdulist[0].header.get("CRVAL%d" % (isx)))
        else:
            print("Warning: No RA info.")

        if isy != False:
            self.header["y"] = self.header_dtype["y"](
                hdulist[0].header.get("CRVAL%d" % (isy)))
        else:
            print("Warning: No Dec info.")

        if isf != False:
            self.header["f"] = self.header_dtype["f"](
                hdulist[0].header.get("CRVAL%d" % (isf)))
            self.header["df"] = self.header_dtype["df"](hdulist[0].header.get(
                "CDELT%d" % (isf)) * hdulist[0].header.get("NAXIS%d" % (isf)))
            self.header["nf"] = self.header_dtype["nf"](1)
            self.header["nfref"] = self.header_dtype["nfref"](1)
        else:
            print("Warning: No image data along the Frequency axis.")

        self.update_fits()

    def update_fits(self,
                    bunit="JY/PIXEL",
                    cctab=True,
                    threshold=None,
                    relative=True,
                    istokes=0, ifreq=0):
        '''
        Reflect current self.data / self.header info to the image FITS data.
        Args:
            bunit (str; default=bunit): unit of the brightness. ["JY/PIXEL", "JY/BEAM"] is available.
            cctab (boolean): If True, AIPS CC table is attached to fits file.
            istokes (integer): index for Stokes Parameter at which the image will be used for CC table.
            ifreq (integer): index for Frequency at which the image will be used for CC table.
            threshold (float): pixels with the absolute intensity smaller than this value will be ignored in CC table.
            relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
        '''

        # CREATE HDULIST
        hdu = pyfits.PrimaryHDU(self.data)
        hdulist = pyfits.HDUList([hdu])

        # GET Current Time
        dtnow = dt.datetime.now()

        # FILL HEADER INFO
        hdulist[0].header.set("OBJECT",   self.header["object"])
        hdulist[0].header.set("TELESCOP", self.header["telescope"])
        hdulist[0].header.set("INSTRUME", self.header["instrument"])
        hdulist[0].header.set("OBSERVER", self.header["observer"])
        hdulist[0].header.set("DATE",     "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("DATE-OBS", self.header["dateobs"])
        hdulist[0].header.set("DATE-MAP", "%04d-%02d-%02d" %
                              (dtnow.year, dtnow.month, dtnow.day))
        hdulist[0].header.set("BSCALE",   np.float64(1.))
        hdulist[0].header.set("BZERO",    np.float64(0.))
        if bunit.upper() == "JY/PIXEL":
            hdulist[0].header.set("BUNIT",    "JY/PIXEL")
            bconv = 1
        elif bunit.upper() == "JY/BEAM":
            hdulist[0].header.set("BUNIT",    "JY/BEAM")
            bconv = util.saconv(
                x1=self.header["dx"],
                y1=self.header["dy"],
                angunit1="deg",
                satype1="pixel",
                x2=self.header["bmaj"],
                y2=self.header["bmin"],
                angunit2="deg",
                satype2="beam",
            )
            hdulist[0].header.set("BMAJ",self.header["bmaj"])
            hdulist[0].header.set("BMIN",self.header["bmin"])
            hdulist[0].header.set("BPA",self.header["bpa"])
        hdulist[0].header.set("EQUINOX",  np.float64(2000.))
        hdulist[0].header.set("OBSRA",    np.float64(self.header["x"]))
        hdulist[0].header.set("OBSDEC",   np.float64(self.header["y"]))
        hdulist[0].header.set("DATAMAX",  self.data.max())
        hdulist[0].header.set("DATAMIN",  self.data.min())
        hdulist[0].header.set("CTYPE1",   "RA---SIN")
        hdulist[0].header.set("CRVAL1",   np.float64(self.header["x"]))
        hdulist[0].header.set("CDELT1",   np.float64(self.header["dx"]))
        hdulist[0].header.set("CRPIX1",   np.float64(self.header["nxref"]))
        hdulist[0].header.set("CROTA1",   np.float64(0.))
        hdulist[0].header.set("CTYPE2",   "DEC--SIN")
        hdulist[0].header.set("CRVAL2",   np.float64(self.header["y"]))
        hdulist[0].header.set("CDELT2",   np.float64(self.header["dy"]))
        hdulist[0].header.set("CRPIX2",   np.float64(self.header["nyref"]))
        hdulist[0].header.set("CROTA2",   np.float64(0.))
        hdulist[0].header.set("CTYPE3",   "FREQ")
        hdulist[0].header.set("CRVAL3",   np.float64(self.header["f"]))
        hdulist[0].header.set("CDELT3",   np.float64(self.header["df"]))
        hdulist[0].header.set("CRPIX3",   np.float64(self.header["nfref"]))
        hdulist[0].header.set("CROTA3",   np.float64(0.))
        hdulist[0].header.set("CTYPE4",   "STOKES")
        hdulist[0].header.set("CRVAL4",   np.int64(self.header["s"]))
        hdulist[0].header.set("CDELT4",   np.int64(self.header["ds"]))
        hdulist[0].header.set("CRPIX4",   np.int64(self.header["nsref"]))
        hdulist[0].header.set("CROTA4",   np.int64(0))

        # scale angunit
        hdulist[0].data *= bconv

        # Add AIPS CC Table
        if cctab:
            aipscctab = self.to_aipscc(threshold=threshold, relative=relative,
                    istokes=istokes, ifreq=ifreq)

            hdulist.append(hdu=aipscctab)

            next = len(hdulist)
            hdulist[next-1].name = 'AIPS CC'

        self.hdulist = hdulist

    def angconv(self, unit1="deg", unit2="deg"):
        '''
        return a conversion factor from unit1 to unit2
        Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
        '''
        return util.angconv(unit1,unit2)

    #-------------------------------------------------------------------------
    # Getting Some information about images
    #-------------------------------------------------------------------------
    def get_beam(self, angunit=None):
        '''
        get beam parameters

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
        '''
        if angunit is None:
            angunit = self.angunit

        conv = util.angconv("deg", angunit)

        outdic = {}
        outdic["majsize"] = self.header["bmaj"] * conv
        outdic["minsize"] = self.header["bmin"] * conv
        outdic["pa"] = self.header["bpa"]
        outdic["angunit"] = angunit
        return outdic

    def get_xygrid(self, twodim=False, angunit=None):
        '''
        calculate the grid of the image

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
          twodim (boolean): It True, the 2D grids will be returned. Otherwise, the 1D arrays will be returned
        '''
        if angunit is None:
            angunit = self.angunit

        dx = self.header["dx"]
        dy = self.header["dy"]
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxref = self.header["nxref"]
        Nyref = self.header["nyref"]
        xg = dx * (np.arange(Nx) - Nxref + 1) * self.angconv("deg", angunit)
        yg = dy * (np.arange(Ny) - Nyref + 1) * self.angconv("deg", angunit)
        if twodim:
            xg, yg = np.meshgrid(xg, yg)
        return xg, yg

    def get_bconv(self,fluxunit="Jy",saunit="beam"):
        '''
        derive a conversion factor to convert the unit of the intensity
        to the one using the specified flux and solid angle units.
        '''
        if fluxunit.lower()!="k":
            fluxconv = util.fluxconv("Jy", fluxunit)

            if saunit.lower() == "beam":
                saconv = util.saconv(
                    x1=self.header["dx"],
                    y1=self.header["dy"],
                    angunit1="deg",
                    satype1="pixel",
                    x2=self.header["bmaj"],
                    y2=self.header["bmin"],
                    angunit2="deg",
                    satype2="beam",
                )
            elif saunit.lower() == "pixel":
                saconv = 1
            else:
                saconv = util.angconv("deg",saunit) ** 2
            return fluxconv * saconv
        else:
            import astropy.constants as ac
            deg2rad = util.angconv("deg", "rad")
            nu = self.header["f"]
            dx = np.abs(self.header["dx"] * deg2rad)
            dy = np.abs(self.header["dy"] * deg2rad)
            jy2k = ac.c.si.value ** 2 / (2 * ac.k_B.si.value * nu **2) / dx / dy * 1e-26
            return jy2k

    def get_imarray(self,fluxunit="Jy",saunit="pixel"):
        '''
        returns np.array of the brightness distributions in a specified unit
        '''
        return self.data * self.get_bconv(fluxunit,saunit)

    def get_imextent(self, angunit=None):
        '''
        calculate the field of view of the image

        Args:
          angunit (string): Angular unit (uas, mas, asec or arcsec, amin or arcmin, degree)
        '''
        if angunit is None:
            angunit = self.angunit

        dx = self.header["dx"]
        dy = self.header["dy"]
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxref = self.header["nxref"]
        Nyref = self.header["nyref"]
        xmax = (1 - Nxref - 0.5) * dx
        xmin = (Nx - Nxref + 0.5) * dx
        ymax = (Ny - Nyref + 0.5) * dy
        ymin = (1 - Nyref - 0.5) * dy
        return np.asarray([xmax, xmin, ymin, ymax]) * self.angconv("deg", angunit)

    def get_angunitlabel(self, angunit=None):
        '''
        Get the angular unit of the specifed angunit. If not given,
        it will be taken by self.angunit
        '''
        if angunit is None:
            angunit = self.angunit

        # Axis Label
        if angunit.lower().find("pixel") == 0:
            unit = "pixel"
        elif angunit.lower().find("uas") == 0:
            unit = r"$\rm \mu$as"
        elif angunit.lower().find("mas") == 0:
            unit = "mas"
        elif angunit.lower().find("arcsec") * angunit.lower().find("asec") == 0:
            unit = "arcsec"
        elif angunit.lower().find("arcmin") * angunit.lower().find("amin") == 0:
            unit = "arcmin"
        elif angunit.lower().find("deg") == 0:
            unit = "deg"
        else:
            unit = "mas"
        return unit

    def get_fluxunitlabel(self, fluxunit="Jy"):
        '''
        Get the angular unit of the specifed angunit. If not given,
        it will be taken by self.angunit
        '''
        # Axis Label
        if   fluxunit.lower().find("jy") == 0:
            unit = "Jy"
        elif fluxunit.lower().find("mjy") == 0:
            unit = "mJy"
        elif fluxunit.lower().find("ujy") == 0:
            unit = r"$\rm \mu$Jy"
        elif fluxunit.lower().find("si") == 0:
            unit = r"W m$^{-2}$ Hz$^{-1}$"
        elif fluxunit.lower().find("cgs") == 0:
            unit = r"erg s cm$^{-2}$ Hz$^{-1}$"
        return unit

    def get_saunitlabel(self, saunit="pixel"):
        '''
        Get the angular unit of the specifed angunit. If not given,
        it will be taken by self.angunit
        '''
        # Axis Label
        if   saunit.lower().find("pixel") == 0:
            unit = r"Pixel$^{-1}$"
        elif saunit.lower().find("beam") == 0:
            unit = r"Beam$^{-1}$"
        else:
            unit = self.get_angunitlabel(angunit=saunit) + r"$^{-2}$"
        return unit

    def peak(self, absolute=False, fluxunit="Jy", saunit="pixel", istokes=0, ifreq=0):
        '''
        calculate the peak intensity of the image in Jy/Pixel

        Args:
          absolute (boolean): if True, it will pick up the peak of the absolute.
          istokes (integer): index for Stokes Parameter at which the peak intensity will be calculated
          ifreq (integer): index for Frequency at which the peak intensity will be calculated
        '''
        if absolute:
            t = np.argmax(np.abs(self.data[istokes, ifreq]))
            t = np.unravel_index(t, [self.header["ny"], self.header["nx"]])
            return self.data[istokes, ifreq][t] * self.get_bconv(fluxunit=fluxunit, saunit=saunit)
        else:
            return self.data[istokes, ifreq].max() * self.get_bconv(fluxunit=fluxunit, saunit=saunit)

    def totalflux(self, fluxunit="Jy", istokes=0, ifreq=0):
        '''
        Calculate the total flux density of the image

        Args:
          istokes (integer): index for Stokes Parameter at which the total flux will be calculated
          ifreq (integer): index for Frequency at which the total flux will be calculated
        '''
        return self.data[istokes, ifreq].sum() * util.fluxconv("Jy", fluxunit)

    def mad(self, imregion=None, fluxunit="Jy", saunit="pixel", istokes=0, ifreq=0):
        '''
        calculate the median absolute deviation of the image

        Args:
          istokes (integer): index for Stokes Parameter at which l1-norm will be calculated
          ifreq (integer): index for Frequency at which l1-norm will be calculated
        '''
        image = self.data[istokes,ifreq]
        if imregion is not None:
            maskimage = imregion.maskimage(self)
            image = image[np.where(maskimage > 0.5)]
        return np.median(np.abs(image - np.median(image))) * self.get_bconv(fluxunit=fluxunit, saunit=saunit)

    def rms(self, imregion=None, fluxunit="Jy", saunit="pixel", istokes=0, ifreq=0):
        '''
        calculate the median absolute deviation of the image

        Args:
          istokes (integer): index for Stokes Parameter at which l1-norm will be calculated
          ifreq (integer): index for Frequency at which l1-norm will be calculated
        '''
        image = self.data[istokes,ifreq]
        if imregion is not None:
            maskimage = imregion.maskimage(self)
            image = image[np.where(maskimage > 0.5)]
        return np.sqrt(np.mean(image**2)) * self.get_bconv(fluxunit=fluxunit, saunit=saunit)

    def compos(self,alpha=1.,angunit=None,ifreq=0, istokes=0):
        '''
        Returns the position of the center of mass in a specified angular unit.

        Arg:
          alpha (float):
            if alpha != 0, then the image is powered by alpha prior to compute
            the center of the mass.
          angunit (string):
            The angular unit for the position. If not specified, it will use
            self.angunit.
          istokes (integer):
            index for Stokes Parameter
          ifreq (integer):
            index for Frequency
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          dictionary for the position
        '''
        if angunit is None:
            angunit = self.angunit
        conv = util.angconv("deg",angunit)

        image = np.abs(self.data[ifreq, istokes])
        if alpha!=1:
            image = np.power(image, alpha)
        pix = sn.measurements.center_of_mass(image)
        x0 = (pix[1]+1-self.header["nxref"])*self.header["dx"]*conv
        y0 = (pix[0]+1-self.header["nyref"])*self.header["dy"]*conv

        x,y = self.get_xygrid(angunit=angunit, twodim=True)
        outdic = {
            "x0": x0,
            "y0": y0,
            "angunit": angunit
        }
        return outdic

    def peakpos(self,angunit=None,ifreq=0,istokes=0):
        '''
        Returns the position of the peak in a specified angular unit.

        Arg:
          angunit (string):
            The angular unit for the position. If not specified, it will use
            self.angunit.
          istokes (integer):
            index for Stokes Parameter
          ifreq (integer):
            index for Frequency
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          dictionary for the position
        '''
        if angunit is None:
            angunit = self.angunit

        image = np.abs(self.data[ifreq, istokes])
        pix = np.unravel_index(np.argmax(image),image.shape)

        x,y = self.get_xygrid(angunit=angunit, twodim=True)
        outdic = {
            "x0": x[pix],
            "y0": y[pix],
            "angunit": angunit
        }
        return outdic

    def get_secondmoment(self, istokes=0, ifreq=0):

        '''
        Return second momentums of images

        Args:
            istokes (int, default=0):
                The ordinal number of stokes parameters
            ifreq (int, default=0):
                The ordinal number of frequency

        Returns:
            Second momentum for xx, yy, and xy components
        '''

        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxy = Nx*Ny
        I1d = np.float64(self.data[istokes, ifreq]).reshape(Nxy)
        xidx = np.arange(Nx)+1
        yidx = np.arange(Ny)+1
        xidx, yidx = np.meshgrid(xidx, yidx)
        dtheta = np.abs(self.angconv("deg",self.angunit)*self.header["dx"])

        Isum = fortlib.image.totalflux(I1d, Nxy)
        print(Isum)
        xycen = fortlib.image.xy_cen(I1d, xidx, yidx, Nxy)
        print(len(xycen))
        print(xycen)
        Sg = fortlib.image.Sigma(I1d, xidx, yidx, Nxy, Isum, xycen[0], xycen[1])
        Sg *= dtheta**2
        return Sg

    def get_sm_character(self, istokes=0, ifreq=0):

        '''
        Returns characteristic quantities for secondary momentum

        Args:
            istokes (int, default=0):
                The ordinal number of stokes parameters
            ifreq (int, default=0):
                The ordinal number of frequency

        Returns:
            FWHMs of major axis, minor axis [same unit of imdata.IMFITS object]
            position angles [rad]
        '''

        Nx = self.header["nx"]
        Ny = self.header["ny"]
        Nxy = Nx*Ny
        I1d = np.float64(self.data[istokes, ifreq]).reshape(Nxy)
        xidx = np.arange(Nx)+1
        yidx = np.arange(Ny)+1
        xidx, yidx = np.meshgrid(xidx, yidx)
        dtheta = np.abs(self.angconv("deg",self.angunit)*self.header["dx"])

        Isum = fortlib.image.totalflux(I1d, Nxy)
        xycen = fortlib.image.xy_cen(I1d, xidx, yidx, Nxy)
        Sg = fortlib.image.Sigma(I1d, xidx, yidx, Nxy, Isum, xycen[0], xycen[1])
        dtheta = np.abs(initimage.angconv("deg",initimage.angunit)*initimage.header["dx"])

        #out_maj, out_min, out_phi
        output = fortlib.image.check_sm_character(Sg)
        #output[0] *= sqrt(8.*log(2.) * dtheta
        #output[1] *= sqrt(8.*log(2.) * dtheta

        return output

    def imagecost(self, func, out, istokes=0, ifreq=0, compower=1.0):
        '''
        return image cost.

        Args:
            func (string): cost function ("l1", "mem", "tv", "tsv", or "com").
            out (string): output cost ("cost", "costmap", "gradmap").
            istokes (integer): index for Stokes Parameter at which l1-norm will be calculated.
            ifreq (integer): index for Frequency at which l1-norm will be calculated.
            compower (float): power if func = "com".
        '''
        # get initial images
        istokes = istokes
        ifreq = ifreq
        nxref = self.header["nxref"]
        nyref = self.header["nyref"]
        Iin = np.float64(self.data[istokes, ifreq])

        if func is "l1":
            costs = fortlib.image.i2d_l1(i2d=np.float64(Iin))
        elif func is "mem":
            costs = fortlib.image.i2d_mem(i2d=np.float64(Iin))
        elif func is "tv":
            costs = fortlib.image.i2d_tv(i2d=np.float64(Iin))
        elif func is "tsv":
            costs = fortlib.image.i2d_tsv(i2d=np.float64(Iin))
        elif func is "com":
            costs = fortlib.image.i2d_com(i2d=np.float64(Iin),
                                          nxref=np.float64(nxref),
                                          nyref=np.float64(nyref),
                                          alpha=np.float64(compower))

        cost = costs[0]
        costmap = costs[1]
        gradmap = costs[2]

        if out is "cost":
            return cost
        elif out is "costmap":
            costmapfits = copy.deepcopy(self)
            costmapfits.data[istokes,ifreq] = costmap
            return costmapfits
        elif out is "gradmap":
            gradmapfits = copy.deepcopy(self)
            gradmapfits.data[istokes,ifreq] = gradmap
            return gradmapfits

    #-------------------------------------------------------------------------
    # Plotting
    #-------------------------------------------------------------------------
    def imshow(self,
            scale="linear",
            dyrange=100,
            gamma=0.5,
            vmax=None,
            vmin=None,
            relative=False,
            fluxunit="jy",
            saunit="pixel",
            restore=False,
            axisoff=False,
            axislabel=True,
            colorbar=False,
            colorbarprm={},
            istokes=0, ifreq=0,
            cmap=cm.afmhot,
            interpolation="none",
            **imshow_args):
        '''
        Plot the image.
        To change the angular unit, please change IMFITS.angunit.

        Args:
          scale (str; default="linear"):
            Transfar function. Availables are "linear", "log", "gamma"
          dyrange (float; default=100):
            Dynamic range of the log color contour.
          gamma (float; default=1/2.):
            Gamma parameter for scale="gamma".
          vmax (float):
            The maximum value of the color contour.
          vmin (float):
            The minimum value of the color contour.
            If logscale=True, dyrange will be used to set vmin.
          relative (boolean, default=True):
            If True, vmin will be the relative value to the peak or vmax.
          fluxunit (string):
            Unit for the flux desity (Jy, mJy, uJy, K, si, cgs)
          saunit (string):
            Angular Unit for the solid angle (pixel, uas, mas, asec or arcsec,
            amin or arcmin, degree, beam). If restore is True, saunit will be
            forced to be "beam".
          restore (boolean, default=False):
            If True, the image will be blurred by a Gaussian specified with
            beam parameters in the header.
          axisoff (boolean, default=False):
            If True, plotting without any axis label, ticks, and lines.
            This option is superior to the axislabel option.
          axislabel (boolean, default=True):
            If True, plotting the axislabel.
          colorbar (boolean, default=False):
            If True, the colorbar will be shown.
          colorbarprm (dic, default={}):
            parameters for pyplot.colorbar
          istokes (integer):
            index for Stokes Parameter at which the image will be plotted
          ifreq (integer):
            index for Frequency at which the image will be plotted
          **imshow_args: Args will be input in matplotlib.pyplot.imshow
        '''
        # Get Image Axis
        angunit = self.angunit
        imextent = self.get_imextent(angunit)

        if fluxunit.lower()=="k":
            saunit="pixel"

        if restore:
            saunit="beam"

        fluxconv = self.get_bconv(fluxunit=fluxunit, saunit="pixel")
        saconv = self.get_bconv(fluxunit="Jy", saunit=saunit)

        # get twodim array
        if restore:
            imarr = self.convolve_gauss(
                majsize=self.header["bmaj"],
                minsize=self.header["bmin"],
                pa=self.header["bpa"],
                angunit="deg"
            )
            imarr = imarr.get_imarray()[istokes,ifreq] * fluxconv * saconv
            if vmax is None:
                peak = imarr.max()
            else:
                peak = vmax
        else:
            imarr = self.get_imarray()[istokes,ifreq] * fluxconv * saconv
            if vmax is None:
                peak = imarr.max()
            else:
                peak = vmax

        if scale.lower()=="log":
            vmin = None
            norm = mcolors.LogNorm(vmin=peak/dyrange, vmax=peak)
            imarr[np.where(imarr<peak/dyrange)] = peak/dyrange
        elif scale.lower()=="gamma":
            if vmin is not None and relative:
                vmin *= peak
            elif vmin is None:
                vmin = 0.
            norm = mcolors.PowerNorm(vmin=peak/dyrange, vmax=peak, gamma=gamma)
            imarr[np.where(np.abs(imarr)<0)] = 0
        elif scale.lower()=="linear":
            if vmin is not None and relative:
                vmin *= peak
            norm = None
        else:
            raise ValueError("Invalid scale parameters. Available: 'linear', 'log', 'gamma'")
        imarr[np.isnan(imarr)] = 0

        im = plt.imshow(
            imarr, origin="lower", extent=imextent, vmin=vmin, vmax=vmax,
            cmap=cmap, interpolation=interpolation, norm=norm,
            **imshow_args
        )

        # Axis Label
        if axislabel:
            angunitlabel = self.get_angunitlabel(angunit)
            plt.xlabel("Relative RA (%s)" % (angunitlabel))
            plt.ylabel("Relative Dec (%s)" % (angunitlabel))

        # Axis off
        if axisoff:
            plt.axis("off")

        # colorbar
        if colorbar:
            clb = self.colorbar(fluxunit=fluxunit, saunit=saunit, **colorbarprm)
            return im,clb
        else:
            return im

    def plot_beam(self, boxfc="black", boxec="white", beamfc="black", beamec="white",
                  lw=1., alpha=0.5, x0=0.05, y0=0.05, boxsize=1.5, zorder=None):
        '''
        Plot beam in the header.
        To change the angular unit, please change IMFITS.angunit.

        Args:
            x0, y0 (float, default=0.05):
                leftmost, lowermost location of the box
                if relative=True, the value is on transAxes coordinates
            relative (boolean, default=True):
                If True, the relative coordinate to the current axis will be
                used to plot data
            boxsize (float, default=1.5):
                Relative size of the box to the major axis size.
            boxfc, boxec (color formatter):
                Face and edge colors of the box
            beamfc, beamec (color formatter):
                Face and edge colors of the beam
            lw (float, default=1): linewidth
            alpha (float, default=0.5): transparency parameter (0<1) for the face color
        '''
        angunit = self.angunit
        angconv = util.angconv("deg",angunit)

        majsize = self.header["bmaj"] * angconv
        minsize = self.header["bmin"] * angconv
        pa = self.header["bpa"]

        offset = np.max([majsize, minsize])/2*boxsize

        # get the current axes
        ax = plt.gca()

        # center
        xedge, yedge = ax.transData.inverted().transform(ax.transAxes.transform((x0,y0)))
        xcen = xedge - offset
        ycen = yedge + offset

        # get ellipce shapes
        xe,ye = _plot_beam_ellipse(majsize, minsize, pa)
        xe += xcen
        ye += ycen

        xb, yb = _plot_beam_box(offset*2, offset*2)
        xb += xcen
        yb += ycen

        plt.fill(xb, yb, fc=boxfc, alpha=alpha , zorder=zorder)
        plt.fill(xe, ye, fc=beamfc, alpha=alpha , zorder=zorder)
        plt.plot(xe, ye, lw, color=beamec, zorder=zorder)
        plt.plot(xb, yb, lw, color=boxec, zorder=zorder)

    def plot_scalebar(self,x,y,length,ha="center",color="white",lw=1,**plotargs):
        '''
        Plot a scale bar

        Args:
            x,y (in the unit of the current plot):
                x,y coordinates of the scalebar
            length (in the unit of the current plot):
                length of the scale bar
            ha (str, default="center"):
                The horizontal alignment of the bar.
                Available options is ["center", "left", "right"]
            plotars:
                Arbital arguments for pyplot.plot.
        '''
        if ha.lower()=="center":
            xmin = x-np.abs(length)/2
            xmax = x+np.abs(length)/2
        elif ha.lower()=="left":
            xmin = x - np.abs(length)
            xmax = x
        elif ha.lower()=="right":
            xmin = x
            xmax = x + np.abs(length)
        else:
            raise ValueError("ha must be center, left or right")
        plt.plot([xmax,xmin],[y,y],color=color,lw=lw,**plotargs)

    def colorbar(self, fluxunit="Jy", saunit="pixel", **colorbarprm):
        '''
        add colorbar
        '''
        clb = plt.colorbar(**colorbarprm)
        if fluxunit.lower()=="k":
            clb.set_label("Brightness Temperature (K)")
        else:
            fluxunitlabel = self.get_fluxunitlabel(fluxunit)
            saunitlabel = self.get_saunitlabel(saunit)
            clb.set_label("Intensity (%s %s)"%(fluxunitlabel, saunitlabel))
        return clb

    def contour(self, cmul=None, relative=True,
                levels=None,
                colors="white", ls="-",
                istokes=0, ifreq=0,
                **contour_args):
        '''
        plot contours of the image

        Args:
          istokes (integer):
            index for Stokes Parameter at which the image will be plotted
          ifreq (integer):
            index for Frequency at which the image will be plotted
          colors (string, array-like):
            colors of contour levels
          cmul (float; default=None):
            The lowest contour level. Default value is 1% of the peak intensity.
          relative (boolean, default=True):
            If true, cmul will be the relative value to the peak intensity.
          levels: contour level. This will be multiplied with cmul.
          **contour_args: Args will be input in matplotlib.pyplot.contour
        '''
        # Get Image Axis
        angunit = self.angunit
        imextent = self.get_imextent(angunit)

        # Get image
        image = self.data[istokes, ifreq]

        if cmul is None:
            vmin = self.peak() * 0.01
        else:
            if relative:
                vmin = cmul * np.abs(image).max()
            else:
                vmin = cmul

        if levels is None:
            clevels = np.power(2, np.arange(10))
        else:
            clevels = np.asarray(levels)
        clevels = vmin * np.asarray(clevels)

        CS = plt.contour(image, extent=imextent, origin="lower",
                         colors=colors, levels=clevels, ls=ls, **contour_args)

        angunitlabel = self.get_angunitlabel(angunit)
        plt.xlabel("Relative RA (%s)" % (angunitlabel))
        plt.ylabel("Relative Dec (%s)" % (angunitlabel))
        return CS

    #-------------------------------------------------------------------------
    # DS9
    #-------------------------------------------------------------------------
    def open_pyds9(self,imregion=None,wait=10):
        '''
        Open pyds9 and plot the region.
        This method uses pyds9.DS9().

        Args:
            imregion (imdata.IMRegion object; default=None):
                If specifed, imregion will be transferred to ds9 as well.
            wait (float, default = 10):
                seconds to wait for ds9 to start.
        '''
        try:
            d = pyds9.DS9(wait=wait)
        except ValueError:
            print("ValueError: try a longer 'wait' time or installation of XPA.")
        except:
            print("Unexpected Error!")
            raise
        else:
            d.set_pyfits(self.hdulist)
            d.set('zoom to fit')
            d.set('cmap heat')
            if imregion is not None:
                for index, row in imregion.iterrows():
                    ds9reg = imregion.reg_to_ds9reg(row,self)
                    d.set("region","image; %s" % ds9reg)

    def load_pyds9(self,angunit=None,wait=10):
        '''
        Load DS9 region to IMRegion.
        This method uses pyds9.DS9().

        Args:
            angunit (str, default = None):
                The angular unit of region. If None, it will take from
                the default angunit of the image
            wait (float, default = 10):
                seconds to wait for ds9 to start.
        Returns:
            imdata.IMRegion object
        '''
        if angunit is None:
            angunit = self.angunit

        try:
            d = pyds9.DS9(wait=wait)
        except ValueError:
            print("ValueError: try longer 'wait' time or installation of XPA.")
        except:
            print("Unexpected Error!")
            raise
        else:
            ds9reg = d.get("regions -system image")
            imregion = imr.ds9reg_to_reg(ds9reg=ds9reg,image=self,angunit=angunit)
            return imregion

    #-------------------------------------------------------------------------
    # Output some information to files
    #-------------------------------------------------------------------------
    def to_fits(self, outfitsfile=None, overwrite=True, bunit="Jy/pixel"):
        '''
        save the image(s) to the image FITS file or HDUList.

        Args:
            outfitsfile (string; default is None):
                FITS file name. If not specified, then HDUList object will be
                returned.
            overwrite (boolean):
                It True, an existing file will be overwritten.
        Returns:
            HDUList object if outfitsfile is None
        '''
        self.update_fits(bunit=bunit)

        if outfitsfile is None:
            return copy.deepcopy(self.hdulist)

        if os.path.isfile(outfitsfile):
            if overwrite:
                os.system("rm -f %s" % (outfitsfile))
                self.hdulist.writeto(outfitsfile)
            else:
                print("Warning: does not overwrite %s" % (outfitsfile))
        else:
            self.hdulist.writeto(outfitsfile)

    def to_aipscc(self, threshold=None, relative=True,
                    istokes=0, ifreq=0):
        '''
        Make AIPS CC table

        Args:
            istokes (integer): index for Stokes Parameter at which the image will be saved
            ifreq (integer): index for Frequency at which the image will be saved
            threshold (float): pixels with the absolute intensity smaller than this value will be ignored.
            relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="deg")
        X, Y = np.meshgrid(xg, yg)
        X = X.reshape(Nx * Ny)
        Y = Y.reshape(Nx * Ny)
        flux = self.data[istokes, ifreq]
        flux = flux.reshape(Nx * Ny)

        # threshold
        if threshold is None:
            thres = np.finfo(np.float64).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        # adopt threshold
        X = X[flux >= thres]
        Y = Y[flux >= thres]
        flux = flux[flux >= thres]

        # make table columns
        c1 = pyfits.Column(name='FLUX', array=flux, format='1E',unit='JY')
        c2 = pyfits.Column(name='DELTAX', array=X, format='1E',unit='DEGREES')
        c3 = pyfits.Column(name='DELTAY', array=Y, format='1E',unit='DEGREES')
        c4 = pyfits.Column(name='MAJOR AX', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c5 = pyfits.Column(name='MINOR AX', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c6 = pyfits.Column(name='POSANGLE', array=np.zeros(len(flux)), format='1E',unit='DEGREES')
        c7 = pyfits.Column(name='TYPE OBJ', array=np.zeros(len(flux)), format='1E',unit='CODE')

        # make CC table
        tab = pyfits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
        return tab

    def to_difmapmod(self, outfile, threshold=None, relative=True,
                     istokes=0, ifreq=0):
        '''
        Save an image into a difmap model file

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be saved
          ifreq (integer): index for Frequency at which the image will be saved
          threshold (float): pixels with the absolute intensity smaller than this value will be ignored.
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        Nx = self.header["nx"]
        Ny = self.header["ny"]
        xg, yg = self.get_xygrid(angunit="mas")
        X, Y = np.meshgrid(xg, yg)
        R = np.sqrt(X * X + Y * Y)
        theta = np.rad2deg(np.arctan2(X, Y))
        flux = self.data[istokes, ifreq]

        R = R.reshape(Nx * Ny)
        theta = theta.reshape(Nx * Ny)
        flux = flux.reshape(Nx * Ny)

        if threshold is None:
            thres = np.finfo(np.float32).eps
        else:
            if relative:
                thres = self.peak(istokes=istokes, ifreq=ifreq) * threshold
            else:
                thres = threshold
        thres = np.abs(thres)

        f = open(outfile, "w")
        for i in np.arange(Nx * Ny):
            if np.abs(flux[i]) < thres:
                continue
            line = "%20e %20e %20e\n" % (flux[i], R[i], theta[i])
            f.write(line)
        f.close()

    def save_fits(self, outfitsfile=None, overwrite=True, bunit="Jy/pixel"):
        '''
        save the image(s) to the image FITS file or HDUList.

        Args:
            outfitsfile (string; default is None):
                FITS file name. If not specified, then HDUList object will be
                returned.
            overwrite (boolean):
                It True, an existing file will be overwritten.
        Returns:
            HDUList object if outfitsfile is None
        '''
        print("Warning: this method will be removed soon. please use the 'to_fits' method.")
        if outfitsfile is None:
            self.to_fits(outfitsfile, overwrite, bunit)
        else:
            return self.to_fits(outfitsfile, overwrite, bunit)

    #-------------------------------------------------------------------------
    # Editing images
    #-------------------------------------------------------------------------
    def winmod(self, imregion, save_totalflux=False):
        '''
        clear brightness distribution outside regions

        Args:
            region (imdata.ImRegTable object):
                region data
            save_totalflux (boolean; default=False):
                if True, keep Totalflux
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if isinstance(imregion,imr.IMRegion):
            imagewin = imregion.imagewin(outfits)
        elif isinstance(imregion,IMFITS):
            imagewin = imregion.data[0,0] > 0.5
        else:
            imagewin = imregion

        for idxs in np.arange(self.header["ns"]):
            for idxf in np.arange(self.header["nf"]):
                image = outfits.data[idxs, idxf]
                masked = imagewin == False
                image[np.where(masked)] = 0
                outfits.data[idxs, idxf] = image
                if save_totalflux:
                    totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / image.sum()
        # Update and Return
        outfits.update_fits()
        return outfits

    def cpimage(self, fitsdata, save_totalflux=False, order=1):
        '''
        Copy the brightness ditribution of the input IMFITS object
        into the image grid of this image data.

        Args:
          fitsdata (imdata.IMFITS object):
            This image will be copied into the image grid of this data.
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.
        Returns:
          imdata.IMFITS object: the copied image data
        '''
        # generate output imfits object
        outfits = copy.deepcopy(self)

        dx0 = fitsdata.header["dx"]
        dy0 = fitsdata.header["dy"]
        Nx0 = fitsdata.header["nx"]
        Ny0 = fitsdata.header["ny"]
        Nxr0 = fitsdata.header["nxref"]
        Nyr0 = fitsdata.header["nyref"]

        dx1 = outfits.header["dx"]
        dy1 = outfits.header["dy"]
        Nx1 = outfits.header["nx"]
        Ny1 = outfits.header["ny"]
        Nxr1 = outfits.header["nxref"]
        Nyr1 = outfits.header["nyref"]

        coord = np.zeros([2, Nx1 * Ny1])
        xgrid = (np.arange(Nx1) + 1 - Nxr1) * dx1 / dx0 + Nxr0 - 1
        ygrid = (np.arange(Ny1) + 1 - Nyr1) * dy1 / dy0 + Nyr0 - 1
        x, y = np.meshgrid(xgrid, ygrid)
        coord[0, :] = y.reshape(Nx1 * Ny1)
        coord[1, :] = x.reshape(Nx1 * Ny1)

        for idxs, idxf in itertools.product(list(range(self.header["ns"])),list(range(self.header["nf"]))):
            outfits.data[idxs, idxf] = sn.map_coordinates(
                fitsdata.data[idxs, idxf], coord, order=order,
                mode='constant', cval=0.0, prefilter=True).reshape([Ny1, Nx1]
                                                                   ) * dx1 * dy1 / dx0 / dy0
            # Flux Scaling
            if save_totalflux:
                totalflux = fitsdata.totalflux(istokes=idxs, ifreq=idxf)
                outfits.data[idxs, idxf] *= totalflux / \
                    outfits.totalflux(istokes=idxs, ifreq=idxf)

        outfits.update_fits()
        return outfits

    def copy(self):
        '''
        Copy the brightness ditribution of the input IMFITS object
        '''

        return copy.deepcopy(self)


    #def convolve(self, kernelimage):

    #def convolve_geomodel(self, geomodel):

    def convolve_gauss(self, majsize, minsize=None, x0=0, y0=0,
                       pa=0., scale=1., angunit=None,
                       set_beam=True,
                       save_totalflux=False):
        '''
        Gaussian Convolution

        Args:
          x0, y0 (float): x, y shift of the convolved image in the unit of "angunit"
          majsize (float): Major Axis Size
          minsize (float): Minor Axis Size. If None, it will be same to the Major Axis Size (Circular Gaussian)
          angunit (string): Angular Unit for the sizes (uas, mas, asec or arcsec, amin or arcmin, degree)
          pa (float): Position Angle of the Gaussian
          scale (float; default=False): The sizes will be multiplied by this value.
          restore (boolean; default=False): if True, omit the flux normalizing factor
          set_beam (boolean; default=True): update header file with this blurring beam
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        Returns:
          imdata.IMFITS object
        '''
        if minsize is None:
            minsize = majsize

        if angunit is None:
            angunit = self.angunit
        angconv = util.angconv("deg", angunit)

        # Create outputdata
        outfits = copy.deepcopy(self)

        # Create Gaussian
        imextent = outfits.get_imextent(angunit)
        if x0 is None:
            x0 = 0.
        if y0 is None:
            y0 = 0.

        X = (np.arange(outfits.header["nx"]) - (outfits.header["nx"]-1)/2.) * outfits.header["dx"] * angconv
        Y = (np.arange(outfits.header["ny"]) - (outfits.header["ny"]-1)/2.) * outfits.header["dy"] * angconv
        X, Y = np.meshgrid(X,Y)
        cospa = np.cos(np.deg2rad(pa))
        sinpa = np.sin(np.deg2rad(pa))
        X1 = (X - x0) * cospa - (Y - y0) * sinpa
        Y1 = (X - x0) * sinpa + (Y - y0) * cospa
        majsig = majsize / np.sqrt(2 * np.log(2)) / 2 * scale
        minsig = minsize / np.sqrt(2 * np.log(2)) / 2 * scale
        gauss = np.exp(-X1 * X1 / 2 / minsig / minsig - Y1 * Y1 / 2 / majsig / majsig)
        #gauss /= 2*np.pi*majsig*minsig

        # Replace nan with zero
        gauss[np.isnan(gauss)] = 0

        # Convolusion (except:gauss is zero array)
        if np.any(gauss != 0):
            for idxs, idxf in itertools.product(list(range(outfits.header["ns"])),list(range(outfits.header["nf"]))):
                orgimage = outfits.data[idxs, idxf]
                newimage = convolve_fft(orgimage, gauss, normalize_kernel=True)
                outfits.data[idxs, idxf] = newimage
                # Flux Scaling
                if save_totalflux:
                    totalflux = self.totalflux(istokes=idxs, ifreq=idxf)
                    outfits.data[idxs, idxf] *= totalflux / outfits.totalflux(istokes=idxs, ifreq=idxf)

        if set_beam:
            outfits.set_beam(majsize=majsize, minsize=minsize, pa=pa, scale=scale, angunit=angunit)

        # Update and Return
        outfits.update_fits()
        return outfits

    def refshift(self,x0=0.,y0=0.,angunit=None,save_totalflux=False):
        '''
        Shift the reference position to the specified coordinate.

        Args:
          x0, y0 (float, default=0):
            RA, Dec coordinate of the reference position
          angunit (string, optional):
            The angular unit of the coordinate. If not specified, self.angunit
            will be used.
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.
        Returns:
          imdata.IMFIT object
        '''
        if angunit is None:
            angunit = self.angunit
        conv = util.angconv(angunit,"deg")

        newimage = copy.deepcopy(self)
        newimage.header["nxref"]+=x0*conv/self.header["dx"]
        newimage.header["nyref"]+=y0*conv/self.header["dy"]
        newimage.update_fits()
        newimage_shift = self.cpimage(newimage, save_totalflux=save_totalflux)

        return newimage_shift

    def comshift(self, alpha=1., save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift the image so that its center-of-mass position coincides with the reference pixel.

        Args:
          alpha (float):
            if alpha != 0, then the image is powered by alpha prior to compute
            the center of the mass.
          istokes (integer):
            index for Stokes Parameter at which the shift will be computed.
          ifreq (integer):
            index for Frequency at which the shift will be computed.
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        pos = self.compos(alpha=alpha,ifreq=ifreq, istokes=istokes)
        return self.refshift(save_totalflux=save_totalflux, **pos)

    def peakshift(self, save_totalflux=False, ifreq=0, istokes=0):
        '''
        Shift the image so that its peak position coincides with the reference pixel.

        Arg:
          istokes (integer):
            index for Stokes Parameter at which the image will be edited
          ifreq (integer):
            index for Frequency at which the image will be edited
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.

        Returns:
          imdata.IMFITS object
        '''
        pos = self.peakpos(ifreq=ifreq, istokes=istokes)
        return self.refshift(save_totalflux=save_totalflux, **pos)

    def rotate(self, angle=0, deg=True, save_totalflux=False):
        '''
        Rotate the input image

        Args:
          angle (float): Rotational Angle. Anti-clockwise direction will be positive (same to the Position Angle).
          deg (boolean): It true, then the unit of angle will be degree. Otherwise, it will be radian.
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        Returns:
          imdata.IMFIT object
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if deg:
            degangle = -angle
            radangle = -np.deg2rad(angle)
        else:
            degangle = -np.rad2deg(angle)
            radangle = -angle
        #cosa = np.cos(radangle)
        #sina = np.sin(radangle)
        Nx = outfits.header["nx"]
        Ny = outfits.header["ny"]
        for istokes, ifreq in itertools.product(list(range(self.header["ns"])),list(range(self.header["nf"]))):
            image = outfits.data[istokes, ifreq]
            # rotate data
            newimage = sn.rotate(image, degangle)
            # get the size of new data
            My = newimage.shape[0]
            Mx = newimage.shape[1]
            # take the center of the rotated image
            outfits.data[istokes, ifreq] = newimage[My // 2 - Ny // 2:My // 2 - Ny // 2 + Ny,
                                                    Mx // 2 - Nx // 2:Mx // 2 - Nx // 2 + Nx]
            # Flux Scaling
            if save_totalflux:
                totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
                outfits.data[istokes, ifreq] *= totalflux / \
                    outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def min_threshold(self, threshold=0.0, replace=0.0,
                      relative=True, save_totalflux=False,
                      istokes=0, ifreq=0):
        '''
        This is thresholding with the mininum value. This is slightly different
        from hard thresholding, since this function resets all of pixels where
        their brightness is smaller than a given threshold. On the other hand,
        hard thresholding resets all of pixels where the absolute of their
        brightness is smaller than the threshold.

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          replace (float): the brightness to be replaced for thresholded pixels.
          relative (boolean): If true, theshold & Replace value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = threshold * self.peak(istokes=istokes, ifreq=ifreq)
            repla = replace * self.peak(istokes=istokes, ifreq=ifreq)
        else:
            thres = threshold
            repla = replace

        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(self.data[istokes, ifreq] < thres)
        image[t] = repla
        outfits.data[istokes, ifreq] = image

        # flux scaling
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def hard_threshold(self, threshold=0.01, relative=True, save_totalflux=False,
                       istokes=0, ifreq=0):
        '''
        Do hard-threshold the input image

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = np.abs(threshold * self.peak(istokes=istokes, ifreq=ifreq))
        else:
            thres = np.abs(threshold)
        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(np.abs(self.data[istokes, ifreq]) < thres)
        image[t] = 0
        outfits.data[istokes, ifreq] = image
        # flux scaling
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits

    def soft_threshold(self, threshold=0.01, relative=True, save_totalflux=False,
                       istokes=0, ifreq=0):
        '''
        Do soft-threshold the input image

        Args:
          istokes (integer): index for Stokes Parameter at which the image will be edited
          ifreq (integer): index for Frequency at which the image will be edited
          threshold (float): threshold
          relative (boolean): If true, theshold value will be normalized with the peak intensity of the image
          save_totalflux (boolean): If true, the total flux of the image will be conserved.
        '''
        # create output fits
        outfits = copy.deepcopy(self)
        if relative:
            thres = np.abs(threshold * self.peak(istokes=istokes, ifreq=ifreq))
        else:
            thres = np.abs(threshold)
        # thresholding
        image = outfits.data[istokes, ifreq]
        t = np.where(np.abs(self.data[istokes, ifreq]) < thres)
        image[t] = 0
        t = np.where(self.data[istokes, ifreq] >= thres)
        image[t] -= thres
        t = np.where(self.data[istokes, ifreq] <= -thres)
        image[t] += thres
        outfits.data[istokes, ifreq] = image
        if save_totalflux:
            totalflux = self.totalflux(istokes=istokes, ifreq=ifreq)
            outfits.data[istokes, ifreq] *= totalflux / \
                outfits.totalflux(istokes=istokes, ifreq=ifreq)
        outfits.update_fits()
        return outfits
    '''
    def add_geomodel(self, geomodel, istokes=0, ifreq=0, overwrite=False, usetheano=False):
        # copy self (for output)
        outfits = copy.deepcopy(self)

        # compile funcitons
        x, y = sp.symbols("x y", real=True)
        expr = geomodel.I(x, y).simplify()
        if usetheano:
            func = theano_function([x,y], [expr], dims={x:2, y:2}, dtypes={x: 'float64', y: 'float64'})
        else:
            func = sp.lambdify([x, y], expr, "numpy")

        # change headers
        dx=np.abs(self.header["dx"])*util.angconv("deg", "rad")
        dy=np.abs(self.header["dy"])*util.angconv("deg", "rad")
        nx=self.header["nx"]
        ny=self.header["ny"]
        x,y = self.get_xygrid(twodim=True, angunit="rad")
        I = func(x,y)*dx*dy

        if overwrite:
            outfits.data[istokes,ifreq,:,:] = I
        else:
            outfits.data[istokes,ifreq,:,:] += I
        return outfits
    '''

    def add_gauss(self, x0=0., y0=0., totalflux=1., majsize=1., minsize=None, scale=1.0,
                  pa=0., overwrite=False, istokes=0, ifreq=0, fluxunit="Jy", angunit=None):
        '''
        Gaussian Convolution

        Args:
          x0, y0 (float): the peak location of the gaussian in the unit of "angunit"
          totalflux (float): total flux density
          majsize (float): Major Axis Size
          minsize (float): Minor Axis Size. If None, it will be same to the Major Axis Size (Circular Gaussian)
          fluxunit (string): unit of the total flux density (Jy, mJy, uJy, si, cgs)
          angunit (string): Angular Unit for the sizes (uas, mas, asec or arcsec, amin or arcmin, degree)
          pa (float): Position Angle of the Gaussian
          scale (float): The sizes will be multiplied by this value.
          overwrite (boolean; default=False): If True, the image will be overwritten by this Gaussian.
        Returns:
          imdata.IMFITS object
        '''
        if angunit is None:
            angunit = self.angunit

        # copy self (for output)
        outfits = copy.deepcopy(self)

        # get size
        thmaj = majsize
        if minsize is None:
            thmin = thmaj
        else:
            thmin = minsize

        # Calc X,Y grid
        X, Y = self.get_xygrid(twodim=True, angunit=angunit)

        # Calc Gaussian Distribution
        X1 = X - x0
        Y1 = Y - y0
        cospa = np.cos(np.deg2rad(pa))
        sinpa = np.sin(np.deg2rad(pa))
        X2 = X1 * cospa - Y1 * sinpa
        Y2 = X1 * sinpa + Y1 * cospa
        majsig = thmaj / np.sqrt(2 * np.log(2)) / 2
        minsig = thmin / np.sqrt(2 * np.log(2)) / 2
        gauss = np.exp(-X2 * X2 / 2 / minsig / minsig -
                       Y2 * Y2 / 2 / majsig / majsig)
        gauss /= gauss.sum()
        gauss *= totalflux * util.fluxconv(fluxunit,"Jy")

        # add to original FITS file
        if overwrite:
            outfits.data[istokes,ifreq,:,:] = gauss
        else:
            outfits.data[istokes,ifreq,:,:] += gauss
        return outfits


    def nxcorr(self,refimage):
        '''
        Computing normalized cross correlation with input image

        Args:
          refimage: Referenced fitsimage used in calculating cross correlation
        Returns:
          imdata.IMFITS object
         '''
        # Adjusting image pixcels of two images for cross corr
        grid_self = refimage.cpimage(self)

        # get 2d arr
        im_arr  = grid_self.data[0,0]
        ref_arr = np.fliplr(np.flipud(refimage.data[0,0]))

        # the maximum peak of autocorr
        fact = 1/np.sqrt(np.sum(im_arr*im_arr) * np.sum(ref_arr*ref_arr))

        # Setting results coodinate to matching reference flame
        image_cc = copy.deepcopy(refimage)
        image_cc.header["nxref"] = refimage.header["nx"]/2+1
        image_cc.header["nyref"] = refimage.header["ny"]/2+1

        normfunc = lambda x: 1
        crosscorr = convolve_fft(ref_arr, im_arr, normalize_kernel=normfunc)*fact
        image_cc.data[0,0] = np.fliplr(np.flipud(crosscorr))
        return image_cc


    def nxcorrpos(self, refimage):
        '''
        Computing the offset position between two images using cross correlation

        Args:
          refimage: Referenced fitsimage used in calculating cross correlation
        Returns:
           dictionary for the offset position
         '''
        nxcorr_image = self.nxcorr(refimage)
        return nxcorr_image.peakpos()

    def nxcorrshift(self, refimage, save_totalflux=True):
        '''
        Automatically obtaining the shifted image using cross correlation

        Args:
          refimage:
            Referenced fitsimage used in calculating cross correlation
          save_totalflux (boolean):
            If true, the total flux of the image will be conserved.
        Returns:
           imdata.IMFITS object
         '''
        shift = self.nxcorrpos(refimage)
        x0=shift['x0']
        y0=shift['y0']
        angunit=shift['angunit']
        return self.refshift(x0=-x0,y0=-y0,angunit=angunit,save_totalflux=save_totalflux)

    #---------------------------------------------------------------------------
    # Feature Extraction
    #---------------------------------------------------------------------------
    def edge_detect(self, method="prewitt", mask=None, sigma=1,
                    low_threshold=0.1, high_threshold=0.2):
        '''
        Output edge-highlighted images.

        Args:
          method (string, default="prewitt"):
            Type of edge filters to be used.
            Availables are ["prewitt","sobel","scharr","roberts","canny"].
          mask (array):
            array for masking
          sigma (integer):
            index for canny
          low_threshold (float):
            index for canny
          high_threshold (float):
            index for canny

        Returns:
          imdata.IMFITS object
        '''
        from skimage.filters import prewitt, sobel, scharr, roberts
        from skimage.feature import canny

        # copy self (for output)
        outfits = copy.deepcopy(self)

        # get information
        nstokes = outfits.header["ns"]
        nif = outfits.header["nf"]
        # detect edge
        # prewitt
        if method == "prewitt":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = prewitt(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = prewitt(
                            outfits.data[idxs, idxf], mask=mask)
        # sobel
        if method == "sobel":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = sobel(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = sobel(
                            outfits.data[idxs, idxf], mask=mask)
        # scharr
        if method == "scharr":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = scharr(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = scharr(
                            outfits.data[idxs, idxf], mask=mask)
        # roberts
        if method == "roberts":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = roberts(
                            outfits.data[idxs, idxf])
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = roberts(
                            outfits.data[idxs, idxf], mask=mask)
        # canny
        if method == "canny":
            if mask is None:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = canny(
                            outfits.data[idxs, idxf], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)
            else:
                for idxs in np.arange(nstokes):
                    for idxf in np.arange(nif):
                        outfits.data[idxs, idxf] = canny(outfits.data[idxs, idxf], mask=mask, sigma=sigma,
                                                         low_threshold=low_threshold, high_threshold=high_threshold, use_quantiles=True)

        outfits.update_fits()
        return outfits

    def circle_hough(self, radius, ntheta=360,
                     angunit=None, istokes=0, ifreq=0):
        '''
        A function calculates the circle Hough transform (CHT) of the input image

        Args:
          radius (array):
            array for radii for which the circle Hough transform is
            calculated. The unit of the radius is specified with angunit.
          Ntheta (optional, integer):
            The number of circular shifts to be used in the circle Hough transform.
            For instance, ntheta=360 (default) gives circular shifts of every 1 deg.
          angunit (optional, string):
            The angular unit for radius and also the output peak profile
          istokes (integer): index for Stokes Parameter at which the CHT to be performed
          ifreq (integer): index for Frequency at which the CHT to be performed

        Returns:
          H (ndarray):
            The Circle Hough Accumulator. This is a three dimensional array of which
            shape is [Nx, Ny, Nr] in *Fortran Order*.
          profile (pd.DataFrame):
            The table for the peak profile Hr(r)=max_r(H(x,y,r)).
        '''
        if angunit is None:
            angunit = self.angunit

        Nr = len(radius)
        Nx = self.header["nx"]
        Ny = self.header["ny"]

        # get xy-coordinates
        xgrid, ygrid = self.get_xygrid(angunit=angunit)
        if self.header["dx"] < 0:
            sgnx = -1
        else:
            sgnx = 1
        if self.header["dy"] < 0:
            sgny = -1
        else:
            sgny = 1

        # calculate circle hough transform
        H = fortlib.houghlib.circle_hough(self.data[istokes, ifreq],
                                           sgnx * xgrid, sgny * ygrid,
                                           radius, np.int32(ntheta))
        isfort = np.isfortran(H)

        # make peak profile
        profile = pd.DataFrame()
        profile["ir"] = np.arange(Nr)
        profile["r"] = radius
        profile["xpeak"] = np.zeros(Nr)
        profile["ypeak"] = np.zeros(Nr)
        profile["ixpeak"] = np.zeros(Nr, dtype=np.int64)
        profile["iypeak"] = np.zeros(Nr, dtype=np.int64)
        profile["hpeak"] = np.zeros(Nr)
        if isfort:
            for i in np.arange(Nr):
                profile.loc[i, "hpeak"] = np.max(H[:, :, i])
                peakxyidx = np.unravel_index(
                    np.argmax(H[:, :, i]), dims=[Ny, Nx])
                profile.loc[i, "xpeak"] = xgrid[peakxyidx[1]]
                profile.loc[i, "ypeak"] = ygrid[peakxyidx[0]]
                profile.loc[i, "ixpeak"] = peakxyidx[1]
                profile.loc[i, "iypeak"] = peakxyidx[0]
        else:
            for i in np.arange(Nr):
                profile.loc[i, "hpeak"] = np.max(H[i, :, :])
                peakxyidx = np.unravel_index(
                    np.argmax(H[i, :, :]), dims=[Ny, Nx])
                profile.loc[i, "xpeak"] = xgrid[peakxyidx[1]]
                profile.loc[i, "ypeak"] = ygrid[peakxyidx[0]]
                profile.loc[i, "ixpeak"] = peakxyidx[1]
                profile.loc[i, "iypeak"] = peakxyidx[0]
        return H, profile


#-------------------------------------------------------------------------
# Calculate Matrix Among Images
#-------------------------------------------------------------------------
def calc_metric(fitsdata, reffitsdata, metric="NRMSE", istokes1=0, ifreq1=0, istokes2=0, ifreq2=0, edgeflag=False):
    '''
    Calculate metrics between two images

    Args:
      fitsdata (imdata.IMFITS object):
        input image

      reffitsdata (imdata.IMFITS object):
        reference image

      metric (string):
        type of a metric to be calculated.
        Availables are ["NRMSE","MSE","SSIM","DSSIM"]

      istokes1 (integer):
        index for the Stokes axis of the input image

      ifreq1 (integer):
        index for the frequency axis of the input image

      istokes2 (integer):
        index for the Stokes axis of the reference image

      ifreq2 (integer):
        index for the frequency axis of the reference image

      edgeflag (boolean):
        calculation of metric on image domain or image gradient domain

    Returns:
      metrics
    '''
    from skimage.filters import prewitt

    # adjust resolution and FOV
    fitsdata2 = copy.deepcopy(fitsdata)
    reffitsdata2 = copy.deepcopy(reffitsdata)
    fitsdata2 = reffitsdata2.cpimage(fitsdata2)
    # edge detection
    if edgeflag:
        fitsdata2 = fitsdata2.edge_detect(method="sobel")
        reffitsdata2 = reffitsdata2.edge_detect(method="sobel")
    # get image data
    inpimage = fitsdata2.data[istokes1, ifreq1]
    refimage = reffitsdata2.data[istokes2, ifreq2]
    # calculate metric
    if metric == "NRMSE" or metric == "MSE":
        metrics = np.sum((inpimage - refimage)**2)
        metrics /= np.sum(refimage**2)
    if metric == "SSIM" or metric == "DSSIM":
        meanI = np.mean(inpimage)
        meanK = np.mean(refimage)
        stdI = np.std(inpimage, ddof=1)
        stdK = np.std(refimage, ddof=1)
        cov = np.sum((inpimage - meanI) * (refimage - meanK)) / \
            (inpimage.size - 1)
        metrics = (2 * meanI * meanK / (meanI**2 + meanK**2)) * \
            (2 * stdI * stdK / (stdI**2 + stdK**2)) * (cov / (stdI * stdK))
    if metric == "NRMSE":
        metrics = np.sqrt(metrics)
    if metric == "DSSIM":
        metrics = 1 / abs(metrics) - 1

    return metrics


def _plot_beam_ellipse(Dmaj, Dmin, PA):
    theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    x = 0.5 * Dmin * np.sin(theta)
    y = 0.5 * Dmaj * np.cos(theta)

    rtheta = -np.radians(PA)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
        ])
    x, y = np.dot(R, np.array([x, y]))
    return x,y

def _plot_beam_box(Lx, Ly):
    x = np.array([0,Lx,Lx,0,0]) - Lx/2.
    y = np.array([0,0,Ly,Ly,0]) - Ly/2.
    return x,y
'''
#-------------------------------------------------------------------------
# Fllowings are subfunctions for ds9flag and read_cleanbox
#-------------------------------------------------------------------------
def get_flagpixels(regfile, X, Y):
    # Read DS9-region file
    f = open(regfile)
    lines = f.readlines()
    f.close()
    keep = np.zeros(X.shape, dtype="Bool")
    # Read each line
    for line in lines:
        # Skipping line
        if line[0] == "#":
            continue
        if "image" in line == True:
            continue
        if "(" in line == False:
            continue
        if "global" in line:
            continue
        # Replacing many characters to empty spaces
        line = line.replace("(", " ")
        line = line.replace(")", " ")
        while "," in line:
            line = line.replace(",", " ")
        # split line to elements
        elements = line.split(" ")
        while "" in elements:
            elements.remove("")
        while "\n" in elements:
            elements.remove("\n")
        if len(elements) < 4:
            continue
        # Check whether the box is for "inclusion" or "exclusion"
        if elements[0][0] == "-":
            elements[0] = elements[0][1:]
            exclusion = True
        else:
            exclusion = False
        if elements[0] == "box":
            tmpkeep = region_box(X, Y,
                                  x0=np.float64(elements[1]),
                                  y0=np.float64(elements[2]),
                                  width=np.float64(elements[3]),
                                  height=np.float64(elements[4]),
                                  angle=np.float64(elements[5]))
        elif elements[0] == "circle":
            tmpkeep = region_circle(X, Y,
                                     x0=np.float64(elements[1]),
                                     y0=np.float64(elements[2]),
                                     radius=np.float64(elements[3]))
        elif elements[0] == "ellipse":
            tmpkeep = region_ellipse(X, Y,
                                      x0=np.float64(elements[1]),
                                      y0=np.float64(elements[2]),
                                      radius1=np.float64(elements[3]),
                                      radius2=np.float64(elements[4]),
                                      angle=np.float64(elements[5]))
        else:
            print("[WARNING] The shape %s is not available." % (elements[0]))
        if not exclusion:
            keep += tmpkeep
        else:
            keep[np.where(tmpkeep)] = False
    return keep


def region_box(X, Y, x0, y0, width, height, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa + dY * sina
    Y1 = -dX * sina + dY * cosa
    region = (Y1 >= -np.abs(height) / 2.)
    region *= (Y1 <= np.abs(height) / 2.)
    region *= (X1 >= -np.abs(width) / 2.)
    region *= (X1 <= np.abs(width) / 2.)
    return region


def region_circle(X, Y, x0, y0, radius):
    return (X - x0) * (X - x0) + (Y - y0) * (Y - y0) <= radius * radius


def region_ellipse(X, Y, x0, y0, radius1, radius2, angle):
    cosa = np.cos(np.deg2rad(angle))
    sina = np.sin(np.deg2rad(angle))
    dX = X - x0
    dY = Y - y0
    X1 = dX * cosa + dY * sina
    Y1 = -dX * sina + dY * cosa
    return X1 * X1 / radius1 / radius1 + Y1 * Y1 / radius2 / radius2 <= 1
'''
