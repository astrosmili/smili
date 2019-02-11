#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of smili handling various types of Visibility data sets.
'''
__author__ = "Smili Developer Team"

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

# internal
from ... import imdata, fortlib
from ...util import prt
from ..uvtable import VisTable
from ..cltable import CLTable

indent = "  "

# ------------------------------------------------------------------------------
# Classes for UVFITS FILE
# ------------------------------------------------------------------------------
stokesDict = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    -1: "RR",
    -2: "LL",
    -3: "RL",
    -4: "LR",
    -5: "XX",
    -6: "YY",
    -7: "XY",
    -8: "YX"
}
stokesDictinv = {}
for key in stokesDict.keys():
    val = stokesDict[key]
    stokesDictinv[val] = key

# ------------------------------------------------------------------------------
# Classes for UVFITS FILE
# ------------------------------------------------------------------------------
class UVFITS(object):
    '''
    This is a class to load, edit, write uvfits data
    '''
    def __init__(self, uvfits):
        '''
        Load an uvfits file. Currently, this function can read only
        single-source uvfits file. The data will be uv-sorted.

        Args:
          infile (string or pyfits.HDUList object): input uvfits data

        Returns:
          uvdata.UVFITS object
        '''
        # check input files
        if type(uvfits) == type(""):
            hdulist = pf.open(uvfits)
        else:
            hdulist = uvfits
        hdulist.info()
        print("")
        hduinfo = hdulist.info(output=False)
        Nhdu = len(hduinfo)

        # Read hdus
        FQtab = None
        ANtabs = {}
        SUtabs = {}
        ghdu = None
        prt("Loading HDUs in the input UVFITS files.")
        for ihdu in xrange(Nhdu):
            hduname = hduinfo[ihdu][1]
            if hduname == "PRIMARY":
                if ghdu is not None:
                    prt("[WARNING] This UVFITS has more than two Primary HDUs.",indent)
                    prt("          The later one will be taken.",indent)
                else:
                    prt("Primary HDU was loaded.", indent)
                ghdu = hdulist[ihdu]
            if hduname == "AIPS FQ":
                if FQtab is not None:
                    prt("[WARNING] This UVFITS has more than two AIPS FQ tables.",indent)
                    prt("          The later one will be taken.",indent)
                else:
                    prt("AIPS FQ Table was loaded.", indent)
                FQtab = hdulist[ihdu]
            if hduname == "AIPS AN":
                subarrid = np.int64(hdulist[ihdu].header.get("EXTVER"))
                if subarrid == -1:
                    subarrid = 1
                if subarrid in ANtabs.keys():
                    prt("[WARNING] There are duplicated subarrays with subarray ID=%d."%(subarrid),indent)
                    pri("          The later one will be adopted.", indent)
                else:
                    prt("Subarray %d was found in an AIPS AN table"%(subarrid), indent)
                ANtabs[subarrid] = hdulist[ihdu]
            if hduname == "AIPS SU":
                suid = np.int64(hdulist[ihdu].header.get("FREQID"))
                if suid in SUtabs.keys():
                    prt("[WARNING] There are more than two SU tables for the same Frequency setup frqselid=%d."%(suid),indent)
                    prt("          The later one will be adopted.",indent)
                else:
                    prt("A SU Table for a frequency Setup %d was found"%(suid),indent)
                SUtabs[suid] = hdulist[ihdu]

        # Check number of AIPS FQ/AN tables loaded.
        print("")
        prt("Checking loaded HDUs.")
        # Group HDU
        if ghdu is None:
            errmsg = "No GroupHDUs are included in the input UVFITS data."
            raise ValueError(errmsg)
        # AIPS FQ Table
        if FQtab is None:
            errmsg = "No FQ tables are included in the input UVFITS data."
            raise ValueError(errmsg)
        # AN Table
        if len(ANtabs)==0:
            errmsg = "No AN tables are included in the input UVFITS data."
            raise ValueError(errmsg)
        else:
            prt("%d Subarray settings are found."%(len(ANtabs)),indent)
        # AIPS SU Table
        if len(SUtabs)==0:
            prt("No AIPS SU tables were found.",indent)
            prt("  Assuming that this is a single source UVFITS file.",indent)
            self.ismultisrc = False
        else:
            prt("AIPS SU tables were found.",indent)
            prt("  Assuming that this is a multi source UVFITS file.",indent)
            self.ismultisrc = True

        #if self.ismultisrc:
        #    raise ImportError("Sorry, this library currently can read only single-source UVFITS files.")

        print("")
        prt("Reading FQ Tables")
        self._read_freqdata(FQtab)

        # Read AN Tables
        print("")
        prt("Reading AN Tables")
        self._read_arraydata(ANtabs)

        # Read SU Tables
        print("")
        if self.ismultisrc:
            prt("Reading SU Tables")
            self._read_srcdata_multi(SUtabs)
        else:
            prt("Reading Source Information from Primary HDU")
            self._read_srcdata_single(ghdu)

        # Load GroupHDU
        print("")
        prt("Reading Primary HDU data")
        self._read_grouphdu(ghdu, timescale="utc")

        # Load Stokes Parameters
        stokesid = np.arange(ghdu.header["NAXIS3"])+1-ghdu.header["CRPIX3"]
        stokesid*= ghdu.header["CDELT3"]
        stokesid+= ghdu.header["CRVAL3"]
        self.stokes = [stokesDict[int(sid)] for sid in stokesid]

        # Load OBSERVER
        #self.observer = ghdu.header["OBSERVER"]

    def _read_arraydata(self, ANtabs):
        subarrays = {}
        subarrids = ANtabs.keys()
        for subarrid in subarrids:
            # Create Array data
            arrdata = ArrayData()

            # AN Table
            ANtab = ANtabs[subarrid]

            # Read AN Table Header
            ANheadkeys = ANtab.header.keys()
            headkeys = arrdata.header.keys()
            for key in ANheadkeys:
                if key in headkeys:
                    arrdata.header[key] = ANtab.header.get(key)
            arrdata.frqsel = ANtab.header.get("FREQID")
            if arrdata.frqsel is None:
                arrdata.frqsel=1
            elif arrdata.frqsel < 1:
                prt("[WARNING] Negative FRQSEL in AIPS AN Table (subarray=%d)"%(subarrid), indent)
                prt("          FRQSEL=1 is set for this subarray", indent)
                arrdata.frqsel=1
            arrdata.subarray = subarrid

            # Read AN Table Data
            arrdata.antable["name"] = ANtab.data["ANNAME"]
            arrdata.antable["x"] = ANtab.data["STABXYZ"][:,0]
            arrdata.antable["y"] = ANtab.data["STABXYZ"][:,1]
            arrdata.antable["z"] = ANtab.data["STABXYZ"][:,2]
            arrdata.antable["id"] = ANtab.data["NOSTA"]
            arrdata.antable["mnttype"] = ANtab.data["MNTSTA"]
            arrdata.antable["axisoffset"] = ANtab.data["STAXOF"]
            arrdata.antable["poltypeA"] = ANtab.data["POLTYA"]
            arrdata.antable["polangA"] = ANtab.data["POLAA"]
            arrdata.antable["poltypeB"] = ANtab.data["POLTYB"]
            arrdata.antable["polangB"] = ANtab.data["POLAB"]
            arrdata.anorbparm = ANtab.data["ORBPARM"]
            arrdata.anpolcalA = ANtab.data["POLCALA"]
            arrdata.anpolcalB = ANtab.data["POLCALB"]
            subarrays[subarrid]=arrdata

            arrdata.check()   # doing sanity check
            prt(arrdata, indent)
        self.subarrays = subarrays

    def _read_freqdata(self, FQtab):
        freqdata = FrequencyData()

        # Load Data
        freqdata.frqsels = FQtab.data["FRQSEL"]
        Nfrqsel = len(freqdata.frqsels)
        for i in xrange(Nfrqsel):
            frqsel = freqdata.frqsels[i]
            fqdic = {
                "if_freq_offset":FQtab.data["IF FREQ"][i],
                "ch_bandwidth":FQtab.data["CH WIDTH"][i],
                "if_bandwidth":FQtab.data["TOTAL BANDWIDTH"][i],
                "sideband":FQtab.data["SIDEBAND"][i]
            }
            if np.isscalar(fqdic["if_freq_offset"]):
                index=[0]
            else:
                index=None
            fqtable = pd.DataFrame(fqdic,columns=freqdata.fqtable_cols,index=index)
            freqdata.fqtables[frqsel] = fqtable
        prt(freqdata, indent)

        self.fqsetup = freqdata
        if Nfrqsel==1:
            self.ismultifrq = False
        else:
            self.ismultifrq = True

    def _read_srcdata_single(self, ghdu):
        # Create Array data
        srcdata = SourceData()

        # Read Header
        srcdata.frqsel = 1
        srcdata.header["VELDEF"] = "RADIO"
        srcdata.header["VELTYP"] = "GEOCENTR"
        srcdata.header["NO_IF"] = ghdu.header.get("NAXIS5")

        # Load Data
        srcdata.sutable["id"] = np.asarray([1], dtype=np.int64)
        srcdata.sutable["source"] = np.asarray([ghdu.header.get("OBJECT")])
        srcdata.sutable["qual"] = np.asarray([0], dtype=np.int64)
        srcdata.sutable["calcode"] = np.asarray([""])
        srcdata.sutable["bandwidth"] = np.asarray([ghdu.header.get("CDELT4")], dtype=np.float64)
        if "EQUINOX" in ghdu.header.keys():
            equinox = ghdu.header.get("EQUINOX")
        elif "EPOCH" in ghdu.header.keys():
            equinox = ghdu.header.get("EPOCH")
        else:
            equinox = 2000.0
        if isinstance(equinox, str) or isinstance(equinox, unicode):
            if "J" in equinox: equinox = equinox.replace("J","")
            if "B" in equinox: equinox = equinox.replace("B","")
            equinox = np.float64(equinox)
        srcdata.sutable["equinox"] = np.asarray([equinox], dtype=np.float64)
        srcdata.sutable["ra_app"] = np.asarray([ghdu.header.get("CRVAL6")], dtype=np.float64)
        srcdata.sutable["dec_app"] = np.asarray([ghdu.header.get("CRVAL7")], dtype=np.float64)
        srcdata.sutable["pmra"] = np.asarray([0.0], dtype=np.float64)
        srcdata.sutable["pmdec"] = np.asarray([0.0], dtype=np.float64)
        srcdata.suiflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.suqflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.suuflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.suvflux = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.sufreqoff = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.sulsrvel = np.zeros([1, srcdata.header["NO_IF"]])
        srcdata.surestfreq = np.zeros([1, srcdata.header["NO_IF"]])

        # Here I note that we assume equinox of coordinates will be J2000.0
        # which is the currend default of acd.SkyCoord (Feb 28 2018)
        radec = acd.SkyCoord(ra=[ghdu.header.get("CRVAL6")], dec=[ghdu.header.get("CRVAL7")],
                             equinox="J%f"%(srcdata.sutable.loc[0,"equinox"]),
                             unit="deg",
                             frame="icrs")
        srcdata.sutable["radec"] = radec.to_string("hmsdms")

        prt(srcdata, indent)
        sources = {}
        sources[1]=srcdata
        self.sources = sources

    def _read_srcdata_multi(self, SUtabs):
        sources = {}
        srclist = []
        frqselids = SUtabs.keys()
        for frqselid in frqselids:
            # Create Array data
            srcdata = SourceData()

            # FQ Table
            SUtab = SUtabs[frqselid]

            # Read Header
            SUheadkeys = SUtab.header.keys()
            headkeys = srcdata.header.keys()
            for key in SUheadkeys:
                if key in headkeys:
                    srcdata.header[key] = SUtab.header.get(key)
            srcdata.frqsel = SUtab.header.get("FREQID")

            # Load Data
            srcdata.sutable["id"] = SUtab.data["ID. NO."]
            srcdata.sutable["source"] = SUtab.data["SOURCE"]
            srcdata.sutable["qual"] = SUtab.data["QUAL"]
            srcdata.sutable["calcode"] = SUtab.data["CALCODE"]
            srcdata.sutable["bandwidth"] = SUtab.data["BANDWIDTH"]
            srcdata.sutable["equinox"] = SUtab.data["EPOCH"]
            srcdata.sutable["ra_app"] = SUtab.data["RAAPP"]
            srcdata.sutable["dec_app"] = SUtab.data["DECAPP"]
            srcdata.sutable["pmra"] = SUtab.data["PMRA"]
            srcdata.sutable["pmdec"] = SUtab.data["PMDEC"]
            srcdata.iflux = SUtab.data["IFLUX"]
            srcdata.qflux = SUtab.data["QFLUX"]
            srcdata.uflux = SUtab.data["UFLUX"]
            srcdata.vflux = SUtab.data["VFLUX"]
            srcdata.freqoff = SUtab.data["FREQOFF"]
            srcdata.lsrvel = SUtab.data["LSRVEL"]
            srcdata.restfreq = SUtab.data["RESTFREQ"]
            radec = acd.SkyCoord(ra=SUtab.data["RAEPO"], dec=SUtab.data["DECEPO"],
                                 equinox="J%f"%(srcdata.sutable.loc[0,"equinox"]),
                                 unit="deg",
                                 frame="icrs")
            srcdata.sutable["radec"] = radec.to_string("hmsdms")

            srclist += list(set(srcdata.sutable["source"]))
            sources[frqselid]=srcdata
            prt(srcdata, indent)
        srclist = list(set(srclist))
        if len(srclist) == 1:
            self.ismultisrc = False
        self.sources = sources

    def _read_grouphdu(self, hdu, timescale="utc"):
        visdata = VisibilityData()

        # Read data
        visdata.data = hdu.data.data
        Ndata = visdata.data.shape[0]

        # Read Random Parameters
        paridxes = [None for i in xrange(9)]
        parnames = hdu.data.parnames
        Npar = len(parnames)
        visdata.coord = pd.DataFrame()
        for i in xrange(Npar):
            parname = parnames[i]
            if "UU" in parname:
                paridxes[0] = i+1
                visdata.coord["usec"] = np.float64(hdu.data.par(i))
            if "VV" in parname:
                paridxes[1] = i+1
                visdata.coord["vsec"] = np.float64(hdu.data.par(i))
            if "WW" in parname:
                paridxes[2] = i+1
                visdata.coord["wsec"] = np.float64(hdu.data.par(i))
            if "DATE" in parname:
                if paridxes[3] is None:
                    paridxes[3] = i+1
                    jd1 = np.float64(hdu.data.par(i))
                elif paridxes[4] is None:
                    paridxes[4] = i+1
                    jd2 = np.float64(hdu.data.par(i))
                else:
                    errmsg = "Random Parameters have too many 'DATE' columns."
                    raise ValueError(errmsg)
            if "BASELINE" in parname:
                paridxes[5] = i+1
                bl = np.float64(hdu.data.par(i))
            if "SOURCE" in parname:
                paridxes[6] = i+1
                visdata.coord["source"] = np.int64(hdu.data.par(i))
            if "INTTIM" in parname:
                paridxes[7] = i+1
                visdata.coord["inttim"] = np.float64(hdu.data.par(i))
            if "FREQSEL" in parname:
                paridxes[8] = i+1
                visdata.coord["freqsel"] = np.int64(hdu.data.par(i))

        # Check Loaded Random Parameters
        #   INTTIM
        if paridxes[7] is None:
            warnmsg = "Warning: this data do not have a random parameter for the integration time\n"
            visdata.coord["inttim"] = np.zeros(visdata.coord.shape[0], dtype=np.float64)
            visdata.coord.loc[:,"inttim"] = -1
            paridxes[7] = -1

        #   Source
        if self.ismultisrc and (paridxes[6] is None):
            errmsg = "Random Parameters do not have 'SOURCE' although UVFITS is for multi sources."
            raise ValueError(errmsg)
        elif (self.ismultisrc is False) and (paridxes[6] is None):
            visdata.coord["source"] = np.asarray([1 for i in xrange(Ndata)])
            paridxes[6] = -1

        #   Frequency
        if self.ismultifrq and (paridxes[8] is None):
            errmsg = "Random Parameters do not have 'FREQSEL' although UVFITS have multi frequency setups."
            raise ValueError(errmsg)
        elif (self.ismultifrq is False) and (paridxes[8] is None):
            visdata.coord["freqsel"] = np.asarray([1 for i in xrange(Ndata)])
            paridxes[8] = -1

        if None in paridxes:
            print(paridxes)
            errmsg = "Random Parameters do not have mandatory columns."
            raise ValueError(errmsg)

        # Time
        timeobj = at.Time(val=jd1, val2=jd2, format="jd", scale=timescale)
        timeobj = timeobj.utc
        visdata.coord["utc"] = timeobj.datetime

        # Baseline
        subarr, bl = np.modf(bl)
        visdata.coord["subarray"] = np.int64(100*(subarr)+1)
        visdata.coord["ant1"] = np.int64(bl//256)
        visdata.coord["ant2"] = np.int64(bl%256)

        # Sort Columns
        visdata.coord = visdata.coord[visdata.coord_cols]
        visdata.sort()
        visdata.check()

        self.visdata = visdata

    def to_uvfits(self, filename=None, overwrite=True):
        '''
        save to uvfits file. If the filename is not given, then
        return HDUList object

        Args:
            filename (str):
                Output uvfits filename
            overwrite (boolean; default=True)
                If True, overwrite when the specified file already exiests.
        Returns:
          astropy.io.fits.HDUList object if filename=None.
        '''
        if self.ismultisrc:
            raise ValueError("Sorry, this library currently can not create multi-source UVFITS data.")

        hdulist = []
        hdulist.append(self._create_ghdu_single())
        hdulist.append(self._create_fqtab())
        hdulist += self._create_antab()
        hdulist = pf.HDUList(hdulist)
        if filename is None:
            return hdulist
        else:
            hdulist.writeto(filename, overwrite=overwrite)

    def write_fits(self, filename=None, overwrite=True):
        print("Warning: this method will be removed soon. Please use `to_uvfits` method")
        self.to_uvfits(filename,overwrite)

    def _create_ghdu_single(self):
        # Generate Randam Group
        parnames = []
        pardata = []

        # Get some information
        # Baseline
        bl = self.visdata.coord["ant1"] * 256 + self.visdata.coord["ant2"]
        bl += (self.visdata.coord["subarray"] - 1) * 0.01
        # First FREQ SETUP ID
        frqsel = self.subarrays[1].frqsel
        #   Ref Freq
        reffreq = self.subarrays[1].header["FREQ"]
        chwidth = self.fqsetup.fqtables[frqsel].loc[0, "ch_bandwidth"]
        #   RADEC & Equinox
        srcname = self.sources[frqsel].sutable.loc[0, "source"]
        radec = self.sources[frqsel].sutable.loc[0, "radec"]
        equinox = self.sources[frqsel].sutable.loc[0, "equinox"]
        radec = acd.SkyCoord(radec, equinox="J%f"%(equinox), frame="icrs")
        #radec = acd.SkyCoord(radec, frame="icrs")
        if len(self.sources[frqsel].surestfreq) != 0:
            restfreq = self.sources[frqsel].surestfreq[0, 0]
        else:
            restfreq = 0.
        #   first date of year
        utc = self.get_utc()
        utc_ref = at.Time(datetime.datetime(utc[0].datetime.year,1,1), scale="utc")
        # U
        parnames.append("UU")
        pardata.append(np.asarray(self.visdata.coord["usec"], dtype=np.float32))
        # V
        parnames.append("VV")
        pardata.append(np.asarray(self.visdata.coord["vsec"], dtype=np.float32))
        # W
        parnames.append("WW")
        pardata.append(np.asarray(self.visdata.coord["wsec"], dtype=np.float32))
        # Baseline
        parnames.append("BASELINE")
        pardata.append(np.asarray(bl, dtype=np.float32))
        # DATE
        parnames.append("DATE")
        parnames.append("DATE")
        pardata.append(np.asarray(utc.jd1-utc_ref.jd1, dtype=np.float64))
        pardata.append(np.asarray(utc.jd2-utc_ref.jd2, dtype=np.float64))
        # inttime
        if self.visdata.coord["inttim"].max() > 0:
            parnames.append("INTTIM")
            pardata.append(np.asarray(self.visdata.coord["inttim"], dtype=np.float32))
        # Frequency Setup Data
        if self.ismultifrq:
            parnames.append("FREQSEL")
            pardata.append(np.asarray(self.visdata.coord["freqsel"], dtype=np.float32))
        # Group HDU
        gdata = pf.GroupData(
            input=np.float32(self.visdata.data),
            parnames=parnames,
            pardata=pardata,
            bscale=1.0,
            bzero=0.0,
            bitpix=-32)
        ghdu = pf.GroupsHDU(gdata)


        # CTYPE HEADER
        cards = []
        # Complex
        cards.append(("CTYPE2","COMPLEX",""))
        cards.append(("CRPIX2",1.0,""))
        cards.append(("CRVAL2",1.0,""))
        cards.append(("CDELT2",1.0,""))
        cards.append(("CROTA2",0.0,""))
        # Stokes
        cards.append(("CTYPE3","STOKES",""))
        cards.append(("CRPIX3",1.0,""))
        cards.append(("CRVAL3",np.float32(stokesDictinv[self.stokes[0]]),""))
        cards.append(("CDELT3",np.float32(np.sign(stokesDictinv[self.stokes[0]])),""))
        cards.append(("CROTA3",0.0,""))
        # FREQ
        self.subarrays[1].header["FREQ"]
        cards.append(("CTYPE4","FREQ",""))
        cards.append(("CRPIX4",1.0,""))
        cards.append(("CRVAL4",reffreq,""))
        cards.append(("CDELT4",chwidth,""))
        cards.append(("CROTA4",0.0,""))
        # Complex
        cards.append(("CTYPE5","IF",""))
        cards.append(("CRPIX5",1.0,""))
        cards.append(("CRVAL5",1.0,""))
        cards.append(("CDELT5",1.0,""))
        cards.append(("CROTA5",0.0,""))
        # RA & Dec
        cards.append(("CTYPE6","RA",""))
        cards.append(("CRPIX6",1.0,""))
        cards.append(("CRVAL6",radec.ra.deg,""))
        cards.append(("CDELT6",1.0,""))
        cards.append(("CROTA6",0.0,""))
        cards.append(("CTYPE7","DEC",""))
        cards.append(("CRPIX7",1.0,""))
        cards.append(("CRVAL7",radec.dec.deg,""))
        cards.append(("CDELT7",1.0,""))
        cards.append(("CROTA7",0.0,""))
        for card in cards:
            ghdu.header.insert("PTYPE1", card)

        # PTYPE HEADER
        for i in xrange(len(parnames)):
            if i == 4:
                pzero = utc_ref.jd1 - 0.5
            elif i == 5:
                pzero = utc_ref.jd2 + 0.5
            else:
                pzero = 0.0
            card = ("PZERO%d"%(i+1), pzero)
            ghdu.header.insert("PTYPE%d"%(i+1), card, after=True)
            card = ("PSCAL%d"%(i+1), 1.0)
            ghdu.header.insert("PTYPE%d"%(i+1), card, after=True)

        # Other Header
        cards = []
        cards.append(("DATE-OBS", utc[0].isot[0:10]))
        cards.append(("TELESCOP", self.subarrays[frqsel].header["ARRNAM"]))
        cards.append(("INSTRUME", self.subarrays[frqsel].header["ARRNAM"]))
        cards.append(("OBSERVER", self.subarrays[frqsel].header["ARRNAM"]))
        cards.append(("OBJECT", srcname))
        cards.append(("EPOCH", equinox))
        cards.append(("BSCALE", 1.0))
        cards.append(("BSZERO", 0.0))
        cards.append(("BUNIT", "UNCALIB"))
        cards.append(("VELREF", 3))
        cards.append(("ALTRVAL", 0.0))
        cards.append(("ALTRPIX", 1.0))
        cards.append(("RESTFREQ", restfreq))
        cards.append(("OBSRA", radec.ra.deg))
        cards.append(("OBSDEC", radec.dec.deg))
        for card in cards:
            ghdu.header.append(card)

        ghdu.header.append()
        return ghdu

    def _create_fqtab(self):
        '''
        Generate FQ Table
        '''
        freqdata = self.fqsetup
        Nfrqsel = len(freqdata.frqsels)

        # Columns
        tables = []
        for i in xrange(Nfrqsel):
            fqtable = freqdata.fqtables[freqdata.frqsels[i]]
            Nif = fqtable.shape[0]
            tables.append(np.asarray(fqtable).transpose().reshape([4,1,Nif]))
        tables = np.concatenate(tables, axis=1)
        c1=pf.Column(
            name="FRQSEL", format='1J', unit=" ",
            array=np.asarray(freqdata.frqsels,dtype=np.int32))
        c2=pf.Column(
            name="IF FREQ", format='%dD'%(Nif), unit="HZ",
            array=np.asarray(tables[0],dtype=np.float64))
        c3=pf.Column(
            name="CH WIDTH", format='%dE'%(Nif), unit="HZ",
            array=np.asarray(tables[1],dtype=np.float32))
        c4=pf.Column(
            name="TOTAL BANDWIDTH", format='%dE'%(Nif), unit="HZ",
            array=np.asarray(tables[2],dtype=np.float32))
        c5=pf.Column(
            name="SIDEBAND", format='%dJ'%(Nif), unit=" ",
            array=np.asarray(tables[3],dtype=np.int16))
        cols = pf.ColDefs([c1, c2, c3, c4, c5])
        hdu = pf.BinTableHDU.from_columns(cols)

        # header for columns
        '''
        hdu.header.comments["TTYPE1"] = "frequency setup ID number"
        hdu.header.comments["TTYPE2"] = "frequency offset"
        hdu.header.comments["TTYPE3"] = "spectral channel separation"
        hdu.header.comments["TTYPE4"] = "total width of spectral wndow"
        hdu.header.comments["TTYPE5"] = "sideband"
        '''

        # keywords
        card = ("EXTNAME","AIPS FQ","")
        hdu.header.insert("TTYPE1", card)
        card = ("EXTVER",1,"")
        hdu.header.insert("TTYPE1", card)
        card = ("EXTLEVEL",1,"")
        hdu.header.insert("TTYPE1", card)
        card = ("NO_IF",np.int32(Nif),"Number IFs")
        hdu.header.append(card=card)

        return hdu

    def _create_antab(self):
        '''
        Generate AN Table
        '''
        hdus = []
        subarrids = self.subarrays.keys()
        for subarrid in subarrids:
            arraydata = self.subarrays[subarrid]
            Nant = arraydata.antable["name"].shape[0]

            # Number of IFs
            if arraydata.header["NO_IF"] is None:
                noif=1
            else:
                noif=arraydata.header["NO_IF"]

            # Columns
            #   ANNAME
            c1=pf.Column(
                name="ANNAME", format='8A', unit=" ",
                array=np.asarray(arraydata.antable["name"],dtype="|S8"))
            #   STABXYZ
            stabxyz = np.zeros([Nant,3],dtype=np.float64)
            stabxyz[:,0] = arraydata.antable["x"]
            stabxyz[:,1] = arraydata.antable["y"]
            stabxyz[:,2] = arraydata.antable["z"]
            c2=pf.Column(
                name="STABXYZ", format='3D', unit="METERS",
                array=stabxyz)
            #   ORBPARM
            c3=pf.Column(
                name="ORBPARM", format='%dD'%(arraydata.header["NUMORB"]), unit=" ",
                array=np.asarray(arraydata.anorbparm,dtype=np.float64))
            #   NOSTA
            c4=pf.Column(
                name="NOSTA", format='1J', unit=" ",
                array=np.asarray(arraydata.antable["id"],dtype=np.int16))
            #   MNTSTA
            c5=pf.Column(
                name="MNTSTA", format='1J', unit=" ",
                array=np.asarray(arraydata.antable["mnttype"],dtype=np.int16))
            #   MNTSTA
            c6=pf.Column(
                name="STAXOF", format='1E', unit="METERS",
                array=np.asarray(arraydata.antable["axisoffset"],dtype=np.float32))
            #   POLTYA
            c7=pf.Column(
                name="POLTYA", format='1A', unit=" ",
                array=np.asarray(arraydata.antable["poltypeA"],dtype="|S1"))
            #   POLTYA
            c8=pf.Column(
                name="POLAA", format='1E', unit="DEGREES",
                array=np.asarray(arraydata.antable["polangA"],dtype=np.float32))
            #   POLTYA
            c9=pf.Column(
                name="POLCALA", format='%dE'%(arraydata.header["NOPCAL"]*noif), unit=" ",
                array=np.asarray(arraydata.anpolcalA,dtype=np.float32))
            #   POLTYA
            c10=pf.Column(
                name="POLTYB", format='1A', unit=" ",
                array=np.asarray(arraydata.antable["poltypeB"],dtype="|S1"))
            #   POLTYA
            c11=pf.Column(
                name="POLAB", format='1E', unit="DEGREES",
                array=np.asarray(arraydata.antable["polangB"],dtype=np.float32))
            #   POLTYA
            c12=pf.Column(
                name="POLCALB", format='%dE'%(arraydata.header["NOPCAL"]*noif), unit=" ",
                array=np.asarray(arraydata.anpolcalB,dtype=np.float32))
            cols = pf.ColDefs([c1, c2, c3, c4, c5, c6,
                            c7, c8, c9, c10, c11, c12])
            hdu = pf.BinTableHDU.from_columns(cols)

            # header for columns
            '''
            hdu.header.comments["TTYPE1"] = "antenna name"
            hdu.header.comments["TTYPE2"] = "antenna station coordinates"
            hdu.header.comments["TTYPE3"] = "orbital parameters"
            hdu.header.comments["TTYPE4"] = "antenna number"
            hdu.header.comments["TTYPE5"] = "mount type"
            hdu.header.comments["TTYPE6"] = "axis offset"
            hdu.header.comments["TTYPE7"] = "feed A: 'R', 'L'"
            hdu.header.comments["TTYPE8"] = "feed A: position angle"
            hdu.header.comments["TTYPE9"] = "feed A: calibration parameters"
            hdu.header.comments["TTYPE10"] = "feed B: 'R', 'L'"
            hdu.header.comments["TTYPE11"] = "feed B: position angle"
            hdu.header.comments["TTYPE12"] = "feed B: calibration parameters"
            '''

            # keywords
            card = ("EXTNAME","AIPS AN","")
            hdu.header.insert("TTYPE1", card)
            card = ("EXTVER",np.int32(arraydata.subarray),"")
            hdu.header.insert("TTYPE1", card)
            card = ("EXTLEVEL",np.int32(arraydata.subarray),"")
            hdu.header.insert("TTYPE1", card)
            #
            keys = "ARRAYX,ARRAYY,ARRAYZ,GSTIA0,DEGPDY,FREQ,RDATE,"
            keys+= "POLARX,POLARY,UT1UTC,DATUTC,TIMSYS,ARRNAM,XYZHAND,FRAME,"
            keys+= "NUMORB,NO_IF,NOPCAL,POLTYPE"
            keys = keys.split(",")
            types = [
                np.float64,np.float64,np.float64,np.float64,np.float64,np.float64,
                str,np.float64,np.float64,np.float64,np.float64,
                str,str,str,str,np.int64,np.int64,np.int64,str,
            ]
            comments = [
                "x coodinates of array center (meters)",
                "y coodinates of array center (meters)",
                "z coodinates of array center (meters)",
                "GST at 0h on reference date (degrees)",
                "Earth's rotation rate (degrees/day)",
                "reference frequency (Hz)",
                "reference date",
                "x coodinates of North Pole (arcseconds)",
                "y coodinates of North Pole (arcseconds)",
                "UT1 - UTC (sec)",
                "time system - UTC (sec)",
                "time system",
                "array name",
                "handedness of station coordinates",
                "coordinate frame",
                "number of orbital parameters in table",
                "number IFs (=Nif)",
                "number of polarization valibration values / Nif",
                "type of polarization calibration",
            ]
            for i in xrange(len(keys)):
                key = keys[i]
                if arraydata.header[key] is not None:
                    #card = (key,types[i](arraydata.header[key]),
                    #        comments[i])
                    card = (key,types[i](arraydata.header[key]))
                    hdu.header.append(card=card)
            #
            if self.ismultifrq:
                card = ("FREQID",np.int32(arraydata.frqsel),"frequency setup number")
                hdu.header.append(card=card)

            # append HDU
            hdus.append(hdu)
        return hdus

    def _create_sutab(self):
        '''
        Generate SU Table
        '''
        hdus = []
        subarrids = self.subarrays.keys()
        for subarrid in subarrids:
            arraydata = self.subarrays[subarrid]
            Nant = arraydata.antable["name"].shape[0]

            # Number of IFs
            if arraydata.header["NO_IF"] is None:
                noif=1
            else:
                noif=arraydata.header["NO_IF"]

            # Columns

            # Create Columns
            cols = []
            for i in xrange(Ncol):
                args = {}
                args["name"] = names[i]
                args["format"] = formats[i]
                if units[i] is not None:
                    args["unit"] = units[i]
                args["array"] = np.asarray(coldata[i],dtype=dtypes[i])
                cols.append(pf.Column(**args))
            cols = fits.ColDefs(cols)

            # create HDU
            hdu = pf.BinTableHDU.from_columns(cols)

            # header for columns
            hdu.header.comments["TTYPE1"] = "source number"
            hdu.header.comments["TTYPE2"] = "source name"
            hdu.header.comments["TTYPE3"] = "source qualifier number"
            hdu.header.comments["TTYPE4"] = "calibration code"
            hdu.header.comments["TTYPE5"] = "Stokes I flux"
            hdu.header.comments["TTYPE6"] = "Stokes Q flux"
            hdu.header.comments["TTYPE7"] = "Stokes U flux"
            hdu.header.comments["TTYPE8"] = "Stokes V flux"
            hdu.header.comments["TTYPE9"] = "frequency offset"
            hdu.header.comments["TTYPE10"] = "spectral channel sepration"
            hdu.header.comments["TTYPE11"] = "RA of equinox"
            hdu.header.comments["TTYPE12"] = "Dec of equinox"
            hdu.header.comments["TTYPE13"] = "equinox"
            hdu.header.comments["TTYPE14"] = "ra of date"
            hdu.header.comments["TTYPE15"] = "dec of date"
            hdu.header.comments["TTYPE16"] = "velocity"
            hdu.header.comments["TTYPE17"] = "rest frequency"
            hdu.header.comments["TTYPE18"] = "proper motion in RA"
            hdu.header.comments["TTYPE19"] = "proper motion in Dec"

            # keywords
            card = ("EXTNAME","AIPS SU","")
            hdu.header.insert("TTYPE1", card)
            card = ("EXTVER",np.int32(srcdata.frqsel),"")
            hdu.header.insert("TTYPE1", card)
            card = ("FREQID",np.int32(arraydata.frqsel),"frequency setup ID number")
            hdu.header.append(card=card)
            card = ("VELDEF",str(arraydata.header["VELDEF"]),"'RADIO' or 'OPTICAL'")
            hdu.header.append(card=card)
            card = ("VELTYP",str(arraydata.header["VELTYP"]),"velocity coordinate reference")
            hdu.header.append(card=card)

            # append HDU
            hdus.append(hdu)
        return hdus

    def get_ant(self,key="name"):
        '''
        This method will make a dictionary of specified antenna information

        Args:
          key (string; default="name"): key name

        Returns:
          Dictionary of the speficied key Value.
          key of the dictionary is (subarrayid, antenna id)
        '''
        Nsubarr = len(self.subarrays)
        subarrs = self.subarrays.keys()
        outdic = {}
        for subarr in subarrs:
            antable=self.subarrays[subarr].antable
            Nant = len(antable["name"])
            for iant in xrange(Nant):
                outdic[(subarr,iant+1)]=antable.loc[iant, key].strip(" ")
        return outdic

    def get_freq(self, center=True):
        '''
        This method will make a dictionary of frequency offsets of each
        channel. The frequency offset is for the center of each channel.

        Args:
            center (boolean; default=True):
                If True, the central frequency of each channel will be returned.
                Otherwise, the beggining frequency of each channel will be returned.

        Returns:
          Dictionary of the speficied key Value.
          key of the dictionary is (frqsel, IFid, CHid)
        '''
        Nif = self.visdata.data.shape[3]
        Nch = self.visdata.data.shape[4]
        subarrs = self.subarrays.keys()
        frqsels = self.fqsetup.frqsels

        outdic = {}
        for frqsel,subarr in itertools.product(frqsels,subarrs):
            fqtable = self.fqsetup.fqtables[frqsel]
            reffreq = self.subarrays[subarr].header["FREQ"]
            for iif,ich in itertools.product(xrange(Nif),xrange(Nch)):
                chbw = fqtable.loc[iif, "ch_bandwidth"]
                fqof = fqtable.loc[iif, "if_freq_offset"]
                sideband = fqtable.loc[iif, "sideband"]
                if center:
                    freq = reffreq + fqof + chbw * ich * sideband
                else:
                    freq = reffreq + fqof + chbw * (ich-1/2.) * sideband
                outdic[(subarr,frqsel,iif+1,ich+1)] = freq
        return outdic

    def get_freq_offset(self, center=True):
        '''
        This method will make a dictionary of frequency offsets of each
        channel. The frequency offset is for the center of each channel.

        Args:
            center (boolean; default=True):
                If True, the central frequency of each channel will be returned.
                Otherwise, the beggining frequency of each channel will be returned.

        Returns:
          Dictionary of the speficied key Value.
          key of the dictionary is (frqsel, IFid, CHid)
        '''
        Nif = self.visdata.data.shape[3]
        Nch = self.visdata.data.shape[4]
        frqsels = self.fqsetup.frqsels

        outdic = {}
        for frqsel in frqsels:
            fqtable = self.fqsetup.fqtables[frqsel]
            for iif,ich in itertools.product(xrange(Nif),xrange(Nch)):
                chbw = fqtable.loc[iif, "ch_bandwidth"]
                fqof = fqtable.loc[iif, "if_freq_offset"]
                if center:
                    freq = fqof + chbw * ich
                else:
                    freq = fqof + chbw * (ich-1/2.)
                outdic[(frqsel,iif+1,ich+1)] = freq
        return outdic

    def get_utc(self):
        '''
        Get the list of UTC in astropy.time.Time object
        '''
        return at.Time(np.datetime_as_string(self.visdata.coord["utc"]), scale="utc")

    def get_gst(self):
        '''
        Get the list of UTC in astropy.time.sidereal_time object
        '''
        return self.get_utc().sidereal_time('apparent', 'greenwich')

    def get_uvw(self):
        '''
        This method will make a matrix of uvw coverage and uv distance of each
        channel, frequency band (IF).
        The frequency offset is for the center of each channel.

        Returns:
          matrix of the Values of u,v,w,uvdistance(=sqrt(u**2+v**2)).
          key of the dictionary is (Nuvw, Ndata, IFid, CHid),
          where Ndata is the number of the uvw data, and Nuvw is the type of the uvw coverages:
          Nuvw=0,1,2, and 3 corresponds to u,v,w, and uvdistance, respectively.
          Further, CHid and IFid is the number of channels and frequency bands.
        '''

        # Number of Data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp=self.visdata.data.shape

        # initialize uvw
        uvw = np.zeros((Nch, Nif, Ndata, 4))

        # freqsel
        freqsel = self.visdata.coord.freqsel.values

        # subarray
        subarray = self.visdata.coord.subarray.values

        # Usec,Vsec,Wsec,UVdsec
        usec   = np.float64(self.visdata.coord.usec.values)
        vsec   = np.float64(self.visdata.coord.vsec.values)
        wsec   = np.float64(self.visdata.coord.wsec.values)
        uvdsec = np.sqrt(usec**2+vsec**2)

        #  frequency for each channel and frequency band
        freqdic =self.get_freq()

        # calculate UVW=(u, v, w, uvdistance)
        for i,j in itertools.product(xrange(Nif),xrange (Nch)): #0<=i<Nif, 0<=j<Nch
            freq = [freqdic[subarray[k],freqsel[k],i+1,j+1] for k in xrange(Ndata)] # 0<=k<Ndata
            uvw[j,i,:,0] = freq[:]*usec
            uvw[j,i,:,1] = freq[:]*vsec
            uvw[j,i,:,2] = freq[:]*wsec
            uvw[j,i,:,3] = freq[:]*uvdsec

        #Transpose UVW
        UVW = uvw.T
        return UVW

    def eval_image(self,iimage,qimage=None,uimage=None,vimage=None):
        '''
        This method will compude model visivilities based on uv-coverages of
        data and the input image.

        Args:
          iimage (imdata.IMFITS object): input Stokes I image data
          qimage (imdata.IMFITS object): input Stokes Q image data
          uimage (imdata.IMFITS object): input Stokes U image data
          vimage (imdata.IMFITS object): input Stokes V image data
          istokes (int, default=0): the stoked index of the image to be used
          ifreq (int, default=0): the frequency index of the image to be used

        Returns:
          uvdata.UVFITS object
        '''
        #data number of u,v,w
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp=self.visdata.data.shape

        # this is the array for each stokes parameter
        ivis = np.zeros([Ndata, Nif, Nch, 2])
        qvis = np.zeros([Ndata, Nif, Nch, 2])
        uvis = np.zeros([Ndata, Nif, Nch, 2])
        vvis = np.zeros([Ndata, Nif, Nch, 2])

        # u,v coordinates
        #   the array size is Ndata, Nif, Nch
        UVW=self.get_uvw()
        uarr = UVW[0,:,:,:].reshape([Ndata, Nif, Nch])
        varr = UVW[1,:,:,:].reshape([Ndata, Nif, Nch])
        del UVW

        # compute visibilities
        iterator = itertools.product(xrange(Nif),xrange(Nch))
        for iif, ich in tqdm.tqdm(iterator):
            print("Compute Model Visibilities for IF=%d, CH=%d"%(iif+1,ich+1))

            # uv coordinates
            utmp = uarr[:,iif,ich]
            vtmp = varr[:,iif,ich]

            # Vreal, Vimag
            Vreal, Vimag = _eval_image_eachfreq(utmp, vtmp, iimage)
            ivis[:,iif,ich,0] = Vreal
            ivis[:,iif,ich,1] = Vimag

            if qimage is not None:
                Vreal, Vimag = _eval_image_eachfreq(utmp, vtmp, qimage)
                qvis[:,iif,ich,0] = Vreal
                qvis[:,iif,ich,1] = Vimag

            if uimage is not None:
                Vreal, Vimag = _eval_image_eachfreq(utmp, vtmp, qimage)
                uvis[:,iif,ich,0] = Vreal
                uvis[:,iif,ich,1] = Vimag

            if vimage is not None:
                Vreal, Vimag = _eval_image_eachfreq(utmp, vtmp, vimage)
                vvis[:,iif,ich,0] = Vreal
                vvis[:,iif,ich,1] = Vimag

        # output uvfits file
        outfits = copy.deepcopy(self)

        # compute stokes visibilities
        stokes_list = outfits.stokes
        for i in xrange(len(stokes_list)):
            stokes = stokes_list[i]
            print("Compute Model Visibilities at Stokes %s"%(stokes))
            if   stokes.upper() == "I":
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = ivis
            elif stokes.upper() == "Q":
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = qvis
            elif stokes.upper() == "U":
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = uvis
            elif stokes.upper() == "V":
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = vvis
            elif stokes.upper() == "RR":
                # RR = I+V
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = ivis + vvis
            elif stokes.upper() == "LL":
                # LL = I-V
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = ivis - vvis
            elif stokes.upper() == "RL":
                # RL = Q+iU = Re(Q)+iIm(Q) + i(Re(U)+iIm(U))
                #  Re(RL) = Re(Q) - Im(U)
                #  Im(RL) = Im(Q) + Re(U)
                outfits.visdata.data[:, 0, 0, :, :, i, 0] = qvis[:,:,:,0] - uvis[:,:,:,1]
                outfits.visdata.data[:, 0, 0, :, :, i, 1] = qvis[:,:,:,1] + uvis[:,:,:,0]
            elif stokes.upper() == "LR":
                # LR = Q-iU = Re(Q)+iIm(Q) - i(Re(U)+iIm(U))
                #  Re(LR) = Re(Q) + Im(U)
                #  Im(LR) = Im(Q) - Re(U)
                outfits.visdata.data[:, 0, 0, :, :, i, 0] = qvis[:,:,:,0] + uvis[:,:,:,1]
                outfits.visdata.data[:, 0, 0, :, :, i, 1] = qvis[:,:,:,1] - uvis[:,:,:,0]
            elif stokes.upper() == "XX":
                # XX = I+Q
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = ivis + qvis
            elif stokes.upper() == "YY":
                # YY = I-Q
                outfits.visdata.data[:, 0, 0, :, :, i, 0:2] = ivis - qvis
            elif stokes.upper() == "XY":
                # XY = U+iV = Re(U)+iIm(U) + i(Re(V)+iIm(V))
                #  Re(XY) = Re(U) - Im(V)
                #  Im(XY) = Im(U) + Re(V)
                outfits.visdata.data[:, 0, 0, :, :, i, 0] = uvis[:,:,:,0] - vvis[:,:,:,1]
                outfits.visdata.data[:, 0, 0, :, :, i, 1] = uvis[:,:,:,1] + vvis[:,:,:,0]
            elif stokes.upper() == "YX":
                # YX = U-iV = Re(U)+iIm(U) - i(Re(V)+iIm(V))
                #  Re(XY) = Re(U) + Im(V)
                #  Im(XY) = Im(U) - Re(V)
                outfits.visdata.data[:, 0, 0, :, :, i, 0] = uvis[:,:,:,0] + vvis[:,:,:,1]
                outfits.visdata.data[:, 0, 0, :, :, i, 1] = uvis[:,:,:,1] - vvis[:,:,:,0]
        return outfits


    def selfcal(self,iimage,qimage=None,uimage=None,vimage=None,std_amp=1,std_pha=100):
        '''
        This function is currently designed for selfcalibration of
        single polarization (LL, RR, I) or dual polarization (RR+LL, XX+YY).

        For dual polarization, we assume that I = LL = RR = XX = YY,
        which means no circlar polariztion for L/R data, and no linear polarization
        for X/Y data.

        Args:
            iimage (imdata.IMFITS object): input Stokes I image data
            qimage (imdata.IMFITS object): input Stokes Q image data
            uimage (imdata.IMFITS object): input Stokes U image data
            vimage (imdata.IMFITS object): input Stokes V image data
            std_amp (float, default=1):
                Standard deviation of Gain amplitudes from unity.
                This standard deviation will used for the Gaussian prior
                of gain amplitudes. The defaul value (Std Gain error=100%)
                provides a weakely infromed prior.
            std_pha (float in radian, default=100):
                Standard deviation of Gain phases from 0.
                This standard deviation will used for the Gaussian prior
                of gain phases. The defaul value provides a very weakely
                infromed prior.

        Return:
            CLTable object
        '''
        print("Initialize CL Table")
        cltable = CLTable(self)

        print("Compute Model Visibilities")
        modeluvfits = self.eval_image(iimage,qimage,uimage,vimage)

        # get utc of data
        utc = np.datetime_as_string(self.visdata.coord["utc"])

        # Run selfcal for each subarray
        subarrids = self.subarrays.keys()
        for subarrid in subarrids:
            print("Subarray %d"%(subarrid))
            # data index of the current subarray
            idx_subarr = self.visdata.coord["subarray"] == subarrid

            # get the number of data along each dimension
            Ntime,Nif,Nch,Nstokes,Nant,Ncomp = cltable.gaintabs[subarrid]["gain"].shape

            # get utc information
            utcset = cltable.gaintabs[subarrid]["utc"]
            for itime,iif,ich,istokes in tqdm.tqdm(itertools.product(xrange(Ntime),xrange(Nif),xrange(Nch),xrange(Nstokes))):
                # data index of the current time
                idx_utc = utc == utcset[itime]
                idx_utc &= idx_subarr
                idx_utc = np.where(idx_utc)

                # get the current full complex visibilities
                fdata=self.visdata.data[idx_utc]
                Vobs_real_itime = fdata[:,0,0,iif,ich,istokes,0]
                Vobs_imag_itime = fdata[:,0,0,iif,ich,istokes,1]
                sigma_itime     = 1/np.sqrt(fdata[:,0,0,iif,ich,istokes,2])
                del fdata

                # same for the models
                fdata=modeluvfits.visdata.data[idx_utc]
                Vmod_real_itime = fdata[:,0,0,iif,ich,istokes,0]
                Vmod_imag_itime = fdata[:,0,0,iif,ich,istokes,1]
                del fdata

                # check if data are not flagged
                idx_flag = np.isnan(sigma_itime)
                idx_flag|= sigma_itime < 0
                idx_flag|= np.isinf(sigma_itime)
                idx_flag = np.where(idx_flag==False)

                # reselect data
                Vobs_real_itime = Vobs_real_itime[idx_flag]
                Vobs_imag_itime = Vobs_imag_itime[idx_flag]
                sigma_itime = sigma_itime[idx_flag]

                # get the corresponding model visibilities
                Vmodel_real_itime = Vmod_real_itime[idx_flag]
                Vmodel_imag_itime = Vmod_imag_itime[idx_flag]

                # antenna ids
                ant1_itime   = self.visdata.coord.ant1.values[idx_utc][idx_flag]
                ant2_itime   = self.visdata.coord.ant2.values[idx_utc][idx_flag]
                Ndata_itime  = len(ant1_itime)

                if Ndata_itime == 0:
                    continue

                # non-redundant set of antenna ids
                antset_itime = sorted(set(ant1_itime.tolist()+ant2_itime.tolist()))
                Nant_itime = len(antset_itime)

                # create dictionary of antenna ids
                ant1id = [antset_itime.index(ant1_itime[i]) for i in xrange(Ndata_itime)]
                ant2id = [antset_itime.index(ant2_itime[i]) for i in xrange(Ndata_itime)]

                # compute wij and Xij
                w_itime = np.sqrt(Vmodel_real_itime**2+Vmodel_imag_itime**2)
                w_itime/= sigma_itime * np.sqrt(Ndata_itime*2)
                X_itime = (Vobs_real_itime+1j*Vobs_imag_itime)/(Vmodel_real_itime+1j*Vmodel_imag_itime)
                del Vobs_real_itime, Vobs_imag_itime, sigma_itime
                del Vmodel_real_itime, Vmodel_imag_itime

                # (Tentative) initilize gains
                gain0 = np.zeros(Nant_itime*2)
                gain0[:Nant_itime] = 1.

                #if Ndata_itime > Nant_itime:
                result = leastsq(
                    _selfcal_error_func, gain0, Dfun=_selfcal_error_dfunc,
                    args=(ant1id,ant2id,w_itime,X_itime,std_amp,std_pha,Nant_itime,Ndata_itime))

                # make cltable
                g = result[0]
                for i in xrange(Nant_itime):
                    cltable.gaintabs[subarrid]["gain"][itime,iif,ich,istokes,antset_itime[i]-1,0]= g[i]
                    cltable.gaintabs[subarrid]["gain"][itime,iif,ich,istokes,antset_itime[i]-1,1]= g[i+Nant_itime]
        return cltable

    def apply_cltable(self,cltable):

        #def get_vis_correction(self,imfits,cltable):
        subarrids = self.subarrays.keys()

        # make uvfits for corrected visibility
        outfits = copy.deepcopy(self)

        # get utc
        utc = np.datetime_as_string(self.visdata.coord["utc"])

        # get full complex visibilities
        fdata=outfits.visdata.data
        Ndata,Ndec,Nra,Nif,Nch,Nstokes,Ncomp = fdata.shape
        Vobs_comp = fdata[:,:,:,:,:,:,0] + 1j*fdata[:,:,:,:,:,:,1]
        weight    = fdata[:,:,:,:,:,:,2]
        del fdata

        # get antenna ids
        subarr = outfits.visdata.coord.subarray.values
        ant1 = outfits.visdata.coord.ant1.values
        ant2 = outfits.visdata.coord.ant2.values

        for subarrid in subarrids:
            # get the number of data along each dimension
            Ntime,dammy,dammy,dammy,Nant,Ncomp = cltable.gaintabs[subarrid]["gain"].shape

            # get utc, utc-set, and utc-group
            utcset = cltable.gaintabs[subarrid]["utc"]
            utcgroup = pd.DataFrame({"utc": utc}).groupby("utc")

            # get gain
            gain = cltable.gaintabs[subarrid]["gain"]
            antset = outfits.subarrays[subarrid].antable.id.values

            for itime, istokes, iant in itertools.product(xrange(Ntime),xrange(Nstokes),xrange(Nant)):
                # data index of the current time and antenna
                idx = set(tuple(utcgroup.groups[utcset[itime]]))

                idx1 = np.where(ant1==antset[iant])
                idx1 = set(idx1[0])
                idx1 &= idx
                idx1 = list(idx1)

                idx2 = np.where(ant2==antset[iant])
                idx2 = set(idx2[0])
                idx2 &= idx
                idx2 = list(idx2)

                # gain of the current time and antenna
                if istokes <2: # dual polarization
                    # compute gains
                    gi = gain[itime,:,:,istokes,iant,0] + 1j*gain[itime,:,:,istokes,iant,1]
                    gjc = np.conj(gi)
                elif istokes == 2: # this must be RL or XY
                    gi = gain[itime,:,:,0,iant,0] + 1j*gain[itime,:,:,0,iant,1]
                    gjc = gain[itime,:,:,1,iant,0] - 1j*gain[itime,:,:,1,iant,1]
                elif istokes == 3: # this must be LR or YX
                    gi = gain[itime,:,:,1,iant,0] + 1j*gain[itime,:,:,1,iant,1]
                    gjc = gain[itime,:,:,0,iant,0] - 1j*gain[itime,:,:,0,iant,1]

                # calibrated visibility
                Vobs_comp[idx1,:,:,:,:,istokes] /= gi
                Vobs_comp[idx2,:,:,:,:,istokes] /= gjc
                weight[idx1,:,:,:,:,istokes] *= np.abs(gi)**2
                weight[idx2,:,:,:,:,istokes] *= np.abs(gjc)**2
        outfits.visdata.data[:,:,:,:,:,:,0] = np.real(Vobs_comp)
        outfits.visdata.data[:,:,:,:,:,:,1] = np.imag(Vobs_comp)
        outfits.visdata.data[:,:,:,:,:,:,2] = weight

        return outfits

    def uvavg(self, solint=10, minpoint=4):
        '''
        This method will weighted-average full complex visibilities in time direction.
        Visibilities will be weighted-average, using weight information of data.
        uvw-coordinates on new time grid will be interpolated with cubic spline interpolation.
        This may give slightly different uvw coordinates from other software, such as DIFMAP
        computing weighted averages or AIPS UVFIX recalculating uvw coordinates.
        Args:
          solint (float; default=10):
            Time averaging interval (in sec)
            minpoint (int; default =2.):
              Number of points required to re-evaluate weights.
              It must be larger than 4, since the software is using a cubic
              interpolation for recalculating uv coordinates.
              If data do not have enough number of points at each time/frequency
              segments specified with dofreq, weight will be set to 0
              meaning that the corresponding point will be flagged out.
        Returns: uvfits.UVFITS object
        '''
        if minpoint < 4:
            raise ValueError("Please specify minpoint >= 4")

        outfits = copy.deepcopy(self)

        # Sort visdata
        print("(1/5) Sort Visibility Data")
        outfits.visdata.sort(by=["subarray","ant1","ant2","source","utc"])
        Ndata = len(outfits.visdata.coord)

        # Check number of Baselines
        print("(2/5) Check Number of Baselines")
        # pick up combinations
        select = outfits.visdata.coord.drop_duplicates(subset=["subarray","ant1","ant2","source","freqsel"])
        combset = zip(
            select.subarray.values,
            select.ant1.values,
            select.ant2.values,
            select.source.values,
            select.freqsel.values
        )
        stlst = np.asarray(select.index.tolist(), dtype=np.int32)+1
        edlst = np.asarray(select.index.tolist()+[Ndata], dtype=np.int32)[1:]
        Nidx = len(stlst)

        print("(3/5) Create Timestamp")
        tsecin = outfits.get_utc().cxcsec
        tsecout = np.arange(tsecin.min()+solint/2,tsecin.max(),solint)
        Nt = len(tsecout)

        # Check number of Baselines
        print("(4/5) Average UV data")
        out = fortlib.uvdata.average(
            uvdata=np.float32(outfits.visdata.data.T),
            u=np.float64(outfits.visdata.coord.usec.values),
            v=np.float64(outfits.visdata.coord.vsec.values),
            w=np.float64(outfits.visdata.coord.wsec.values),
            tin=np.float64(tsecin),
            tout=np.float64(tsecout),
            start=np.int32(stlst),
            end=np.int32(edlst),
            solint=solint,
            minpoint=minpoint,
        )
        usec = out[1]
        vsec = out[2]
        wsec = out[3]
        isdata = np.asarray(out[4],dtype=np.bool)
        outfits.visdata.data = np.ascontiguousarray(out[0].T)[np.where(isdata)]
        del out

        print("(5/5) Forming UV data")
        utc = [tsecout for i in xrange(Nidx)]
        #usec = [0.0 for i in xrange(Nidx*Nt)]
        #vsec = [0.0 for i in xrange(Nidx*Nt)]
        #wsec = [0.0 for i in xrange(Nidx*Nt)]
        subarray = [[combset[idx][0] for i in xrange(Nt)] for idx in xrange(Nidx)]
        ant1 = [[combset[idx][1] for i in xrange(Nt)] for idx in xrange(Nidx)]
        ant2 = [[combset[idx][2] for i in xrange(Nt)] for idx in xrange(Nidx)]
        source = [[combset[idx][3] for i in xrange(Nt)] for idx in xrange(Nidx)]
        inttim = [solint for i in xrange(Nidx*Nt)]
        freqsel = [[combset[idx][4] for i in xrange(Nt)] for idx in xrange(Nidx)]
        utc = at.Time(np.concatenate(utc), format="cxcsec", scale="utc").datetime

        outfits.visdata.coord = pd.DataFrame()
        outfits.visdata.coord["utc"] = utc
        outfits.visdata.coord["usec"] = np.float64(usec)
        outfits.visdata.coord["vsec"] = np.float64(vsec)
        outfits.visdata.coord["wsec"] = np.float64(wsec)
        outfits.visdata.coord["subarray"] = np.int64(np.concatenate(subarray))
        outfits.visdata.coord["ant1"] = np.int64(np.concatenate(ant1))
        outfits.visdata.coord["ant2"] = np.int64(np.concatenate(ant2))
        outfits.visdata.coord["source"] = np.int64(np.concatenate(source))
        outfits.visdata.coord["inttim"] = np.float64(inttim)
        outfits.visdata.coord["freqsel"] = np.int64(np.concatenate(freqsel))
        del utc,usec,vsec,wsec,subarray,ant1,ant2,source,inttim,freqsel
        del combset,stlst,edlst,Nt,Nidx
        outfits.visdata.coord = outfits.visdata.coord.loc[isdata,:]
        outfits.visdata.coord.reset_index(drop=True,inplace=True)
        outfits.visdata.sort()

        #print("(6/6) UVW recaluation")
        #outfits = outfits.uvw_recalc()
        return outfits

    def avspc(self, dofreq=0, minpoint=2):
        '''
        This method will weighted-average full complex visibilities in frequency directions.
        Args:
          dofreq (int; default = 0):
            Parameter for multi-frequency data.
              dofreq = 0: average over IFs and channels
              dofreq = 1: average over channels at each IF
          minpoint (int; default =2.):
            Number of points required to re-evaluate weights.
            If data do not have enough number of points at each time/frequency
            segments specified with dofreq, weight will be set to 0
            meaning that the corresponding point will be flagged out.
        Returns: uvfits.UVFITS object
        '''
        outfits = copy.deepcopy(self)

        # Update visibilities
        if np.int32(dofreq) > 0.5:
            outfits.visdata.data = np.ascontiguousarray(
                fortlib.uvdata.avspc_dofreq1(
                    uvdata=np.asarray(self.visdata.data.T, dtype=np.float32))).T
        else:
            outfits.visdata.data = np.ascontiguousarray(
                fortlib.uvdata.avspc_dofreq0(
                    uvdata=np.asarray(self.visdata.data.T, dtype=np.float32))).T

        # remake frequency tables
        if np.int32(dofreq) > 0.5:
            for frqsel in outfits.fqsetup.frqsels:
                outfits.fqsetup.fqtables[frqsel].loc[:,"ch_bandwidth"]=outfits.fqsetup.fqtables[frqsel]["if_bandwidth"]
        else:
            for frqsel in outfits.fqsetup.frqsels:
                bw = outfits.fqsetup.fqtables[frqsel]["if_bandwidth"].sum()
                cf = outfits.fqsetup.fqtables[frqsel]["if_freq_offset"] + \
                     outfits.fqsetup.fqtables[frqsel]["if_bandwidth"]/2
                sf = cf.mean() - bw/2
                sb = np.int32(np.median(outfits.fqsetup.fqtables[frqsel]["sideband"]))
                newtable = pd.DataFrame(
                    {"if_freq_offset": [sf],
                     "ch_bandwidth": [bw],
                     "if_bandwidth": [bw],
                     "sideband": [sb]},
                     columns=outfits.fqsetup.fqtable_cols
                )
                outfits.fqsetup.fqtables[frqsel] = newtable

        # update antenna Tables
        for arrid in outfits.subarrays.keys():
            outfits.subarrays[arrid].avspc(dofreq=dofreq)

        return outfits

    def uvw_recalc(self):
        '''
        This method will recalculate uvw coordinates, using utc information,
        source coordinates, and station locations.

        This function would be not accurate as uvw-recalculation functions in
        AIPS (UVFIX) and VEDA, which are using the latest parameters like EOPs.

        So, we do not guarantee that this function will provide accurate
        uvw coordinates enough for astrometric VLBI observations.
        '''
        print("Start UVW recalculation")

        # Copy data
        outfits = copy.deepcopy(self)
        utctime = self.get_utc()
        frqsels = sorted(set(outfits.visdata.coord["freqsel"]))
        Ndata = outfits.visdata.coord.shape[0]

        # arrays to be used to compute uvw
        alpha = np.zeros(Ndata, dtype=np.float64)
        delta = np.zeros(Ndata, dtype=np.float64)
        dx = np.zeros(Ndata, dtype=np.float64)
        dy = np.zeros(Ndata, dtype=np.float64)
        dz = np.zeros(Ndata, dtype=np.float64)

        # calc GST
        print("  (1/4) Compute GST from UTC")
        gsthour = np.float64(utctime.sidereal_time('apparent', 'greenwich').hour)

        # compute alpha & delta
        print("  (2/4) Compute RA, DEC of the Sources in GCRS")
        for frqsel in frqsels:
            idx1 = outfits.visdata.coord["freqsel"]==frqsel
            srcs = sorted(set(outfits.visdata.coord.loc[idx1,"source"]))
            for src in srcs:
                srctab = outfits.sources[frqsel].sutable
                radec = srctab.loc[srctab["id"]==src, "radec"].values
                equinox = srctab.loc[srctab["id"]==src, "equinox"].values
                radec = acd.SkyCoord(radec, equinox="J%f"%(equinox), frame="icrs")
                #
                idx2 = outfits.visdata.coord["source"]==src
                idx3 = np.where(idx1&idx2)
                utctmp = utctime[idx3]
                radec = radec.transform_to(acd.GCRS(obstime=utctmp))
                alpha[idx3] = radec.ra.rad
                delta[idx3] = radec.dec.rad

        # compute baseline vectors
        print("  (3/4) Compute Baseline Vectors")
        subarrays = sorted(set(outfits.visdata.coord["subarray"]))
        for subarray in subarrays:
            idx1 = outfits.visdata.coord["subarray"] == subarray

            # get antenna ids
            ants = sorted(set(outfits.visdata.coord.loc[idx1,"ant1"]))
            ants+= sorted(set(outfits.visdata.coord.loc[idx1,"ant2"]))
            ants = sorted(set(ants))

            # antenna table
            arrdata = outfits.subarrays[subarray].antable
            for ant in ants:
                idx2 = outfits.visdata.coord["ant1"] == ant
                idx3 = outfits.visdata.coord["ant2"] == ant
                idx2 = idx1 & idx2
                idx3 = idx1 & idx3

                # get xyz coordinates
                x,y,z = arrdata.loc[arrdata["id"]==ant, ["x","y","z"]].values[0]/ac.c.si.value
                if True in idx2:
                    idx2 = np.where(idx2)
                    dx[idx2]+=x
                    dy[idx2]+=y
                    dz[idx2]+=z
                if True in idx3:
                    idx3 = np.where(idx3)
                    dx[idx3]-=x
                    dy[idx3]-=y
                    dz[idx3]-=z

        # compute uvw vectors
        print("  (4/4) Compute uvw coordinates")
        u,v,w = fortlib.coord.calc_uvw(gsthour,alpha,delta,dx,dy,dz)
        outfits.visdata.coord["usec"] = u
        outfits.visdata.coord["vsec"] = v
        outfits.visdata.coord["wsec"] = w

        return outfits

    def weightcal(self, dofreq=0, solint=60., minpoint=2):
        '''
        This method will recalculate sigmas and weights of data from scatter
        in full complex visibilities over specified frequency and time segments.

        Args:
          dofreq (int; default = 0):
            Parameter for multi-frequency data.
              dofreq = 0: calculate weights and sigmas over IFs and channels
              dofreq = 1: calculate weights and sigmas over channels at each IF
              dofreq = 2: calculate weights and sigmas at each IF and Channel

          solint (float; default = 60.):
            solution interval in sec

        Returns: uvfits.UVFITS object
        '''
        # Save and Return re-weighted uv-data
        outfits = copy.deepcopy(self)
        tsec = outfits.get_utc().cxcsec
        outfits.visdata.data = np.ascontiguousarray(fortlib.uvdata.weightcal(
                uvdata=np.asarray(self.visdata.data.T, dtype=np.float32),
                tsec=np.array(tsec, dtype=np.float64),
                ant1=np.asarray(self.visdata.coord["ant1"], dtype=np.int32),
                ant2=np.asarray(self.visdata.coord["ant2"], dtype=np.int32),
                subarray=np.asarray(self.visdata.coord["subarray"], dtype=np.int32),
                source=np.asarray(self.visdata.coord["source"], dtype=np.int32),
                solint=np.float64(solint),
                dofreq=np.int32(dofreq),
                minpoint=np.int32(minpoint)).T)
        return outfits

    def select_stokes(self, stokes="I"):
        '''
        Pick up single polarization data

        Args:
          stokes (string; default="I"):
            Output stokes parameters.
            Availables are ["I", "Q", "U", "V",
                            "LL", "RR", "RL", "LR",
                            "XX", "YY", "XY", "YX"].

        Returns: uvdata.UVFITS object
        '''
        # get stokes data
        stokesorg = self.stokes

        # create output data
        outfits = copy.deepcopy(self)
        dshape = list(outfits.visdata.data.shape)
        dshape[5] = 1

        if stokes == "I":
            outfits.stokes = ["I"]
            if ("I" in stokesorg):  # I <- I
                print("Stokes I data will be copied from the input data")
                idx = stokesorg.index("I")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RR" in stokesorg) and ("LL" in stokesorg):  # I <- (RR + LL)/2
                print("Stokes I data will be calculated from input RR and LL data")
                idx1 = stokesorg.index("RR")
                idx2 = stokesorg.index("LL")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            elif ("RR" in stokesorg):  # I <- RR
                print("Stokes I data will be copied from input RR data")
                idx = stokesorg.index("RR")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("LL" in stokesorg):  # I <- LL
                print("Stokes I data will be copied from input LL data")
                idx = stokesorg.index("LL")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("XX" in stokesorg) and ("YY" in stokesorg):  # I <- (XX + YY)/2
                print("Stokes I data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XX")
                idx2 = stokesorg.index("YY")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            elif ("XX" in stokesorg):  # I <- XX
                print("Stokes I data will be copied from input XX data")
                idx = stokesorg.index("XX")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("YY" in stokesorg):  # I <- YY
                print("Stokes I data will be copied from input YY data")
                idx = stokesorg.index("YY")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "Q":
            outfits.stokes = ["Q"]
            if ("Q" in stokesorg):  # Q <- Q
                print("Stokes Q data will be copied from the input data")
                idx = stokesorg.index("Q")
                outfits.visdata.data = self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RL" in stokesorg) and ("LR" in stokesorg):  # Q <- (RL + LR)/2
                print("Stokes Q data will be calculated from input RL and LR data")
                idx1 = stokesorg.index("RL")
                idx2 = stokesorg.index("LR")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            elif ("XX" in stokesorg) and ("YY" in stokesorg):  # Q <- (XX - YY)/2
                print("Stokes Q data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XX")
                idx2 = stokesorg.index("YY")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=-0.5)
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "U":
            outfits.stokes = ["U"]
            if ("U" in stokesorg):  # U <- U
                print("Stokes U data will be copied from the input data")
                idx = stokesorg.index("U")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RL" in stokesorg) and ("LR" in stokesorg):  # U <- (RL - LR)/2i = (- RL + LR)i/2
                print("Stokes U data will be calculated from input RL and LR data")
                idx1 = stokesorg.index("RL")
                idx2 = stokesorg.index("LR")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=-0.5j, factr2=0.5j)
            elif ("XY" in stokesorg) and ("YX" in stokesorg):  # U <- (XY + YX)/2
                print("Stokes U data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XY")
                idx2 = stokesorg.index("YX")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=0.5)
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "V":
            outfits.stokes = ["V"]
            if ("V" in stokesorg):  # V <- V
                print("Stokes V data will be copied from the input data")
                idx = stokesorg.index("V")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            elif ("RR" in stokesorg) and ("LL" in stokesorg):  # V <- (RR - LL)/2
                print("Stokes V data will be calculated from input RR and LL data")
                idx1 = stokesorg.index("RR")
                idx2 = stokesorg.index("LL")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=0.5, factr2=-0.5)
            elif ("XY" in stokesorg) and ("YX" in stokesorg):  # V <- (XY - YX)/2i = (-XY + YX)i/2
                print("Stokes V data will be calculated from input XX and YY data")
                idx1 = stokesorg.index("XY")
                idx2 = stokesorg.index("YX")
                outfits.visdata.data = _bindstokes(
                    self.visdata.data,
                    stokes1=idx1, stokes2=idx2,
                    factr1=-0.5j, factr2=0.5j)
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "RR":
            outfits.stokes = ["RR"]
            if ("RR" in stokesorg):
                print("Stokes RR data will be copied from the input data")
                idx = stokesorg.index("RR")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "LL":
            outfits.stokes = ["LL"]
            if ("LL" in stokesorg):
                print("Stokes LL data will be copied from the input data")
                idx = stokesorg.index("LL")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "RL":
            outfits.stokes = ["RL"]
            if ("RL" in stokesorg):
                print("Stokes RL data will be copied from the input data")
                idx = stokesorg.index("RL")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        elif stokes == "LR":
            outfits.stokes = ["LR"]
            if ("LR" in stokesorg):
                print("Stokes LR data will be copied from the input data")
                idx = stokesorg.index("LR")
                outfits.visdata.data =  self.visdata.data[:, :, :, :, :, idx, :]
            else:
                errmsg="[WARNING] No data are available to calculate Stokes %s"%(stokes)
                raise ValueError(errmsg)
        else:
            errmsg="[WARNING] Currently Stokes %s is not supported in this function."%(stokes)
            raise ValueError(errmsg)
        outfits.visdata.data = outfits.visdata.data.reshape(dshape)
        return outfits

    def uv_rotate(self, dPA, deg=True):
        '''
        Rotate uv-coordinates by a specified rotation angle

        Args:
            dPA (float):
                Rotation angle.
            deg (boolean, default=True):
                The unit of dPA. If True, it will be degree. Otherwise,
                it will be radian
        Returns:
            uvdata.UVFITS object
        '''
        # warning messages
        print("[WARNING] uvdata.UVFITS.uvrotate")
        print("This makes your uv-coodinates no longer consistent with the antenna position informations.")

        # output data
        outdata = copy.deepcopy(self)

        # rotation angle and cos/sin of them
        if deg:
            theta = np.deg2rad(dPA)
        else:
            theta = dPA
        cost = np.cos(theta)
        sint = np.sin(theta)

        # take uv-coordinates
        u = self.visdata.coord["usec"]
        v = self.visdata.coord["vsec"]

        # rotate the uv-coordinates
        outdata.visdata.coord["vsec"] = v * cost - u * sint
        outdata.visdata.coord["usec"] = v * sint + u * sint

        return outdata

    def add_frac_error(self, ferror, quadrature=True):
        '''
        Increase errors by specified fractional values of amplitudes

        Args:
            ferror (float):
                fractional error of amplitudes to be added.
            quadrature (boolean; default=True):
                if True, error will be added to sigma in quadrature.
                Otherwise, it will be added directly to the current sigma.
        '''
        # Data to be output
        outfits = copy.deepcopy(self)

        # Number of Data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = self.visdata.data.shape

        # amplitudes
        Vreal = np.sqrt(self.visdata.data[:,:,:,:,:,:,0])
        Vimag = np.sqrt(self.visdata.data[:,:,:,:,:,:,1])
        Vweig = np.sqrt(self.visdata.data[:,:,:,:,:,:,2])
        Vamp = np.sqrt(Vreal**2 + Vimag**2)
        Vsig = 1./np.sqrt(np.abs(Vweig))

        # compute the new sigma
        if quadrature:
            Vsig_new = np.sqrt(Vsig**2 + (ferror * Vamp)**2)
        else:
            Vsig_new = Vsig + ferror * Vamp

        # recompute weights
        Vweig_new = 1./(Vsig_new**2) * np.sign(Vweig)
        Vweig_new[np.where(np.isnan(Vweig_new))] = 0.
        Vweig_new[np.where(np.isinf(Vweig_new))] = 0.
        Vweig_new[np.where(Vweig_new<0)] = 0.

        # update uvfits object to be output
        outfits.visdata.data[:,:,:,:,:,:,2] = Vweig_new[:,:,:,:,:,:]
        return outfits

    def make_vistable(self, flag=True):
        '''
        Convert visibility data to a two dimentional table.
        Args:
          flag (boolean):
            if flag=True, data with weights <= 0 or sigma <=0 will be ignored.
        Returns:
          uvdata.VisTable object
        '''
        # Number of Data
        Ndata, Ndec, Nra, Nif, Nch, Nstokes, Ncomp = self.visdata.data.shape

        # Time
        utctime = self.get_utc()
        gsthour = np.float64(utctime.sidereal_time('apparent', 'greenwich').hour)
        utctime = utctime.datetime

        # UVW
        u = np.float64(self.visdata.coord.usec.values)
        v = np.float64(self.visdata.coord.vsec.values)
        w = np.float64(self.visdata.coord.wsec.values)
        uvd = np.sqrt(u**2+v**2)

        # Antenna Name
        namedic = self.get_ant()
        subarr = np.int32(self.visdata.coord.subarray)
        st1 = np.int32(self.visdata.coord.ant1.values)
        st2 = np.int32(self.visdata.coord.ant2.values)
        st1name = np.asarray([namedic[(subarr[i],st1[i])] for i in xrange(Ndata)])
        st2name = np.asarray([namedic[(subarr[i],st2[i])] for i in xrange(Ndata)])

        # other parameters
        frqsel = self.visdata.coord.freqsel.values

        # frequency
        freqdic = self.get_freq()

        # Create Tables
        outdata = VisTable()
        for idec, ira, iif, ich, istokes in itertools.product(xrange(Ndec),
                                                             xrange(Nra),
                                                             xrange(Nif),
                                                             xrange(Nch),
                                                             xrange(Nstokes)):
            tmpdata = VisTable()

            # Time
            tmpdata["utc"] = utctime
            tmpdata["gsthour"] = gsthour

            #  Freq
            tmpdata["freq"] = np.zeros(Ndata, dtype=np.float32)

            # Stokes ID
            tmpdata["stokesid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "stokesid"] = np.int32(stokesDictinv[self.stokes[istokes]])

            # ch/if id, frequency
            tmpdata["ifid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "ifid"] = np.int32(iif)
            tmpdata["chid"] = np.zeros(Ndata, dtype=np.int32)
            tmpdata.loc[:, "chid"] = np.int32(ich)
            tmpdata["ch"] = tmpdata["ifid"] + tmpdata["chid"] * Nif

            tmpdata["freq"] = [freqdic[(subarr[i],frqsel[i],iif+1,ich+1)] for i in xrange(Ndata)]

            # uvw
            tmpdata["u"] = u*tmpdata["freq"]
            tmpdata["v"] = v*tmpdata["freq"]
            tmpdata["w"] = w*tmpdata["freq"]
            tmpdata["uvdist"] = uvd*tmpdata["freq"]

            # station number
            tmpdata["subarray"] = subarr
            tmpdata["st1"] = st1
            tmpdata["st2"] = st2
            tmpdata["st1name"] = st1name
            tmpdata["st2name"] = st2name

            visreal = np.float64(
                self.visdata.data[:, idec, ira, iif, ich, istokes, 0])
            visimag = np.float64(
                self.visdata.data[:, idec, ira, iif, ich, istokes, 1])
            visweig = np.float64(
                self.visdata.data[:, idec, ira, iif, ich, istokes, 2])
            tmpdata["amp"] = np.sqrt(visreal * visreal + visimag * visimag)
            tmpdata["phase"] = np.rad2deg(np.arctan2(visimag, visreal))
            tmpdata["weight"] = visweig
            tmpdata["sigma"] = np.sqrt(1. / visweig)

            outdata = pd.concat([outdata, tmpdata])

        if flag:
            select = outdata["weight"] > 0
            select &= outdata["sigma"] > 0
            select &= np.isnan(outdata["weight"]) == False
            select &= np.isnan(outdata["sigma"]) == False
            select &= np.isinf(outdata["weight"]) == False
            select &= np.isinf(outdata["sigma"]) == False
            outdata = outdata.loc[select, :].reset_index(drop=True)

        return outdata

    def uvplot(self, uvunit=None, conj=True,
               ls="none", marker=".", **plotargs):
        '''
        Plot uv-plot on the current axes.
        This method uses matplotlib.pyplot.plot().

        Args:
          uvunit (str, default = None):
            The unit of the baseline length.

          conj (boolean, default = True):
            if conj=True, it will plot complex conjugate components (i.e. (-u, -v)).

          **plotargs:
            You can set parameters of matplotlib.pyplot.plot.
            Defaults are {'ls': "none", 'marker': "."}
        '''
        self.make_vistable().uvplot(
          uvunit=uvunit, conj=conj, ls=ls, marker=marker, **plotargs)

    def radplot(self, uvunit=None, datatype="amp", normerror=False, errorbar=True,
                ls="none", marker=".", **plotargs):
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
        self.make_vistable().radplot(uvunit=uvunit, datatype=datatype,
                    normerror=normerror, errorbar=errorbar,
                    ls=ls, marker=marker, **plotargs)


#-------------------------------------------------------------------------
# Subfunctions for UVFITS
#-------------------------------------------------------------------------
def _bindstokes(data, stokes1, stokes2, factr1, factr2):
    '''
    This is a subfunction for uvdata.UVFITS.
    '''
    vcomp1 = data[:, :, :, :, :, stokes1, 0] + \
        1j * data[:, :, :, :, :, stokes1, 1]
    vweig1 = data[:, :, :, :, :, stokes1, 2]
    vcomp2 = data[:, :, :, :, :, stokes2, 0] + \
        1j * data[:, :, :, :, :, stokes2, 1]
    vweig2 = data[:, :, :, :, :, stokes2, 2]

    vcomp = factr1 * vcomp1 + factr2 * vcomp2
    vweig = np.power(np.abs(factr1)**2 / vweig1 +
                     np.abs(factr2)**2 / vweig2, -1)

    select  = vweig1 <= 0
    select |= vweig2 <= 0
    select |= vweig <= 0
    select |= np.isnan(vweig1)
    select |= np.isnan(vweig2)
    select |= np.isnan(vweig)
    select |= np.isinf(vweig1)
    select |= np.isinf(vweig2)
    select |= np.isinf(vweig)
    vweig[np.where(select)] = 0.0

    outdata = data[:, :, :, :, :, stokes1, :]
    outdata[:, :, :, :, :, 0] = np.real(vcomp)
    outdata[:, :, :, :, :, 1] = np.imag(vcomp)
    outdata[:, :, :, :, :, 2] = vweig
    return outdata


class VisibilityData(object):
    # Default Variables
    def __init__(self):
        self.data = np.zeros([0,1,1,1,1,1,3])

        columns = "utc,usec,vsec,wsec,subarray,ant1,ant2,source,inttim,freqsel"
        self.coord_cols = columns.split(",")
        self.coord = pd.DataFrame(columns=self.coord_cols)

    def check(self):
        # Check Dimension
        Ndim = len(self.data.shape)
        if Ndim != 7:
            errmsg = "VisData.check: Dimension %d is not available (must be 7)."%(Ndim)
            raise ValueError(errmsg)
        if self.data.shape[6]!=3:
            errmsg = "VisData.check: COMPLEX should have NAXIS=3 (currently NAXIS=%d)."%(self.data.shape[6])
            raise ValueError(errmsg)

    def sort(self, by=["utc","ant1","ant2","subarray"]):
        # Check if ant1 > ant2
        self.coord.reset_index(drop=True, inplace=True)
        idx = self.coord["ant1"] > self.coord["ant2"]
        if True in idx:
            self.coord.loc[idx, ["usec", "vsec", "wsec"]] *= -1
            ant2 = self.coord.loc[idx, "ant2"]
            self.coord.loc[idx, "ant2"] = self.coord.loc[idx, "ant1"]
            self.coord.loc[idx, "ant1"] = ant2
            where = np.where(idx)[0]
            self.data[where,:,:,:,:,:,1] *= -1
            prt("VisData.sort: %d indexes have wrong station orders (ant1 > ant2)."%(len(where)),indent)
        else:
            prt("VisData.sort: Data have correct station orders (ant1 < ant2).",indent)

        # Sort Data
        self.coord.reset_index(drop=True, inplace=True)
        self.coord = self.coord.sort_values(by=by)
        rows = np.asarray(self.coord.index)
        self.data = self.data[rows,:,:,:,:,:,:]
        self.coord.reset_index(drop=True, inplace=True)
        prt("VisData.sort: Data have been sorted by %s"%(", ".join(by)),indent)

class FrequencyData(object):
    def __init__(self):
        # Frequency Setup Number
        self.frqsels=[]

        # Table
        fqtable_cols="if_freq_offset,ch_bandwidth,if_bandwidth,sideband"
        fqtable_cols=fqtable_cols.split(",")
        self.fqtables={}
        self.fqtable_cols=fqtable_cols

    def __repr__(self):
        lines = []
        for i in xrange(len(self.frqsels)):
            frqsel = self.frqsels[i]
            fqtable = self.fqtables[frqsel]
            lines.append("Frequency Setup ID: %d"%(frqsel))
            lines.append("  IF Freq setups (Hz):")
            lines.append(prt(fqtable,indent*2,output=True))
        lines.append("  Note: Central Frequency of ch=i at IF=j (where i,j=1,2,3...)")
        lines.append("     freq(i,j) = reffreq + (i-1) * ch_bandwidth(j) * sideband + if_freq_offset(j)")
        return "\n".join(lines)


class ArrayData(object):
    def __init__(self):
        self.subarray = 1
        self.frqsel = 1     #
        # Initialize Header
        #   Keywords
        keys = ""
        #     Originaly from AN Table Header
        keys+= "SUBARRAY,ARRAYX,ARRAYY,ARRAYZ,GSTIA0,DEGPDY,FREQ,RDATE,"
        keys+= "POLARX,POLARY,UT1UTC,DATUTC,TIMSYS,ARRNAM,XYZHAND,FRAME,"
        keys+= "NUMORB,NO_IF,NOPCAL,POLTYPE"
        keys = keys.split(",")
        #   Initialize
        header = collections.OrderedDict()
        for key in keys:
            header[key] = None
        self.header = header

        # Antenna Information
        self.antable_cols = "id,name,x,y,z,mnttype,axisoffset,"
        self.antable_cols+= "poltypeA,polangA,poltypeB,polangB"
        self.antable_cols = self.antable_cols.split(",")
        self.antable = pd.DataFrame(columns=self.antable_cols)
        self.anorbparm = np.zeros([0,0]) # Nant, Orbital Parmeters
        self.anpolcalA = np.zeros([0,0]) # Nant, Npcal*NO_IF
        self.anpolcalB = np.zeros([0,0]) # Nant, Npcal*NO_IF

    def check(self):
        '''
        Keep consistensy between header and data of AN Table
        '''
        # if number of orbital parameter is equals to zero,
        # then reset orbparm
        if self.header["NUMORB"] == 0:
            self.anorbparm = np.array([], dtype=np.float32).reshape([0,0])

        # if number of pcal is zero, then reset POLCALA/B
        if self.header["NOPCAL"] == 0:
            self.anpolcalA = np.array([], dtype=np.float32).reshape([0,0,0])
            self.anpolcalB = np.array([], dtype=np.float32).reshape([0,0,0])

    def avspc(self, dofreq=0):
        if dofreq == 0:
            self.header["NO_IF"]=1
            if self.header["NOPCAL"] != 0:
                Nant, Nif = self.anpolcalA.shape
                Npcal = self.header["NOPCAL"]
                Nif = Nif//Npcal
                self.anpolcalA = self.anpolcalA.reshape([Nant,Npcal*Nif])
                self.anpolcalB = self.anpolcalB.reshape([Nant,Npcal*Nif])


    def __repr__(self):
        lines = []
        lines.append("Sub Array ID: %d"%(self.frqsel))
        lines.append("  Frequency Setup ID: %d"%(self.frqsel))
        lines.append("  Reference Frequency: %.0f Hz"%(self.header["FREQ"]))
        lines.append("  Reference Date: %s"%(self.header["RDATE"]))
        lines.append("  AN Table Contents:")
        lines.append(prt(self.antable["id,name,x,y,z,mnttype".split(",")],indent*2,output=True))
        return "\n".join(lines)

class SourceData(object):
    def __init__(self):
        # Frequency Setup Number
        self.frqsel=1

        # Initialize Header
        #   Keywords
        keys = "NO_IF,VELDEF,VELTYP"
        keys = keys.split(",")
        #   Initialize
        header = collections.OrderedDict()
        for key in keys:
            header[key] = None
        self.header = header

        # Table
        sutable_cols ="id,source,qual,calcode,bandwidth,radec,equinox,"
        sutable_cols+="raapp,decapp,pmra,pmdec"
        sutable_cols=sutable_cols.split(",")
        self.sutable=pd.DataFrame(columns=sutable_cols)
        self.sutable_cols=sutable_cols
        self.suiflux=np.zeros([0,0])
        self.suqflux=np.zeros([0,0])
        self.suuflux=np.zeros([0,0])
        self.suvflux=np.zeros([0,0])
        self.sufreqoff=np.zeros([0,0])
        self.sulsrvel=np.zeros([0,0])
        self.surestfreq=np.zeros([0,0])

    def __repr__(self):
        lines = []
        lines.append("Frequency Setup ID: %d"%(self.frqsel))
        lines.append("  Sources:")
        lines.append(prt(self.sutable["id,source,radec,equinox".split(",")],indent*2,output=True))
        return "\n".join(lines)

def _selfcal_error_func(gain,ant1,ant2,w,X,std_amp,std_pha,Nant,Ndata):
    g1 = np.asarray([gain[ant1[i]]+1j*gain[Nant+ant1[i]] for i in xrange(Ndata)])
    g2 = np.asarray([gain[ant2[i]]-1j*gain[Nant+ant2[i]] for i in xrange(Ndata)])
    dV = w*(X-g1*g2)
    Pamp = [(np.sqrt(gain[i]**2+gain[i+Nant]**2)-1)/std_amp for i in xrange(Nant)]
    Ppha = [np.arctan2(gain[i+Nant],gain[i])/std_pha for i in xrange(Nant)]
    return np.hstack([np.real(dV),np.imag(dV),Pamp,Ppha])

def _selfcal_error_dfunc(gain,ant1,ant2,w,X,std_amp,std_pha,Nant,Ndata):
    ddV = np.zeros([Ndata*2+Nant*2,Nant*2])
    for idata in xrange(Ndata):
        # antenna id
        i = ant1[idata]
        j = ant2[idata]

        # gains
        gr_i = gain[i]
        gi_i = gain[i+Nant]
        gr_j = gain[j]
        gi_j = gain[j+Nant]

        ddV[idata,       i]      = -w[idata]*gr_j # d(Vreal)/d(gr_i)
        ddV[idata,       j]      = -w[idata]*gr_i # d(Vreal)/d(gr_j)
        ddV[idata,       i+Nant] = -w[idata]*gi_j # d(Vreal)/d(gi_i)
        ddV[idata,       j+Nant] = -w[idata]*gi_i # d(Vreal)/d(gi_j)

        ddV[idata+Ndata, i]      = +w[idata]*gi_j # d(Vimag)/d(gr_i)
        ddV[idata+Ndata, j]      = -w[idata]*gi_i # d(Vimag)/d(gr_j)
        ddV[idata+Ndata, i+Nant] = -w[idata]*gr_j # d(Vimag)/d(gi_i)
        ddV[idata+Ndata, j+Nant] = +w[idata]*gr_i # d(Vimag)/d(gi_j)
    for i in xrange(Nant):
        ampsq = gain[i]**2+gain[i+Nant]**2
        amp = np.sqrt(ampsq)
        ddV[i+Ndata*2, i]      = gain[i]/amp
        ddV[i+Ndata*2, i+Nant] = gain[i+Nant]/amp
        ddV[i+Nant+Ndata*2, i]      = gain[i]/ampsq
        ddV[i+Nant+Ndata*2, i+Nant] =-gain[i+Nant]/ampsq
    return ddV

def _eval_image_eachfreq(u, v, image):
    # make a loop for Nif, Nch
    Nx    = image.header["nx"]
    Ny    = image.header["ny"]
    Nxref = image.header["nxref"]
    Nyref = image.header["nyref"]

    # dx_rad,dy_rad
    dx_rad = np.deg2rad(image.header["dx"])
    dy_rad = np.deg2rad(image.header["dy"])

    # normalize u, v coordinates
    u_scaled = np.float64(2*np.pi*dx_rad*u)
    v_scaled = np.float64(2*np.pi*dy_rad*v)

    # model intensity
    I2d = np.float64(image.data[0, 0])
    I2d = np.asfortranarray(I2d.T)

    # model visibility
    Vreal,Vimag = fortlib.fftlib.nufft_fwd_real(u_scaled,v_scaled,I2d)

    # phase shift (the image center to the reference pixel)
    #    pixel deviation of the center of
    #    the image from the reference pixel
    ix = Nx/2. + 1 - Nxref
    iy = Ny/2. + 1 - Nyref

    # complex visbility
    Vcmp = Vreal+1j*Vimag

    # new visibility due to the deviation (ix,iy)
    Vcmp *= np.exp(1j*(u_scaled*ix+v_scaled*iy))
    Vreal = np.real(Vcmp)
    Vimag = np.imag(Vcmp)

    return Vreal,Vimag
