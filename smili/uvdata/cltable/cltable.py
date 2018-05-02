#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
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
#import matplotlib.pyplot as plt
import astropy.time as at

# internal
#from ..uvtable   import UVTable, UVSeries
#from ..gvistable import GVisTable, GVisSeries
#from ..catable   import CATable, CASeries
#from ..bstable   import BSTable, BSSeries
#from ... import imdata


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
        subarrids = uvfits.subarrays.keys()
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
        subarrids = self.gaintabs.keys()
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
        subarrids = self.gaintabs.keys()
        for subarrid in subarrids:
            # get gain amplitude
            greal = self.gaintabs[subarrid]["gain"][:,:,:,:,:,0]
            gimag = self.gaintabs[subarrid]["gain"][:,:,:,:,:,1]
            gainamp = np.sqrt(greal*greal + gimag*gimag)

            # make normalized gain
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,0] = greal/gainamp
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,1] = gimag/gainamp

        return out

    def get_gaintable(self):
        '''
        This method make a gain table with pandas.DataFrame.
        '''
        out = pd.DataFrame()

        # make gain table for each subarray
        subarrids = self.gaintabs.keys()
        for subarrid in subarrids:

            # get the number of data along each dimension
            Ntime,Nif,Nch,Nstokes,Nant,Ncomp = self.gaintabs[subarrid]["gain"].shape
            for iif,ich,istokes in itertools.product(xrange(Nif),xrange(Nch),xrange(Nstokes)):

                # get complex gain
                greal = self.gaintabs[subarrid]["gain"][:,iif,ich,istokes,:,0]
                gimag = self.gaintabs[subarrid]["gain"][:,iif,ich,istokes,:,1]
                gain = greal + 1j*gimag
                gain = pd.DataFrame(gain)
                gain.columns = xrange(1,Nant+1,1)

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

    def rescale(self):
        '''
        This method re-scales gain amplitude with mean value.
        '''
        out = copy.deepcopy(self)

        # make gain table for each subarray
        subarrids = self.gaintabs.keys()
        for subarrid in subarrids:
            # get mean amplitude
            greal = self.gaintabs[subarrid]["gain"][:,:,:,:,:,0]
            gimag = self.gaintabs[subarrid]["gain"][:,:,:,:,:,1]
            meanamp = np.average(np.sqrt(greal*greal + gimag*gimag))
            print(meanamp)

            # make rescaled gain
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,0] = greal/meanamp
            out.gaintabs[subarrid]["gain"][:,:,:,:,:,1] = gimag/meanamp

        return out
