#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module describes UVTable/UVseries object, which is a basis class of
table data sets in this library
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# standard modules
import collections
import numpy as np
import pandas as pd
import astropy.constants as ac
import astropy.time as at
import datetime as dt


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class UVTable(pd.DataFrame):
    '''
    This is a class describing common variables and methods of VisTable,
    BSTable and CATable.
    '''
    @property
    def _constructor(self):
        return UVTable

    @property
    def _constructor_sliced(self):
        return UVSeries

    def gencvtables(self, nfold=10, seed=0):
        '''
        This method generates data sets for N-fold cross varidations.

        Args:
            nfolds (int): the number of folds
            seed (int): the seed number of pseudo ramdam numbers

        Returns:
            collections.OrderedDict object that lists all training/validating data
        '''
        Ndata = self.shape[0]  # Number of data points
        Nval = Ndata // nfold    # Number of Varidation data

        # Make shuffled data
        shuffled = self.sample(Ndata,
                               replace=False,
                               random_state=np.int64(seed))

        # Name
        out = collections.OrderedDict()
        for icv in xrange(nfold):
            trainkeyname = "t%d" % (icv)
            validkeyname = "v%d" % (icv)
            if Nval * (icv + 1) == Ndata:
                train = shuffled.loc[:Nval*icv,:]
            else:
                train = pd.concat([shuffled.loc[:Nval*icv,:],
                                   shuffled.loc[Nval*(icv+1):,:]])
            valid = shuffled[Nval*icv:Nval*(icv+1)]
            out[trainkeyname] = train
            out[validkeyname] = valid
        return out

    def uvunitconv(self, unit1="l", unit2="l"):
        '''
        Derive a conversion factor of units for the baseline length from unit1
        to unit2. Available angular units are l[ambda], kl[ambda], ml[ambda],
        gl[ambda], m[eter] and km[eter].

        Args:
          unit1 (str): the first unit
          unit2 (str): the second unit

        Returns:
          conversion factor from unit1 to unit2 in float.
        '''
        if unit1 == unit2:
            return 1.

        # Convert from unit1 to lambda
        if unit1.lower().find("l") == 0:
            conv = 1.
        elif unit1.lower().find("kl") == 0:
            conv = 1e3
        elif unit1.lower().find("ml") == 0:
            conv = 1e6
        elif unit1.lower().find("gl") == 0:
            conv = 1e9
        elif unit1.lower().find("m") == 0:
            conv = 1/(ac.c.si.value / self["freq"])
        elif unit1.lower().find("km") == 0:
            conv = 1/(ac.c.si.value / self["freq"] / 1e3)
        else:
            print("Error: unit1=%s is not supported" % (unit1))
            return -1

        # Convert from lambda to unit2
        if unit2.lower().find("l") == 0:
            conv /= 1.
        elif unit2.lower().find("kl") == 0:
            conv /= 1e3
        elif unit2.lower().find("ml") == 0:
            conv /= 1e6
        elif unit2.lower().find("gl") == 0:
            conv /= 1e9
        elif unit2.lower().find("m") == 0:
            conv *= ac.c.si.value / self["freq"]
        elif unit2.lower().find("km") == 0:
            conv *= ac.c.si.value / self["freq"] / 1e3
        else:
            print("Error: unit2=%s is not supported" % (unit2))
            return -1

        return conv

    def get_uvunitlabel(self, uvunit=None):
        '''
        Get a unit label name for uvunits.
        Available units are l[ambda], kl[ambda], ml[ambda], gl[ambda], m[eter]
        and km[eter].

        Args:
          uvunit (str, default is None):
            The input unit. If uvunit is None, it will use self.uvunit.

        Returns:
          The unit label name in string.
        '''
        if uvunit is None:
            uvunit = self.uvunit

        if uvunit.lower().find("l") == 0:
            unitlabel = r"$\lambda$"
        elif uvunit.lower().find("kl") == 0:
            unitlabel = r"$10^3 \lambda$"
        elif uvunit.lower().find("ml") == 0:
            unitlabel = r"$10^6 \lambda$"
        elif uvunit.lower().find("gl") == 0:
            unitlabel = r"$10^9 \lambda$"
        elif uvunit.lower().find("m") == 0:
            unitlabel = "m"
        elif uvunit.lower().find("km") == 0:
            unitlabel = "km"
        else:
            print("Error: uvunit=%s is not supported" % (unit2))
            return -1
        return unitlabel

    def utc_astropytime(self):
        return at.Time(np.datetime_as_string(self.utc.values))

    def gst_datetime(self, continuous=True, wraphour=0):
        '''
        get GST in datetime
        '''
        Ndata = len(self.utc)

        utc = self.utc_astropytime()
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

    def to_csv(self, filename, float_format=r"%22.16e", index=False,
               index_label=False, **args):
        '''
        Output table into csv files using pd.DataFrame.to_csv().
        Default parameter will be
          float_format=r"%22.16e"
          index=False
          index_label=False.
        see DocStrings for pd.DataFrame.to_csv().

        Args:
            filename (string or filehandle): output filename
            **args: other arguments of pd.DataFrame.to_csv()
        '''
        super(UVTable, self).to_csv(filename,
                                    index=False,
                                    index_label=False,
                                    **args)


class UVSeries(pd.Series):
    @property
    def _constructor(self):
        return UVSeries

    @property
    def _constructor_expanddim(self):
        return UVTable
