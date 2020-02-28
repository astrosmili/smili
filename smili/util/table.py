#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili. This module saves some common functions,
variables, and data types in the smili module.
'''
from pandas as DataFrame, Series

class DataTable(DataFrame):
    '''
    This is a class describing common variables and methods of VisTable,
    BSTable and CATable.
    '''
    @property
    def _constructor(self):
        return DataTable

    @property
    def _constructor_sliced(self):
        return DataSeries

class DataSeries(Series):
    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataTable