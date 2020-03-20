#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Definition and Global Variables
# ------------------------------------------------------------------------------
def set_global(name, default):
    '''
    '''
    if name not in globals():
        globals()[name] = default

set_global("__smili_nproc", 1)