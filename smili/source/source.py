#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
# astropy
from astropy.coordinates import SkyCoord


# ------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------
class Source(object):
    # name
    name = "mysource"
    skycoord = None
    
    def __init__(self, name, skycoord=None):
        self.name = name
        
        if skycoord is None:
            self.skycoord = SkyCoord.from_name("M87")
        elif type(skycoord) == SkyCoord:
            self.skycoord = SkyCoord.copy()

    def __repr__(self):
        output = "%s %s"%(self.name, self.skycoord.to_string("hmsdms"))
        return output
    
    def _repr_html_(self):
        output = "%s %s"%(self.name, self.skycoord.to_string("hmsdms"))
        return output