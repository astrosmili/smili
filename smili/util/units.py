#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This module provides a quick shortcut to the major units
'''
__author__ = "Smili Developer Team"

import astropy.units as au

# Dimension Less
DIMLESS = au.Unit("")

# Length
M = au.Unit("m")
MM = au.Unit("mm")
CM = au.Unit("cm")
KM = au.Unit("km")

# Radio Flux
JY = au.Unit("Jy")
MJY = au.Unit("mJy")
UJY = au.Unit("uJy")

# Angular Size
UAS = au.Unit("uas")
MAS = au.Unit("mas")
ASEC = au.Unit("arcsec")
AMIN = au.unit("arcmin")
DEG = au.Unit("deg")
RAD = au.Unit("rad")

# Time
SEC = au.Unit("second")
MIN = au.Unit("minute")
HOUR = au.Unit("hour")
DAY = au.Unit("day")
YEAR = au.Unit("year")

# Mass
MSUN = au.Unit("Msun")