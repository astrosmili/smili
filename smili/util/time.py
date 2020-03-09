#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
'''
__author__ = "Smili Developer Team"
# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
from numpy import arange, concatenate

# internal
from astropy.time import Time, TimeDelta
from .units import MIN, SEC

def create_utc_array(tstart, tend, tscan=6.*MIN, tint=20*MIN, tap=30*SEC):
    '''
    Create a list of UTC based on input scan patterns

    Args:
        tstart (astropy.time.Time): The start time of the observations 
        tend (astropy.time.Time): The end time of the observations 
        tscan (astropy.units.Quantity): Scan length (default=6 minutes)
        tint (astropy.units.Quantity): Scan interval (default=20 minutes)
        tap (astropy.units.Quantity): Accumulation periods (default=30 seconds)
    Returns:
        utc (astropy.time.Time): UTC of each time segment
    '''
    ttotal = tend - tstart
    ttotal_sec = ttotal.sec
    tscan_sec = tscan.to_value(unit=SEC)
    tint_sec = tint.to_value(unit=SEC)
    tap_sec = tap.to_value(unit=SEC)

    tscan_start_sec = arange(0,ttotal_sec,tscan_sec+tint_sec)
    tscan_seg_sec = arange(tap_sec/2,tscan_sec,tap_sec)

    utc = tstart + TimeDelta(concatenate([tscan_start+tscan_seg_sec for tscan_start in tscan_start_sec]), format="sec")
    return utc
