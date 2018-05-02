#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
'''
This is a submodule of smili. This module saves some common functions,
variables, and data types in the smili module.
'''
import numpy as np

def matplotlibrc(nrows=1,ncols=1,width=250,height=250):
    import matplotlib

    # Get this from LaTeX using \showthe\columnwidth
    fig_width_pt  = width*ncols
    fig_height_pt = height*nrows
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width     = fig_width_pt*inches_per_pt  # width in inches
    fig_height    = fig_height_pt*inches_per_pt # height in inches
    fig_size      = [fig_width,fig_height]
    params = {'axes.labelsize': 13,
              'axes.titlesize': 13,
              'legend.fontsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'figure.figsize': fig_size,
              'figure.dpi'    : 300
    }
    matplotlib.rcParams.update(params)


def angconv(unit1="deg", unit2="deg"):
    '''
    return a conversion factor from unit1 to unit2
    Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
    '''
    if unit1 == unit2:
        return 1

    # Convert from unit1 to "arcsec"
    if unit1 == "deg":
        conv = 3600.
    elif unit1 == "rad":
        conv = 180. * 3600. / np.pi
    elif unit1 == "arcmin" or unit1 == "amin":
        conv = 60.
    elif unit1 == "arcsec" or unit1 == "asec":
        conv = 1.
    elif unit1 == "mas":
        conv = 1e-3
    elif unit1 == "uas":
        conv = 1e-6
    else:
        print("Error: unit1=%s is not supported" % (unit1))
        return -1

    # Convert from "arcsec" to unit2
    if unit2 == "deg":
        conv /= 3600.
    elif unit2 == "rad":
        conv /= (180. * 3600. / np.pi)
    elif unit2 == "arcmin" or unit2 == "amin":
        conv /= 60.
    elif unit2 == "arcsec" or unit2 == "asec":
        pass
    elif unit2 == "mas":
        conv *= 1e3
    elif unit2 == "uas":
        conv *= 1e6
    else:
        print("Error: unit2=%s is not supported" % (unit2))
        return -1

    return conv


def prt(obj, indent="", output=False):
    '''
    a useful print function
    '''
    if   type(obj) == type(""):
        lines = obj.split("\n")
    else:
        if hasattr(obj, '__str__'):
            lines = obj.__str__().split("\n")
        elif hasattr(obj, '__repr__'):
            lines = obj.__repr__().split("\n")
        else:
            lines = [""]
    for i in xrange(len(lines)):
        lines[i] = indent + lines[i]
    if output:
        return "\n".join(lines)
    else:
        print("\n".join(lines))
