#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This module describes data classes and related functions to handle UVFITS data.
'''
__author__ = "Smili Developer Team"

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
import matplotlib

def matplotlibrc(nrows=1,ncols=1,width=250,height=250):
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

def reset_matplotlibrc():
    matplotlib.rcdefaults()