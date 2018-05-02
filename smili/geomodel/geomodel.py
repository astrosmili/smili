#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This is a submodule of smili.geomodel, containing functions to
calculate visibilities and images of some geometric models.

Here, we note that the whole design of this module is imspired by
Lindy Blackburn's python module 'modmod', and
we would like to thank Lindy to share his idea.
'''
import numpy as np
import theano
import theano.tensor as T
from .. import util

# initial variables
x_T, y_T = T.scalars("x","y")
u_T, v_T = T.scalars("u","v")
u1_T, v1_T = T.scalars("u1","v1")
u2_T, v2_T = T.scalars("u2","v2")
u3_T, v3_T = T.scalars("u3","v3")
u4_T, v4_T = T.scalars("u4","v4")

class GeoModel(object):
    def __init__(self, Vreal=None, Vimag=None, I=None):
        if Vreal is None:
            self.Vreal = lambda u=u_T, v=v_T: (u-u)*(v-v)
        else:
            self.Vreal = Vreal

        if Vimag is None:
            self.Vimag = lambda u=u_T, v=v_T: (u-u)*(v-v)
        else:
            self.Vimag = Vimag

        if I is None:
            self.I = lambda x=x_T, y=y_T: (x-x)*(y-y)
        else:
            self.I = I

    def __add__(self, other):
        if type(self) == type(other):
            Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) + other.Vreal(u,v)
            Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) + other.Vimag(u,v)
            I = lambda x=x_T, y=y_T: self.I(x,y) + other.I(x,y)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Addition can be calculated only between the same type of objects")

    def __iadd__(self, other):
        if type(self) == type(other):
            Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) + other.Vreal(u,v)
            Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) + other.Vimag(u,v)
            I = lambda x=x_T, y=y_T: self.I(x,y) + other.I(x,y)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Addition can be calculated only between the same type of objects")

    def __sub__(self, other):
        if type(self) == type(other):
            Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) - other.Vreal(u,v)
            Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) - other.Vimag(u,v)
            I = lambda x=x_T, y=y_T: self.I(x,y) - other.I(x,y)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Subtraction can be calculated only between the same type of objects")


    def __isub__(self, other):
        if type(self) == type(other):
            Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) - other.Vreal(u,v)
            Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) - other.Vimag(u,v)
            I = lambda x=x_T, y=y_T: self.I(x,y) - other.I(x,y)
            return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)
        else:
            raise ValueError("Subtraction can be calculated only between the same type of objects")


    def __mul__(self, other):
        Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) * other
        Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) * other
        I = lambda x=x_T, y=y_T: self.I(x,y) * other
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def __imul__(self, other):
        Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) * other
        Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) * other
        I = lambda x=x_T, y=y_T: self.I(x,y) * other
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def __truediv__(self, other):
        Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) / other
        Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) / other
        I = lambda x=x_T, y=y_T: self.I(x,y) / other
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def __itruediv__(self, other):
        Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) / other
        Vimag = lambda u=u_T, v=v_T: self.Vimag(u,v) / other
        I = lambda x=x_T, y=y_T: self.I(x,y) / other
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def shift(self, deltax=0., deltay=0., angunit="mas"):
        angunit2rad = angconv(angunit, "rad")
        dx = deltax * angunit2rad
        dy = deltay * angunit2rad
        Vreal = lambda u=u_T, v=v_T: self.Vreal(u,v) * T.cos(2*np.pi*(u*dx+v*dy)) - self.Vimag(u,v) * T.sin(2*np.pi*(u*dx+v*dy))
        Vimag = lambda u=u_T, v=v_T: self.Vreal(u,v) * T.sin(2*np.pi*(u*dx+v*dy)) + self.Vimag(u,v) * T.cos(2*np.pi*(u*dx+v*dy))
        I = lambda x=x_T, y=y_T: self.I(x-dx,y-dy)
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)

    def rotate(self, deltaPA=0., deg=True):
        if deg:
            dPA = deltaPA * np.pi / 180
        else:
            dPA = deltaPA
        cosdpa = T.cos(dPA)
        sindpa = T.sin(dPA)
        x1 = lambda x=x_T, y=y_T: x * cosdpa - y * sindpa
        y1 = lambda x=x_T, y=y_T: x * sindpa + y * cosdpa
        Vreal = lambda u=u_T, v=v_T: self.Vreal(x1(u,v),y1(u,v))
        Vimag = lambda u=u_T, v=v_T: self.Vimag(x1(u,v),y1(u,v))
        I = lambda x=x_T, y=y_T: self.I(x1(x,y),y1(x,y))
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def scale(self, hx=1., hy=None):
        if hy is None:
            hy = hx
        Vreal = lambda u=u_T, v=v_T: self.Vreal(u*hx, v*hy)
        Vimag = lambda u=u_T, v=v_T: self.Vimag(u*hx, v*hy)
        I = lambda x=x_T, y=y_T: self.I(x/hx,y/hy)/hx/hy
        return GeoModel(Vreal=Vreal, Vimag=Vimag, I=I)


    def Vamp(self, u=u_T, v=v_T):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        Vre = self.Vreal(u,v)
        Vim = self.Vimag(u,v)
        return T.sqrt(Vre*Vre + Vim*Vim)


    def logVamp(self, u=u_T, v=v_T):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        return T.log(self.Vamp(u,v))


    def Vphase(self, u=u_T, v=v_T):
        '''
        Return theano symbolic represenation of the visibility phase

        Args:
            u, v: uv-coordinates
        Return:
            phase in radian
        '''
        return T.arctan2(self.Vimag(u,v), self.Vreal(u,v))


    # Bi-spectrum
    def Bre(self, u1=u1_T, v1=v1_T, u2=u2_T, v2=v2_T, u3=u3_T, v3=v3_T):
        '''
        Return theano symbolic represenation of the real part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            real part of the bi-spectrum
        '''
        Vre1 = self.Vreal(u1,v1)
        Vim1 = self.Vimag(u1,v1)
        Vre2 = self.Vreal(u2,v2)
        Vim2 = self.Vimag(u2,v2)
        Vre3 = self.Vreal(u3,v3)
        Vim3 = self.Vimag(u3,v3)
        Bre =  -Vim1*Vim2*Vre3 - Vim1*Vim3*Vre2 - Vim2*Vim3*Vre1 + Vre1*Vre2*Vre3
        return Bre


    def Bim(self, u1=u1_T, v1=v1_T, u2=u2_T, v2=v2_T, u3=u3_T, v3=v3_T):
        '''
        Return theano symbolic represenation of the imaginary part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            imaginary part of the bi-spectrum
        '''
        Vre1 = self.Vreal(u1,v1)
        Vim1 = self.Vimag(u1,v1)
        Vre2 = self.Vreal(u2,v2)
        Vim2 = self.Vimag(u2,v2)
        Vre3 = self.Vreal(u3,v3)
        Vim3 = self.Vimag(u3,v3)
        Bim = -Vim1*Vim2*Vim3 + Vim1*Vre2*Vre3 + Vim2*Vre1*Vre3 + Vim3*Vre1*Vre2
        return Bim


    def Bamp(self, u1=u1_T, v1=v1_T, u2=u2_T, v2=v2_T, u3=u3_T, v3=v3_T):
        '''
        Return theano symbolic represenation of the amplitude of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            amplitude of the bi-spectrum
        '''
        Bre = self.Bre(u1, v1, u2, v2, u3, v3)
        Bim = self.Bim(u1, v1, u2, v2, u3, v3)
        Bamp = T.sqrt(Bre*Bre+Bim*Bim)
        return Bamp


    def Bphase(self, u1=u1_T, v1=v1_T, u2=u2_T, v2=v2_T, u3=u3_T, v3=v3_T):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        Bre = self.Bre(u1, v1, u2, v2, u3, v3)
        Bim = self.Bim(u1, v1, u2, v2, u3, v3)
        Bphase = T.arctan2(Bim, Bre)
        return Bphase


    # Closure Amplitudes
    def Camp(self, u1=u1_T, v1=v1_T, u2=u2_T, v2=v2_T, u3=u3_T, v3=v3_T, u4=u4_T, v4=v4_T):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        Vamp1 = self.Vamp(u1, v1)
        Vamp2 = self.Vamp(u2, v2)
        Vamp3 = self.Vamp(u3, v3)
        Vamp4 = self.Vamp(u4, v4)
        return Vamp1*Vamp2/Vamp3/Vamp4


    def logCamp(self, u1=u1_T, v1=v1_T, u2=u2_T, v2=v2_T, u3=u3_T, v3=v3_T, u4=u4_T, v4=v4_T):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        logVamp1 = self.logVamp(u1, v1)
        logVamp2 = self.logVamp(u2, v2)
        logVamp3 = self.logVamp(u3, v3)
        logVamp4 = self.logVamp(u4, v4)
        return logVamp1+logVamp2-logVamp3-logVamp4


#-------------------------------------------------------------------------------
# Some calculations
#-------------------------------------------------------------------------------
def dphase(phase1, phase2):
    dphase = phase2 - phase1
    return T.arctan2(T.sin(dphase), T.cos(dphase))

def angconv(unit1="deg", unit2="deg"):
    '''
    return a conversion factor from unit1 to unit2
    Available angular units are uas, mas, asec or arcsec, amin or arcmin and degree.
    '''
    if unit1 == unit2:
        return 1

    # Convert from unit1 to "arcsec"
    if unit1 == "deg":
        conv = 3600
    elif unit1 == "rad":
        conv = 180 * 3600 / np.pi
    elif unit1 == "arcmin" or unit1 == "amin":
        conv = 60
    elif unit1 == "arcsec" or unit1 == "asec":
        conv = 1
    elif unit1 == "mas":
        conv = 1e-3
    elif unit1 == "uas":
        conv = 1e-6
    else:
        print("Error: unit1=%s is not supported" % (unit1))
        return -1

    # Convert from "arcsec" to unit2
    if unit2 == "deg":
        conv /= 3600
    elif unit2 == "rad":
        conv /= (180 * 3600 / np.pi)
    elif unit2 == "arcmin" or unit2 == "amin":
        conv /= 60
    elif unit2 == "arcsec" or unit2 == "asec":
        pass
    elif unit2 == "mas":
        conv *= 1000
    elif unit2 == "uas":
        conv *= 1000000
    else:
        print("Error: unit2=%s is not supported" % (unit2))
        return -1

    return conv


#-------------------------------------------------------------------------------
# Phase for symmetric sources
#-------------------------------------------------------------------------------
def phaseshift(u,v,x0=0,y0=0,angunit="mas"):
    '''
    Phase of a symmetric object (Gaussians, Point Sources, etc).
    This function also can be used to compute a phase shift due to positional shift.
    Args:
        u, v (mandatory): uv coordinates in lambda
        x0=0, y0=0: position of centorid or positional shift in angunit.
        angunit="mas": angular unit of x0, y0 (uas, mas, asec, amin, deg, rad)
    return:
        phase in rad
    '''
    return 2*np.pi*(u*x0+v*y0)*angconv(angunit, "rad")


#-------------------------------------------------------------------------------
# Point Source for symmetric sources
#-------------------------------------------------------------------------------
def Gaussian(x0=0,y0=0,totalflux=1,majsize=1,minsize=None,pa=0,angunit="mas"):
    '''
    Create geomodel.geomodel.GeoModel Object for the specified Gaussian
    Args:
        x0=0.0, y0=0.0: position of the centorid
        totalflux=1.0: total flux of the Gaussian
        majsize, minsize: Major/Minor-axis FWHM size of the Gaussian
        pa: Position Angle of the Gaussian in degree
        angunit="mas": angular unit of x0, y0, majsize, minsize (uas, mas, asec, amin, deg, rad)
    Returns:
    Returns:
        geomodel.geomodel.GeoModel Object for the specified Gaussian
    '''
    if minsize is None:
        minsize = majsize

    # define a Gaussian with F=1 jy, size = 1 (angunit)
    sigma = angconv(angunit, "rad")/np.sqrt(8*np.log(2))
    Vreal = lambda u=u_T, v=v_T: T.exp(-2*np.pi*np.pi*(u*u+v*v)*sigma*sigma)
    I = lambda x=x_T, y=y_T: 1/2/np.pi/sigma/sigma*T.exp(-(x*x+y*y)/2/sigma/sigma)
    output = GeoModel(Vreal=Vreal, I=I)

    # transform Gaussian, so that it will be elliptical Gaussian
    output = output * totalflux
    output = output.scale(hx=minsize, hy=majsize)
    output = output.rotate(deltaPA=pa, deg=True)
    output = output.shift(deltax=x0, deltay=y0, angunit=angunit)
    return output
