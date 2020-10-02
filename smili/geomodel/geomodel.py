#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili.geomodel, containing functions to
calculate visibilities and images of some geometric models.

Here, we note that the whole design of this module is imspired by
Lindy Blackburn's python module 'modmod', and
we would like to thank Lindy to share his idea.
'''
import numpy as np
from .. import util

class GeoModel(object):
    def __init__(self, V=None, I=None):
        if V is None:
            self.V = lambda u, v: (u-u)*(v-v)
        else:
            self.V = V

        if I is None:
            self.I = lambda x, y: (x-x)*(y-y)
        else:
            self.I = I

    def __add__(self, other):
        if type(self) == type(other):
            V = lambda u, v: self.V(u,v) + other.V(u,v)
            I = lambda x, y: self.I(x,y) + other.I(x,y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError("Addition can be calculated only between the same type of objects")

    def __iadd__(self, other):
        if type(self) == type(other):
            V = lambda u, v: self.V(u,v) + other.V(u,v)
            I = lambda x, y: self.I(x,y) + other.I(x,y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError("Addition can be calculated only between the same type of objects")

    def __sub__(self, other):
        if type(self) == type(other):
            V = lambda u, v: self.V(u,v) - other.V(u,v)
            I = lambda x, y: self.I(x,y) - other.I(x,y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError("Subtraction can be calculated only between the same type of objects")


    def __isub__(self, other):
        if type(self) == type(other):
            V = lambda u, v: self.V(u,v) - other.V(u,v)
            I = lambda x, y: self.I(x,y) - other.I(x,y)
            return GeoModel(V=V, I=I)
        else:
            raise ValueError("Subtraction can be calculated only between the same type of objects")


    def __mul__(self, other):
        V = lambda u, v: self.V(u,v) * other
        I = lambda x, y: self.I(x,y) * other
        return GeoModel(V=V, I=I)


    def __imul__(self, other):
        V = lambda u, v: self.V(u,v) * other
        I = lambda x, y: self.I(x,y) * other
        return GeoModel(V=V, I=I)


    def __truediv__(self, other):
        V = lambda u, v: self.V(u,v) / other
        I = lambda x, y: self.I(x,y) / other
        return GeoModel(V=V, I=I)


    def __itruediv__(self, other):
        V = lambda u, v: self.V(u,v) / other
        I = lambda x, y: self.I(x,y) / other
        return GeoModel(V=V, I=I)


    def shift(self, deltax=0., deltay=0., angunit="mas"):
        angunit2rad = angconv(angunit, "rad")
        dx = deltax * angunit2rad
        dy = deltay * angunit2rad
        dphase = 2*np.pi*(u*dx+v*dy)
        V = lambda u, v: self.V(u,v) * np.exp(1j*dphase)
        I = lambda x, y: self.I(x-dx,y-dy)
        return GeoModel(V=V, I=I)

    def rotate(self, deltaPA=0., deg=True):
        if deg:
            dPA = deltaPA * np.pi / 180
        else:
            dPA = deltaPA
        cosdpa = np.cos(dPA)
        sindpa = np.sin(dPA)
        x1 = lambda x, y: x * cosdpa - y * sindpa
        y1 = lambda x, y: x * sindpa + y * cosdpa
        V = lambda u, v: self.V(x1(u,v),y1(u,v))
        I = lambda x, y: self.I(x1(x,y),y1(x,y))
        return GeoModel(V=V, I=I)


    def scale(self, hx=1., hy=None):
        if hy is None:
            hy = hx
        V = lambda u, v: self.V(u*hx, v*hy)
        I = lambda x, y: self.I(x/hx,y/hy)/hx/hy
        return GeoModel(V=V, I=I)

    def Vre(self, u, v):
        return np.real(self.V(u,v))

    def Vim(self, u, v):
        return np.imag(self.V(u,v))

    def Vamp(self, u, v):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        return np.abs(self.V(u,v))


    def logVamp(self, u, v):
        '''
        Return theano symbolic represenation of the visibility amplitude

        Args:
            u, v: uv-coordinates
        Return:
            amplitude
        '''
        return np.log(self.Vamp(u,v))


    def Vphase(self, u, v):
        '''
        Return theano symbolic represenation of the visibility phase

        Args:
            u, v: uv-coordinates
        Return:
            phase in radian
        '''
        return np.angle(self.V(u,v))


    # Bi-spectrum
    def B(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the real part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            real part of the bi-spectrum
        '''
        V1 = self.V(u1,v1)
        V2 = self.V(u2,v2)
        V3 = self.V(u3,v3)
        return V1*V2*V3

    def Bre(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the imaginary part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            imaginary part of the bi-spectrum
        '''
        return np.real(self.B(u1,v1,u2,v2,u3,v3))

    def Bim(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the imaginary part of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            imaginary part of the bi-spectrum
        '''
        return np.imag(self.B(u1,v1,u2,v2,u3,v3))


    def Bamp(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the amplitude of the Bi-spectrum.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            amplitude of the bi-spectrum
        '''
        return np.abs(self.B(u1, v1, u2, v2, u3, v3))


    def Bphase(self, u1, v1, u2, v2, u3, v3):
        '''
        Return theano symbolic represenation of the phase of the Bi-spectrum.
        if given uv-coodinates are closed, this will be the closure phase.

        Args:
            un, vn (n=1, 2, 3): uv-coordinates
        Return:
            phase of the bi-spectrum
        '''
        return np.angle(self.B(u1, v1, u2, v2, u3, v3))


    # Closure Amplitudes
    def Camp(self, u1, v1, u2, v2, u3, v3, u4, v4):
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


    def logCamp(self, u1, v1, u2, v2, u3, v3, u4, v4):
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
    return np.arctan2(np.sin(dphase), np.cos(dphase))

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
    This function also can be used to compute a phase shift due to positional shifnp.
    Args:
        u, v (mandatory): uv coordinates in lambda
        x0=0, y0=0: position of centorid or positional shift in anguninp.
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
    V = lambda u, v: np.exp(-2*np.pi*np.pi*(u*u+v*v)*sigma*sigma)
    I = lambda x, y: 1/2/np.pi/sigma/sigma*np.exp(-(x*x+y*y)/2/sigma/sigma)
    output = GeoModel(V=V, I=I)

    # transform Gaussian, so that it will be elliptical Gaussian
    if totalflux != 1:
        output = output * totalflux
    if majsize != 1 or minsize != 1:
        output = output.scale(hx=minsize, hy=majsize)
    if pa != 0:
        output = output.rotate(deltaPA=pa, deg=True)
    if x0 != 0 or y0 != 0:
        output = output.shift(deltax=x0, deltay=y0, angunit=angunit)
    return output


def Rectangular(totalflux=1,x0=0,y0=0,dx=1,dy=None,Lx=1.,Ly=None,angunit="mas"):
    '''
    Create geomodel.geomodel.GeoModel Object for the specified RectAngular function
    Args:
        x0=0.0, y0=0.0: position of the centorid
        totalflux=1.0: total flux of the model in Jy
        Lx, Ly: the size of the rectangular function.
        dx, dy: the pixel size of the image
        angunit="mas": angular unit of x0, y0, dx, dy (uas, mas, asec, amin, deg, rad)
    Returns:
        geomodel.geomodel.GeoModel Object for the specified RectAngular function
    '''
    from numpy import where, abs, sinc, pi, exp, isnan
    
    if dy is None:
        dy = dx
    if Ly is None:
        Ly = Lx

    # define a Gaussian with F=1 jy, size = 1 (angunit)
    factor = angconv(angunit, "rad")

    x0rad = x0 * factor
    y0rad = y0 * factor
    Lxrad = abs(factor*Lx)
    Lyrad = abs(factor*Ly)
    dxrad = abs(factor*dx)
    dyrad = abs(factor*dy)

    # get the mean intensity = Total flux / (the number of pixels in the rect angular)
    Imean = totalflux / Lxrad / Lyrad * dxrad * dyrad

    def I(x, y):
        xnorm = abs((x-x0rad)/Lxrad)
        ynorm = abs((y-y0rad)/Lyrad)
        return where(xnorm <= 0.5, 1, 0)*where(ynorm <= 0.5, 1, 0)*Imean

    def V(u, v):
        unorm = Lxrad*u
        vnorm = Lyrad*v
        amp = totalflux * sinc(unorm)*sinc(vnorm)
        phase = 2*pi*(u*x0rad+v*y0rad)
        return amp*exp(1j*phase)

    return GeoModel(V=V, I=I)


def Triangular(totalflux=1,x0=0,y0=0,dx=1,dy=None,angunit="mas"):
    '''
    Create geomodel.geomodel.GeoModel Object for the specified Triangular function
    Args:
        x0=0.0, y0=0.0: position of the centroid
        totalflux=1.0: total flux of the Gaussian
        dx, dy: the size of the Triangular function
        angunit="mas": angular unit of x0, y0, dx, dy (uas, mas, asec, amin, deg, rad)
    Returns:
        geomodel.geomodel.GeoModel Object for the specified Triangular function
    '''
    if dy is None:
        dy = dx

    # define a Gaussian with F=1 jy, size = 1 (angunit)
    conv = angconv(angunit, "rad")
    dxrad = np.abs(conv*dx)
    dyrad = np.abs(conv*dy)
    dxyinv = 1./dxrad/dyrad

    V = lambda u, v: np.exp(-2*np.pi*np.pi*(u*u+v*v)*sigma*sigma)

    def I(x,y):
        xnorm = np.abs(x/dxrad)
        ynorm = np.abs(y/dyrad)
        return np.where(xnorm<=1, (1-xnorm)/dxrad, 0)*np.where(ynorm<=1, (1-ynorm)/dyrad, 0)

    def V(u,v):
        unorm = np.pi*dxrad*u
        vnorm = np.pi*dyrad*v
        return np.square(np.sin(unorm)*np.sin(vnorm)/unorm/vnorm)

    output = GeoModel(V=V, I=I)
    if totalflux != 1:
        output = output * totalflux
    if x0 != 0 or y0 != 0:
        output = output.shift(deltax=x0, deltay=y0, angunit=angunit)
    return output
