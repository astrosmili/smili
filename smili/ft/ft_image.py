#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
This module wrap up functions for imaging using various optimizers.
'''
__author__ = "Smili Developer Team"

class FT_Image(object):
    def __init__(self,u,v,dx,dy,Nx,Ny):
        '''
        Initialize the Fourier Transform functions

        Args:
            u, v (1d numpy array): spatial frequencies in lambda
            dx, dy (float): the pixel sizes of input images along RA, Dec axises, respectively, in radians
            Nx, Ny (float): the number of pixels of input images along RA, Dec axises, respectively.

        Returns:
            FT_Image object
        '''
        # Sanity Check
        if u.size != v.size:
            raise ValueError("u and v must have the same size")

        if dx > 0:
            print("[Warining] the pixel increment dx for RA is positive. (usually negative for astronomical images)")

        if dy < 0:
            print("[Warining] the pixel increment dy for Dec is negative. (usually positive for astronomical images)")

        # the size of u & v vectors
        Nuv = u.size

        # scale u, v coordinates
        #     the negative factor is coming from the fact that the NFFT's fourier exponent sign is opposite to the radio astronomy convension.
        #     see NFFT's documentation.
        u_nfft = u * dx * -1
        v_nfft = v * dy * -1

        # initialize nfft routines
        self.plan = NFFT([Ny, Nx], Nuv, d=2)
        self.plan.x = np.vstack([v_nfft,u_nfft]).T
        self.plan.precompute()

        # set forward / inverse FT functions
        self.forward = self.nfft2d_simple_forward
        self.adjoint = self.nfft2d_simple_adjoint
        self.adjoint_real = self.nfft2d_simple_adjoint_real

    def nfft2d_simple_forward(self, I2d):
        '''
        Two-dimensional Forward Non-uniform Fast Fourier Transform

        Args:
            image in two dimensional numpy array

        Returns:
            complex visibilities in one dimensional numpy array
        '''
        self.plan.f_hat = I2d
        return self.plan.trafo()

    def nfft2d_simple_adjoint(self, Vcmp):
        '''
        Two-dimensional Adjoint Non-uniform Fast Fourier Transform

        Args:
            complex visibilities in one dimensional numpy array

        Returns:
            image in two dimensional numpy array
        '''
        self.plan.f = Vcmp
        return self.plan.adjoint()

    def nfft2d_simple_adjoint_real(self, Vcmp):
        '''
        Two-dimensional Adjoint Non-uniform Fast Fourier Transform

        Args:
            complex visibilities in one dimensional numpy array

        Returns:
            image in two dimensional numpy array
        '''
        self.plan.f = Vcmp
        return np.real(self.plan.adjoint())
