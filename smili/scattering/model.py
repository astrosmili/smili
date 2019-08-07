#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili.scattering, containing model classes to compute
image-domain and/or Fourier-domain effects of interstellar scattering.
'''
import copy
import itertools
import tqdm

# numerical packages
import numpy as np
import pandas as pd
from scipy import optimize, linalg
from scipy.optimize import minimize
import scipy.special as ss

# astropy
import astropy.constants as ac

# internal module
from .. import geomodel, util

# import astropy

class P18Model(object):
    '''
    The unisotropic scattering model presented in Psaltis & Johnson et al. 2018.
    Currently, this class can compute the scattering kernel based on
    the dipole model described in Johnson & Narayan 2016, and Psaltis et al. 2018.
    '''
    def __init__(self, alpha=1.38, theta_maj_ref=1.380, theta_min_ref=0.703, D_pc=2.862e3, R_pc=5.4e3, r_in=800.0e5, wavelength_ref=1., pos_ang=81.9):
        '''
        alpha: power-law index of phase structure function, D_phi
        theta_maj_ref: scattered image size along the major axis at reference wavelength (mas)
        theta_min_ref: scattered image size along the minor axis at reference wavelength (mas)
        D_pc: observer-screen distance (pc; e.g., 2.862e3)
        R_pc: source-screen distance (pc; e.g., 5.53e3)
        r_in: inner scale (cm)
        wavelength_ref: reference wavelength (cm)
        pos_ang: position angle of major axis of scattering kernel (deg)
        '''
        self.alpha = alpha
        self.r_in = r_in
        self.D = D_pc/ac.pc.cgs.value  # convert in cm
        self.R = R_pc/ac.pc.cgs.value  # convert in cm
        self.M = D_pc/R_pc      # magnification of the scattering screen
        self.wavelength_ref = wavelength_ref
        self.pos_ang = pos_ang
        self.phi0 = (90. - self.pos_ang)

        # Unit conversion (to radian)
        theta_maj_ref_rad = theta_maj_ref*util.angconv('mas', 'rad')
        theta_min_ref_rad = theta_min_ref*util.angconv('mas', 'rad')

        # Degree of anisotropy, when b << r_in (Psaltis et al. 2018)
        self.A = theta_maj_ref/theta_min_ref
        # dipole model (Johnson & Narayan 2016; Psaltis et al. 2018)
        self.zeta0 = (self.A**2 - 1) / (self.A**2 + 1)

        # Get kzeta (with dipole model, refering to eht-imaging-library; Johnson & Narayan 2016; Psaltis et al. 2018)
        def dipole_Anisotropy(kzeta):
            return np.abs( ss.hyp2f1((self.alpha+2)/2., 0.5, 2., -kzeta)/ss.hyp2f1((self.alpha+2)/2., 1.5, 2., -kzeta) - self.A**2 )

        self.kzeta = minimize(dipole_Anisotropy, self.A, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x     # 3.52 in Psaltis et al. 2018
        self.P_phi_prefac = 1.0/(2.0*np.pi*ss.hyp2f1((self.alpha + 2)/2., 0.5, 1.0, -self.kzeta))

        # Get B parameters
        self.Bmaj = 2**(4-self.alpha) / (self.alpha**2 * (1+self.zeta0) * ss.gamma(self.alpha/2) * np.sqrt(1+self.kzeta) ) / ss.hyp2f1(0.5, (2+self.alpha)/2., 1, -1*self.kzeta)
        self.Bmin = 2**(4-self.alpha) / (self.alpha**2 * (1-self.zeta0) * ss.gamma(self.alpha/2) ) / ss.hyp2f1(0.5, -1*self.alpha/2., 1, -1*self.kzeta)

        # Get C parameter
        FWHM_fac = (2.0 * np.log(2.0))**0.5/np.pi
        self.Qbar0 = 2.0/ss.gamma((2-self.alpha)/2.) * (self.r_in**2*(1.0 + self.M)/(FWHM_fac*(self.wavelength_ref/(2.0*np.pi))**2) )**2 * ( (theta_maj_ref_rad**2 + theta_min_ref_rad**2))
        self.C0 = (self.wavelength_ref/(2.0*np.pi))**2 * self.Qbar0*ss.gamma(1.0 - self.alpha/2.0)/(8.0*np.pi**2*self.r_in**2)

    def rF(self, wavelength_cm=1.):
        rF = np.sqrt(self.D*wavelength_cm / ((1+self.M)*2*np.pi))
        return rF

    def Qbar(self, wavelength_cm=1.):
        ''' Qbar parameter
        '''
        return self.Qbar0 * (wavelength_cm/self.wavelength_ref)**4

    def Cparm(self, wavelength_cm=1.):
        ''' C paramter at wavelength_cm
        '''
        return self.C0 * (wavelength_cm/self.wavelength_ref)**2

    def Dmaj(self, r, wavelength_cm=1.):
        '''
        r: uv-distance in unit of lambda
        wavelength_cm: observing wavelength (cm)
        '''
        r_cm = r*wavelength_cm

        tmp1 = self.C0*(1+self.zeta0)*0.5*self.Bmaj*(2./(self.alpha*self.Bmaj))**(-self.alpha/(2-self.alpha))
        tmp2 = (1+(2./(self.alpha*self.Bmaj))**(2./(2-self.alpha)) * (r_cm/(self.r_in))**2 )**(self.alpha/2.) - 1
        Dmaj = tmp1*tmp2 * (wavelength_cm/self.wavelength_ref)**2
        return Dmaj

    def Dmin(self, r, wavelength_cm=1.):
        '''
        r: uv-distance in unit of lambda
        wavelength_cm: observing wavelength (cm)
        '''
        r_cm = r*wavelength_cm

        tmp1 = self.C0*(1-self.zeta0)*0.5*self.Bmin*(2./(self.alpha*self.Bmin))**(-self.alpha/(2-self.alpha))
        tmp2 = (1+(2./(self.alpha*self.Bmin))**(2./(2-self.alpha)) * (r_cm/(self.r_in))**2 )**(self.alpha/2.) - 1
        Dmin = tmp1*tmp2 * (wavelength_cm/self.wavelength_ref)**2
        return Dmin

    def V_kernel(self, u, v, wavelength_cm=1.):
        '''
        Scatter kernel visibility
        '''
        uvdist = np.sqrt(u**2 + v**2)
        r = uvdist/(1+self.M)
        phi = np.rad2deg(np.arctan2(v, u))

        Dphi = (self.Dmaj(r, wavelength_cm) + self.Dmin(r, wavelength_cm))/2. + (self.Dmaj(r, wavelength_cm) - self.Dmin(r, wavelength_cm))/2. * np.cos(2*np.deg2rad(phi-self.phi0))
        kernel = np.exp(-0.5*Dphi)
        return kernel

    def deblur_vis(self, vistable, u, v, wavelength_cm=1.):
        '''
        Deblur vistable
        '''
        dbtable = copy.deepcopy(vistable)
        scatt_kernel = V_kernel(u, v, wavelength_cm)

        dbtable["amp"] /= scatt_kernel
        dbtable["sigma"] /= scatt_kernel
        return dbtable

    def gen_geomodel(self, wavelength_cm=1.):
        ''' (under construction)
        generate GeoModel
        '''
        def I(x,y):
            return np.zeros(len(x))

        def V(u,v):
            return self.V_kernel(u, v, wavelength_cm=wavelength_cm)

        self.geomodel = geomodel.GeoModel(V=V, I=I)
