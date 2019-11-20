#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili to handle some processing with Gaussian Processing.
'''
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def gp_interp(xin,yin,sigmain=None,xpred=None,length_scale=None,fit_noise=True,n_restarts_optimizer=1, return_std=False, return_gp=False, **gprargs):
    '''
    Interpolation / Smoothing with Gaussian processing using an RBF kernel.
    This function will use the Gaussian Process Regressor in scikit-learn.

    Args:
        xin, yin (1d array-like) : input signal
        sigmain (optional, 1d array-like) : 1 sigma uncertainty of yin
        xpred (optional, 1d array-like) : x-axis value of the output signal. If not specified, xin will be used.
        length_scale (optional, float) : a length scale of the RBF kernel to be used. The default value is the minimum interval of xin.
        fit_noise (optional, boolean) : If true, the global noise will be fitted as well.
        n_restarts_optimizer (optional, integer) : GP fit will be repeated for this value to avoid local minima.
        return_std (optional, boolean) : If true, the uncertainty of ypred will be output.
        return_gp (optional, boolean) : If true, the instance of the Gaussian Process Regressor will be returned.
    Returns:
        ypred: predicted y-value on xpred.
        sigmapred (optional; if return_std==True): sigma estimators on ypred
        gp (optional; if return_gp==True): the instance of the Gaussian Process Regressor
    '''
    # set uncertainties on the input data
    if sigmain is None:
        alpha = 0
    else:
        alpha = sigmain**2

    # initialize kernel
    if not length_scale:
        length_scale = np.min(np.abs(np.diff(xin)))
    if fit_noise:
        noise_level = np.median(np.abs(np.diff(yin)))
        kernel = ConstantKernel() * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level*np.asarray([0.01,100]))
    else:
        kernel = ConstantKernel() * RBF(length_scale=length_scale)

    # fit with Gaussian Process
    X = xin.reshape([-1,1])
    y = yin.copy()
    gp = gaussian_process.GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        **gprargs
    )
    gp.fit(X, y)

    # output results
    if not xpred:
        xpred = xin.copy()

    if return_gp:
        return gp.predict(xpred.reshape([-1,1]), return_std=return_std), gp
    else:
        return gp.predict(xpred.reshape([-1,1]), return_std=return_std)
