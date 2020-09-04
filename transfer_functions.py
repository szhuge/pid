# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 20:54:29 2020

@author: ScottZhuge
"""

import numpy as np

def fopdt(time = None, gain = None, tau = None, delay = None):
    delayed_t = time - delay

    if delayed_t < 0:
        return 0

    return gain * (1 - np.exp(-(delayed_t) / tau)) if tau > 0 else gain

def sopdt(time = None, gain = None, tau1 = None, tau2 = None, delay = None):
    delayed_t = time - delay

    if delayed_t < 0:
        return 0

    if tau1 == tau2:
        if tau1 == 0:
            return gain
        return gain * (1 - np.exp(-(delayed_t) / tau1) - delayed_t * np.exp(-(delayed_t) / tau1) / tau1)
    
    ret = 1
    if tau1 != 0:
        ret += -1 * tau1 * np.exp(-(delayed_t) / tau1) / (tau1 - tau2)
    if tau2 != 0:
        ret += tau2 * np.exp(-(delayed_t) / tau2) / (tau1 - tau2)
    
    return gain * ret