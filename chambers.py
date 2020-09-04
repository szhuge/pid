# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:26:23 2020

@author: ScottZhuge
"""

import numpy as np

from transfer_functions import fopdt, sopdt

# *_transfer_func() takes two parameters: time, temperature
class Chamber:
    def __init__(self,
                 pv_start = None, output_start = 0, x_back = None, t_step = None,
                 heat_transfer_func = None,
                 cool_transfer_func = None,
                 neutral_transfer_func = None,
    ):                
        self.x_back = x_back
        self.t_step = t_step

        self.heat_transfer_func = heat_transfer_func
        self.cool_transfer_func = cool_transfer_func
        self.neutral_transfer_func = neutral_transfer_func
    
    def seed(self, pv_start = None, output_start = None):
        if type(pv_start) == list and type(output_start) == list:
            assert len(pv_start) == len(output_start)
            self.outputs = output_start
            self.pvs = pv_start
        else:
            self.outputs = [output_start]
            self.pvs = [pv_start]
    
    def add_output(self, output = None, index = None):
        assert len(self.pvs) > 0
        assert len(self.pvs) == len(self.outputs)
        assert index == len(self.pvs)
        
        self.outputs.append(output)
        self.calculate_next_pv()
        
        assert len(self.pvs) == len(self.outputs)

    # side effect: pv is appended to self.pvs
    def calculate_next_pv(self):
        i = len(self.pvs)

        assert i == len(self.outputs) - 1

        pv_back = 1
        pv = self.pvs[i - pv_back]
        
        for j in range(0, self.x_back):
            if i - j - 1 < 0:
                break

            x = self.outputs[i - j]
            temperature = self.pvs[i - j - 1]
            
            t_1 = j * self.t_step
            t_0 = (j - pv_back) * self.t_step
            
            if self.heat_transfer_func == None:
                heat_gain = 0
            else:
                heat_gain = self.heat_transfer_func(t_1, temperature) - self.heat_transfer_func(t_0, temperature)
            
            if self.cool_transfer_func == None:
                cool_gain = 0
            else:
                cool_gain = self.cool_transfer_func(t_1, temperature) - self.cool_transfer_func(t_0, temperature)
            
            if self.neutral_transfer_func == None:
                neutral_gain = 0
            else:
                neutral_gain = self.neutral_transfer_func(t_1, temperature) - self.neutral_transfer_func(t_0, temperature)
            
            if x >= 0:
                gain = abs(x) * heat_gain + (1 - abs(x)) * neutral_gain
            else:
                gain = abs(x) * cool_gain + (1 - abs(x)) * neutral_gain
            
            pv += gain
            
            # pv += np.random.normal() * 0.0015 * self.t_step

        self.pvs.append(pv)
        return pv

    def get_pv(self, index = None):
        return self.pvs[index]

# *_gain_func takes one parameter: temperature
class ChamberTauDelay(Chamber):
    def __init__(self,
                 pv_start = None, output_start = 0, x_back = None, t_step = None,
                 heat_gain_func = None, heat_tau = None, heat_delay = None,
                 cool_gain_func = None, cool_tau = None, cool_delay = None,
                 neutral_gain_func = None, neutral_tau = None, neutral_delay = None,
                 ):

        heat_transfer_func = (lambda time, temp : fopdt(time = time, gain = heat_gain_func(temp), tau = heat_tau, delay = heat_delay))
        cool_transfer_func = (lambda time, temp : fopdt(time = time, gain = cool_gain_func(temp), tau = cool_tau, delay = cool_delay))
        neutral_transfer_func = (lambda time, temp : fopdt(time = time, gain = neutral_gain_func(temp), tau = neutral_tau, delay = neutral_delay))

        Chamber.__init__(self,
                         pv_start = pv_start, output_start = output_start, x_back = x_back, t_step = t_step,
                         heat_transfer_func = heat_transfer_func,
                         cool_transfer_func = cool_transfer_func,
                         neutral_transfer_func = neutral_transfer_func,
        )

class ChamberFull(ChamberTauDelay):
    def __init__(self,
                  pv_start = None, output_start = 0, x_back = None, t_step = None,
                  heat_gain_temp = None, heat_gain_tau = None, heat_tau = None, heat_delay = None,
                  cool_gain_temp = None, cool_gain_tau = None, cool_tau = None, cool_delay = None,
                  neutral_gain_temp = None, neutral_gain_tau = None, neutral_tau = None, neutral_delay = None,
                  ):
        
        neutral_gain_func = (lambda temp: np.exp((neutral_gain_temp - temp) / neutral_gain_tau) * t_step)
        heat_gain_func = (lambda temp: (heat_gain_temp - temp) * (1 - np.exp(-t_step / heat_gain_tau)) + neutral_gain_func(temp))
        cool_gain_func = (lambda temp: (cool_gain_temp - temp) * (1 - np.exp(-t_step / cool_gain_tau)) + neutral_gain_func(temp))
        
        ChamberTauDelay.__init__(self,
            pv_start = pv_start, output_start = output_start, x_back = x_back, t_step = t_step,
            heat_gain_func = heat_gain_func,
            heat_tau = heat_tau, heat_delay = heat_delay,
            cool_gain_func = cool_gain_func,
            cool_tau = cool_tau, cool_delay = cool_delay,
            neutral_gain_func = neutral_gain_func,
            neutral_tau = neutral_tau, neutral_delay = neutral_delay
        )
