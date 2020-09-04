# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:18:09 2020

@author: ScottZhuge
"""

import numpy as np

class ControllerHardCode:
    def __init__(self, outputs = []):
        self.outputs = outputs
    
    def seed(self, pv_start = None, output_start = None):
        return
    
    def add_pv(self, pv = None, index = None):
        return
    
    def get_pid_output(self, index = None):
        return self.outputs[index % len(self.outputs)]

class ControllerBangBang:
    def __init__(self, setpoint = None, setpoints = [], heater_off = False, heat_output = 1, cool_output = -1):
        self.setpoint = setpoint
        self.setpoints = setpoints
        self.heater_off = heater_off
        
        self.heat_output = heat_output
        self.cool_output = cool_output
    
    def seed(self, pv_start = None, output_start = None):
        self.errors = []
        self.outputs = [output_start]

        self.add_pv(pv = pv_start, index = 0)
    
    def add_pv(self, pv = None, index = None):
        assert index == len(self.errors)

        err = (self.setpoint or self.setpoints[index]) - pv
        self.errors.append(err)

        self.calculate_next_pid_output()

    def calculate_next_pid_output(self):
        assert len(self.outputs) == len(self.errors)

        next_output = np.sign(self.errors[-1])
        
        if np.sign(self.errors[-1]) > 0:
            if self.heater_off:
                next_output = 0
            else:
                next_output = self.heat_output
        elif np.sign(self.errors[-1]) < 0:
            next_output = self.cool_output
        else:
            next_output = 0

        self.outputs.append(next_output)
        return next_output

    def get_pid_output(self, index = None):
        assert self.outputs[index] == self.heat_output or self.outputs[index] == self.cool_output or self.outputs[index] == 0

        return self.outputs[index]

class Controller:
    def __init__(self, setpoints = [], kp = None, ti = None, td = None, ki = None, kd = None, t_step = None, anti_windup = True):
        self.setpoints = setpoints
        self.kp = kp
        
        if ki != None:
            self.ki = ki
        else:
            assert ti != None
            self.ki = kp / ti if ti > 0 else 0
            
        if kd != None:
            self.kd = kd
        else:
            assert td != None
            self.kd = kp * td
        self.t_step = t_step
        
        self.anti_windup = anti_windup
    
    def seed(self, pv_start = None, output_start = None):
        self.errors = []
        self.accum_errors = []

        self.outputs = [output_start]
        self.add_pv(pv = pv_start, index = 0)
    
    def add_pv(self, pv = None, index = None):
        assert index == len(self.errors)
        assert len(self.outputs) == len(self.errors) + 1
        assert len(self.outputs) == len(self.accum_errors) + 1

        err = self.setpoints[index] - pv
        
        if index == 0:
            accum_err = err
        else:
            accum_err = self.accum_errors[-1] + err

        # integral anti-windup
        if self.anti_windup and (len(self.outputs) > 0) and ((self.outputs[-1] >= 1 and err > 0) or (self.outputs[-1] <= -1 and err < 0)):
            accum_err = accum_err - err
        
        self.errors.append(err)
        self.accum_errors.append(accum_err)
        
        self.calculate_next_pid_output()
        
        assert len(self.outputs) == len(self.errors) + 1
        assert len(self.outputs) == len(self.accum_errors) + 1
    
    # side effect: new output is appended to self.outputs
    def calculate_next_pid_output(self):
        index = len(self.outputs)

        assert index == len(self.errors)
        assert index == len(self.accum_errors)
        
        error_index = index - 1

        err = self.errors[error_index]
        sum_err = self.accum_errors[error_index]
        
        prev_error_index = error_index - 1
        if prev_error_index < 0:
            d_err = 0
        else:
            d_err = (self.errors[error_index] - self.errors[prev_error_index]) / (error_index - prev_error_index)
            assert d_err == err - self.errors[error_index - 1]         
        
        p = self.kp * err
        i = self.ki * sum_err * self.t_step
        d = self.kd * d_err / self.t_step

        next_output = p + i + d

        self.outputs.append(next_output)
        return next_output
    
    def get_pid_output(self, index = None):
        return max(-1.0, min(1.0, self.outputs[index]))

class ControllerCalcT(Controller):
    def __init__(self, setpoints = [], kp = None, ti = None, td = None, heat_calc_t = None, cool_calc_t = None):
        Controller.__init__(self,
                            setpoints = setpoints,
                            kp = kp, ti = ti, td = td,
        )
        
        self.heat_calc_t = heat_calc_t
        self.cool_calc_t = cool_calc_t
        self.output_queue = []
    
    # example: output = 2.5, calc_t = 5
    # return: [1, 1, 0.5, 0, 0]
    def create_output_queue(self, output = None, calc_t = None):
        output_queue = []

        sign = np.sign(output)
        output = output * calc_t
        while (abs(output) > 1):
            output_queue.append(sign * 1.0)
            output = output - sign * 1.0
        
        output_queue.append(output)

        while (len(output_queue) < calc_t):
            output_queue.append(0)

        return output_queue
    
    def get_pid_output(self, index = None):
        if (len(self.output_queue) > 0):
            return self.output_queue.pop(0)
        
        # Empty output_queue
        output = Controller.get_pid_output(self, index = index)
        calc_t = self.heat_calc_t if output >= 0 else self.cool_calc_t
        
        self.output_queue = self.create_output_queue(output = output, calc_t = calc_t)

        return self.get_pid_output(index = index)

class ControllerCalcT2(Controller):    
    def __init__(self, setpoints = [], kp = None, ti = None, td = None, heat_calc_t = None, cool_calc_t = None):
        Controller.__init__(self,
                            setpoints = setpoints,
                            kp = kp, ti = ti, td = td,
        )
        
        self.heat_calc_t = heat_calc_t
        self.cool_calc_t = cool_calc_t
        
        self.on = True
        self.mode = 0
        self.count = 0

    def get_pid_output(self, index = None):        
        output = Controller.get_pid_output(self, index = index)

        if self.count == 0:
            self.on = True
            self.mode = np.sign(output)

        if output == 0 or np.sign(output) != np.sign(self.mode):
            self.on = False

        calc_t = self.heat_calc_t if self.mode >= 0 else self.cool_calc_t
        
        ret_output = None
        if self.on:
            ret_output = min(1, max(0, abs(output) * calc_t - self.count)) * np.sign(self.mode)
        else:
            ret_output = 0
        
        if ret_output == 0:
            self.on = False

        self.count = (self.count + 1) % calc_t
        return ret_output
