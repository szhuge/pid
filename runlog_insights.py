# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:20:07 2020

@author: ScottZhuge
"""
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
import numpy as np

from runlog_parser import get_fields_from_file
from controllers import *
from chambers import *
from transfer_functions import fopdt, sopdt

linewidth = 3

def truncate_list(temperatures = [], temp_range = [None, None]):
    min_temp, max_temp = temp_range
    
    all_index_ranges = []
    min_index = 0
    max_index = 0
    
    for i in range(len(temperatures)):
        max_index = i
        
        if temperatures[i] > max_temp or temperatures[i] < min_temp:
            all_index_ranges.append((min_index, max_index))
            min_index = i + 1
            max_index = i + 1
    else:
        all_index_ranges.append((min_index, len(temperatures)))

    best_min_index, best_max_index = max(all_index_ranges, key = lambda index_range : index_range[1] - index_range[0])
    
    return temperatures[best_min_index : best_max_index]

def get_min_max_indices_for_temperature(times = [], temperatures = [], initial_index = None, time_radius = 0):
    temperatures = [round(t, 1) for t in temperatures]
    i = initial_index

    current_temp = temperatures[i]
    i_min = i
    while i_min >= 0 and temperatures[i_min] == current_temp:
        i_min -= 1
    else:
        i_min += 1

    current_time = times[i_min]
    while i_min >= 0 and times[i_min] >= current_time - time_radius:
        i_min -= 1
    else:
        i_min += 1

    i_max = i
    while i_max < len(times) and temperatures[i_max] == current_temp:
        i_max += 1
    else:
        i_max -= 1
    
    current_time = times[i_max]
    while i_max < len(times) and times[i_max] <= current_time + time_radius:
        i_max += 1
    else:
        i_max -= 1
    
    return i_min, i_max

def gain_vs_temperature(fname = None, t_step = None, temperatures = [], temp_range = [None, None], time_radius = 5):
    if fname != None:
        assert t_step != None
        temperatures, = get_fields_from_file(fname = fname, fields = ['TemperaturePv'], relays = [])

    temperatures = truncate_list(temperatures = temperatures, temp_range = temp_range)
    times = [i * t_step for i in range(len(temperatures))]

    gains = {}
    
    i = 0
    while i < len(times):
        i_min, i_max = get_min_max_indices_for_temperature(times = times, temperatures = temperatures, initial_index = i, time_radius = time_radius)
        (m, _), _ = curve_fit(
            (lambda x, m, b : m * x + b),
            times[i_min:i_max + 1],
            temperatures[i_min:i_max + 1],
        )
        
        temperature_key = round(temperatures[i], 1)
        gains[temperature_key] = m
        
        _, i_next = get_min_max_indices_for_temperature(times = times, temperatures = temperatures, initial_index = i, time_radius = 0)
        i = i_next + 1

    return gains

def plot_file(fname = None, temp_range = [None, None], t_step = None, label = ''):
    temperatures, = get_fields_from_file(fname = fname, fields = ['TemperaturePv'], relays = [])
    temperatures = truncate_list(temperatures = temperatures, temp_range = temp_range)
    times = [i * t_step for i in range(len(temperatures))]
    
    plt.plot(times, temperatures, '-', linewidth = linewidth, label = label)

def gain_dict_subtract(gain_dict1, gain_dict2):
    diff_dict = {}
    for key in gain_dict1.keys():
        if key in gain_dict2:
            diff_dict[key] = gain_dict1[key] - gain_dict2[key]
    return diff_dict

def gain_dict_lambda(my_lambda, gain_dicts):
    ret_dict = {}
    for key in gain_dicts[0]:
        values = []
        for gain_dict in gain_dicts:
            if key in gain_dict:
                values.append(gain_dict[key])
            else:
                values.append(None)
        if None in values:
            continue
        else:
            ret_dict[key] = my_lambda(*values)
    return ret_dict

def plot_fit_gain_dict(gain_dict, fit_lambda = None, p0 = [1, 1, 1, 1], plot_dict = True, plot_fit = True, label = ''):
    gain_keys = list(gain_dict.keys())
    gain_values = list(gain_dict.values())
    
    if plot_dict:
        plt.plot(list(gain_dict.keys()), list(gain_dict.values()), '-', label = label)
    
    if plot_fit:
        optimal_params, _ = curve_fit(
            fit_lambda,
            gain_keys,
            gain_values,
            p0 = p0,
        )
        
        print(optimal_params)

    if plot_fit:
        plt.plot(gain_keys, [fit_lambda(x, *optimal_params) for x in gain_keys], '-', label = label)

def optimize_fit(gains_and_rates = [], fit_lambda = None, initial_params = None):
    def error_parameters(params):
        print(params)
        residuals = []
        for gain_dict, rate in gains_and_rates:
            for temperature in gain_dict:
                residuals.append(abs(gain_dict[temperature] - fit_lambda(temperature, rate, *params)))
        return np.array(residuals)

    res_1 = least_squares(error_parameters, np.array(initial_params))
    
    return res_1.x

def get_gains_and_rates(fnames_and_rates = [], temp_range = [], time_radius = 5):
    print('gains_and_rates start')
    
    gains_and_rates = []
    for (fname, rate) in fnames_and_rates:
        print(fname)
        gain_dict = gain_vs_temperature(
            fname = fname,
            temp_range = temp_range,
            t_step = 0.5,
            time_radius = time_radius,
        )
        gains_and_rates.append((gain_dict, rate))
    
    print('gains_and_rates done')
    return gains_and_rates

def optimize_and_plot_fnames_and_rates(fnames_and_rates = [], temp_range = []):   
    gains_and_rates = get_gains_and_rates(fnames_and_rates = fnames_and_rates, temp_range = temp_range)

    plt.clf()
    
    plt.subplot(2, 1, 1)
    for (fname, rate) in fnames_and_rates:
        plot_file(
            fname = fname,
            temp_range = temp_range,
            t_step = 0.5,
        )
    
    plt.subplot(2, 1, 2)
    
    rate_lambda = lambda rate, c : (np.exp(1 / c) * (1 - np.exp(-rate / c)) / (np.exp(1 / c) - 1))
    rate_lambda = lambda rate, c : 1 - np.power(1 - rate, c)
    fit_lambda = lambda x, rate, a, b, c: (b - x) * a * rate_lambda(rate, c) + np.exp(-(x + 1.01371152e+02) / 1.02374864e+01) - 3.00303865e-04 * x + 1.28737349e-02
    fit_lambda = lambda x, rate, a, b, c, d, e, f: (b - x) * a * rate_lambda(rate, c) + np.exp((d - x) / e) + f
    fit_lambda = lambda x, rate, a, b, c, d, e: ((b - x) / a + np.exp((d - x) / e)) * rate_lambda(rate, c)
    
    optimal_params = optimize_fit(
        gains_and_rates = gains_and_rates,
        fit_lambda = fit_lambda,
        initial_params = [100, -200, 0.17837194, -100, 100]
    )
    
    for (gain_dict, rate) in gains_and_rates:
        temperatures = list(gain_dict.keys())
        gains = list(gain_dict.values())
        plt.plot(temperatures, gains, '-')
        plt.plot(temperatures, [fit_lambda(x, rate, *optimal_params) for x in temperatures], '-')

def get_temperatures_and_rates(fnames_and_rates = [], temp_range = []):
    temperatures_and_rates = []
    for (fname, rate) in fnames_and_rates:
        temperatures, = get_fields_from_file(fname = fname, fields = ['TemperaturePv'], relays = [])
        temperatures = truncate_list(temperatures = temperatures, temp_range = temp_range)
        temperatures_and_rates.append((temperatures, rate))
    return temperatures_and_rates

def simulate_temperatures(temperatures = [], rate = None, fit_lambda = None, params = None, t_step = None):
    sim_temperatures = []
    for i in range(len(temperatures)):
        if i == 0:
            sim_temperatures.append(temperatures[i])
        else:
            sim_temperatures.append(sim_temperatures[-1] + t_step * fit_lambda(sim_temperatures[-1], rate, *params))
    return sim_temperatures

def optimize_and_plot_temperatures_and_rates(
        temperatures_and_rates = [],
        t_step = 0.5,
        fit_lambda = None,
        params_and_bounds = None,
        optimize = True,
    ):
    def error_parameters(params):
        print(params)
        residuals = []
        for temperatures, rate in temperatures_and_rates:
            sim_temperatures = simulate_temperatures(temperatures = temperatures, rate = rate, fit_lambda = fit_lambda, params = params, t_step = t_step)
            for i in range(len(temperatures)):
                residuals.append(abs(sim_temperatures[i] - temperatures[i]))
        return np.array(residuals)

    if optimize:
        initial_params, bounds_min, bounds_max = zip(*params_and_bounds)
        
        res = least_squares(error_parameters, np.array(initial_params), bounds = (bounds_min, bounds_max))
        optimal_params = res.x
    else:
        optimal_params = None

    plt.clf()

    for (temperatures, rate) in temperatures_and_rates:
        plt.plot([t_step * i for i in range(len(temperatures))], temperatures, '-')
        
        if optimize:
            sim_temperatures = simulate_temperatures(temperatures = temperatures, rate = rate, fit_lambda = fit_lambda, params = optimal_params, t_step = t_step)
            plt.plot([t_step * i for i in range(len(sim_temperatures))], sim_temperatures, '-', linewidth = 3)

    return optimal_params

def main_gains_and_rates():
    fnames_and_rates = [
        ('SetTemp_20200821112538_cool', 1),
        # ('SetTemp_20200821133114', 0.5),
        # ('SetTemp_20200821144154', 0.25),
        # ('SetTemp_20200821153125', 0.125),
        # ('SetTemp_20200821170300', 0.0625),
        # ('SetTemp_20200824110236', 0.0625),
        # ('SetTemp_20200824135143', 0.03125),
        # ('SetTemp_20200821112538_room', 0),
    ]
    temp_range = [-70, 22]
    
    # fnames_and_rates = [
    #     # All temps 100% heating
    #     # ('SetTemp_20200716114040.log', 1)
    #     # (8/26)
    #     # ('SetTemp_20200826104430.log', 1),
    #     # ('SetTemp_20200826111854.log', 0.5),
    #     # ('SetTemp_20200826121515.log', 0.25),
    #     # ('SetTemp_20200826135551.log', 0.125),
    #     # (8/25)
    #     # ('SetTemp_20200825151219.log', 0.5),
    #     # ('SetTemp_20200825123334.log', 0.25),
    #     # ('SetTemp_20200825104224.log', 0.0),
    # ]
    # temp_range = [10, 140]
    
    gains_and_rates = get_gains_and_rates(fnames_and_rates = fnames_and_rates, temp_range = temp_range, time_radius = 60)
    
    plt.clf()
    for (gain_dict, rate) in gains_and_rates:
        plt.plot(list(gain_dict.keys()), list(gain_dict.values()), '-', label = str(rate))

def main_optimize_and_plot_temperatures():
    fnames_and_rates = [
        # ('SetTemp_20200821112538_cool', 1),
        # ('SetTemp_20200821133114', 0.5),
        # ('SetTemp_20200821144154', 0.25),
        ('SetTemp_20200821153125', 0.125),
        # ('SetTemp_20200821170300', 0.0625),
        ('SetTemp_20200824110236', 0.0625),
        ('SetTemp_20200824135143', 0.03125),
        # ('SetTemp_20200821112538_room', 0),
    ]
    temp_range = [-70, 22]
    
    # fnames_and_rates = [
    #     # # 100% heating (8/26)
    #     ('SetTemp_20200826104430.log', 1),
    #     # 50% heating (8/26)
    #     ('SetTemp_20200826111854.log', 0.5),
    #     # 25% heating (8/26)
    #     ('SetTemp_20200826121515.log', 0.25),
    #     # 12.5% heating (8/26)
    #     ('SetTemp_20200826135551.log', 0.125),
    #     # 50% heating (8/25)
    #     ('SetTemp_20200825151219.log', 0.5),
    #     # 25% heating (8/25)
    #     ('SetTemp_20200825123334.log', 0.25),
    #     # 0% heating (8/25)
    #     ('SetTemp_20200825104224.log', 0.0),
    # ]
    # temp_range = [30, 140]
    
    # optimize_and_plot_fnames_and_rates(
    #     fnames_and_rates = fnames_and_rates,
    #     temp_range = temp_range,
    # )
    
    # gains_and_rates = get_gains_and_rates(fnames_and_rates = fnames_and_rates, temp_range = temp_range, time_radius = 60)
    
    # plt.clf()
    # for (gain_dict, rate) in gains_and_rates:
    #     prop_dict = gain_dict_lambda(
    #         lambda room, cool, mine: (mine - room) / (cool - room),
    #         # lambda room, cool, mine: (mine) / (cool),
    #         [gains_and_rates[-1][0], gains_and_rates[0][0], gain_dict]
    #     )
    #     plt.plot(list(prop_dict.keys()), list(prop_dict.values()), '-', label = str(rate))
    
    # Direct time domain modeling
    # Singular (heating / cooling)
    # 100% cooling with (-202.65956408, 46.00742582): [-166.41447389 1093.94964967]
    fit_lambda = lambda x, rate, a1, b1: (a1 - x) / b1 + 0 * np.exp((-202.65956408 - x) / 46.00742582)
    initial_params = [-80, 100]
    bounds = (
        [-np.inf, 0],
        [np.inf, np.inf],
    )
    
    # Singular (room)
    # Cooling: -202.65956408, 46.00742582
    fit_lambda = lambda x, rate, a1, b1: np.exp((a1 - x) / b1)
    initial_params = [0, 100]
    bounds = (
        [-np.inf, 0],
        [np.inf, np.inf],
    )
    
    # Best structure so far (heating)
    rate_lambda = lambda rate, c : 1 - np.power(1 - rate, c)
    fit_lambda = lambda x, rate, a, b, c, d: (a - x + (1 - rate) * c) / (b) * rate_lambda(rate, d) + np.exp((-237.93686291 - x) / 70.92528626)
    params_and_bounds = [
        (600, 0, np.inf),
        (6000, 0, np.inf),
        (-500, -np.inf, np.inf),
        (5, -np.inf, np.inf),
    ]
    
    # Best structure so far (cooling)
    fit_lambda = lambda x, rate, a, b, c: (a - x) / b * (rate + c)
    params_and_bounds = [
        (-77, -np.inf, 0),
        (100, 0, np.inf),
        (0, -1, 1),
    ]
    
    optimal_params = optimize_and_plot_temperatures_and_rates(
        fit_lambda = fit_lambda,
        params_and_bounds = params_and_bounds,
        temperatures_and_rates = get_temperatures_and_rates(fnames_and_rates = fnames_and_rates, temp_range = temp_range),
        t_step = 0.5,
        optimize = True,
    )

def get_oscillation_indices(temperatures = None, setpoint = None, oscillation_number = None):
    up_crosses = []
    down_crosses = []

    for i in range(1, len(temperatures)):
        if temperatures[i - 1] < setpoint and temperatures[i] > setpoint:
            up_crosses.append(i)
        if temperatures[i - 1] > setpoint and temperatures[i] < setpoint:
            down_crosses.append(i)
    
    # full cycle starting from up_cross
    ret_arr = up_crosses
    if oscillation_number + 1 < len(ret_arr):
        return ret_arr[oscillation_number], ret_arr[oscillation_number + 1]
    
    # half cycle starting from up_cross
    # if oscillation_number < len(up_crosses):
    #     i = 0
    #     while i < len(down_crosses):
    #         if down_crosses[i] < up_crosses[oscillation_number]:
    #             i += 1
    #         else:
    #             return up_crosses[oscillation_number], down_crosses[i]

    return 0, len(temperatures)

def simulate_chamber_oscillations(
        test_length = None,
        setpoint = None,
        pv_start = None,
        params = None,
        get_chamber = None,
        oscillation_number = None,
    ):
    chamber = get_chamber(params)
    controller = ControllerBangBang(setpoint = setpoint)

    controller.seed(pv_start = pv_start, output_start = 0)
    chamber.seed(pv_start = pv_start, output_start = 0)

    up_crosses = []
    for i in range(1, test_length):
        chamber.add_output(
            output = controller.get_pid_output(index = i),
            index = i
        )
        
        pv = chamber.get_pv(index = i)
        controller.add_pv(
            pv = pv,
            index = i
        )
        
        if chamber.pvs[i-1] < setpoint and chamber.pvs[i] > setpoint:
            up_crosses.append(i)
        if oscillation_number != None and len(up_crosses) > oscillation_number:
            break

    return chamber.pvs

def optimize_and_plot_oscillations(
        temperatures = [],
        t_step = 0.5,
        get_chamber = None,
        params_and_bounds = None,
        optimize = True,
        setpoint = None,
        measured_oscillation_number = 4,
        simulated_oscillation_number = 4,
        pv_start = None,
    ):
    start_index, end_index = get_oscillation_indices(
        temperatures = temperatures,
        setpoint = setpoint,
        oscillation_number = measured_oscillation_number
    )
    temperatures_trunc = temperatures[start_index : end_index]

    def error_parameters(params):
        print(params)
        sim_temperatures = simulate_chamber_oscillations(
            test_length = int(end_index * 3 / 2),
            pv_start = pv_start,
            setpoint = setpoint,
            params = params,
            get_chamber = get_chamber,
            oscillation_number = measured_oscillation_number
        )
        
        sim_start_index, sim_end_index = get_oscillation_indices(
            temperatures = sim_temperatures,
            setpoint = setpoint,
            oscillation_number = measured_oscillation_number
        )
        
        residuals = []
        # for i in range(len(temperatures_trunc)):
        #     # if temperatures_trunc[i] != min(temperatures_trunc) and temperatures_trunc[i] != max(temperatures_trunc):
        #     #     continue
        #     if sim_start_index + i >= sim_end_index:
        #         sim_temperature = setpoint
        #     else:
        #         sim_temperature = sim_temperatures[sim_start_index + i]
        #     residuals.append(abs(temperatures_trunc[i] - sim_temperature))
        residuals.append(abs((sim_end_index - sim_start_index) - (end_index - start_index)))
        print(residuals)
        return np.array(residuals)

    initial_params, bounds_min, bounds_max = zip(*params_and_bounds)
    if optimize:
        res = least_squares(
            error_parameters, np.array(initial_params),
            bounds = (bounds_min, bounds_max),
            max_nfev = 5,
            method = 'dogbox',
            diff_step = [0.1] * len(initial_params),
        )
        optimal_params = res.x
    else:
        optimal_params = initial_params
    
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot([t_step * i for i in range(len(temperatures))], temperatures, '-', linewidth = 1)
    plt.plot([t_step * i for i in range(start_index, end_index)], temperatures_trunc, linewidth = 3)
    
    plt.subplot(2, 1, 2)
    sim_temperatures = simulate_chamber_oscillations(
        test_length = int(end_index * 3 / 2),
        pv_start = pv_start,
        setpoint = setpoint,
        params = optimal_params,
        get_chamber = get_chamber
    )
    
    sim_start_index, _ = get_oscillation_indices(
        temperatures = sim_temperatures,
        setpoint = setpoint,
        oscillation_number = measured_oscillation_number
    )
    plt.plot([t_step * i for i in range(len(sim_temperatures))], sim_temperatures, '-')
    plt.plot([t_step * i for i in range(sim_start_index, sim_start_index + (end_index - start_index))], temperatures_trunc, '-')

def get_outputs(heat_outputs = [], cool_outputs = []):
    assert len(heat_outputs) == len(cool_outputs)
    outputs = []
    for i in range(len(heat_outputs)):
        if heat_outputs[i] > 0:
            outputs.append(heat_outputs[i])
        elif cool_outputs[i] > 0:
            outputs.append(-1 * cool_outputs[i])
        else:
            outputs.append(0)
    return outputs

def simulate_chamber_with_measured_outputs(
        pv_start = None,
        output_start = None,
        outputs = None,
        params = None,
        get_chamber = None,
):
    len_output_start = len(output_start)
    chamber = get_chamber(params)
    chamber.seed(pv_start = pv_start, output_start = output_start)
    
    for i in range(len(outputs)):
        chamber.add_output(
            output = outputs[i],
            index = len_output_start + i,
        )

    return chamber.pvs

def optimize_and_plot_oscillations_with_measured_outputs(
        temperatures = [],
        outputs = [],
        t_step = None,
        get_chamber = None,
        params_and_bounds = None,
        optimize = False,
        oscillation_number = None,
):
    start_index, end_index = get_oscillation_indices(
        temperatures = outputs,
        setpoint = 0.1,
        oscillation_number = oscillation_number
    )
    
    def error_parameters(params):
        print(params)
        end_index = len(temperatures)
        sim_temperatures = simulate_chamber_with_measured_outputs(
            pv_start = temperatures[0:start_index],
            output_start = outputs[0:start_index],
            outputs = outputs[start_index:end_index],
            params = params,
            get_chamber = get_chamber,
        )
        measured_temperatures = temperatures[0:end_index]
        
        assert len(measured_temperatures) == end_index
        assert len(measured_temperatures) == len(sim_temperatures)
        
        residuals = [abs(measured_temperatures[i] - sim_temperatures[i]) for i in range(len(measured_temperatures))]
        return np.array(residuals)
    
    initial_params, bounds_min, bounds_max = zip(*params_and_bounds)
    if optimize:
        res = least_squares(
            error_parameters, np.array(initial_params),
            bounds = (bounds_min, bounds_max),
            # max_nfev = 5,
            # method = 'dogbox',
            # diff_step = [0.1] * len(initial_params),
        )
        optimal_params = res.x
    else:
        optimal_params = initial_params
    
    plt.clf()
    plt.subplot(2, 1, 1)
    # plt.plot([t_step * i for i in range(len(temperatures))], temperatures, '-', linewidth = 1)
    # plt.plot([t_step * i for i in range(start_index, end_index)], temperatures[start_index:end_index], linewidth = 3)
    plt.plot([t_step * i for i in range(len(outputs))], outputs, '-')
    
    plt.subplot(2, 1, 2)
    sim_temperatures = simulate_chamber_with_measured_outputs(
        pv_start = temperatures[0:start_index],
        output_start = outputs[0:start_index],
        outputs = outputs[start_index:-1],
        params = optimal_params,
        get_chamber = get_chamber,
    )
    plt.plot([t_step * i for i in range(len(temperatures))], temperatures, '-', linewidth = 1)
    plt.plot([t_step * i for i in range(len(sim_temperatures))], sim_temperatures, linewidth = 3)

fname, setpoint = ('SetTemp_20200828151053', -40)
pv_start = -42
temperatures, heat_outputs, cool_outputs = get_fields_from_file(fname = fname, fields = ['TemperaturePv'], relays = [1 - 1, 12 - 1])

# optimize_and_plot_oscillations(
#     temperatures = temperatures,
#     t_step = 0.5,
#     get_chamber = lambda params : Chamber(
#         pv_start = None, output_start = 0, x_back = 50, t_step = 0.5,
#         heat_transfer_func = (lambda time, temp : sopdt(time = time, gain = 0.174, tau1 = params[0], tau2 = params[1], delay = params[2])),
#         cool_transfer_func = (lambda time, temp : sopdt(time = time, gain = -0.096, tau1 = params[3], tau2 = params[4], delay = params[5])),
#     ),
#     params_and_bounds = [
#         # (0.174, 0, 1),
#         (18, 0, np.inf),
#         (12, 0, np.inf),
#         (5, 0, np.inf),
#         # (-0.096, -1, 0),
#         (7, 0, np.inf),
#         (1, 0, np.inf),
#         (2, 0, np.inf),
#     ],
#     # [ 0.17456026  1.12846769 23.86365589 -0.09591477  5.03194779 20.79518335]
#     # [15 10 5 15 10 5]
#     # [6.43995421 4.48157045 2.29420257 5.63158277 5.71600249 8.50939316]  
#     # [12.4595461  14.48571446  3.66502118  6.49868623  1.23800345  6.26149932]
#     optimize = False,
#     setpoint = setpoint,
#     pv_start = pv_start,
#     measured_oscillation_number = 3,
#     simulated_oscillation_number = 3,
# )

optimize_and_plot_oscillations_with_measured_outputs(
    temperatures = temperatures,
    outputs = get_outputs(heat_outputs = heat_outputs, cool_outputs = cool_outputs),
    t_step = 0.5,
    get_chamber = lambda params : Chamber(
        pv_start = None, output_start = 0, x_back = 50, t_step = 0.5,
        heat_transfer_func = (lambda time, temp : sopdt(time = time, gain = 0.174, tau1 = params[0], tau2 = 0, delay = params[1])),
        cool_transfer_func = (lambda time, temp : sopdt(time = time, gain = -0.096, tau1 = params[2], tau2 = 0, delay = params[3])),
    ),
    params_and_bounds = [
        # (0.174, 0, 1),
        (1.28064063, 0, np.inf),
        # (12, 1, np.inf),
        (24, 0, np.inf),
        # (-0.096, -1, 0),
        (1, 0, np.inf),
        # (1, 1, np.inf),
        (24, 0, np.inf),
    ],
    # SOPDT
    # [4.62052136e-02 1.20849093e+00 2.40002542e+01 2.14794724e-02 1.00000000e+00 2.37528464e+01]
    # FOPDT
    # [ 0.96974957 24.13102366  0.67542312 24.00093017]
    optimize = True,
    oscillation_number = 4,
)

# plt.clf()
# plt.plot(heat_outputs, '-')
# plt.plot([1 * i for i in cool_outputs], '-')
# main_gains_and_rates()