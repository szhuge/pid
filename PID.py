# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:26:23 2020

@author: ScottZhuge
"""
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
import numpy as np

from runlog_parser import get_fields_from_file
from controllers import *
from chambers import *

def simulate_controller_and_chamber(
        test_length = None,
        pv_start = None,
        output_start = None,
        controller = None,
        chamber = None,
        plot = False
    ):
    controller.seed(pv_start = pv_start, output_start = output_start)
    chamber.seed(pv_start = pv_start, output_start = output_start)
    
    for index in range(1, test_length):
        chamber.add_output(
            output = controller.get_pid_output(index = index),
            index = index
        )
        
        controller.add_pv(
            pv = chamber.get_pv(index = index),
            index = index
        )

    pvars = chamber.pvs
    outputs = chamber.outputs
    
    if plot:
        plt.clf()    
        plt.subplot(2, 1, 1)
        plt.plot(outputs, '-')    
        plt.subplot(2, 1, 2)
        plt.plot(pvars, '-')
    
    return controller, chamber

sti_chamber = ChamberFull(
    x_back = 50, t_step = 1,
    heat_gain_temp = 418.0161664606, heat_gain_tau = 3171.325288956,
    heat_tau = 10, heat_delay = 10,
    cool_gain_temp = -135.2900683091, cool_gain_tau = 868.2204691822,
    cool_tau = 10, cool_delay = 10,
    neutral_gain_temp = -202.6128099492, neutral_gain_tau = 46.1964481429,
    neutral_tau = 10, neutral_delay = 10,
)

def get_label(kp, ti, td):
    kp_string = 'Kp = ' + str(kp)
    ti_string = '' if ti == 0 else ', Ti = ' + str(ti)
    td_string = '' if td == 0 else ', Td = ' + str(td)
    
    return 'PV (' + kp_string + ti_string + td_string + ')'

def educate_screenshot_kp_ti(
        kps = [], tis = [], tds = [], pv_starts = [3], anti_windups = [True],
        ncols = None, nrows = None,
        ylim = None,
        display_outputs = False
    ):
    
    test_length = 1200
    setpoints = [0] * test_length
    output_start = 0
    
    plt.clf()
    
    num_plots = len(pv_starts) * len(kps) * len(tis) * len(tds) * len(anti_windups)
    
    if ncols == None:
        ncols = np.ceil(np.sqrt(num_plots))
    if nrows == None:
        nrows = int(np.ceil(num_plots / ncols))
    if display_outputs:
        nrows = nrows * 2

    index = 1
    for pv_start in pv_starts:
        for kp in kps:
            for ti in tis:
                for td in tds:            
                    for anti_windup in anti_windups:
                        plt.subplot(nrows, ncols, index)
                        
                        controller, chamber = simulate_controller_and_chamber(
                            test_length = test_length,
                            pv_start = pv_start,
                            output_start = output_start,
                            controller = Controller(
                                setpoints = setpoints,
                                kp = kp, ti = ti, td = td,
                                t_step = t_step,
                                anti_windup = anti_windup
                            ),
                            chamber = sti_chamber,
                            plot = False
                        )
                        
                        pvars = chamber.pvs

                        plt.plot(setpoints, '-', label="Setpoints", linewidth=3)
                        plt.plot(pvars, '-', label = get_label(kp, ti, td), linewidth=3)
                        plt.ylim(ylim)
                        plt.legend(loc = "upper right", fontsize = 20)
                        plt.tight_layout()
                        
                        if display_outputs:
                            outputs = controller.accum_errors

                            plt.subplot(nrows, ncols, index + ncols)
                            plt.plot(outputs, '-', label="Accumulated Error", linewidth=3)
                            plt.ylim([-1100, 150])
                            plt.legend(loc = "upper right", fontsize = 20)
                            plt.tight_layout()
                        
                        index += 1

    plt.tight_layout()

def relay_tuning(heat_output = 1, cool_output = -1):
    test_length = 1200
    setpoints = [0] * test_length
    linewidth = 3
    
    controller, chamber = simulate_controller_and_chamber(
        test_length = test_length,
        pv_start = 0,
        output_start = 0,
        controller = ControllerBangBang(
            setpoints = setpoints,
            heat_output = heat_output,
            cool_output = cool_output,
        ),
        chamber = sti_chamber,
    )
    
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(setpoints, '-', label="Setpoints", linewidth = linewidth)
    plt.plot(chamber.pvs, '-', label="Process Variable", linewidth = linewidth)
    
    plt.legend(loc = "upper right", fontsize = 20)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(chamber.outputs, '-', label="Outputs", linewidth = linewidth)
    
    plt.legend(loc = "upper right", fontsize = 20)
    plt.tight_layout()
    
    a = max(chamber.pvs) - min(chamber.pvs)
    d = heat_output - cool_output
    
    print('Amplitude: ' + str(a))
    print('Ultimate Gain: ' + str(4 * d / (np.pi * a)))
    
    crossing_indices = []
    for i in range(int(test_length / 2), test_length):
        if chamber.outputs[i - 1] < 0 and chamber.outputs[i] > 0:
            crossing_indices.append(i)
            
            if len(crossing_indices) == 2:
                print('Ultimate period: ' + str(crossing_indices[1] - crossing_indices[0]))
                break

def overlay_pids(
        pids = [], pv_start = None, output_start = None, test_length = None,
    ):

    setpoints = [0] * test_length
    linewidth = 3
    
    plt.clf()
    plt.plot(setpoints, '-', label="Setpoints", linewidth = linewidth)

    for pid in pids:
        kp, ti, td, label = pid
        
        controller, chamber = simulate_controller_and_chamber(
            test_length = test_length,
            pv_start = pv_start,
            output_start = output_start,
            controller = Controller(
                setpoints = setpoints,
                kp = kp, ti = ti, td = td,
                t_step = 1,
            ),
            chamber = sti_chamber,
            plot = False
        )
        
        pvars = chamber.pvs


        plt.plot(pvars, '-', label = label, linewidth = linewidth)

    plt.legend(loc = "upper right", fontsize = 20)
    plt.tight_layout()

# educate_screenshot_kp_ti(
#     kps = [
#         0.39,
#     ],
#     tis = [
#         40,
#     ],
#     tds = [
#         10,
#     ],
#     # ncols = 2,
#     pv_starts = [15],
#     anti_windups = [True],
#     # ylim = [-1, 1],
#     display_outputs = False,
# )

# relay_tuning(
#     heat_output = 0.5,
#     cool_output = -1,
# )

# ku = .65
# tu = 80
# overlay_pids(
#     pids = [
#         (0.6 * ku, tu / 2, tu / 8, 'Classic ZN'),
#         (0.7 * ku, 0.4 * tu, .15 * tu, 'Pessen Integral Rule'),
#         (ku / 3, tu / 2, tu / 3, 'Moderate Overshoot'),
#         (ku / 5, tu / 2, tu / 3, 'No Overshoot'),
#         # (0.2 * ku, tu / 2, tu / 8, 'Classic ZN 0.2'),
#         # (0.6 * ku, tu / 2, tu / 8, 'Classic ZN 0.6'),
#         # (1.0 * ku, tu / 2, tu / 8, 'Classic ZN 1.0'),
#         # (1.15 * ku, tu / 2, tu / 8, 'Classic ZN 1.15'),
#         # (1.6 * ku, tu / 2, tu / 8, 'Classic ZN 1.6'),
#     ],
#     test_length = 1200,
#     pv_start = 10,
#     output_start = 0,
# )