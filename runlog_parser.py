# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:58:28 2020

@author: ScottZhuge
"""


import csv
import matplotlib.pyplot as plt

def read_csv(fname):
    data = {}

    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        headers = []

        time_index = 0
        for row_unstripped in reader:
            row = [r.strip() for r in row_unstripped]
            
            # StartTime / EndTime
            if ('StartTime' in row[0] or 'EndTime' in row[0]):
                continue
            
            # headers row
            if (not headers) and ('Time' in row[0]):
                headers = row
                continue
        
            # read row body
            row_hash = {}
            for i in range(len(headers)):
                if (headers[i] in ['Time', 'TempHex', 'OEMType']):
                    row_hash[headers[i]] = row[i]
                elif (headers[i] in ['RelayOut']):
                    row_hash[headers[i]] = int(row[i], 16)
                else:
                    row_hash[headers[i]] = float(row[i])

            data[time_index] = row_hash
            time_index += 1
    return data

def from_data(data, field_name):
    ydata = [d[field_name] for d in data.values()]
    return ydata

def get_k_bit(binary_number, k):
    # remove '0b' from beginning
    binary = bin(binary_number)[2:]
    
    if k >= len(binary):
        return 0
    
    return int(binary[len(binary) - k - 1])

def relay_out_data(data, k, shift = 0):
    ydata = [get_k_bit(d['RelayOut'], k) + shift for d in data.values()]
    return ydata

# Time, TemperatureSv, TemperaturePv, HumiditySv, HumidityPv, TempHex, OldTemp,
# Dio, RelayOut, TempHot, TempCool, HumiOut, DehumiOut, OEMType, HeatErrLast,
# HeatErrSum, HeatOutput, CoolErrLast, CoolErrSum, CoolOutput, HumErrLast,
# HumErrSum, HumOutput, DehumErrLast, DehumErrSum, DehumOutput, CoolAvgOutput, DehumAvgOutput, 
    
# 1: Heating, 3: System Fan, 11: C2V1, 12: C2V2, 14: C2 Heat Bypass, 16: Dhum, 17: Dhum compressor valve, 18: DHum compressor heat bypass
# 25: Dehum small valve
def get_fields_from_file(fname = None, fields = ['TemperaturePv', 'TempHot', 'TempCool'], relays = []):
    path = "C:/Users/ScottZhuge/Dropbox (Team CI)/Scott Zhuge Personal/THV Test Runs/Run Logs/"
    
    if fname[-1 * len('.log'):] != '.log':
        fname = fname + '.log'
    
    data = read_csv(path + fname)
    return [from_data(data, field) for field in fields] + [relay_out_data(data, k) for k in relays]

def plot_nested_fields(fname = None, list_of_fields = [], relays = [], clear_plot = True):
    num_plots = len(list_of_fields) + len(relays)
    plot_index = 1

    if clear_plot:
        plt.clf()

    for i in range(0, len(list_of_fields)):
        plt.subplot(num_plots, 1, plot_index)
        plot_index += 1
        
        maybe_list = list_of_fields[i]

        if type(maybe_list) == list:
            fields = get_fields_from_file(fname = fname, fields = maybe_list)
        else:
            fields = get_fields_from_file(fname = fname, fields = [maybe_list])
        
        for field in fields:
            plt.plot(field, '-')
    
    relay_data = get_fields_from_file(fname = fname, fields = [], relays = relays)
    for i in range(len(relay_data)):
        plt.subplot(num_plots, 1, plot_index)
        plot_index += 1
        
        plt.plot(relay_data[i], '-')

    plt.tight_layout()

# fname = 'SetTemp_20200731102705.log' # first -30 run with normal logic file
# fname = 'SetTemp_20200731111737.log' # second -30 run with normal logic file
# fname = 'SetTemp_20200731120010.log' # -30 run with (0.01, 25, 28)
# fname = 'SetTemp_20200731123904.log' # -30 run, playing with PID parameters
# fname = 'SetTemp_20200731133843.log' # attempt tuning with 1 valve only no heat
# fname = 'SetTemp_20200803165336.log' # -50 C hold attempt
# fname = 'SetTemp_20200814124803' # -50 to -70 transition, not working
# fname = 'SetTemp_20200814130546' # -70 hold
# fname1 = 'SetTemp_20200814115952' # hold at -50, Kp = 0.0284, Ti = 48.25, Td = 32.1667
# fname2 = 'SetTemp_20200814122714' # hold at -50, Kp = 0.0170, Ti = 48.25, Td = 12.0625
# fname = 'SetTemp_20200821112538' # 100% open loop cooling

# plot_nested_fields(
#     fname = fname,
#     list_of_fields=[
#         [
#           'TemperaturePv',
#           'TemperatureSv',
#         ],
#         'CoolOutput',
#         # 'CoolErrSum',
#     ],
#     relays=[
#         11-1,
#         12-1,
#         13-1,
#     ],
# )

# plot_nested_fields(
#     fname = fname2,
#     list_of_fields=[
#         [
#           'TemperaturePv',
#           'TemperatureSv',
#         ],
#         'CoolOutput',
#         # 'CoolErrSum',
#     ],
#     relays=[
#         # 11-1,
#     ],
#     clear_plot = False,
# )

# kp = 0.02
# ti = 35
# period = 2
# print((((get_fields_from_file(fname, fields=['CoolErrSum']))[0])[-1]) * kp / ti * period)
