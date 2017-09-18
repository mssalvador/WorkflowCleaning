from functools import reduce
import re
import sys
import os
import getpass
import datetime
import logging
import pandas as pd


def convert_time_to_second(duration):
    time = list(map(lambda x: cast_to_val(x), filter(lambda x: x is not '', re.split("([0-9]\.+)", duration))))
    result = 0
    factor = 0
    for idx, val in enumerate(time[::-1]):
        if idx % 2 == 0:
            if val == 's':
                factor = 1
            elif val == 'min':
                factor = 60
            elif val == 'h':
                factor = 60**2
            elif val == 'ms':
                factor = 0.001
        else:
            result += factor*val
    return result


def cast_to_val(x):
    try:
        return float(x)
    except ValueError:
        return x


def divide_string(array):
    # dive between the timestamp from the string.
    ts_split = [re.findall(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})', i) + re.split(':', i)[3:] for i in array]
    cols = ['ts',
            'log-type',
            'program',
            'iteration',
            'data-size',
            'method',
            'avg-training-duration',
            'std-training-duration',
            'runs',
            'loops']

    # del data

    # print(ts_split)
    for_split = list(map(lambda l: l[:3] + re.split('\s for\s', l[3]) + l[4:], ts_split[:-1]))
    result_split = []
    size = None
    for idx, line in enumerate(for_split):
        if idx % 3 == 0:
            # first line in data point
            size = re.findall(r'\d+', line[-1])
            iteration = re.findall(r'\d+', line[-2])
            # print(size)
        else:
            line.insert(4, int(*size))
            line[3] = int(iteration[0])
            elements = [i.replace(' ', '') for i in re.split(r'±|per', line[-1])]
            # elements = [i.replace(' ','') for i in re.findall(r'(\d+\w+\s\d+\w+|\d+\s\w+)',line[-1])]
            elements[0] = convert_time_to_second(elements[0])
            elements[1] = convert_time_to_second(elements[1])
            loop_runs = re.findall(r'(\d+)', string=elements[-1])
            del elements[2:]
            del line[-1]
            line = line + elements + loop_runs
            result_split.append(line)
    return pd.DataFrame(data=result_split, columns=cols)