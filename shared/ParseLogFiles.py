
import re
import sys
import os
from datetime import datetime
import pandas as pd

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def convert(list_time):
    '''
    Converts a list of numbers and time symbols into seconds s
    :param list_time: list of time e.g. [23, 'min', 33, 's', 321, 'ms']
    :return:
    '''
    factor = 0.0
    duration = 0.0

    for value, time_unit in pairwise(list_time):
        try:
            if time_unit == 's':
                factor = 1
            elif time_unit == 'min':
                factor = 60
            elif time_unit == 'h':
                factor = 60**2
            elif time_unit == 'ms':
                factor = 0.001
            duration += factor * float(value)
        except IndexError as e:
            print('Does not comupte got: value {} and time unit {}'
                  .format(value, time_unit)
                  )
            break
    return duration


def fix_logfile_mess(pdf):
    '''
    Takes a raw pandas custom logfile and makes into a nice pandas dataframe
    :param pdf:  raw pandas dataframe
    :return: nice pandas dataframe
    '''

    make_partition_int = lambda x: re.search('[0-9]+',x).group()
    split_to_mean_std = lambda x: re.split(r'±',x)
    get_std = lambda x: re.search(r'([a-zA-Z0-9\.\s]+)+ per',x).group(1)
    get_n = lambda x: int(re.search(r'\d+', x).group())

    def split_to(x):
        return re.split(r'±',x)[0], re.split(r'±',x)[1]


    partitions_df = (pdf[[
        'ts','type','function',
        'Iteration','Number of partions']].dropna()
    )
    partitions_df['Number of partions'] = (partitions_df['Number of partions']
        .map(make_partition_int)
    )
    data_df = (pdf[['Iteration','data']].dropna()
    )
    train_mod_df = (pdf[['Iteration','Training model time']].dropna()
    )
    train_mod_df['mean'], train_mod_df['std']  = zip(
        *train_mod_df['Training model time'].map(split_to)
    )
    train_mod_df['std'] = train_mod_df['std'].map(get_std)
    merged_df = partitions_df.merge(data_df,on=['Iteration'])
    merged_df = merged_df.merge(
        other=train_mod_df[['Iteration','mean','std']],
        on=['Iteration']
    )
    merged_df['mean'] = (merged_df['mean']
        .map(lambda x: re.findall('[a-zA-Z]+|\\d+\.\d+|\\d+',x))
        .map(lambda x: convert(x))
    )
    merged_df['std'] = (merged_df['std']
        .map(lambda x: re.findall('[a-zA-Z]+|\\d+\.\d+|\\d+',x))
        .map(lambda x: convert(x))
    )
    merged_df['n'] = merged_df['data'].map(get_n)
    merged_df['partitions'] = merged_df['Number of partions']
    del merged_df['data']
    del merged_df['Number of partions']
    return merged_df


def split_logfile(log_file):
    """
    Splits the logfile into a list of list. Each child list contains the structured information
    :param log_file: The log file imported
    :return: list of lists.
    """
    # split of date from rest
    tmp = list(map(
        func=lambda l: l[1:],
        iter1=map(
            func=lambda x: re.split(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})', x),
            iter1=log_file))
    )
    tmp_1 = list(map(
        func=lambda line: (datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S,%f'), line[1][1:]),
        iter1=tmp)
    )
    # split on :
    tmp_2 = list(map(
        func=lambda line: (line[0], re.split(r':', line[1])),
        iter1=tmp_1)
    )
    labels = ('ts', 'type', 'function')
    temp_3 = list(map(
        func=lambda x: [x[0]] + x[1].split(':'),
        iter1=tmp_1)
    )
    temp_4 = list(map(
        func=lambda x: list(zip(labels, x[:3]))
                       + [':'.join(x[3:])],
        iter1=temp_3)
    )
    temp_5 = list(map(
        func=lambda x: x[:3] + x[3].split(' - '),
        iter1=temp_4)
    )
    raw_data = list(map(
        func=lambda l: l[:3]
                       + [tuple(re.split(':', l[3]))]
                       + [tuple(re.split(':', l[4]))],
        iter1=temp_5)
    )
    filtered_raw_data = []
    for t in raw_data:
        try:
            filtered_raw_data.append(dict(t))
        except ValueError as e:
            continue

    return filtered_raw_data