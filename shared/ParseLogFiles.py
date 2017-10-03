
import re
import sys
import os
import datetime
import pandas as pd


def convert(list_time):
    '''
    Converts a list of numbers and time symbols into seconds s
    :param list_time: list of time e.g. [23, 'min', 33, 's', 321, 'ms']
    :return:
    '''
    factor = 0.0
    durration = 0.0

    for idx, val in enumerate(list_time[::2]):
        try:
            if list_time[idx+1] == 's':
                factor = 1
            elif list_time[idx+1] == 'min':
                factor = 60
            elif list_time[idx+1] == 'h':
                factor = 60**2
            elif list_time[idx+1] == 'ms':
                factor = 0.001
            durration += factor*float(val)
        except IndexError as e:
            print(val+" "+list_time[idx+1])
            break
    return durration


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


    partitions_df = (pdf[['ts','type','function','Iteration','Number of partions']]
                     .dropna())
    partitions_df['Number of partions'] = partitions_df['Number of partions'].map(make_partition_int)

    data_df = (pdf[['Iteration','data']]
                     .dropna())
    train_mod_df = (pdf[['Iteration','Training model time']]
                     .dropna())
    train_mod_df['mean'], train_mod_df['std']  = zip(*train_mod_df['Training model time'].map(split_to))
    train_mod_df['std'] = train_mod_df['std'].map(get_std)
    merged_df = partitions_df.merge(data_df,on=['Iteration'])
    merged_df = merged_df.merge(train_mod_df[['Iteration','mean','std']],on=['Iteration'])
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
    tmp = list(map(lambda l: l[1:],
                   map(lambda x: re.split(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})', x),
                       log_file)))
    tmp_1 = list(map(
        lambda line: (datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S,%f'), line[1][1:])
        , tmp))

    # split on :
    tmp_2 = list(map(
        lambda line: (line[0], re.split(r':', line[1]))
        , tmp_1))

    labels = ('ts', 'type', 'function')

    temp_3 = list(map(lambda x: [x[0]] + x[1].split(':'), tmp_1))
    temp_4 = list(map(lambda x: list(zip(labels, x[:3])) + [':'.join(x[3:])], temp_3))
    temp_5 = list(map(lambda x: x[:3] + x[3].split(' - '), temp_4))

    raw_data = list(map(
        lambda l: l[:3] + [tuple(re.split(':', l[3]))] + [tuple(re.split(':', l[4]))],
        temp_5))

    filtered_raw_data = []
    for t in raw_data:
        try:
            filtered_raw_data.append(dict(t))
        except ValueError as e:
            continue

    return filtered_raw_data