import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.fft import fft, fftfreq

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_1','Bearing2_2','Bearing2_3',
'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_1','Bearing3_2','Bearing3_3']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']

def get_rms(records):

    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel_means, vaccel_means, haccel_maxs, vaccel_maxs, haccel_mins, vaccel_mins, haccel_p2ps, vaccel_p2ps, haccel_vars, vaccel_vars, haccel_kurts, vaccel_kurts, haccel_skews, vaccel_skews,haccel_rmss, vaccel_rmss = []
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        haccel_means.append(df['haccel'].mean())
        vaccel_means.append(df['vaccel'].mean())
        haccel_maxs.append(df['haccel'].max())
        vaccel_maxs.append(df['vaccel'].max())
        haccel_mins.append(df['haccel'].min())
        vaccel_mins.append(df['vaccel'].min())
        haccel_p2ps.append(df['haccel'].max()-df['haccel'].min())
        vaccel_p2ps.append(df['vaccel'].max()-df['vaccel'].min())       
        haccel_vars.append(df['haccel'].var())
        vaccel_vars.append(df['vaccel'].var())
        haccel_kurts.append(df['haccel'].kurt())
        vaccel_kurts.append(df['vaccel'].kurt())
        haccel_skews.append(df['haccel'].skew())
        vaccel_skews.append(df['vaccel'].skew())
        haccel_rmss.append(get_rms(df['haccel']))
        vaccel_rmss.append(get_rms(df['vaccel']))
        i += 1

    times = [10*_ for _ in range(len(haccel_means))]

    df = pd.DataFrame({'time': times,
                       'haccel_mean': haccel_means,
                       'vaccel_mean': vaccel_means,
                       'haccel_max': haccel_maxs,
                       'vaccel_max': vaccel_maxs,
                       'haccel_min': haccel_mins,
                       'vaccel_min': vaccel_mins,
                       'haccel_p2p': haccel_p2ps,
                       'vaccel_p2p': vaccel_p2ps,
                       'haccel_var': haccel_vars,
                       'vaccel_var': vaccel_vars,
                       'haccel_kurt': haccel_kurts,
                       'vaccel_kurt': vaccel_kurts,
                       'haccel_skew': haccel_skews,
                       'vaccel_skew': vaccel_skews,
                       'haccel_rms': haccel_rmss,
                       'vaccel_rms': vaccel_rmss,})

    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_time_domain_features.csv',  encoding='utf-8',index = None)
