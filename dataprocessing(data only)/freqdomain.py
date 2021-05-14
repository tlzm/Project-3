import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_1','Bearing2_2','Bearing2_3',
'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_1','Bearing3_2','Bearing3_3']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']
num_points = 2560

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel_freq_means, vaccel_freq_means = [], []
    haccel_freq_maxs, vaccel_freq_maxs = [], []
    haccel_freq_stds, vaccel_freq_stds = [], []
    haccel_freq_vars, vaccel_freq_vars = [], []
    haccel_freq_kurts, vaccel_freq_kurts = [], []
    haccel_freq_skews, vaccel_freq_skews = [], []


    plot_num = 1
    for fname in glob.glob('acc*.csv'):
        # if i % 100 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        freqs = fftfreq(num_points)

        #fft
        haccel_fft=np.abs(fft(df['haccel'],num_points))
        vaccel_fft=np.abs(fft(df['vaccel'],num_points))

        #kurt cal.
        haccel_fft_mean=np.mean(haccel_fft)
        haccel_freq_kurt=np.mean((haccel_fft-haccel_fft_mean)**3)
        vaccel_fft_mean=np.mean(vaccel_fft)
        vaccel_freq_kurt=np.mean((vaccel_fft-vaccel_fft_mean)**3)

        #skew cal.
        haccel_fft_var=np.var(haccel_fft)
        haccel_freq_skew=np.mean((haccel_fft-haccel_fft_mean)**4)/pow(haccel_fft_var,2)
        vaccel_fft_var=np.var(vaccel_fft)
        vaccel_freq_skew=np.mean((vaccel_fft-vaccel_fft_mean)**4)/pow(vaccel_fft_var,2)

        #fft feature
        haccel_freq_means.append(haccel_fft.mean())
        vaccel_freq_means.append(vaccel_fft.mean())
        haccel_freq_maxs.append(haccel_fft.max())
        vaccel_freq_maxs.append(vaccel_fft.max())
        haccel_freq_stds.append(haccel_fft.std())
        vaccel_freq_stds.append(vaccel_fft.std())
        haccel_freq_vars.append(haccel_fft.var())
        vaccel_freq_vars.append(vaccel_fft.var())
        haccel_freq_kurts.append(haccel_freq_kurt)
        vaccel_freq_kurts.append(vaccel_freq_kurt)
        haccel_freq_skews.append(haccel_freq_skew)
        vaccel_freq_skews.append(vaccel_freq_skew)


        # plot_num += 1
        i += 1

    times = [10*_ for _ in range(len(haccel_freq_means))]

    df = pd.DataFrame({'time': times,
                       'haccel_freq_means': haccel_freq_means,
                       'vaccel_freq_means': vaccel_freq_means,
                       'haccel_freq_maxs': haccel_freq_maxs,
                       'vaccel_freq_maxs': vaccel_freq_maxs,
                       'haccel_freq_stds': haccel_freq_stds,
                       'vaccel_freq_stds': vaccel_freq_stds,
                       'haccel_freq_vars': haccel_freq_vars,
                       'vaccel_freq_vars': vaccel_freq_vars,
                       'haccel_freq_kurts': haccel_freq_kurts,
                       'vaccel_freq_kurts': vaccel_freq_kurts,
                       'haccel_freq_skews': haccel_freq_skews,
                       'vaccel_freq_skews': vaccel_freq_skews
    })


    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_freq_domain_features.csv',  encoding='utf-8',index = None)
