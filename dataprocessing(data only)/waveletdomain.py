import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import pywt

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_1','Bearing2_2','Bearing2_3',
'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_1','Bearing3_2','Bearing3_3']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']
num_points = 2560

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel_waveletcoeff1_means, vaccel_waveletcoeff1_means = [], []
    haccel_waveletcoeff1_maxs, vaccel_waveletcoeff1_maxs = [], []
    haccel_waveletcoeff1_mins, vaccel_waveletcoeff1_mins = [], []
    haccel_waveletcoeff1_vars, vaccel_waveletcoeff1_vars = [], []
    haccel_waveletcoeff1_kurts, vaccel_waveletcoeff1_kurts = [], []
    haccel_waveletcoeff1_skews, vaccel_waveletcoeff1_skews = [], []

    haccel_waveletcoeff2_means, vaccel_waveletcoeff2_means = [], []
    haccel_waveletcoeff2_maxs, vaccel_waveletcoeff2_maxs = [], []
    haccel_waveletcoeff2_mins, vaccel_waveletcoeff2_mins = [], []
    haccel_waveletcoeff2_vars, vaccel_waveletcoeff2_vars = [], []
    haccel_waveletcoeff2_kurts, vaccel_waveletcoeff2_kurts = [], []
    haccel_waveletcoeff2_skews, vaccel_waveletcoeff2_skews = [], []

    haccel_waveletcoeff3_means, vaccel_waveletcoeff3_means = [], []
    haccel_waveletcoeff3_maxs, vaccel_waveletcoeff3_maxs = [], []
    haccel_waveletcoeff3_mins, vaccel_waveletcoeff3_mins = [], []
    haccel_waveletcoeff3_vars, vaccel_waveletcoeff3_vars = [], []
    haccel_waveletcoeff3_kurts, vaccel_waveletcoeff3_kurts = [], []
    haccel_waveletcoeff3_skews, vaccel_waveletcoeff3_skews = [], []

    plot_num = 1
    for fname in glob.glob('acc*.csv'):
        # if i % 100 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        freqs = fftfreq(num_points)

        haccel_coeffs = pywt.wavedec(df['haccel'], 'bior3.7', level = 5)
        vaccel_coeffs = pywt.wavedec(df['vaccel'], 'bior3.7', level = 5)

        #1st level coeffs
        #kurt cal.
        haccel_waveletcoeff1_mean=np.mean(haccel_coeffs[1])
        haccel_waveletcoeff1_kurt=np.mean((haccel_coeffs[1]-haccel_waveletcoeff1_mean)**3)
        vaccel_waveletcoeff1_mean=np.mean(vaccel_coeffs[1])
        vaccel_waveletcoeff1_kurt=np.mean((vaccel_coeffs[1]-vaccel_waveletcoeff1_mean)**3)

        #skew cal.
        haccel_waveletcoeff1_var=np.var(haccel_coeffs[1])
        haccel_waveletcoeff1_skew=np.mean((haccel_coeffs[1]-haccel_waveletcoeff1_mean)**4)/pow(haccel_waveletcoeff1_var,2)
        vaccel_waveletcoeff1_var=np.var(vaccel_coeffs[1])
        vaccel_waveletcoeff1_skew=np.mean((vaccel_coeffs[1]-vaccel_waveletcoeff1_mean)**4)/pow(vaccel_waveletcoeff1_var,2)

        haccel_waveletcoeff1_means.append(haccel_waveletcoeff1_mean)
        vaccel_waveletcoeff1_means.append(vaccel_waveletcoeff1_mean)
        haccel_waveletcoeff1_maxs.append(np.max(haccel_coeffs[1]))
        vaccel_waveletcoeff1_maxs.append(np.max(vaccel_coeffs[1]))
        haccel_waveletcoeff1_mins.append(np.min(haccel_coeffs[1]))
        vaccel_waveletcoeff1_mins.append(np.min(vaccel_coeffs[1]))
        haccel_waveletcoeff1_vars.append(haccel_waveletcoeff1_var)
        vaccel_waveletcoeff1_vars.append(vaccel_waveletcoeff1_var)
        haccel_waveletcoeff1_kurts.append(haccel_waveletcoeff1_kurt)
        vaccel_waveletcoeff1_kurts.append(vaccel_waveletcoeff1_kurt)
        haccel_waveletcoeff1_skews.append(haccel_waveletcoeff1_skew)
        vaccel_waveletcoeff1_skews.append(vaccel_waveletcoeff1_skew)

        #2nd level coeffs
        #kurt cal.
        haccel_waveletcoeff2_mean=np.mean(haccel_coeffs[2])
        haccel_waveletcoeff2_kurt=np.mean((haccel_coeffs[2]-haccel_waveletcoeff2_mean)**3)
        vaccel_waveletcoeff2_mean=np.mean(vaccel_coeffs[2])
        vaccel_waveletcoeff2_kurt=np.mean((vaccel_coeffs[2]-vaccel_waveletcoeff2_mean)**3)

        #skew cal.
        haccel_waveletcoeff2_var=np.var(haccel_coeffs[2])
        haccel_waveletcoeff2_skew=np.mean((haccel_coeffs[2]-haccel_waveletcoeff2_mean)**4)/pow(haccel_waveletcoeff2_var,2)
        vaccel_waveletcoeff2_var=np.var(vaccel_coeffs[2])
        vaccel_waveletcoeff2_skew=np.mean((vaccel_coeffs[2]-vaccel_waveletcoeff2_mean)**4)/pow(vaccel_waveletcoeff2_var,2)

        haccel_waveletcoeff2_means.append(haccel_waveletcoeff2_mean)
        vaccel_waveletcoeff2_means.append(vaccel_waveletcoeff2_mean)
        haccel_waveletcoeff2_maxs.append(np.max(haccel_coeffs[2]))
        vaccel_waveletcoeff2_maxs.append(np.max(vaccel_coeffs[2]))
        haccel_waveletcoeff2_mins.append(np.min(haccel_coeffs[2]))
        vaccel_waveletcoeff2_mins.append(np.min(vaccel_coeffs[2]))
        haccel_waveletcoeff2_vars.append(haccel_waveletcoeff2_var)
        vaccel_waveletcoeff2_vars.append(vaccel_waveletcoeff2_var)
        haccel_waveletcoeff2_kurts.append(haccel_waveletcoeff2_kurt)
        vaccel_waveletcoeff2_kurts.append(vaccel_waveletcoeff2_kurt)
        haccel_waveletcoeff2_skews.append(haccel_waveletcoeff2_skew)
        vaccel_waveletcoeff2_skews.append(vaccel_waveletcoeff2_skew)

        #3rd level coeffs
        #kurt cal.
        haccel_waveletcoeff3_mean=np.mean(haccel_coeffs[3])
        haccel_waveletcoeff3_kurt=np.mean((haccel_coeffs[3]-haccel_waveletcoeff3_mean)**3)
        vaccel_waveletcoeff3_mean=np.mean(vaccel_coeffs[3])
        vaccel_waveletcoeff3_kurt=np.mean((vaccel_coeffs[3]-vaccel_waveletcoeff3_mean)**3)

        #skew cal.
        haccel_waveletcoeff3_var=np.var(haccel_coeffs[3])
        haccel_waveletcoeff3_skew=np.mean((haccel_coeffs[3]-haccel_waveletcoeff3_mean)**4)/pow(haccel_waveletcoeff3_var,2)
        vaccel_waveletcoeff3_var=np.var(vaccel_coeffs[3])
        vaccel_waveletcoeff3_skew=np.mean((vaccel_coeffs[3]-vaccel_waveletcoeff3_mean)**4)/pow(vaccel_waveletcoeff3_var,2)

        haccel_waveletcoeff3_means.append(haccel_waveletcoeff3_mean)
        vaccel_waveletcoeff3_means.append(vaccel_waveletcoeff3_mean)
        haccel_waveletcoeff3_maxs.append(np.max(haccel_coeffs[3]))
        vaccel_waveletcoeff3_maxs.append(np.max(vaccel_coeffs[3]))
        haccel_waveletcoeff3_mins.append(np.min(haccel_coeffs[3]))
        vaccel_waveletcoeff3_mins.append(np.min(vaccel_coeffs[3]))
        haccel_waveletcoeff3_vars.append(haccel_waveletcoeff3_var)
        vaccel_waveletcoeff3_vars.append(vaccel_waveletcoeff3_var)
        haccel_waveletcoeff3_kurts.append(haccel_waveletcoeff3_kurt)
        vaccel_waveletcoeff3_kurts.append(vaccel_waveletcoeff3_kurt)
        haccel_waveletcoeff3_skews.append(haccel_waveletcoeff3_skew)
        vaccel_waveletcoeff3_skews.append(vaccel_waveletcoeff3_skew)

        # plot_num += 1
        i += 1


    times = [10*_ for _ in range(len(haccel_waveletcoeff1_means))]

    df = pd.DataFrame({'time': times,
                       'haccel_waveletcoeff1_means': haccel_waveletcoeff1_means,
                       'vaccel_waveletcoeff1_means': vaccel_waveletcoeff1_means,
                       'haccel_waveletcoeff1_maxs': haccel_waveletcoeff1_maxs,
                       'vaccel_waveletcoeff1_maxs': vaccel_waveletcoeff1_maxs,
                       'haccel_waveletcoeff1_mins': haccel_waveletcoeff1_mins,
                       'vaccel_waveletcoeff1_mins': vaccel_waveletcoeff1_mins,
                       'haccel_waveletcoeff1_vars': haccel_waveletcoeff1_vars,
                       'vaccel_waveletcoeff1_vars': vaccel_waveletcoeff1_vars,
                       'haccel_waveletcoeff1_kurts': haccel_waveletcoeff1_kurts,
                       'vaccel_waveletcoeff1_kurts': vaccel_waveletcoeff1_kurts,
                       'haccel_waveletcoeff1_skews': haccel_waveletcoeff1_skews,
                       'vaccel_waveletcoeff1_skews': vaccel_waveletcoeff1_skews,

                       'haccel_waveletcoeff2_means': haccel_waveletcoeff2_means,
                       'vaccel_waveletcoeff2_means': vaccel_waveletcoeff2_means,
                       'haccel_waveletcoeff2_maxs': haccel_waveletcoeff2_maxs,
                       'vaccel_waveletcoeff2_maxs': vaccel_waveletcoeff2_maxs,
                       'haccel_waveletcoeff2_mins': haccel_waveletcoeff2_mins,
                       'vaccel_waveletcoeff2_mins': vaccel_waveletcoeff2_mins,
                       'haccel_waveletcoeff2_vars': haccel_waveletcoeff2_vars,
                       'vaccel_waveletcoeff2_vars': vaccel_waveletcoeff2_vars,
                       'haccel_waveletcoeff2_kurts': haccel_waveletcoeff2_kurts,
                       'vaccel_waveletcoeff2_kurts': vaccel_waveletcoeff2_kurts,
                       'haccel_waveletcoeff2_skews': haccel_waveletcoeff2_skews,
                       'vaccel_waveletcoeff2_skews': vaccel_waveletcoeff2_skews,
                       
                       'haccel_waveletcoeff3_means': haccel_waveletcoeff3_means,
                       'vaccel_waveletcoeff3_means': vaccel_waveletcoeff3_means,
                       'haccel_waveletcoeff3_maxs': haccel_waveletcoeff3_maxs,
                       'vaccel_waveletcoeff3_maxs': vaccel_waveletcoeff3_maxs,
                       'haccel_waveletcoeff3_mins': haccel_waveletcoeff3_mins,
                       'vaccel_waveletcoeff3_mins': vaccel_waveletcoeff3_mins,
                       'haccel_waveletcoeff3_vars': haccel_waveletcoeff3_vars,
                       'vaccel_waveletcoeff3_vars': vaccel_waveletcoeff3_vars,
                       'haccel_waveletcoeff3_kurts': haccel_waveletcoeff3_kurts,
                       'vaccel_waveletcoeff3_kurts': vaccel_waveletcoeff3_kurts,
                       'haccel_waveletcoeff3_skews': haccel_waveletcoeff3_skews,
                       'vaccel_waveletcoeff3_skews': vaccel_waveletcoeff3_skews
    })


    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_timefreq_domain_features.csv',  encoding='utf-8',index = None)
