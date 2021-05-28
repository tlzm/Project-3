from numpy.core.fromnumeric import shape
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.fft import fft, fftfreq
import scipy.signal as signal

bearings = ['Bearing1_1']
#'Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_1','Bearing2_2','Bearing2_3',
#'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_1','Bearing3_2','Bearing3_3']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']

#t1,t2,t3,t4,t5,t6,t7 = [[] for _ in range(16)]

for i in range(1,22):
    locals()["t"+str(i)]=[]
    
times=[]
#print(t21)
position=[]

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel_means, vaccel_means, haccel_maxs, vaccel_maxs, haccel_mins, vaccel_mins, haccel_p2ps, vaccel_p2ps, haccel_vars, vaccel_vars, haccel_kurts, vaccel_kurts, haccel_skews, vaccel_skews,haccel_rmss, vaccel_rmss = [[] for _ in range(16)]
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        fh, th, ndh = signal.stft(df['haccel'],fs = 25600,window ='hann',nperseg =None)
        fv, tv, ndv = signal.stft(df['vaccel'],fs = 25600,window ='hann',nperseg =None)
        
        freqnum=ndh.shape[0]
        timenum=ndh.shape[1]        
#        print(ndh.shape[0])
#        print(ndh.shape[1])

#        plt.pcolormesh(t, f, np.abs(nd), vmin = 0, vmax = 0.1)
#        plt.title('STFT')
#        plt.ylabel('frequency')
#        plt.xlabel('time')
#        plt.show()
#        haccel_means.append(df['haccel'].mean())

        i += 1
#        for j in range(1,22):
#            locals()["t"+str(j)].append(nd[:,j-1])
        
        for j in range (freqnum):
#            print(j)
            for k in range(21):
                locals()["t"+str(k+1)].append(np.abs(ndh)[j,k])            
            times.append(i)
            position.append('haccel')
        for j in range (freqnum):
#            print(j)
            for k in range(21):
                locals()["t"+str(k+1)].append(np.abs(ndv)[j,k])            
            times.append(i)
            position.append('vaccel')

    #times = [10*_ for _ in range(len(haccel_means))]

    df = pd.DataFrame({'time': times,
                       'postion':position,
                       't1':t1,
                       't2':t2,
                       't3':t3,
                       't4':t4,
                       't5':t5,
                       't6':t6,
                       't7':t7,
                       't8':t8,
                       't9':t9,
                       't10':t10,
                       't11':t11,
                       't12':t12,
                       't13':t13,
                       't14':t14,
                       't15':t15,
                       't16':t16,
                       't17':t17,
                       't18':t18,
                       't19':t19,
                       't20':t20,
                       't21':t21})

    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_STFT.csv',  encoding='utf-8',index = None)