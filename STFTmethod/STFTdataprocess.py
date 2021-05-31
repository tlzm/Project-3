from numpy.core.fromnumeric import shape
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.fft import fft, fftfreq
import scipy.signal as signal

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_1','Bearing2_2','Bearing2_3',
'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_1','Bearing3_2','Bearing3_3']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']


finalRUL = [0,0,573,290,161,146,757,0,0,753,139,309,129,58,0,0,82]

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    pos = bearings.index(bearing)
    i = 0
    times=[]
    position=[]
    for j in range(1,22):
        locals()["t"+str(j)]=[]

   
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

    n_time = np.array(times).max() 

    df = pd.DataFrame({'id': times,
#                       'postion':position,
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
                       't21':t21,
                       'RUL': finalRUL[pos] + n_time - times})
    df['RUL'] = np.where(df['RUL'] >= 1001, 1001, df['RUL'] )
    
#    df=df[~(df['RUL'].isin([801]))]

    os.chdir('../..')  # save csv to project directory
    df.to_csv('input/'+bearing + '_STFT.csv',  encoding='utf-8',index = None)
    print(bearing+'_STFT.csv has saved.')


d11 = pd.read_csv('input/Bearing1_1_STFT.csv')
d12 = pd.read_csv('input/Bearing1_2_STFT.csv')
d13 = pd.read_csv('input/Bearing1_3_STFT.csv')
d14 = pd.read_csv('input/Bearing1_4_STFT.csv')
d15 = pd.read_csv('input/Bearing1_5_STFT.csv')
d16 = pd.read_csv('input/Bearing1_6_STFT.csv')
d17 = pd.read_csv('input/Bearing1_7_STFT.csv')
d21 = pd.read_csv('input/Bearing2_1_STFT.csv')
d22 = pd.read_csv('input/Bearing2_2_STFT.csv')
d23 = pd.read_csv('input/Bearing2_3_STFT.csv')
d24 = pd.read_csv('input/Bearing2_4_STFT.csv')
d25 = pd.read_csv('input/Bearing2_5_STFT.csv')
d26 = pd.read_csv('input/Bearing2_6_STFT.csv')
d27 = pd.read_csv('input/Bearing2_7_STFT.csv')
d31 = pd.read_csv('input/Bearing3_1_STFT.csv')
d32 = pd.read_csv('input/Bearing3_2_STFT.csv')
d33 = pd.read_csv('input/Bearing3_3_STFT.csv')


d12['id']=d12['id']+2803
d14['id']=d14['id']+1802
d15['id']=d15['id']+2941
d16['id']=d16['id']+5243
d17['id']=d17['id']+7545
d22['id']=d22['id']+911
d24['id']=d24['id']+1202
d25['id']=d25['id']+1814
d26['id']=d26['id']+3816
d27['id']=d27['id']+4388
d32['id']=d32['id']+515

#bearing No.
d11['bearing']=1
d12['bearing']=2
d13['bearing']=1
d14['bearing']=2
d15['bearing']=3
d16['bearing']=4
d17['bearing']=5
d21['bearing']=1
d22['bearing']=2
d23['bearing']=1
d24['bearing']=2
d25['bearing']=3
d26['bearing']=4
d27['bearing']=5
d31['bearing']=1
d32['bearing']=2
d33['bearing']=1

L1 = pd.concat ([d11,d12],ignore_index=True)
L2 = pd.concat ([d21,d22],ignore_index=True)
L3 = pd.concat ([d31,d32],ignore_index=True)
T1 = pd.concat ([d13,d14,d15,d16,d17],ignore_index=True)
T2 = pd.concat ([d23,d24,d25,d26,d27],ignore_index=True)
T3 = d33

L1.to_csv('input/LearningSet1.csv',  encoding='utf-8',index = None)
L2.to_csv('input/LearningSet2.csv',  encoding='utf-8',index = None)
L3.to_csv('input/LearningSet3.csv',  encoding='utf-8',index = None)
T1.to_csv('input/TestingSet1.csv',  encoding='utf-8',index = None)
T2.to_csv('input/TestingSet2.csv',  encoding='utf-8',index = None)
T3.to_csv('input/TestingSet3.csv',  encoding='utf-8',index = None) 
