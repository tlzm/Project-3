from numpy.core.numeric import ones
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_1','Bearing2_2','Bearing2_3',
'Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_1','Bearing3_2','Bearing3_3']
names = ['hour', 'minute', 'second', 'micro', 'haccel', 'vaccel']

for bearing in bearings:
    os.chdir('Data_set/' + bearing)
    i = 0
    haccel, vaccel,times = [[] for _ in range(3)]
    bearing_data=np.array([])
    for fname in glob.glob('acc*.csv'):
        # if i % 10 != 0:
        #     i += 1
        #     continue
        df = pd.read_csv(fname, names=names)
        data = np.array(df.loc[:])
#        print(data.shape)
        data=data[:,4:6]
        i += 1
        time = i * ones(1)
        data=np.insert(data, 2, time, axis=1)
#        print(data.shape)
        

        if bearing_data.size == 0:
            bearing_data = data
        else:
            bearing_data = np.append(bearing_data,data,axis=0)


    #df = pd.DataFrame({'time': times,})

    bearing_csv = pd.DataFrame(bearing_data)
    os.chdir('../..')  # save csv to project directory
    bearing_csv.to_csv('input/'+bearing + '_rawdata.csv',  encoding='utf-8',index = None)

finalRUL = [0,0,573,290,161,146,757,0,0,753,139,309,129,58,0,0,82]

for bearing in bearings:
    df = pd.read_csv('input/'+bearing + '_rawdata.csv')
    
    pos = bearings.index(bearing)
    n_time = df['2'].unique().max()    

    df = pd.DataFrame({'id': df['2'],
                       'haccel':df['0'],
                       'vaccel':df['1'],
                       'RUL': finalRUL[pos] + n_time - df['2']})
    
    # Set RUL label
    df['RUL'] = np.where(df['RUL'] >= 800, 800, df['RUL'] )
    
    # MinMax normalization (from 0 to 1)
    cols_normalize = df.columns.difference(['RUL','id'])
    scaler = MinMaxScaler()
    # scaler = MaxAbsScaler()
    # scaler = StandardScaler()
    norm_data = pd.DataFrame(scaler.fit_transform(df[cols_normalize]),columns=cols_normalize, index=df.index)
    join_data = df[df.columns.difference(cols_normalize)].join(norm_data)
    df = join_data.reindex(columns = df.columns)
    df.to_csv('input/'+bearing + '_rawdata_withlabel.csv',  encoding='utf-8',index = None)

d11 = pd.read_csv('input/Bearing1_1_rawdata_withlabel.csv')
d12 = pd.read_csv('input/Bearing1_2_rawdata_withlabel.csv')
d13 = pd.read_csv('input/Bearing1_3_rawdata_withlabel.csv')
d14 = pd.read_csv('input/Bearing1_4_rawdata_withlabel.csv')
d15 = pd.read_csv('input/Bearing1_5_rawdata_withlabel.csv')
d16 = pd.read_csv('input/Bearing1_6_rawdata_withlabel.csv')
d17 = pd.read_csv('input/Bearing1_7_rawdata_withlabel.csv')
d21 = pd.read_csv('input/Bearing2_1_rawdata_withlabel.csv')
d22 = pd.read_csv('input/Bearing2_2_rawdata_withlabel.csv')
d23 = pd.read_csv('input/Bearing2_3_rawdata_withlabel.csv')
d24 = pd.read_csv('input/Bearing2_4_rawdata_withlabel.csv')
d25 = pd.read_csv('input/Bearing2_5_rawdata_withlabel.csv')
d26 = pd.read_csv('input/Bearing2_6_rawdata_withlabel.csv')
d27 = pd.read_csv('input/Bearing2_7_rawdata_withlabel.csv')
d31 = pd.read_csv('input/Bearing3_1_rawdata_withlabel.csv')
d32 = pd.read_csv('input/Bearing3_2_rawdata_withlabel.csv')
d33 = pd.read_csv('input/Bearing3_3_rawdata_withlabel.csv')

d12 = pd.DataFrame({'id': d12['id']+2803,'haccel':d12['haccel'],'vaccel':d12['vaccel'],'RUL':d12['RUL']})
d14 = pd.DataFrame({'id': d14['id']+1802,'haccel':d14['haccel'],'vaccel':d14['vaccel'],'RUL':d14['RUL']})
d15 = pd.DataFrame({'id': d15['id']+2941,'haccel':d15['haccel'],'vaccel':d15['vaccel'],'RUL':d15['RUL']})
d16 = pd.DataFrame({'id': d16['id']+5243,'haccel':d16['haccel'],'vaccel':d16['vaccel'],'RUL':d16['RUL']})
d17 = pd.DataFrame({'id': d17['id']+7545,'haccel':d17['haccel'],'vaccel':d17['vaccel'],'RUL':d17['RUL']})
d22 = pd.DataFrame({'id': d22['id']+911,'haccel':d22['haccel'],'vaccel':d22['vaccel'],'RUL':d22['RUL']})
d24 = pd.DataFrame({'id': d24['id']+1202,'haccel':d24['haccel'],'vaccel':d24['vaccel'],'RUL':d24['RUL']})
d25 = pd.DataFrame({'id': d25['id']+1814,'haccel':d25['haccel'],'vaccel':d25['vaccel'],'RUL':d25['RUL']})
d26 = pd.DataFrame({'id': d26['id']+3816,'haccel':d26['haccel'],'vaccel':d26['vaccel'],'RUL':d26['RUL']})
d27 = pd.DataFrame({'id': d27['id']+4388,'haccel':d27['haccel'],'vaccel':d27['vaccel'],'RUL':d27['RUL']})
d32 = pd.DataFrame({'id': d32['id']+515,'haccel':d32['haccel'],'vaccel':d32['vaccel'],'RUL':d32['RUL']})

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