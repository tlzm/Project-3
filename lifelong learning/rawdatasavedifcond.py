from numpy.core.numeric import ones
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler

bearings = ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing2_1','Bearing2_2','Bearing2_3'
,'Bearing3_1','Bearing3_2','Bearing3_3']
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




for bearing in bearings:
    df = pd.read_csv('input/'+bearing + '_rawdata.csv')

    n_time = df['2'].unique().max()    

    df = pd.DataFrame({'id': df['2'],
                       'haccel':df['0'],
                       'vaccel':df['1'],
                       'RUL':  n_time - df['2']})
    
    # Set RUL label
    
#    MinMax normalization (from 0 to 1)
    cols_normalize = df.columns.difference(['RUL','id'])
#    scaler = MinMaxScaler()
    scaler = MaxAbsScaler()
#    scaler = StandardScaler()
    norm_data = pd.DataFrame(scaler.fit_transform(df[cols_normalize]),columns=cols_normalize, index=df.index)
    join_data = df[df.columns.difference(cols_normalize)].join(norm_data)
    df = join_data.reindex(columns = df.columns)

    df['RUL'] = np.where(df['RUL'] >= 801, 801, df['RUL'] )
    
    df=df[~(df['RUL'].isin([801]))]

    df.to_csv('input/'+bearing + '_rawdata_withlabel.csv',  encoding='utf-8',index = None)


d11 = pd.read_csv('input/Bearing1_1_rawdata_withlabel.csv')
d12 = pd.read_csv('input/Bearing1_2_rawdata_withlabel.csv')
d13 = pd.read_csv('input/Bearing1_3_rawdata_withlabel.csv')
d21 = pd.read_csv('input/Bearing2_1_rawdata_withlabel.csv')
d22 = pd.read_csv('input/Bearing2_2_rawdata_withlabel.csv')
d23 = pd.read_csv('input/Bearing2_3_rawdata_withlabel.csv')
d31 = pd.read_csv('input/Bearing3_1_rawdata_withlabel.csv')
d32 = pd.read_csv('input/Bearing3_2_rawdata_withlabel.csv')
d33 = pd.read_csv('input/Bearing3_3_rawdata_withlabel.csv')


d11 = pd.DataFrame({'id': d11['id']-2002,'haccel':d11['haccel'],'vaccel':d11['vaccel'],'RUL':d11['RUL']})
d12 = pd.DataFrame({'id': d12['id']+730,'haccel':d12['haccel'],'vaccel':d12['vaccel'],'RUL':d12['RUL']})
d13 = pd.DataFrame({'id': d13['id']+26,'haccel':d13['haccel'],'vaccel':d13['vaccel'],'RUL':d13['RUL']})
d21 = pd.DataFrame({'id': d21['id']-110,'haccel':d21['haccel'],'vaccel':d21['vaccel'],'RUL':d21['RUL']})
d22 = pd.DataFrame({'id': d22['id']+800,'haccel':d22['haccel'],'vaccel':d22['vaccel'],'RUL':d22['RUL']})
d23 = pd.DataFrame({'id': d23['id']+442,'haccel':d23['haccel'],'vaccel':d23['vaccel'],'RUL':d23['RUL']})
d31 = pd.DataFrame({'id': d31['id'],'haccel':d31['haccel'],'vaccel':d31['vaccel'],'RUL':d31['RUL']})
d32 = pd.DataFrame({'id': d32['id']-322,'haccel':d32['haccel'],'vaccel':d32['vaccel'],'RUL':d32['RUL']})
d33 = pd.DataFrame({'id': d33['id']+1314,'haccel':d33['haccel'],'vaccel':d33['vaccel'],'RUL':d33['RUL']})

L1 = pd.concat ([d11,d12,d13],ignore_index=True)
L2 = pd.concat ([d21,d22,d23],ignore_index=True)
L3 = pd.concat ([d31,d32,d33],ignore_index=True)

L1.to_csv('input/LearningSet1.csv',  encoding='utf-8',index = None)
L2.to_csv('input/LearningSet2.csv',  encoding='utf-8',index = None)
L3.to_csv('input/LearningSet3.csv',  encoding='utf-8',index = None)


