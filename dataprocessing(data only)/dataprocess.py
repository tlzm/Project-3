from docutils.parsers.rst.directives import positive_int
import pandas as pd
import numpy as np

learningbearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
testingbearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_3']
finalRUL = [5730,339,1610,1460,7570,7530,1390,3090,1290,580,820]

for bearing in learningbearings:
    df_td = pd.read_csv('input/'+bearing + '_time_domain_features.csv')
    df_fd = pd.read_csv('input/'+bearing + '_freq_domain_features.csv')
    df_tfd = pd.read_csv('input/'+bearing + '_timefreq_domain_features.csv')
    df = pd.DataFrame({'time': df_td['time'],
                       'f1': df_td['haccel_mean'],
                       'f2': df_td['vaccel_mean'],
                       'f3': df_td['vaccel_max'],
                       'f4': df_td['vaccel_max'],
                       'f5': df_td['vaccel_min'],
                       'f6': df_td['vaccel_min'],
                       'f7': df_td['vaccel_std'],
                       'f8': df_td['vaccel_std'],
                       'f9': df_td['vaccel_var'],
                       'f10': df_td['vaccel_var'],
                       'f11': df_td['vaccel_kurt'],
                       'f12': df_td['vaccel_kurt'],
                       'f13': df_td['vaccel_skew'],
                       'f14': df_td['vaccel_skew'],

                       'f15': df_fd['haccel_freq_means'],
                       'f16': df_fd['vaccel_freq_means'],
                       'f17': df_fd['haccel_freq_maxs'],
                       'f18': df_fd['vaccel_freq_maxs'],
                       'f19': df_fd['haccel_freq_stds'],
                       'f20': df_fd['vaccel_freq_stds'],
                       'f21': df_fd['haccel_freq_vars'],
                       'f22': df_fd['vaccel_freq_vars'],
                       'f23': df_fd['haccel_freq_kurts'],
                       'f24': df_fd['vaccel_freq_kurts'],
                       'f25': df_fd['haccel_freq_skews'],
                       'f26': df_fd['vaccel_freq_skews'],

                       'f27': df_tfd['haccel_waveletcoeff1_means'],
                       'f28': df_tfd['vaccel_waveletcoeff1_means'],
                       'f29': df_tfd['haccel_waveletcoeff1_maxs'],
                       'f30': df_tfd['vaccel_waveletcoeff1_maxs'],
                       'f31': df_tfd['haccel_waveletcoeff1_mins'],
                       'f32': df_tfd['vaccel_waveletcoeff1_mins'],
                       'f33': df_tfd['haccel_waveletcoeff1_vars'],
                       'f34': df_tfd['vaccel_waveletcoeff1_vars'],
                       'f35': df_tfd['haccel_waveletcoeff1_kurts'],
                       'f36': df_tfd['vaccel_waveletcoeff1_kurts'],
                       'f37': df_tfd['haccel_waveletcoeff1_skews'],
                       'f38': df_tfd['vaccel_waveletcoeff1_skews'],

                       'f39': df_tfd['haccel_waveletcoeff2_means'],
                       'f40': df_tfd['vaccel_waveletcoeff2_means'],
                       'f41': df_tfd['haccel_waveletcoeff2_maxs'],
                       'f42': df_tfd['vaccel_waveletcoeff2_maxs'],
                       'f43': df_tfd['haccel_waveletcoeff2_mins'],
                       'f44': df_tfd['vaccel_waveletcoeff2_mins'],
                       'f45': df_tfd['haccel_waveletcoeff2_vars'],
                       'f46': df_tfd['vaccel_waveletcoeff2_vars'],
                       'f47': df_tfd['haccel_waveletcoeff2_kurts'],
                       'f48': df_tfd['vaccel_waveletcoeff2_kurts'],
                       'f49': df_tfd['haccel_waveletcoeff2_skews'],
                       'f50': df_tfd['vaccel_waveletcoeff2_skews'],
                       
                       'f51': df_tfd['haccel_waveletcoeff3_means'],
                       'f52': df_tfd['vaccel_waveletcoeff3_means'],
                       'f53': df_tfd['haccel_waveletcoeff3_maxs'],
                       'f54': df_tfd['vaccel_waveletcoeff3_maxs'],
                       'f55': df_tfd['haccel_waveletcoeff3_mins'],
                       'f56': df_tfd['vaccel_waveletcoeff3_mins'],
                       'f57': df_tfd['haccel_waveletcoeff3_vars'],
                       'f58': df_tfd['vaccel_waveletcoeff3_vars'],
                       'f59': df_tfd['haccel_waveletcoeff3_kurts'],
                       'f60': df_tfd['vaccel_waveletcoeff3_kurts'],
                       'f61': df_tfd['haccel_waveletcoeff3_skews'],
                       'f62': df_tfd['vaccel_waveletcoeff3_skews'],

                       'RUL': 10*(len(df_td)-1) - df_td['time']})
    df.to_csv('input/'+bearing + '_final_features.csv',  encoding='utf-8',index = None)


for bearing in testingbearings:
    df_td = pd.read_csv('input/'+bearing + '_time_domain_features.csv')
    df_fd = pd.read_csv('input/'+bearing + '_freq_domain_features.csv')
    df_tfd = pd.read_csv('input/'+bearing + '_timefreq_domain_features.csv')
    
    pos = testingbearings.index(bearing)

    df = pd.DataFrame({'time': df_td['time'],
                       'f1': df_td['haccel_mean'],
                       'f2': df_td['vaccel_mean'],
                       'f3': df_td['vaccel_max'],
                       'f4': df_td['vaccel_max'],
                       'f5': df_td['vaccel_min'],
                       'f6': df_td['vaccel_min'],
                       'f7': df_td['vaccel_std'],
                       'f8': df_td['vaccel_std'],
                       'f9': df_td['vaccel_var'],
                       'f10': df_td['vaccel_var'],
                       'f11': df_td['vaccel_kurt'],
                       'f12': df_td['vaccel_kurt'],
                       'f13': df_td['vaccel_skew'],
                       'f14': df_td['vaccel_skew'],

                       'f15': df_fd['haccel_freq_means'],
                       'f16': df_fd['vaccel_freq_means'],
                       'f17': df_fd['haccel_freq_maxs'],
                       'f18': df_fd['vaccel_freq_maxs'],
                       'f19': df_fd['haccel_freq_stds'],
                       'f20': df_fd['vaccel_freq_stds'],
                       'f21': df_fd['haccel_freq_vars'],
                       'f22': df_fd['vaccel_freq_vars'],
                       'f23': df_fd['haccel_freq_kurts'],
                       'f24': df_fd['vaccel_freq_kurts'],
                       'f25': df_fd['haccel_freq_skews'],
                       'f26': df_fd['vaccel_freq_skews'],

                       'f27': df_tfd['haccel_waveletcoeff1_means'],
                       'f28': df_tfd['vaccel_waveletcoeff1_means'],
                       'f29': df_tfd['haccel_waveletcoeff1_maxs'],
                       'f30': df_tfd['vaccel_waveletcoeff1_maxs'],
                       'f31': df_tfd['haccel_waveletcoeff1_mins'],
                       'f32': df_tfd['vaccel_waveletcoeff1_mins'],
                       'f33': df_tfd['haccel_waveletcoeff1_vars'],
                       'f34': df_tfd['vaccel_waveletcoeff1_vars'],
                       'f35': df_tfd['haccel_waveletcoeff1_kurts'],
                       'f36': df_tfd['vaccel_waveletcoeff1_kurts'],
                       'f37': df_tfd['haccel_waveletcoeff1_skews'],
                       'f38': df_tfd['vaccel_waveletcoeff1_skews'],

                       'f39': df_tfd['haccel_waveletcoeff2_means'],
                       'f40': df_tfd['vaccel_waveletcoeff2_means'],
                       'f41': df_tfd['haccel_waveletcoeff2_maxs'],
                       'f42': df_tfd['vaccel_waveletcoeff2_maxs'],
                       'f43': df_tfd['haccel_waveletcoeff2_mins'],
                       'f44': df_tfd['vaccel_waveletcoeff2_mins'],
                       'f45': df_tfd['haccel_waveletcoeff2_vars'],
                       'f46': df_tfd['vaccel_waveletcoeff2_vars'],
                       'f47': df_tfd['haccel_waveletcoeff2_kurts'],
                       'f48': df_tfd['vaccel_waveletcoeff2_kurts'],
                       'f49': df_tfd['haccel_waveletcoeff2_skews'],
                       'f50': df_tfd['vaccel_waveletcoeff2_skews'],
                       
                       'f51': df_tfd['haccel_waveletcoeff3_means'],
                       'f52': df_tfd['vaccel_waveletcoeff3_means'],
                       'f53': df_tfd['haccel_waveletcoeff3_maxs'],
                       'f54': df_tfd['vaccel_waveletcoeff3_maxs'],
                       'f55': df_tfd['haccel_waveletcoeff3_mins'],
                       'f56': df_tfd['vaccel_waveletcoeff3_mins'],
                       'f57': df_tfd['haccel_waveletcoeff3_vars'],
                       'f58': df_tfd['vaccel_waveletcoeff3_vars'],
                       'f59': df_tfd['haccel_waveletcoeff3_kurts'],
                       'f60': df_tfd['vaccel_waveletcoeff3_kurts'],
                       'f61': df_tfd['haccel_waveletcoeff3_skews'],
                       'f62': df_tfd['vaccel_waveletcoeff3_skews'],

                       'RUL': finalRUL[pos]+10*(len(df_td)-1) - df_td['time']})
    df.to_csv('input/'+bearing + '_final_features.csv',  encoding='utf-8',index = None)


d11 = pd.read_csv('input/Bearing1_1_final_features.csv')
d12 = pd.read_csv('input/Bearing1_2_final_features.csv')
d13 = pd.read_csv('input/Bearing1_3_final_features.csv')
d14 = pd.read_csv('input/Bearing1_4_final_features.csv')
d15 = pd.read_csv('input/Bearing1_5_final_features.csv')
d16 = pd.read_csv('input/Bearing1_6_final_features.csv')
d17 = pd.read_csv('input/Bearing1_7_final_features.csv')
d21 = pd.read_csv('input/Bearing2_1_final_features.csv')
d22 = pd.read_csv('input/Bearing2_2_final_features.csv')
d23 = pd.read_csv('input/Bearing2_3_final_features.csv')
d24 = pd.read_csv('input/Bearing2_4_final_features.csv')
d25 = pd.read_csv('input/Bearing2_5_final_features.csv')
d26 = pd.read_csv('input/Bearing2_6_final_features.csv')
d27 = pd.read_csv('input/Bearing2_7_final_features.csv')
d31 = pd.read_csv('input/Bearing3_1_final_features.csv')
d32 = pd.read_csv('input/Bearing3_2_final_features.csv')
d33 = pd.read_csv('input/Bearing3_3_final_features.csv')

#convinient for time windows
d11['id']=1
d12['id']=2
d13['id']=1
d14['id']=2
d15['id']=3
d16['id']=4
d17['id']=5
d21['id']=1
d22['id']=2
d23['id']=1
d24['id']=2
d25['id']=3
d26['id']=4
d27['id']=5
d31['id']=1
d32['id']=2
d33['id']=1

#Split the dataset into training sets and testing sets
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
