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
                       'haccel_mean': df_td['haccel_mean'],
                       'vaccel_mean': df_td['vaccel_mean'],
                       'haccel_max': df_td['vaccel_max'],
                       'vaccel_max': df_td['vaccel_max'],
                       'haccel_min': df_td['vaccel_min'],
                       'vaccel_min': df_td['vaccel_min'],
                       'haccel_std': df_td['vaccel_std'],
                       'vaccel_std': df_td['vaccel_std'],
                       'haccel_var': df_td['vaccel_var'],
                       'vaccel_var': df_td['vaccel_var'],
                       'haccel_kurt': df_td['vaccel_kurt'],
                       'vaccel_kurt': df_td['vaccel_kurt'],
                       'haccel_skew': df_td['vaccel_skew'],
                       'vaccel_skew': df_td['vaccel_skew'],

                       'haccel_freq_means': df_fd['haccel_freq_means'],
                       'vaccel_freq_means': df_fd['vaccel_freq_means'],
                       'haccel_freq_maxs': df_fd['haccel_freq_maxs'],
                       'vaccel_freq_maxs': df_fd['vaccel_freq_maxs'],
                       'haccel_freq_stds': df_fd['haccel_freq_stds'],
                       'vaccel_freq_stds': df_fd['vaccel_freq_stds'],
                       'haccel_freq_vars': df_fd['haccel_freq_vars'],
                       'vaccel_freq_vars': df_fd['vaccel_freq_vars'],
                       'haccel_freq_kurts': df_fd['haccel_freq_kurts'],
                       'vaccel_freq_kurts': df_fd['vaccel_freq_kurts'],
                       'haccel_freq_skews': df_fd['haccel_freq_skews'],
                       'vaccel_freq_skews': df_fd['vaccel_freq_skews'],

                       'haccel_waveletcoeff1_means': df_tfd['haccel_waveletcoeff1_means'],
                       'vaccel_waveletcoeff1_means': df_tfd['vaccel_waveletcoeff1_means'],
                       'haccel_waveletcoeff1_maxs': df_tfd['haccel_waveletcoeff1_maxs'],
                       'vaccel_waveletcoeff1_maxs': df_tfd['vaccel_waveletcoeff1_maxs'],
                       'haccel_waveletcoeff1_mins': df_tfd['haccel_waveletcoeff1_mins'],
                       'vaccel_waveletcoeff1_mins': df_tfd['vaccel_waveletcoeff1_mins'],
                       'haccel_waveletcoeff1_vars': df_tfd['haccel_waveletcoeff1_vars'],
                       'vaccel_waveletcoeff1_vars': df_tfd['vaccel_waveletcoeff1_vars'],
                       'haccel_waveletcoeff1_kurts': df_tfd['haccel_waveletcoeff1_kurts'],
                       'vaccel_waveletcoeff1_kurts': df_tfd['vaccel_waveletcoeff1_kurts'],
                       'haccel_waveletcoeff1_skews': df_tfd['haccel_waveletcoeff1_skews'],
                       'vaccel_waveletcoeff1_skews': df_tfd['vaccel_waveletcoeff1_skews'],

                       'haccel_waveletcoeff2_means': df_tfd['haccel_waveletcoeff2_means'],
                       'vaccel_waveletcoeff2_means': df_tfd['vaccel_waveletcoeff2_means'],
                       'haccel_waveletcoeff2_maxs': df_tfd['haccel_waveletcoeff2_maxs'],
                       'vaccel_waveletcoeff2_maxs': df_tfd['vaccel_waveletcoeff2_maxs'],
                       'haccel_waveletcoeff2_mins': df_tfd['haccel_waveletcoeff2_mins'],
                       'vaccel_waveletcoeff2_mins': df_tfd['vaccel_waveletcoeff2_mins'],
                       'haccel_waveletcoeff2_vars': df_tfd['haccel_waveletcoeff2_vars'],
                       'vaccel_waveletcoeff2_vars': df_tfd['vaccel_waveletcoeff2_vars'],
                       'haccel_waveletcoeff2_kurts': df_tfd['haccel_waveletcoeff2_kurts'],
                       'vaccel_waveletcoeff2_kurts': df_tfd['vaccel_waveletcoeff2_kurts'],
                       'haccel_waveletcoeff2_skews': df_tfd['haccel_waveletcoeff2_skews'],
                       'vaccel_waveletcoeff2_skews': df_tfd['vaccel_waveletcoeff2_skews'],
                       
                       'haccel_waveletcoeff3_means': df_tfd['haccel_waveletcoeff3_means'],
                       'vaccel_waveletcoeff3_means': df_tfd['vaccel_waveletcoeff3_means'],
                       'haccel_waveletcoeff3_maxs': df_tfd['haccel_waveletcoeff3_maxs'],
                       'vaccel_waveletcoeff3_maxs': df_tfd['vaccel_waveletcoeff3_maxs'],
                       'haccel_waveletcoeff3_mins': df_tfd['haccel_waveletcoeff3_mins'],
                       'vaccel_waveletcoeff3_mins': df_tfd['vaccel_waveletcoeff3_mins'],
                       'haccel_waveletcoeff3_vars': df_tfd['haccel_waveletcoeff3_vars'],
                       'vaccel_waveletcoeff3_vars': df_tfd['vaccel_waveletcoeff3_vars'],
                       'haccel_waveletcoeff3_kurts': df_tfd['haccel_waveletcoeff3_kurts'],
                       'vaccel_waveletcoeff3_kurts': df_tfd['vaccel_waveletcoeff3_kurts'],
                       'haccel_waveletcoeff3_skews': df_tfd['haccel_waveletcoeff3_skews'],
                       'vaccel_waveletcoeff3_skews': df_tfd['vaccel_waveletcoeff3_skews'],

                       'RUL': 10*(len(df_td)-1) - df_td['time']})
    df.to_csv('input/'+bearing + '_final_features.csv',  encoding='utf-8',index = None)


for bearing in testingbearings:
    df_td = pd.read_csv('input/'+bearing + '_time_domain_features.csv')
    df_fd = pd.read_csv('input/'+bearing + '_freq_domain_features.csv')
    df_tfd = pd.read_csv('input/'+bearing + '_timefreq_domain_features.csv')
    
    pos = testingbearings.index(bearing)

    df = pd.DataFrame({'time': df_td['time'],
                       'haccel_mean': df_td['haccel_mean'],
                       'vaccel_mean': df_td['vaccel_mean'],
                       'haccel_max': df_td['vaccel_max'],
                       'vaccel_max': df_td['vaccel_max'],
                       'haccel_min': df_td['vaccel_min'],
                       'vaccel_min': df_td['vaccel_min'],
                       'haccel_std': df_td['vaccel_std'],
                       'vaccel_std': df_td['vaccel_std'],
                       'haccel_var': df_td['vaccel_var'],
                       'vaccel_var': df_td['vaccel_var'],
                       'haccel_kurt': df_td['vaccel_kurt'],
                       'vaccel_kurt': df_td['vaccel_kurt'],
                       'haccel_skew': df_td['vaccel_skew'],
                       'vaccel_skew': df_td['vaccel_skew'],

                       'haccel_freq_means': df_fd['haccel_freq_means'],
                       'vaccel_freq_means': df_fd['vaccel_freq_means'],
                       'haccel_freq_maxs': df_fd['haccel_freq_maxs'],
                       'vaccel_freq_maxs': df_fd['vaccel_freq_maxs'],
                       'haccel_freq_stds': df_fd['haccel_freq_stds'],
                       'vaccel_freq_stds': df_fd['vaccel_freq_stds'],
                       'haccel_freq_vars': df_fd['haccel_freq_vars'],
                       'vaccel_freq_vars': df_fd['vaccel_freq_vars'],
                       'haccel_freq_kurts': df_fd['haccel_freq_kurts'],
                       'vaccel_freq_kurts': df_fd['vaccel_freq_kurts'],
                       'haccel_freq_skews': df_fd['haccel_freq_skews'],
                       'vaccel_freq_skews': df_fd['vaccel_freq_skews'],

                       'haccel_waveletcoeff1_means': df_tfd['haccel_waveletcoeff1_means'],
                       'vaccel_waveletcoeff1_means': df_tfd['vaccel_waveletcoeff1_means'],
                       'haccel_waveletcoeff1_maxs': df_tfd['haccel_waveletcoeff1_maxs'],
                       'vaccel_waveletcoeff1_maxs': df_tfd['vaccel_waveletcoeff1_maxs'],
                       'haccel_waveletcoeff1_mins': df_tfd['haccel_waveletcoeff1_mins'],
                       'vaccel_waveletcoeff1_mins': df_tfd['vaccel_waveletcoeff1_mins'],
                       'haccel_waveletcoeff1_vars': df_tfd['haccel_waveletcoeff1_vars'],
                       'vaccel_waveletcoeff1_vars': df_tfd['vaccel_waveletcoeff1_vars'],
                       'haccel_waveletcoeff1_kurts': df_tfd['haccel_waveletcoeff1_kurts'],
                       'vaccel_waveletcoeff1_kurts': df_tfd['vaccel_waveletcoeff1_kurts'],
                       'haccel_waveletcoeff1_skews': df_tfd['haccel_waveletcoeff1_skews'],
                       'vaccel_waveletcoeff1_skews': df_tfd['vaccel_waveletcoeff1_skews'],

                       'haccel_waveletcoeff2_means': df_tfd['haccel_waveletcoeff2_means'],
                       'vaccel_waveletcoeff2_means': df_tfd['vaccel_waveletcoeff2_means'],
                       'haccel_waveletcoeff2_maxs': df_tfd['haccel_waveletcoeff2_maxs'],
                       'vaccel_waveletcoeff2_maxs': df_tfd['vaccel_waveletcoeff2_maxs'],
                       'haccel_waveletcoeff2_mins': df_tfd['haccel_waveletcoeff2_mins'],
                       'vaccel_waveletcoeff2_mins': df_tfd['vaccel_waveletcoeff2_mins'],
                       'haccel_waveletcoeff2_vars': df_tfd['haccel_waveletcoeff2_vars'],
                       'vaccel_waveletcoeff2_vars': df_tfd['vaccel_waveletcoeff2_vars'],
                       'haccel_waveletcoeff2_kurts': df_tfd['haccel_waveletcoeff2_kurts'],
                       'vaccel_waveletcoeff2_kurts': df_tfd['vaccel_waveletcoeff2_kurts'],
                       'haccel_waveletcoeff2_skews': df_tfd['haccel_waveletcoeff2_skews'],
                       'vaccel_waveletcoeff2_skews': df_tfd['vaccel_waveletcoeff2_skews'],
                       
                       'haccel_waveletcoeff3_means': df_tfd['haccel_waveletcoeff3_means'],
                       'vaccel_waveletcoeff3_means': df_tfd['vaccel_waveletcoeff3_means'],
                       'haccel_waveletcoeff3_maxs': df_tfd['haccel_waveletcoeff3_maxs'],
                       'vaccel_waveletcoeff3_maxs': df_tfd['vaccel_waveletcoeff3_maxs'],
                       'haccel_waveletcoeff3_mins': df_tfd['haccel_waveletcoeff3_mins'],
                       'vaccel_waveletcoeff3_mins': df_tfd['vaccel_waveletcoeff3_mins'],
                       'haccel_waveletcoeff3_vars': df_tfd['haccel_waveletcoeff3_vars'],
                       'vaccel_waveletcoeff3_vars': df_tfd['vaccel_waveletcoeff3_vars'],
                       'haccel_waveletcoeff3_kurts': df_tfd['haccel_waveletcoeff3_kurts'],
                       'vaccel_waveletcoeff3_kurts': df_tfd['vaccel_waveletcoeff3_kurts'],
                       'haccel_waveletcoeff3_skews': df_tfd['haccel_waveletcoeff3_skews'],
                       'vaccel_waveletcoeff3_skews': df_tfd['vaccel_waveletcoeff3_skews'],

                       'RUL': finalRUL[pos]+10*(len(df_td)-1) - df_td['time']})
    df.to_csv('input/'+bearing + '_final_features.csv',  encoding='utf-8',index = None)

#Split the dataset into training sets and testing sets
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