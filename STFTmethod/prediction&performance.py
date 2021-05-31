import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from keras.models import load_model
import keras.backend as K
from keras.layers import Dense, Dropout, LSTM,BatchNormalization, Activation,Conv1D,multiply,Flatten,TimeDistributed,Attention

output_path = 'model/regression_model_v0.h5'
nb_freqs = 258
test_data = pd.read_csv("input/TestingSet1.csv")
#test_data = pd.read_csv("input/TestingSet2.csv")
#test_data = pd.read_csv("input/TestingSet3.csv")

#n_time = int(test_data['id'].unique().max())

n_time = int(test_data['bearing'].unique().max())

time_cols = ['t' + str(i) for i in range(1,22)]


seq_array_test_last = [test_data[test_data['id']==id][time_cols].values[-nb_freqs:] 
                       for id in range(1,n_time + 1) if len(test_data[test_data['id']==id]) >= nb_freqs]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)




y_mask = [len(test_data[test_data['bearing']==bearing]) >= nb_freqs for bearing in test_data['bearing'].unique()]
#y_mask = [len(test_data[test_data['id']==id]) >= nb_freqs for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('bearing')['RUL'].nth(-1)[y_mask].values
#label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

def root_mean_squared_error(y_true, y_pred): 
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def exps(y_true, y_pred):
    return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000

# load the model
if os.path.isfile(output_path):
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error,'exps':exps})

    y_pred_test = estimator.predict(seq_array_test_last)
    
    y_true_test = label_array_test_last
    
    print(y_pred_test,y_true_test)

    # test metrics
#    scores_test = estimator.evaluate(seq_array_test_last,label_array_test_last,verbose = False)
#    print('\nMSE: {}'.format(scores_test[0]))
#print(root_mean_squared_error(y_true_test,y_pred_test))
