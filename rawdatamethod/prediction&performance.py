import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from keras.models import load_model
import keras.backend as K
from keras.layers import Dense, Dropout, LSTM,BatchNormalization, Activation,Conv1D,multiply,Flatten

output_path = 'model/regression_model_v0.h5'
sequence_length = 2
test_data = pd.read_csv("input/TestingSet1.csv")
#test_data = pd.read_csv("input/TestingSet2.csv")
#test_data = pd.read_csv("input/TestingSet3.csv")

n_time = int(test_data['id'].unique().max())

n_bearing = int(test_data['bearing'].unique().max())

feature_cols = ['haccel','vaccel']


seq_array_test_last = [test_data[test_data['id']==id][feature_cols].values[-sequence_length:] 
                       for id in range(1,n_time + 1) if len(test_data[test_data['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

print("This is the shape of the test set: {} TWs, {} cycles and {} features.".format(
    seq_array_test_last.shape[0], seq_array_test_last.shape[1], seq_array_test_last.shape[2]))

print("There is only {} TWs out of {} as {} TWs didn't have more than {} cycles.".format(
    seq_array_test_last.shape[0], n_time, n_time - seq_array_test_last.shape[0], sequence_length))




y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]

label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values

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
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last,verbose = 2)
    print('\nMSE: {}'.format(scores_test[0]))

