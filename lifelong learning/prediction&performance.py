import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from keras.models import load_model
import keras.backend as K
from keras.layers import Dense, Dropout, LSTM,BatchNormalization, Activation,Conv1D,multiply,Flatten,TimeDistributed,Attention,Permute

""" output_path = 'model/ATCN_regression_model_v0.h5'

sequence_length=2511 """

output_path = 'model/RCNN_regression_model_v0.h5'

sequence_length=2560


def reshapeFeatures(id_df, seq_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    """
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] # 输出行数
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop,:]

# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1: num_elements, :]

def root_mean_squared_error(y_true, y_pred): 
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def exps(y_true, y_pred):
    return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000

# pick the feature columns 
feature_col = ['haccel','vaccel']

if os.path.isfile(output_path):
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error,'exps':exps})

for i in range(1,8):
    test_data=pd.read_csv("input/LearningSet"+str(i)+".csv")
    n_time = int(test_data['id'].unique().max())
    
    # generator for the sequences
    fea_gen = (list(reshapeFeatures(test_data[test_data['id']==id], sequence_length, feature_col)) for id in range(1, n_time + 1))
    fea_array = np.concatenate(list(fea_gen)).astype(np.float32)
    print("The data set has now shape: {} entries, {} time windows and {} features.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))

    # generate labels
    label_gen = [reshapeLabel(test_data[test_data['id']==id]) for id in range(1, n_time + 1)]
    label_array = np.concatenate(label_gen).astype(np.float32)
    print(label_array.shape)

    y_pred_test = estimator.predict(fea_array)
    y_pred_test = np.transpose(y_pred_test) 
    y_true_test = label_array
    y_true_test = np.transpose(y_true_test) 

    print(root_mean_squared_error(y_true_test,y_pred_test))


""" 


#print(feature_col)

# generator for the sequences
fea_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, feature_col)) for id in range(1, n_time + 1))

# generate sequences and convert to numpy array
fea_array = np.concatenate(list(fea_gen)).astype(np.float32)

print("The data set has now shape: {} entries, {} time windows and {} features.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))



# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_time + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)

print(label_array.shape)



# load the model
if os.path.isfile(output_path):
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error,'exps':exps})

    y_pred_test = estimator.predict(seq_array_test_last)
    
    y_true_test = label_array_test_last
    
    print(y_pred_test,y_true_test)

    # test metrics
#    scores_test = estimator.evaluate(seq_array_test_last,label_array_test_last,verbose = False)
#    print('\nMSE: {}'.format(scores_test[0]))
#print(root_mean_squared_error(y_true_test,y_pred_test)) """
