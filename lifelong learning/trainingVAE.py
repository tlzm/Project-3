from keras.layers.merge import add
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM,Conv1D,Reshape,Flatten,BatchNormalization,Permute,TimeDistributed,Attention,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
import os
from keras.models import load_model

output_path = 'model/VAE_v0.h5'
output_path1 = 'model/encoder_model_v0.h5'
output_path2 = 'model/generater_model_v0.h5'
train_data = pd.read_csv('input/LearningSet1.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')



# time windows
sequence_length = 2541
fea_length = 2

n_time = int(train_data['id'].unique().max())

def reshapeFeatures(id_df, sequence_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    """
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] #
    for start, stop in zip(range(0, num_elements-sequence_length+1), range(sequence_length, num_elements+1)):
        yield np.transpose(data_matrix[start:stop,:])

# pick the feature columns 
feature_col = ['haccel','vaccel']
#print(feature_col)

# generator for the sequences
fea_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, feature_col)) for id in range(1, n_time + 1))
# generate sequences and convert to numpy array

fea_array = np.concatenate(list(fea_gen)).astype(np.float32)

print("The data set has now shape: {} entries, {} features and {} time windows.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))


# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


def exps(y_true, y_pred):
        return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000




original_dim = 2541

intermediate_dim = 500

intermediate_dim1 = 100

latent_dim = 10

epsilon_std = 1.0

#my tips:encoding

x = keras.Input(shape=(fea_length,sequence_length))

h = Dense(intermediate_dim, activation='relu')(x)

hd = Dropout(0.2)(h)

h1 = Dense(intermediate_dim1, activation='relu')(hd)

h1d = Dropout(0.2)(h1)

z_mean = Dense(latent_dim)(h1d)

z_log_var = Dense(latent_dim)(h1d)

#my tips:Gauss sampling,sample Z

def sampling(args): 

    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend

# my tips:get sample z(encoded)

z = Lambda(sampling, output_shape=(fea_length,latent_dim))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later

decoder_h = Dense(intermediate_dim1, activation='relu')

d1=Dropout(0.2)

decoder_h1 = Dense(intermediate_dim, activation='relu')

d2=Dropout(0.2)

decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded =decoder_h(z)

h_decodedd = d1(h_decoded)

h1_decoded = decoder_h1(h_decodedd)

h1_decodedd = d2(h1_decoded)

x_decoded_mean = decoder_mean(h1_decodedd)

vae = keras.Model(x, x_decoded_mean)




optimizer =adam_v2.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
vae.compile(optimizer=optimizer,loss='mse')


encoder = keras.Model(x, z_mean)


decoder_input = keras.Input(shape=(fea_length,latent_dim))
_h_decoded = decoder_h(decoder_input)
_h_decodedd = d1(_h_decoded)
_h1_decoded = decoder_h1(_h_decodedd)
_h1_decodedd = d2(_h1_decoded)
_x_decoded_mean = decoder_mean(_h1_decodedd)

generator = keras.Model(decoder_input, _x_decoded_mean)

vae.fit(fea_array, fea_array,epochs=500,batch_size=512,shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )


encoder.save(output_path1)

generator.save(output_path2)


