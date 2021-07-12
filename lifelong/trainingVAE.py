from keras.layers.merge import add
import tensorflow
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM,Conv1D,Reshape,Flatten,BatchNormalization,Permute,TimeDistributed,Attention,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.python.keras.layers.core import Flatten
from keras.optimizers import Adam

output_path = 'model/VAE_v0.h5'
output_path1 = 'model/encoder_model_v0.h5'
output_path2 = 'model/generater_model_v0.h5'
train_data = pd.read_csv('input/LearningSet1.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')



# time windows
sequence_length = 2500

n_time = int(train_data['id'].unique().max())

def reshapeFeatures(id_df, seq_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    """
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] # 输出行数
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop,:]

# pick the feature columns 
feature_col = ['haccel','vaccel']
#print(feature_col)

# generator for the sequences
fea_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, feature_col)) for id in range(1, n_time + 1))

# generate sequences and convert to numpy array
fea_array = np.concatenate(list(fea_gen)).astype(np.float32)

print("The data set has now shape: {} entries, {} time windows and {} features.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))

# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1: num_elements, :]

# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_time + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)

print(label_array.shape)

# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


def exps(y_true, y_pred):
        return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000


nb_features = fea_array.shape[2]
nb_out = label_array.shape[1]


original_dim = 2500

intermediate_dim1 = 2000

intermediate_dim2 = 1500

latent_dim = 1000

epsilon_std = 1.0

#my tips:encoding

x = keras.Input(shape=(sequence_length, nb_features))

x1 = Permute((2,1))(x)

h1 = Dense(intermediate_dim1, activation='relu')(x1)

h2 = Dense(intermediate_dim2, activation='relu')(h1)

z_mean = Dense(latent_dim)(h2)

z_log_var = Dense(latent_dim)(h2)

#my tips:Gauss sampling,sample Z

def sampling(args): 

    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=(sequence_length,latent_dim), mean=0.)

    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend

# my tips:get sample z(encoded)

z = Lambda(sampling, output_shape=(sequence_length,latent_dim))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later

h_decoded1 =Dense(intermediate_dim2, activation='relu')(z)

h_decoded2 =Dense(intermediate_dim1, activation='relu')(h_decoded1)

x_decoded_mean = Dense(original_dim, activation='relu')(h_decoded2)

vae = keras.Model(x, x_decoded_mean)

vae.compile(optimizer=Adam(lr=1e-2), loss='mse')

vae.fit(fea_array, fea_array,epochs=5000,batch_size=512,shuffle=True,validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )
 

# build a model to project inputs on the latent space

encoder = keras.Model(x, z_mean)

generater = keras.Model(z_mean,x_decoded_mean)

encoder.save(output_path1)

generater.save(output_path2)