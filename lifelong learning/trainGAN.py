from logging import disable
from keras.layers.merge import add
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU, Dropout, LSTM,Conv1D,Reshape,Flatten,BatchNormalization,Permute,TimeDistributed,Attention,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import adam_v2
import os
from keras.models import load_model


output_path1 = 'model/Discriminator_v0.h5'
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

latent_dim=10

# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


opt = adam_v2.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
dopt =adam_v2.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


gen=Sequential()


gen.add(Dense(100))
gen.add(Dropout(0.2))
gen.add(Dense(500))
gen.add(Dropout(0.2))
gen.add(Dense(2541))

Gen_input = keras.Input(shape=(fea_length,latent_dim))

GR=gen(Gen_input)

generator= keras.Model(Gen_input,GR)
generator.compile(loss='mse', optimizer=opt)

if os.path.isfile(output_path2):
#    model.load_weights(output_path)
    generators=load_model(output_path2,custom_objects={'root_mean_squared_error':root_mean_squared_error})
    print('generative model loaded')


dis=Sequential()

dis.add(Flatten())
dis.add(Dense(512))
dis.add(LeakyReLU(alpha=0.2))
dis.add(Dense(16))
dis.add(LeakyReLU(alpha=0.2))
dis.add(Dense(1, activation='sigmoid'))

dis_input = keras.Input(shape=(fea_length,sequence_length))

validity = dis(dis_input)

discriminator = keras.Model(dis_input,validity)
discriminator.compile(loss='mse', optimizer=dopt)

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)


gan_input = keras.Input(shape=(fea_length,latent_dim))
H = generator(gan_input)
gan_V = discriminator(H)
GAN = keras.Model(gan_input, gan_V)
GAN.compile(loss='mse', optimizer=opt)


batch_size=512
epochs=8000

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, fea_array.shape[0], batch_size)
    imgs = fea_array[idx]

    noise = np.random.randn (batch_size, fea_length,latent_dim)
    
    gen_imgs = generator.predict(noise)

    array = np.concatenate((imgs,gen_imgs))

    label = np.concatenate((valid,fake))

    d_loss= discriminator.train_on_batch(array,label)

    discriminator.save(output_path1)

    noise = np.random.randn (batch_size, fea_length,latent_dim)

   # noise = np.random.normal(0, 1, (batch_size, fea_length,latent_dim))
    g_loss = GAN.train_on_batch(noise, valid)

    generator.save(output_path2)
    print(epoch,d_loss,g_loss)

