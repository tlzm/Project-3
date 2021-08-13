import keras
from keras import optimizers
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense,add, Dropout,Conv1D,Reshape,Flatten,BatchNormalization,Permute,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import tensorflow as tf
from keras.optimizers import adam_v2

input_path = 'model/generater_model_v0.h5'
output_path = 'model/ATCN_regression_model_v0.h5'
train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


def exps(y_true, y_pred):
        return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000
# time windows
sequence_length = 2511

n_time = int(train_data['id'].unique().max())

def reshapeFeatures(id_df, seq_length, Feature):
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] 
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
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1: num_elements, :]

# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_time + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)

print(label_array.shape)

entries = fea_array.shape[0]


if os.path.isfile(input_path):
#    model.load_weights(output_path)
    generators=load_model(input_path,custom_objects={'root_mean_squared_error':root_mean_squared_error,'exps':exps})
    model=load_model(output_path,custom_objects={'root_mean_squared_error':root_mean_squared_error,'exps':exps})
    print('generative model loaded')
    input = [2*np.random.rand(2,1500)-1]
    for i in range(2):
        x = [2*np.random.rand(2,1500)-1]
        input=np.append(input,x,axis=0)
    y=generators.predict(input)
    y1=Permute((2,1))(y)
    z=model.predict(y1)
    fea_array=np.append(fea_array,y1,axis=0)
    label_array=np.append(label_array,z,axis=0)


print("The combine data set has now shape: {} entries, {} time windows and {} features.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))
print(label_array.shape)

# MODEL




SINGLE_ATTENTION_VECTOR = True

def attention_3d_block(inputs,TIME_STEPS):

    input_dim = int(inputs.shape[2]) 

    a = Permute((2, 1))(inputs) 

    a = Reshape((input_dim, TIME_STEPS))(a) 

    a = Dense(TIME_STEPS, activation='softmax')(a)  

    if SINGLE_ATTENTION_VECTOR:

        a = Lambda(lambda x: K.mean(x, axis=1))(a)  

        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a) 

    output_attention_mul = multiply([inputs, a_probs])  

    return output_attention_mul


def ResBlock(x,filters,kernel_size,dilation_rate):
    r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='swish')(x) 
    r=LayerNormalization()(r)
    r=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='swish')(r)
    r=LayerNormalization()(r)
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,kernel_size,padding='same')(x) 
    o=add([r,shortcut])
#    o=LayerNormalization()(o)
    o=Activation('relu')(o)  
    return o



nb_features = fea_array.shape[2]
nb_out = label_array.shape[1]

x = keras.Input(shape=(sequence_length, nb_features))
TIME_STEPS = int(x.shape[1])
x1=attention_3d_block(x,TIME_STEPS)
x2=ResBlock(x1,filters=32,kernel_size=5,dilation_rate=1)
x3=ResBlock(x2,filters=32,kernel_size=5,dilation_rate=2)
x4=ResBlock(x3,filters=16,kernel_size=5,dilation_rate=4)

attentionTCN = keras.Model(x,x4)


model = Sequential()
model.add(attentionTCN)
model.add(Flatten())
#model.add(Dense(units=200, name="dense_0"))
#model.add(Dropout(0.2, name="dropout_1"))
model.add(Dense(units=200, name="dense_1"))
model.add(Dropout(0.2, name="dropout_1")) 
model.add(Dense(units=10, name="dense_2"))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(units=nb_out,activation='relu', name="dense_3"))
#model.add(Activation("swish", name="activation_0"))
if os.path.isfile(output_path):
#    model.load_weights(output_path)
    model=load_model(output_path,custom_objects={'root_mean_squared_error':root_mean_squared_error,'exps':exps})
    print('model loaded')
optimizer =adam_v2.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
model.compile(optimizer=optimizer,loss='mse', metrics=[root_mean_squared_error])

print(model.summary())

epochs = 500
batch_size = 256                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

class SaveModel(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_root_mean_squared_error")/2 + logs.get("root_mean_squared_error") 
        self.D_loss.append(logs.get("val_root_mean_squared_error")/2+logs.get("root_mean_squared_error"))
        if current_D_loss <= min(self.D_loss):
            print('Find lowest val_loss. Saving entire model.')
            model.save(output_path) # < ----- Here 

save_model = SaveModel()
# fit the network
history = model.fit(fea_array, label_array, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.2, verbose=1,
          callbacks = save_model#[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=800,verbose=0, mode='min'),
#                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_root_mean_squared_error',save_best_only=True, mode='min', verbose=0)]
#                       keras.callbacks.LambdaCallback(on_epoch_end=saveModel)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))


#generator
output_path1 = 'model/encoder_model_v0.h5'
output_path2 = 'model/VAE_v0.h5'

fea_length=2

fea_array_gen=Permute((2,1))(fea_array)

original_dim = 2511

intermediate_dim = 2000

latent_dim = 1500

epsilon_std = 1.0

#my tips:encoding

x = keras.Input(shape=(fea_length,sequence_length))


h = Dense(intermediate_dim, activation='relu')(x)

z_mean = Dense(latent_dim)(h)

z_log_var = Dense(latent_dim)(h)

#my tips:Gauss sampling,sample Z

def sampling(args): 

    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=(fea_length,latent_dim), mean=0.)

    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend

# my tips:get sample z(encoded)

z = Lambda(sampling, output_shape=(fea_length,latent_dim))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later

decoder_h = Dense(intermediate_dim, activation='relu')

decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded =decoder_h(z)

x_decoded_mean = decoder_mean(h_decoded)

vae = keras.Model(x, x_decoded_mean)

optimizer =adam_v2.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
vae.compile(optimizer=optimizer,loss='mse')

if os.path.isfile(output_path2):
    vae.load_weights(output_path2)
    

vae.fit(fea_array_gen, fea_array_gen,epochs=500,batch_size=128,shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )
 
vae=keras.Model(x,x_decoded_mean)

vae.save_weights(output_path)
# build a model to project inputs on the latent space

encoder = keras.Model(x, z_mean)


decoder_input = keras.Input(shape=(fea_length,latent_dim))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)

generator = keras.Model(decoder_input, _x_decoded_mean)

encoder.save(output_path1)

generator.save(input_path)