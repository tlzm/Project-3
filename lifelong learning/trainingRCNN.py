from keras.layers.merge import add
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D, Dropout, Reshape,Flatten,BatchNormalization,Permute,TimeDistributed,ConvLSTM1D,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from keras.optimizers import adam_v2

input_path = 'model/generater_model_v0.h5'
output_path = 'model/RCNN_regression_model_v0.h5'
train_data = pd.read_csv('input/LearningSet1.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')

# time windows
sequence_length = 2541

n_time = int(train_data['id'].unique().max())

def reshapeFeatures(id_df, seq_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    """
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

x = keras.Input(shape=(sequence_length, nb_features))
x1=Reshape((sequence_length, nb_features,1))(x)

reshape = keras.Model(x,x1)


model = Sequential()
model.add(reshape)
model.add(ConvLSTM1D(8,2,padding='same',return_sequences=True))
model.add(LayerNormalization())
model.add(ConvLSTM1D(8,2,return_sequences=True))
model.add(LayerNormalization())
model.add(Flatten())
model.add(Dense(units=200, name="dense_0"))
model.add(Dropout(0.2, name="dropout_1"))
model.add(Dense(units=10, name="dense_2"))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(units=nb_out,activation='relu', name="dense_3"))
#model.add(Activation("swish", name="activation_0"))

if os.path.isfile(output_path):
#    model.load_weights(output_path)
    model=load_model(output_path,custom_objects={'root_mean_squared_error':root_mean_squared_error,'exps':exps})
optimizer =adam_v2.Adam(learning_rate=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
model.compile(optimizer=optimizer,loss='mse', metrics=[root_mean_squared_error])

print(model.summary())

epochs = 100
batch_size = 64

class SaveModel(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_root_mean_squared_error") + logs.get("root_mean_squared_error") 
        self.D_loss.append(logs.get("val_root_mean_squared_error")+logs.get("root_mean_squared_error"))
        if current_D_loss <= min(self.D_loss):
            print('Find lowest val_loss. Saving entire model.')
            model.save(output_path) # < ----- Here 
save_model = SaveModel()

# fit the network
history = model.fit(fea_array, label_array, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.2, verbose=1,
          callbacks = save_model
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))
