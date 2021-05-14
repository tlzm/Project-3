import tensorflow
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_path = 'model/regression_model_v0.h5'
train_data = pd.read_csv('input/LearningSet1.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')

n_bearing = train_data['id'].unique().max()

# time windows
sequence_length = 40

def reshapeFeatures(id_df, seq_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.

    """
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] # 输出行数
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop, :]

# pick the feature columns 
feature_col = ['f' + str(i) for i in range(1,63)]
#print(feature_col)

# generator for the sequences
fea_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, feature_col)) for id in range(1, n_bearing + 1))

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
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_bearing + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)

print(label_array.shape)

# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

nb_features = fea_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(input_shape=(sequence_length, nb_features), units=128, return_sequences=True, name="lstm_0"))
model.add(Dropout(0.2, name="dropout_0")) 
model.add(LSTM(units=64, return_sequences=True, name="lstm_1"))
model.add(Dropout(0.2, name="dropout_1"))
#model.add(Dense(units=32, name="dense_0"))
#model.add(Dropout(0.2, name="dropout_2"))
model.add(Dense(units=16, name="dense_1"))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(units=nb_out, name="dense_2"))
model.add(Activation("relu", name="activation_0"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[root_mean_squared_error, 'mae'])

print(model.summary())

epochs = 100
batch_size = 200

# fit the network
history = model.fit(fea_array, label_array, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))