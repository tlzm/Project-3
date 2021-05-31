from keras.layers.merge import add
import tensorflow
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM,Conv1D,Reshape,Flatten,BatchNormalization,Permute,TimeDistributed,Attention,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add,Conv2D
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.python.keras.layers.core import Flatten



output_path = 'model/regression_model_v0.h5'
train_data = pd.read_csv('input/LearningSet1.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')



# time windows
nb_freqs = 258

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
time_col = ['t' + str(i) for i in range(1,22)]
#print(feature_col)

# generator for the sequences
fea_gen = (list(reshapeFeatures(train_data[train_data['id']==id], nb_freqs, time_col)) for id in range(1, n_time + 1))

# generate sequences and convert to numpy array
fea_array = np.concatenate(list(fea_gen)).astype(np.float32)

print("The data set has now shape: {} entries, {} time windows and {} features.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))

# function to generate label
def reshapeLabel(id_df, seq_length=nb_freqs, label=['RUL']):
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


SINGLE_ATTENTION_VECTOR = True

def attention_3d_block(inputs,TIME_STEPS):

    input_dim = int(inputs.shape[2])  # input_dim = 100

#    TIME_STEPS = int(inputs.shape[1])

    a = Permute((2, 1))(inputs)  # a.shape = (?, 100, ?)

    a = Reshape((input_dim, TIME_STEPS))(a)  # a.shape = (?, 100, 30)this line is not useful. It's just to know which dimension is what.

    a = Dense(TIME_STEPS, activation='softmax')(a)  # a.shape = (?, 100, 30)

    if SINGLE_ATTENTION_VECTOR:

        a = Lambda(lambda x: K.mean(x, axis=1))(a)  # a.shape = (?, 30)

        a = RepeatVector(input_dim)(a)  # a.shape = (?, 100, 30) RepeatVector层将输入重复n次

    a_probs = Permute((2, 1))(a)  # a.shape = (?, 30, 100)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')

    output_attention_mul = multiply([inputs, a_probs])  # [?, 30, 100]

    return output_attention_mul


nb_times = fea_array.shape[2]
nb_out = label_array.shape[1]

x = keras.Input(shape=(nb_freqs,nb_times))
x1=attention_3d_block(x,nb_freqs)

x2=Permute((2,1))(x1)
x3=Reshape((nb_times,nb_freqs,1))(x2)

x4=TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,padding='causal',activation='swish'))(x3)
x5=LayerNormalization()(x4)
x6=TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,dilation_rate=2,padding='causal',activation='swish'))(x5)
x7=LayerNormalization()(x6)
x8=TimeDistributed(SeparableConv1D(1, kernel_size=1, padding='same'))(x7)

x9=Reshape((nb_times,nb_freqs))(x8)
x10=Permute((2,1))(x9)
x11=keras.layers.Add()([x10,x])

y1=Permute((2,1))(x)
y2=attention_3d_block(y1,nb_times)

y3=Permute((2,1))(y2)
y4=Reshape((nb_freqs,nb_times,1))(y3)

y5=TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,padding='causal',activation='swish'))(y4)
y6=LayerNormalization()(y5)
y7=TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,dilation_rate=2,padding='causal',activation='swish'))(y6)
y8=LayerNormalization()(y7)
y9=TimeDistributed(SeparableConv1D(1, kernel_size=1, padding='same'))(y8)

y10=Reshape((nb_freqs,nb_times))(y9)

y11=keras.layers.Add()([y10,x])
y=keras.layers.Concatenate(axis= 1)([x11,y11])

attentionDTCN = keras.Model(x,y)


model = Sequential()
#model.add(Reshape((50,100),input_shape=(sequence_length, nb_features)))
model.add(attentionDTCN)
#model.add(Permute((2,1)))
#model.add(Reshape((2,2500,1)))
#model.add(TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,padding='causal',activation='swish')))
#model.add(LayerNormalization())
#model.add(TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,dilation_rate=2,padding='causal',activation='swish')))
#model.add(LayerNormalization())
#model.add(TimeDistributed(Conv1D(filters=64,kernel_size=5,strides=1,dilation_rate=4,padding='causal',activation='swish')))
#model.add(LayerNormalization())
#model.add(TimeDistributed(SeparableConv1D(1, kernel_size=1, padding='same')))
#model.add(Reshape((2,2500)))
model.add(Flatten())
model.add(Dense(units=200, name="dense_0"))
model.add(Dropout(0.2, name="dropout_1"))
#model.add(Dense(units=100, name="dense_1"))
#model.add(Dropout(0.2, name="dropout_2"))
model.add(Dense(units=20, name="dense_2"))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(units=nb_out, name="dense_3"))
model.add(Activation("swish", name="activation_0"))
model.compile(loss='mse', optimizer='adam', metrics=[root_mean_squared_error,exps, 'mae'])

print(model.summary())

epochs = 5000
batch_size = 200

# fit the network
history = model.fit(fea_array, label_array, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2000,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_root_mean_squared_error',save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))
