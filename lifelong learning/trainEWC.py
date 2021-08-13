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
from copy import deepcopy
from tensorflow.keras.losses import sparse_categorical_crossentropy,mse

input_path = 'model/generater_model_v0.h5'
output_path = 'model/ATCN_regression_model_v0.h5'
train_data_old = pd.read_csv('input/LearningSet6.csv')
train_data = pd.read_csv('input/LearningSet7.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')


# time windows
sequence_length = 2511

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
fea_gen_old = (list(reshapeFeatures(train_data_old[train_data_old['id']==id], sequence_length, feature_col)) for id in range(1, n_time + 1))
# generate sequences and convert to numpy array
fea_array = np.concatenate(list(fea_gen)).astype(np.float32)
fea_array_old = np.concatenate(list(fea_gen_old)).astype(np.float32)
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
label_gen_old = [reshapeLabel(train_data_old[train_data_old['id']==id]) for id in range(1, n_time + 1)]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array_old = np.concatenate(label_gen_old).astype(np.float32)
print(label_array.shape)

# EWC

def fisher_matrix(model, dataset, samples):
    inputs, labels = dataset
    weights = model.trainable_weights
    variance = [tf.zeros_like(tensor) for tensor in weights]

    for _ in range(samples):
        # Select a random element from the dataset.
        index = np.random.randint(len(inputs))
        data = inputs[index]

        # When extracting from the array we lost a dimension so put it back.
        data = tf.expand_dims(data, axis=0)

        # Collect gradients.
        with tf.GradientTape() as tape:
            output = model(data)
            log_likelihood = tf.math.log(output)

        gradients = tape.gradient(log_likelihood, weights)

        # If the model has converged, we can assume that the current weights
        # are the mean, and each gradient we see is a deviation. The variance is
        # the average of the square of this deviation.
        variance = [var + (grad ** 2) for var, grad in zip(variance, gradients)]

    fisher_diagonal = [tensor / samples for tensor in variance]
    return fisher_diagonal


def ewc_loss(lam, model, dataset, samples):

    optimal_weights = deepcopy(model.trainable_weights)
    fisher_diagonal = fisher_matrix(model, dataset, samples)

    def loss_fn(new_model):
        # We're computing:
        # sum [(lambda / 2) * F * (current weights - optimal weights)^2]
        loss = 0
        current = new_model.trainable_weights
        for f, c, o in zip(fisher_diagonal, current, optimal_weights):
            loss += tf.reduce_mean(f * ((c - o) ** 2))

        return loss * (lam / 2)

    return loss_fn




def custom_loss(y_true, y_pred):
    loss = root_mean_squared_error(y_true, y_pred)/100
    extra_losses = regularisers
    for fn in extra_losses:
        loss += fn(model)

    return loss



def apply_mask(gradients, mask):

    if mask is not None:
        gradients = [grad * tf.cast(mask, tf.float32)
                     for grad, mask in zip(gradients, mask)]
    return gradients


# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


def exps(y_true, y_pred):
        return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000


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
    model=load_model(output_path,custom_objects={'custom_loss':custom_loss,'root_mean_squared_error':root_mean_squared_error,'exps':exps})
    print('model loaded')
optimizer =adam_v2.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

regularisers = []
gradient_mask = []

loss_fn = ewc_loss(0.5, model, (fea_array_old, label_array_old),100)
regularisers.append(loss_fn)
model.compile(optimizer=optimizer,loss='mse', metrics=[root_mean_squared_error,custom_loss])
training_data = (fea_array, label_array)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(256)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))






print(model.summary())

epochs = 200
batch_size = 128                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

class SaveModel(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_custom_loss")/2 + logs.get("custom_loss") 
        self.D_loss.append(logs.get("val_custom_loss")/2+logs.get("custom_loss"))
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