import keras
from keras import optimizers
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense,add, Dropout,Conv1D,LeakyReLU,Reshape,Flatten,BatchNormalization,Permute,SeparableConv1D,Lambda,RepeatVector,multiply,Multiply,LayerNormalization,Add
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import tensorflow as tf
from keras.optimizers import adam_v2

input_path = 'model/generater_model_v0.h5'
output_path1 = 'model/encoder_model_v0.h5'
output_path2 = 'model/VAE_v0.h5'
output_path3 = 'model/Discriminator_v0.h5'
output_path4 = 'model/generater_model_v0.h5'
output_path = 'model/ATCN_regression_model_v0.h5'
train_data = pd.read_csv('input/LearningSet3.csv')
#train_data = pd.read_csv('input/LearningSet2.csv')
#train_data = pd.read_csv('input/LearningSet3.csv')

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


def exps(y_true, y_pred):
        return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000
# time windows
sequence_length = 2541

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
    input = [np.random.randn(2,10)]
    for i in range(entries-1):
        x = [np.random.randn(2,10)]
        input=np.append(input,x,axis=0)
#        y=generators.predict(input)
 #       y1=Permute((2,1))(y)
  #      z=model.predict(y1)
   #     print(z)
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
batch_size = 512                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

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

fea_length=2

fea_array_gen=Permute((2,1))(fea_array)

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



vae.fit(fea_array_gen, fea_array_gen,epochs=500,batch_size=512,shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path2, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

encoder.save(output_path1)

generator.save(input_path)


n_time = int(train_data['id'].unique().max())

opt = adam_v2.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
dopt =adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


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
    generators=load_model(input_path,custom_objects={'root_mean_squared_error':root_mean_squared_error})
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

if os.path.isfile(output_path3):
#    model.load_weights(output_path)
    generators=load_model(output_path3,custom_objects={'root_mean_squared_error':root_mean_squared_error})
    print('discriminator model loaded')


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
epochs=6000

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, fea_array.shape[0], batch_size)
    imgs = fea_array[idx]

    imgs_re = Permute((2,1))(imgs)

    noise = np.random.randn (batch_size, fea_length,latent_dim)
    
    gen_imgs = generator.predict(noise)

    array = np.concatenate((imgs_re,gen_imgs))

    label = np.concatenate((valid,fake))

    d_loss= discriminator.train_on_batch(array,label)


    discriminator.save(output_path3)

    noise = np.random.randn (batch_size, fea_length,latent_dim)
    g_loss = GAN.train_on_batch(noise, valid)

    generator.save(input_path)
    print(epoch,d_loss,g_loss)
