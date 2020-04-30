# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 02:42:35 2019

@author: Harish
"""

import glob
import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape
from keras.layers import AveragePooling3D
from keras.layers import Bidirectional


path_train_x =r'D:\8sem_project\cnn_lstm_10_10_data\train_x' # use your path
path_test_x=r'D:\8sem_project\cnn_lstm_10_10_data\test_x'

allFiles_1 = glob.glob(path_train_x + "/*.csv")
allFiles_2=glob.glob(path_test_x+"/*.csv")

y_train_data=pd.read_csv("labels.csv")
y_train=y_train_data.values

y_test_data=pd.read_csv('test_y.csv')
y_test=y_test_data.values

frame = pd.DataFrame()
count=0
x_train=np.zeros((682,99,100))
x_test=np.zeros((20,99,100))

for file_ in allFiles_1:
    df = pd.read_csv(file_)
    x_train[count,:,:]=df
    count=count+1
count=0
for file_ in allFiles_2:
    df = pd.read_csv(file_)
    x_test[count,:,:]=df
    count=count+1
    


batch_size = 5
num_classes = 2
epochs =5

x_train = x_train.reshape(682,99,100,1)
x_test = x_test.reshape(20,99,100,1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
abc=[]

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(99,100,1)))
cnn.add(BatchNormalization())
#cnn.add(Dropout(0.3))
#cnn.add(Conv2D(32, (3, 3), activation='relu'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.3))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(BatchNormalization())
#cnn.add(Dropout(0.1))
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
cnn.add(Flatten())
"""l=cnn.layers
s=l[7].output_shape
a=s[1]
cnn.add(Reshape((1,a)))
cnn.add(Bidirectional(LSTM(64,batch_input_shape=(None,1,a),return_sequences=True)))
cnn.add(Bidirectional(LSTM(128,return_sequences=False)))"""
cnn.add(Dense(64, activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(128,activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(128,activation='relu'))

cnn.add(Dense(2, activation='sigmoid'))
cnn.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
cnn.fit(x_train, y_train,epochs=epochs,batch_size=batch_size,verbose=1,validation_split=0.05)
cnn.summary()
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

cnn.save("cnn_model.h5")
cnn.save("cnn_model.model")
"""model=Sequential()
model.add(TimeDistributed(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(9,10,1))))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(x_train, y_train,epochs=epochs,validation_data=(x_test, y_test))"""

"""model.add(LSTM(16,input_shape=[9,10],return_sequences=False))

model.add(Dense(64, activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(x_train, y_train,epochs=epochs,validation_data=(x_test, y_test))
model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])"""


"""model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=(None, 9, 10, 1),padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),padding='same', return_sequences=False))
#model.add(AveragePooling3D((1, 9,10)))
#model.add(Reshape((-1,32)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(x_train, y_train,epochs=epochs,validation_data=(x_test, y_test))
model.summary()"""

