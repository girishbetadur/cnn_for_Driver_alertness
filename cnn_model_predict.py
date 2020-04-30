# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 06:24:17 2019

@author: Harish
"""
import glob
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


path_test_x=r'D:\8sem_project\cnn_lstm_10_10_data\test_x'
allFiles_2=glob.glob(path_test_x+"/*.csv")
y_test_data=pd.read_csv('test_y.csv')
y_test=y_test_data.values
count=0
x_test=np.zeros((20,99,100))
x_test_array=np.zeros((1,99,100))
count=0
for file_ in allFiles_2:
    df = pd.read_csv(file_)
    x_test[count,:,:]=df
    #plt.subplot(20,2,count+1)
    plt.plot(x_test[1,:,:])
    count=count+1
    
x_test = x_test.reshape(20,99,100,1)
x_test_array=x_test_array.reshape(1,99,100,1)
model=load_model("cnn_model.h5")

list1=[]
for i in range(20):
    x_test_array=x_test[i,:,:,:]
    x_test_array=x_test_array.reshape(1,99,100,1)

    pred=model.predict(x_test_array)
    value=np.argmax(pred)
    prediction={'prediction':int(value)}
    list1.append(value)
list1=np.array(list1)
print(confusion_matrix(y_test,list1))  
"""x_test=pd.read_csv("wake341.csv")
x_test=x_test.values
model=load_model("cnn_model.h5")
x_test=x_test.reshape(1,99,100,1)
pred=model.predict(x_test)
value=np.argmax(pred)
prediction={'prediction':int(value)}
print(prediction)"""