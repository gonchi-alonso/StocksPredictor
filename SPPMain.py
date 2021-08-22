# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 05:15:18 2021

@author: galon
"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt

plt.style.use('fivethirtyeight')

#datetime for start and end date of the data collection
start = dt.datetime(2012, 1, 1)
end = dt.datetime.now()

#collect data from the web, specifically yahoo
df = web.DataReader('AAPL', data_source = 'yahoo', start=start, end=end)


# plt.figure(figsize = (16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize = 18)
# plt.ylabel('Close Price USD', fontsize = 18)
# plt.show()

closing = df.filter(['Close'])
closingData = closing.values

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(closingData)

#Create training data set
trainingDataLen = math.ceil(len(closingData) * 0.8)
trainData = scaledData[0:trainingDataLen,:]
xtrain = []
ytrain = []

#Create array 
for i in range(60, len(trainData)):
    xtrain.append(trainData[i-60:i, 0])
    ytrain.append(trainData[i, 0])
    if i <= 60:
        print(xtrain)
        print(ytrain)
        print()

#convert x and y trains into numpy arrays
xtrain, ytrain = np.array(xtrain), np.array(ytrain)

#reshape the data
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (xtrain.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
#train the model
model.fit(xtrain, ytrain, batch_size = 1, epochs = 1)

#create the testing dataset
#new array containing scaled value from index 1543 to 2003
testData = scaledData[trainingDataLen - 60: , :]
xTest = []
yTest = closingData[trainingDataLen: , :]
for i in range(60, len(testData)):
    xTest.append(testData[i-60:i, 0])
    
#convert the data to a numpy array and reshape, so it first LSTM
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

#get the models predicted price values
prediction = model.predict(xTest)
prediction = scaler.inverse_transform(prediction)

#evaluate the model by getting root mean square error (RMSE)
rmse = np.sqrt(np.mean(prediction - yTest)**2 )
print(rmse)

#plot the data
train = closing[:trainingDataLen]
valid = closing[trainingDataLen:]
valid['Predictions'] = prediction

#Visuals
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Show the predictions vs the Actual Values
valid


    
    
    
    