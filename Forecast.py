# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:56:50 2023

@author: ltmat
"""
#This program reads preprocesses and analyzes time sereies future market price data using a stacked Bidirectional LSTM model 
#Last price point recorded is 08/17/2023  

#Importing needed modules
import tensorflow as tf
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import time

n_steps = 4 #number of steps fed to LSTM model
tf.random.set_seed(69)
look_back = 200 #forecast window

#sequence function for data preprocessing 
def split_seq(sequences,n_steps):
    x, y = list(), list()
    for i in range (len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix,:-1], sequences[end_ix-1,-1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


#data preprocessing
ts1 = pd.read_csv("C:\\Users\\ltmat\\Dropbox\\Fractical Analysis\\Nov23 Soybean Futures data hurstND.csv")
ts1['Date'] = pd.to_datetime(ts1.Date)
ts1.set_index('Date',inplace=True)
ts1.sort_index(inplace=True)
ts1=ts1.iloc[2991:] #select only 2013-2023

#data scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ts1)
ts=pd.DataFrame(scaled_data)

#full preprocessed timeseries
mes = np.array(ts)
mes[:, [5, 0]] = mes[:, [0, 5]]
mesx,mesy = split_seq(mes,n_steps)



#traing data------------------------------
train = ts.iloc[0:(len(ts1)-look_back)] 
trx1 = train.iloc[:,:1]
trx2 = train.iloc[:,1:2]
trx3 = train.iloc[:,2:3]
trx4 = train.iloc[:,3:4]
trx5 = train.iloc[:,4:5]
trx6 = train.iloc[:,5:6]
trx1 = np.array(trx1).reshape(len(trx1),1)
trx2 = np.array(trx2).reshape(len(trx2),1)
trx3 = np.array(trx3).reshape(len(trx3),1)
trx4 = np.array(trx4).reshape(len(trx4),1)

train_data = np.hstack((trx2,trx3,trx4,trx1))
X,y = split_seq(train_data,n_steps)

#testing data-------------------------------------
test =ts.iloc[len(y):round(len(mesy)*(look_back/len(mesy)))+len(y)]
tex1 = test.iloc[:,:1]
tex2 = test.iloc[:,1:2]
tex3 = test.iloc[:,2:3]
tex4 = test.iloc[:,3:4]
tex5 = test.iloc[:,4:5]
tex6 = test.iloc[:,5:6]
tex1 = np.array(tex1).reshape(len(tex1),1)
tex2 = np.array(tex2).reshape(len(tex2),1)
tex3 = np.array(tex3).reshape(len(tex3),1)
tex4 = np.array(tex4).reshape(len(tex4),1)
test_data = np.hstack((tex3,tex2,tex4,tex1)) 
X_test,y_test = split_seq(test_data, n_steps)


n_features = mesx.shape[2] #number features for input shape

#model training--------------------------------

#hyperparameters
opt = SGD(learning_rate=0.001, momentum=0.9) #SDG optimizer
epochs = 50
alpha = 2 #scaling factor for optimal neurons for hidden layer
neurons = 6#neuron calculation

#neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                        (neurons,
                                         activation='elu',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001,l2=0.0001),
                                         return_sequences=True,
                                         input_shape=(n_steps,n_features))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                        ((round(neurons/2)),
                                         activation='elu',
                                         return_sequences=False
                                         )))
    
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=opt,loss='mean_absolute_error')
history = model.fit(mesx,mesy,epochs=epochs)
train_mse = model.evaluate(mesx, mesy, verbose=0)
#----------------------------------------------

#forecasting with training data
last_values = mesx[-len(y_test):]
forecast = model.predict(last_values)

#data inverse scaling
y_test = y_test.reshape(-1, 1)
scaler_pred = MinMaxScaler()
scaler_pred.min_, scaler_pred.scale_ = scaler.min_[1], scaler.scale_[1]
forecast = scaler_pred.inverse_transform(forecast)[:, [0]]
y_test = scaler_pred.inverse_transform(y_test)[:, [0]]

#forcast visualization
pyplot.plot(y_test,label='Testing data',color='green')
pyplot.plot(forecast,label='Estimated Forecast',color='orange')
pyplot.legend()
pyplot.ylabel('Price')
pyplot.xlabel('Timesteps')
pyplot.show()

#model preformance metrics
print("""
      ----------------------------------------
      
      """)
print(str(epochs)+' epochs')
print('r^2 score is ' + str(r2_score(y_test,forecast)))
print('train mse is ' + str(train_mse))
print('test mse score is ' + str(mean_squared_error(y_test,forecast)))
print('explained variance is ' + str(explained_variance_score(y_test,forecast)))
print("""
      ----------------------------------------
      
      """)
      
#future forcasting

lag = 226

last_values2 = mesx[-int(lag):]
forecast2 = model.predict(last_values2)
forecast2 = scaler_pred.inverse_transform(forecast2)[:, [0]]
mean = np.mean(forecast2[0:])
print(mean)



