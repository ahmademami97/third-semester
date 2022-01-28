#!/usr/bin/env python
# coding: utf-8

# In[63]:


import math
import pandas as pd
import numpy as np
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.stats.diagnostic import acorr_ljungbox 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,plot_predict
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
plt.style.use('fivethirtyeight')
from datetime import datetime , date
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
import pylab
import scipy.stats as stats
from scipy.stats import norm
from math import log
from pmdarima.utils import diff_inv


# In[199]:


def create_dataset(df):
    x = []
    y = []
    for i in range(30, df.shape[0]):
        x.append(df[i-30:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y


# In[65]:


# reading data from csv file
NFLX=pd.read_csv('D:\\R-documantary\\NFLX.csv',index_col='date',parse_dates=True)
dataset=NFLX[NFLX.index>='2018-01-01']
dataset=dataset.dropna()


# In[67]:


# take a look at the dataset
dataset


# In[71]:


#plotting our data to see the patterns and see if it needs transformation for stationarity
plt.figure(figsize=(16,8))
plt.plot(dataset,color='darkcyan')
plt.xlabel('Date')
plt.ylabel('price of stock')
plt.title('netflix closed stock price ')
plt.show()


# In[72]:


dataset.describe()


# In[74]:


round(0.9 * len(dataset))


# In[73]:


#defining train and test datasets
train_size=round(0.9 * len(dataset))
train=dataset[:train_size]
test=dataset[train_size:]
test.head()


# In[75]:


#differenced datasets
differenced_train=dataset.diff().dropna()[:train_size]
differenced_test=dataset.diff().dropna()[train_size:]
differenced_test.head()


# In[78]:


plt.figure(figsize=(16,8))
plt.plot(dataset.diff().dropna(),color='darkcyan')
plt.xlabel('Date')
plt.ylabel('amount of increase in price of stock')
plt.title('differnced dataset ')
plt.show()


# In[83]:


# stationarity test
result=adfuller(differenced_train['close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[80]:


returns=dataset.pct_change().dropna()
returns.head()
train_returns=returns[:train_size]
test_returns=returns[train_size:]


# In[79]:


plt.figure(figsize=(16,8))
plt.plot(returns,color='darkcyan')
plt.xlabel('Date')
plt.ylabel('return')
plt.title('returns')
plt.show()


# # ARIMA MODEL

# In[82]:


plot_acf(differenced_train)
plot_pacf(differenced_train)
plt.show()


# In[84]:


auto_arima(train['close'],trace=True,stepwise=False,max_p=10,max_q=10)


# In[85]:


arima_model=ARIMA(train['close'],order=(0,1,5))
arima_model=arima_model.fit()
print(arima_model.summary())


# In[87]:


train.head()


# In[89]:


plt.figure(figsize=(10,6))
plt.hist(arima_model.resid,density=True)
plt.title('histogram of residuals')
mu, std = norm.fit(arima_model.resid)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


# In[91]:


plt.figure(figsize=(12,8))
stats.probplot(arima_model.resid, dist="norm", plot=pylab)
plt.show()


# In[95]:


plot_acf(arima_model.resid,lags=40)
plot_pacf(arima_model.resid,lags=40)
plt.show()


# In[97]:


acorr_ljungbox(arima_model.resid,return_df=True,lags=20,boxpierce=True)


# In[98]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('residual')
plt.plot(arima_model.resid)
plt.title('plot of residuals ARIMA(0,1,5)')
plt.show()


# In[99]:


arima_pred=[]
for i in range(len(test)):
    model=ARIMA(dataset[:train_size+i],order=(0,1,5))
    model=model.fit()
    start=len(train)+i
    arima_pred.append(model.predict(start,start,typ='levels'))


# In[103]:


test['arima(1 step ahead)']=arima_pred


# In[113]:


arima_pred2=arima_model.forecast(steps=84,alpha=0.05)
test['upper_interval']=arima_pred2[2][:,1]
test['lower_interval']=arima_pred2[2][:,0]
test['prediction(without updating model)']=arima_pred2[0]


# In[114]:


test


# In[148]:


plt.figure(figsize=(16,8))
plt.plot(test['close'],label='actual stock price',color='lightskyblue')
plt.plot(test['arima(1 step ahead)'],label='one step ahead prediction',color='red')
plt.plot(test['prediction(without updating model)'],label='prediction (without updating model)',color='darkcyan')
plt.title('prediction result using arima(0,1,5)')
plt.xlabel('Date')
plt.ylabel('price($)')
plt.legend()
plt.show()


# In[153]:


plt.figure(figsize=(16,8))
plt.plot(test['close'],label='actual stock price',color='lightskyblue')
plt.plot(test['arima(1 step ahead)'],label='one step ahead prediction',color='red')
plt.plot(test['prediction(without updating model)'],label='prediction (without updating model)',color='darkcyan')
plt.plot(test['upper_interval'],label='95 percent conf interval',color='yellowgreen')
plt.plot(test['lower_interval'],color='yellowgreen')
plt.title('prediction result using arima(0,1,5)')
plt.xlabel('Date')
plt.ylabel('price($)')
plt.legend()
plt.show()


# # calculate MSE of one step ahead prediction

# In[119]:


mean_squared_error(test['close'],test['arima(1 step ahead)'])


# # exponential smoothing models (simple model)

# In[120]:


alpha=(0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95,1.00)


# In[124]:


sum_of_squared_error=[]
for i in alpha:
    model=SimpleExpSmoothing(train['close']).fit(smoothing_level=i,initial_level=train['close'][0])
    sum_of_squared_error.append(model.sse)


# In[130]:


plt.figure(figsize=(12,8))
plt.plot(alpha,sum_of_squared_error)
plt.title('alpha vs sum of squared error')
plt.ylabel('sum of squared error')
plt.xlabel('alpha')
plt.show()


# In[132]:


simple_model=SimpleExpSmoothing(train['close']).fit()
print(simple_model.summary())


# In[136]:


train['SES fitted values']=simple_model.fittedvalues


# In[145]:


plt.figure(figsize=(16,8))
plt.plot(train['close'],label='actual price',color='navy')
plt.plot(train['SES fitted values'],label='smoothed prices',color='tomato')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('simple exponential smoothing')
plt.legend()
plt.show()


# In[147]:


plt.figure(figsize=(16,8))
plt.plot(simple_model.resid,color='darkcyan')
plt.xlabel('Date')
plt.ylabel('residuals')
plt.title('residuals plot')
plt.show()


# In[154]:


acorr_ljungbox(simple_model.resid,lags=20,return_df=True,boxpierce=True)


# In[283]:


plt.figure(figsize=(10,6))
stats.probplot(simple_model.resid, dist="norm", plot=pylab)
plt.show()


# In[156]:


plt.figure(figsize=(10,6))
plt.hist(simple_model.resid,density=True)
plt.title('histogram of residuals')
mu, std = norm.fit(simple_model.resid)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


# In[157]:


plot_acf(simple_model.resid)
plot_pacf(simple_model.resid)
plt.show()


# In[158]:


acorr_ljungbox(simple_model.resid,return_df=True,lags=10)


# In[159]:


pred=simple_model.forecast(84)
pred.index=test.index


# In[162]:


pred_simple_model=simple_model.forecast()
for i in range(83):
    simple_model=SimpleExpSmoothing(dataset[:train_size+i+1]['close']).fit()
    pred_simple_model=pred_simple_model.append(simple_model.forecast())



# In[163]:


pred_simple_model.index=test.index


# In[164]:


test['simple exp smoothing prediction']=pred_simple_model
test


# In[165]:


mean_squared_error(test['close'],pred_simple_model)


# In[166]:


plt.figure(figsize=(16,8))
plt.plot(test['close'],label='actual')
plt.plot(pred_simple_model,label='prediction using simple exp smoothing')
plt.plot(pred,label='normal prediction(not updating the model at each period)',color='darkgreen')
plt.title('actual vs prediction')
plt.xlabel('Date')
plt.ylabel('price')
plt.legend()
plt.show()


# In[168]:


mse=[]
for i in alpha:
    smodel=SimpleExpSmoothing(train['close']).fit(smoothing_level=i,optimized=True)
    pred_simple_model=smodel.forecast()
    for j in range(83):
        smodel=SimpleExpSmoothing(dataset[:train_size+j+1]['close']).fit(smoothing_level=i)
        pred_simple_model=pred_simple_model.append(smodel.forecast())
    mse.append(mean_squared_error(pred_simple_model,test['close']))
    print('simple exponential model with landa =',i,'  MSE will be ',mean_squared_error(pred_simple_model,test['close']))


# In[171]:


plt.figure(figsize=(10,6))
plt.plot(alpha,mse)
plt.title('SSE vs alpha for forecast')
plt.ylabel('MSE')
plt.xlabel('lambda')
plt.show()


# #  double exponential model

# In[175]:


double_model=ExponentialSmoothing(train['close'],trend='additive',damped=False,seasonal=None).fit()
double_model.summary()


# In[178]:


train['double_exp_fitted']=double_model.fittedvalues


# In[179]:


plt.figure(figsize=(16,8))
plt.plot(train['close'],label='actual prices',color='blue')
plt.plot(train['double_exp_fitted'],label='smoothed prices',color='tomato')
plt.xlabel('date')
plt.ylabel('price')
plt.title('double exponential smoothing')
plt.legend()
plt.show()


# In[180]:


plt.figure(figsize=(16,8))
plt.plot(double_model.resid,color='darkcyan')
plt.xlabel('Date')
plt.ylabel('residuals')
plt.title('double exponential residuals plot')
plt.show()


# In[52]:


acorr_ljungbox(double_model.resid,lags=20,return_df=True,boxpierce=True)


# In[181]:


plt.figure(figsize=(10,6))
stats.probplot(double_model.resid, dist="norm", plot=pylab)
plt.show()


# In[184]:


plt.figure(figsize=(10,6))
plt.hist(double_model.resid,density=True)
plt.title('histogram of residuals')
mu, std = norm.fit(double_model.resid)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


# In[185]:


plot_acf(double_model.resid)
plot_pacf(double_model.resid)
plt.show()


# In[186]:


mean_squared_error(train['close'],train['double_exp'])


# In[188]:


mse_double=[]
for i in alpha:
    dmodel=ExponentialSmoothing(train['close'],trend='additive',damped=False,seasonal=None).fit(
        smoothing_level=i)
    pred_double_model=dmodel.forecast()
    for j in range(83):
        dmodel=ExponentialSmoothing(dataset[:train_size+j+1]['close'],trend='additive',damped=False,seasonal=None).fit(
            smoothing_level=i)
        pred_double_model=pred_double_model.append(dmodel.forecast())
    mse_double.append(mean_squared_error(pred_double_model,test['close']))
    print('double exponential model with landa =',i,'and beta= ',dmodel.params['smoothing_trend'],'  MSE will be ',mean_squared_error(pred_double_model,test['close']))


# In[189]:


pred_double=double_model.forecast(84)
pred_double.index=test.index


# In[193]:


pred_double_1step=double_model.forecast()
for i in range(83):
    double_model=ExponentialSmoothing(dataset[:train_size+i+1]['close'],trend='additive',damped=True,seasonal=None).fit()
    pred_double_1step=pred_double_1step.append(double_model.forecast())


# In[195]:


pred_double_1step.index=test.index


# In[196]:


plt.figure(figsize=(16,8))
plt.plot(test['close'],label='actual')
plt.xlabel('Date')
plt.ylabel('price')
plt.plot(pred_double_1step,label='prediction using double exp smoothing')
plt.plot(pred_double,label='normal prediction(without updating the model at each period)',color='darkgreen')
plt.title('double exponential model prediction results')
plt.legend()
plt.show()


# # mean squared error of one step ahead prediction

# In[265]:


mean_squared_error(pred_double_1step,test['close'])


# # GARCH MODEL (PREDICTING VOLATILITY)

# In[197]:


from arch import arch_model


# In[198]:


plt.figure(figsize=(16,8))
plt.plot(returns,color='darkcyan')
plt.title('netflix stock returns over time')
plt.xlabel('Date')
plt.ylabel('return')
plt.show()


# In[21]:


plot_acf(train_returns**2)
plot_pacf(train_returns**2)
plt.show()


# In[271]:


gmodel=arch_model(train_returns,p=1,q=1,mean='constant')
gmodel=gmodel.fit(disp='off')
gmodel.summary()


# In[272]:


plt.figure(figsize=(16,8))
plt.title('garch model residuals')
plt.xlabel('Date')
plt.ylabel('error')
plt.plot(gmodel.std_resid)


# In[273]:


plot_acf(gmodel.std_resid**2)
plot_pacf(gmodel.std_resid**2)
plt.show()


# In[274]:


acorr_ljungbox(gmodel.std_resid**2,lags=30,return_df=True)


# In[275]:


plt.figure(figsize=(10,6))
stats.probplot(gmodel.std_resid, dist="norm", plot=pylab)
plt.show()


# In[276]:


plt.figure(figsize=(10,6))
plt.hist(gmodel.std_resid,density=True)
plt.title('histogram of residuals')
mu, std = norm.fit(gmodel.std_resid)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


# # predict future volatility

# In[277]:


garch_prediction=[]
test_size=len(test_returns)
for i in range(test_size):
    gtrain=returns[:-(test_size-i)]
    gmodel=arch_model(gtrain,p=1,q=1)
    gmodel_fit=gmodel.fit(disp='off')
    pred=gmodel_fit.forecast(horizon=1)
    garch_prediction.append(np.sqrt(pred.variance.values[-1,:][0]))


# In[278]:


garch_prediction = pd.Series(garch_prediction, index=returns.index[-test_size:])


# In[279]:


plt.figure(figsize=(16,8))
true, = plt.plot(returns[-test_size:])
preds, = plt.plot(garch_prediction)
plt.title('Volatility Prediction', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# # RECURRENT NEURAL NETWORK

# In[206]:


from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.preprocessing.sequence import TimeseriesGenerator


# In[209]:


dataset_test = np.array(dataset[int(dataset.shape[0]*0.9):])


# In[210]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_dataset=scaler.fit_transform(dataset)


# In[211]:


dataset_test=scaler.transform(dataset_test)
x_test, y_test = create_dataset(dataset_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[212]:


scaled_dataset.shape


# In[213]:


test_size=84
train_rnn=dataset[:-test_size]
test_rnn=dataset[len(train_rnn):]
scaled_train=scaled_dataset[:-test_size]
scaled_test=scaled_dataset[len(scaled_train):]


# In[214]:


scaled_test[:10]


# In[216]:


plt.figure(figsize=(16,8))
plt.plot(RNN_dataset.index,scaled_dataset,color='darkcyan')
plt.title('plot of scaled  dataset')
plt.xlabel('Date')
plt.ylabel('scaled prices')
plt.show()


# In[217]:


#define generator
generator=TimeseriesGenerator(scaled_train,scaled_train,length=30,batch_size=1)


# In[218]:


x,y=generator[0]
print(f'given the array : \n{x.flatten()}')
print(f'predict this y : \n{y}')


# In[219]:


del model


# In[220]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[221]:


model=Sequential()
model.add(LSTM(units=50,return_sequences=True,activation='relu',input_shape=(30,1)))
model.add(LSTM(units=50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[222]:


model.compile(optimizer='adam',loss='mean_squared_error',metrics='accuracy')


# In[253]:


LSTM_model=model.fit(generator,epochs=20,validation_data=(x_test,y_test))


# In[224]:


model.summary()


# In[227]:


plt.figure(figsize=(16,8))
plt.plot(LSTM_model.history['loss'],label='training mean squared error')
plt.plot(LSTM_model.history['val_loss'],label='validation mean squared error',color='darkcyan')
plt.title('model validation')
plt.legend()
plt.show()


# In[226]:


fitted=[]
for i in range(len(generator)):
    fitted.append(model.predict(generator[i][0])[0])


# In[228]:


scaler.inverse_transform(fitted)


# In[232]:


train2=train_rnn[30:]
train2


# In[233]:


train2['fitted']=scaler.inverse_transform(fitted)
resid=train2['close']-train2['fitted']


# In[234]:


plt.figure(figsize=(16,8))
plt.plot(train2['close'],label='actual')
plt.plot(train2['fitted'],label='fitted value using LSTM')
plt.xlabel('Date')
plt.ylabel('price')
plt.title('fitted vs actual')
plt.legend()
plt.show()


# In[235]:


mean_squared_error(train2['fitted'],train2['close'])


# In[236]:


train2.head()


# In[238]:


plt.figure(figsize=(16,8))
plt.plot(resid,color='darkcyan')
plt.title('residuals of LSTM model')
plt.xlabel('Date')
plt.ylabel('residual')
plt.show()


# In[356]:


plot_acf(resid,lags=40)
plot_pacf(resid,lags=40)
plt.show()


# In[357]:


acorr_ljungbox(resid,return_df=True,lags=20,boxpierce=True)


# In[358]:


plt.figure(figsize=(10,6))
stats.probplot(resid, dist="norm", plot=pylab)
plt.show()


# In[239]:


plt.figure(figsize=(10,6))
plt.hist(resid,density=True)
plt.title('histogram of residuals')
mu, std = norm.fit(resid)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


# In[254]:


last_train_batch=scaled_train[-30:]
last_train_batch=last_train_batch.reshape((1,30,1))


# In[255]:


model.predict(last_train_batch)


# In[256]:


scaled_test[0]


# In[257]:


test_prediction=[]
current_batch=last_train_batch
for i in range(len(test_rnn)):
    current_pred=model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    #update current batch
    current_batch=np.append(current_batch[:,1:],scaled_test[i].reshape((1,1,1)),axis=1)


# In[258]:


test_prediction[:5]


# In[259]:


test_prediction=scaler.inverse_transform(test_prediction)


# In[260]:


test_rnn['prediction']=test_prediction


# In[261]:


test_rnn.head()


# In[263]:


plt.figure(figsize=(16,8))
plt.plot(test_rnn['close'],label='actual price',color='lightskyblue')
plt.plot(test_rnn['prediction'],label='prediction using RNN',color='red')
plt.title('prediction vs actual prices')
plt.xlabel('date')
plt.ylabel('price($)')
plt.legend()
plt.show()


#  # mean squared error of one step ahead prediction

# In[264]:


mean_squared_error(test_rnn['close'],test_rnn['prediction'])

