#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:15:08 2022

@author: max
"""
iyear = 2001
window = 5
num_groups = 10
valid_window = 3


import pandas as pd
df = pd.read_csv('/users/max/Downloads/rank_signals7 (1).csv', parse_dates=['DATE'])
#df = pd.read_csv('../../Data/Class7/rank_signals_all_nplus1.csv', 
parse_dates=['DATE']
df['RET']=df['RET']
df['year']=pd.DatetimeIndex(df['DATE']).year
df=df.dropna()
#As in the previous code, limit attention to 5 signals for computing ease.
df = df[['DATE','year','mvlag','permno','RET','CF_PRC','ROE','REL_PRC','FDISP','LTM_AT']]
column_name_list=df.columns.tolist()
column_name_list.remove('year')
column_name_list.remove('mvlag')
column_name_list.remove('permno')
column_name_list.remove('DATE')
column_name_list.remove('RET')
#Train, validation, and test samples.  Train:  2001-2005, Valid:  2006-2008, 
#Test:  2009
X_train = df[(df['year']<2017) & (df['year']>=iyear)][column_name_list]
y_train = df[(df['year']<2017) & (df['year']>=iyear)]['RET']
X_valid = df[(df['year']<2019) & (df['year']>=2017)][column_name_list]
y_valid = df[(df['year']<2019) & (df['year']>=2017)]['RET']
X_test = df[(df['year']<2021) & (df['year']>=2019)][column_name_list]
y_test = df[(df['year']<2021) & (df['year']>=2019)]['RET']

out_of_sample = df[(df['year']<2021) & (df['year']>=2019)]
import numpy as np
#tensorflow is the background package for neural nets
import tensorflow as tf

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import regularizers
nnet1 = Sequential()
nnet1.add(Dense(32, input_dim = X_train.shape[1],activation='relu'))
nnet1.add(Dense(16, input_dim = X_train.shape[1],activation='relu'))
nnet1.add(Dense(1, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=1e-5)
nnet1.compile(optimizer=opt, loss='mse')
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
stop = EarlyStopping(monitor='val_loss', patience = 5, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history1 = nnet1.fit(X_train, y_train, validation_data = (X_valid, y_valid), 
                     epochs = 500, batch_size=10000,  callbacks = [stop, mc])
best_model=load_model('best_model.h5')
from sklearn.metrics import r2_score
train_preds = best_model.predict(X_train)
valid_preds = best_model.predict(X_valid)
test_preds = nnet1.predict(X_test)
r2 = r2_score(y_test, test_preds)
print(r2_score(y_train, train_preds))
print(r2_score(y_valid, valid_preds))

import matplotlib.pyplot as plt
plt.plot(history1.history['loss'])
plt.show()


out_of_sample['pred_neural']=np.reshape(np.array(test_preds),(-1,1))
out_of_sample['ERDecile1']=out_of_sample.groupby(['DATE'])['pred_neural'].transform(lambda x: pd.qcut(x,10, labels=False, duplicates='drop'))
out_of_sample['ERDecile1']=np.abs(out_of_sample['ERDecile1']-(10-1))

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum()/ w.sum()
    except ZeroDivisionError:
        return np.nan
    
benchmark = out_of_sample.groupby(['DATE'],as_index=False).apply(wavg,'RET','mvlag')
benchmark = benchmark.rename(columns={None:'benchmark'})    

erret1 = out_of_sample.groupby(['DATE','ERDecile1'],as_index=False).apply(wavg,'RET','mvlag')
erret1 = erret1.rename(columns={None:'erret1','ERDecile1':'Decile'})

port_ret = benchmark.merge(erret1, how='inner', on='DATE')
port_ret['rb_gross']=1+port_ret['benchmark']/100
port_ret['r1_gross']=1+port_ret['erret1']/100
port_ret=port_ret.set_index('DATE')
port_ret['rcumb'] = port_ret.groupby(['Decile'])['rb_gross'].transform('cumprod')
port_ret['rcum1'] = port_ret.groupby(['Decile'])['r1_gross'].transform('cumprod')
port_ret = port_ret.sort_values(['DATE','Decile'])

from numpy import mean

port_ret[port_ret['Decile']==0]['rcumb'].plot(label='Benchmark')
bench_return = mean(port_ret[port_ret['Decile']==0]['rcumb'])
port_ret[port_ret['Decile']==0]['rcum1'].plot(label='Neural Net')
winner_return = mean(port_ret[port_ret['Decile']==0]['rcum1'])
plt.title('Cumulative Strategy Return')
plt.legend()
plt.show()

#The top return portfolio will be Decile 0
winner = port_ret[port_ret['Decile']==0]

rf = pd.read_csv('/users/max/Downloads/Rf.csv',parse_dates=['DATE'])
winner = winner.merge(rf,how='inner',on='DATE')
winner['rfp1']=1+winner['Rf']
winner['cumrf']=winner['rfp1'].transform('cumprod')

import statsmodels.formula.api as sm
newreg1 = sm.ols('erret1~benchmark',data=winner).fit()
beta1 = newreg1.params[1]

geom_avg1 = winner['rcum1'][len(winner)-1]**(1/(20))-1
geom_avgb = winner['rcumb'][len(winner)-1]**(1/(20))-1
geom_avgf = winner['cumrf'][len(winner)-1]**(1/(20))-1

alpha1 = geom_avg1 - geom_avgf - beta1*(geom_avgb-geom_avgf)

drawdown1 = []
drawdown3 = []
drawdown12 = []
for i in range(12,len(winner)):
    retm = winner['rcum1'][i]/winner['rcum1'][i-1]-1
    retq = winner['rcum1'][i]/winner['rcum1'][i-3]-1
    reta = winner['rcum1'][i]/winner['rcum1'][i-12]-1
    drawdown1 = np.append(drawdown1,retm)
    drawdown3 = np.append(drawdown3,retq)
    drawdown12 = np.append(drawdown12, reta)

print("Maximum 1-Month Drawdown: {}".format(min(drawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(drawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(drawdown12)))

from statistics import stdev
import math

std1 = stdev(drawdown1)*math.sqrt(12)
sharpe1 = (geom_avg1-geom_avgf)/(np.sqrt(12*winner['rcum1'].std()))

#Sort the importance of signals 
importances = nnet1.feature_importances_
sorted_index = np.argsort(importances)[::-1]
print(sorted_index)
print(importances)

#Create a bar graph
x = range(len(importances))
labels = np.array(column_name_list)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()

df2 = pd.DataFrame({ 'Total return' : winner_return, 'Benchmark return' : bench_return, 'R-squared' : r2, 'Geometric average' : geom_avg1, 'Alpha' : alpha1, 'Beta' : beta1, '1-month drawdowns' : min(drawdown1), '3-month drawdowns' : min(drawdown3), '12-month drawdowns' : min(drawdown12), 'Sharpe ratio' : sharpe1}, index=[0])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df2)


