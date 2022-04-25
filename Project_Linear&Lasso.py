#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

iyear = 2006
window = 5
num_groups = 10
valid_window = 3
gridpoints = 100

df = pd.read_csv('c:/Users/qiany/OneDrive/桌面/Senior/Fin 427/Final Project/rank_signals7.csv', parse_dates=['DATE'])
df['RET']=df['RET']*100
df['year']=pd.DatetimeIndex(df['DATE']).year
df=df.dropna()

column_name_list=df.columns.tolist()
column_name_list.remove('mvlag')
column_name_list.remove('permno')
column_name_list.remove('DATE')
column_name_list.remove('RET')
column_name_list.remove('year')


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

out_of_sample = df[df['year']>=iyear+valid_window]


# In[7]:


#expanding window 

window = 0
pred_reg_expanding = []
pred_lasso_expanding = []
for i in range(iyear,2021-valid_window):
    
    if window > 0:
        X_train = df[(df['year']<i) & (df['year']>=(i-window))][column_name_list]
        y_train = df[(df['year']<i) & (df['year']>=(i-window))]['RET'].values
    else:
        X_train = df[df['year']<i][column_name_list]
        y_train = df[df['year']<i]['RET'].values    
    
    X_valid = df[(df['year']>=i) & (df['year']<i+valid_window)][column_name_list]
    y_valid = df[(df['year']>=i) & (df['year']<i+valid_window)]['RET']
    
    X_test = df[df['year']==i+valid_window][column_name_list]
    y_test = df[df['year']==i+valid_window]['RET']     
  
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    y_pred_reg = reg.predict(X_test)
    
    alpha_opt = 0.00005
    obj = -1
    for j in range(1,gridpoints):
        alpha_cand = j/100
        lasso = Lasso(alpha = alpha_cand)
        lfit = lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_valid)
        u = ((y_valid-y_pred)**2).sum()
        v = ((y_valid)**2).sum()
        r2 = 1-u/v
        
        if r2 > obj:
            alpha_opt = alpha_cand
            obj = r2
            
#Fitting the model
    lasso = Lasso(alpha=alpha_opt)
    lfit = lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    
    pred_reg_expanding.extend(y_pred_reg)
    pred_lasso_expanding.extend(y_pred_lasso)
    y_pred_reg = reg.predict(X_test)


# In[8]:


#c Expanding window R square
from sklearn.metrics import r2_score 
r2linew = r2_score(y_test, y_pred_reg)
r2lasew = r2_score(y_test, y_pred_lasso)

print("c.Test-sample R-square using linear regression (expanding window) is ", r2linew)
print("c.Test-sample R-square using lasso regression (expanding window) is ", r2lasew)


# In[10]:


#Rolling window 
window = 5
pred_reg_roll = []
pred_lasso_roll = []
for i in range(iyear,2021-valid_window):
    
    
    if window > 0:
        X_train = df[(df['year']<i) & (df['year']>=(i-window))][column_name_list]
        y_train = df[(df['year']<i) & (df['year']>=(i-window))]['RET'].values
    else:
        X_train = df[df['year']<i][column_name_list]
        y_train = df[df['year']<i]['RET'].values    
    
    X_valid = df[(df['year']>=i) & (df['year']<i+valid_window)][column_name_list]
    y_valid = df[(df['year']>=i) & (df['year']<i+valid_window)]['RET']
    
    X_test = df[df['year']==i+valid_window][column_name_list]
    y_test = df[df['year']==i+valid_window]['RET']     
  
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    y_pred_reg = reg.predict(X_test)
    
    alpha_opt = 0.00005
    obj = -1
    for j in range(1,gridpoints):
        alpha_cand = j/100
        lasso = Lasso(alpha = alpha_cand)
        lfit = lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_valid)
        u = ((y_valid-y_pred)**2).sum()
        v = ((y_valid)**2).sum()
        r2 = 1-u/v
        if r2 > obj:
            alpha_opt = alpha_cand
            obj = r2
            
    lasso = Lasso(alpha=alpha_opt)
    lfit = lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    
    pred_reg_roll.extend(y_pred_reg)
    pred_lasso_roll.extend(y_pred_lasso)


# In[11]:


#Rolling window R square

r2linroll = r2_score(y_test, y_pred_reg)
r2lasroll = r2_score(y_test, y_pred_lasso)

print("c.Test-sample R-square using linear regression (rolling window) is ", r2linroll)
print("c.Test-sample R-square using lasso regression (rolling window) is ", r2lasroll)


# In[12]:


#Out of sample results
out_of_sample['pred_reg_expand']=np.reshape(np.array(pred_reg_expanding),(-1,1))
out_of_sample['pred_lasso_expand']=np.reshape(np.array(pred_lasso_expanding),(-1,1))
out_of_sample['pred_reg_roll']=np.reshape(np.array(pred_reg_roll),(-1,1))
out_of_sample['pred_lasso_roll']=np.reshape(np.array(pred_lasso_roll),(-1,1))


# In[13]:


#Decile 0 has the highest predicted expected returns

out_of_sample['ERDecile1']=out_of_sample.groupby(['DATE'])['pred_reg_expand'].    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile1']=np.abs(out_of_sample['ERDecile1']-(num_groups-1))

out_of_sample['ERDecile2']=out_of_sample.groupby(['DATE'])['pred_lasso_expand'].    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile2']=np.abs(out_of_sample['ERDecile2']-(num_groups-1))

out_of_sample['ERDecile3']=out_of_sample.groupby(['DATE'])['pred_reg_roll'].    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile3']=np.abs(out_of_sample['ERDecile3']-(num_groups-1))

out_of_sample['ERDecile4']=out_of_sample.groupby(['DATE'])['pred_lasso_roll'].    transform(lambda x: pd.qcut(x,num_groups, labels=False, duplicates='drop'))
out_of_sample['ERDecile4']=np.abs(out_of_sample['ERDecile4']-(num_groups-1))


# In[14]:


#Forming portfolios

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum()/ w.sum()
    except ZeroDivisionError:
        return np.nan
    
#4 portfolios:  Lasso rolling, Lasso expanding, linear regression rolling, and linear regression expanding

benchmark = out_of_sample.groupby(['DATE'],as_index=False).apply(wavg,'RET','mvlag')
benchmark = benchmark.rename(columns={None:'benchmark'})    

erret1 = out_of_sample.groupby(['DATE','ERDecile1'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret1 = erret1.rename(columns={None:'erret1','ERDecile1':'Decile'})

erret2 = out_of_sample.groupby(['DATE','ERDecile2'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret2 = erret2.rename(columns={None:'erret2','ERDecile2':'Decile'})

erret3 = out_of_sample.groupby(['DATE','ERDecile3'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret3 = erret3.rename(columns={None:'erret3','ERDecile3':'Decile'})

erret4 = out_of_sample.groupby(['DATE','ERDecile4'],as_index=False).apply(wavg, 
                                                            'RET','mvlag')
erret4 = erret4.rename(columns={None:'erret4','ERDecile4':'Decile'})


port_ret = benchmark.merge(erret1, how='inner', on='DATE')
port_ret = port_ret.merge(erret2, how='inner', on=['DATE','Decile'])
port_ret = port_ret.merge(erret3, how='inner', on=['DATE','Decile'])
port_ret = port_ret.merge(erret4, how='inner', on=['DATE','Decile'])


port_ret['rb_gross']=1+port_ret['benchmark']/100
port_ret['r1_gross']=1+port_ret['erret1']/100
port_ret['r2_gross']=1+port_ret['erret2']/100
port_ret['r3_gross']=1+port_ret['erret3']/100
port_ret['r4_gross']=1+port_ret['erret4']/100

port_ret=port_ret.set_index('DATE')

port_ret['rcumb'] = port_ret.groupby(['Decile'])['rb_gross'].transform('cumprod')
port_ret['rcum1'] = port_ret.groupby(['Decile'])['r1_gross'].transform('cumprod')
port_ret['rcum2'] = port_ret.groupby(['Decile'])['r2_gross'].transform('cumprod')
port_ret['rcum3'] = port_ret.groupby(['Decile'])['r3_gross'].transform('cumprod')
port_ret['rcum4'] = port_ret.groupby(['Decile'])['r4_gross'].transform('cumprod')

port_ret = port_ret.sort_values(['DATE','Decile'])


# In[15]:


#P.S. the chart is extra
import matplotlib.pyplot as plt

port_ret[port_ret['Decile']==0]['rcumb'].plot(label='Benchmark')
port_ret[port_ret['Decile']==0]['rcum1'].plot(label='Regression Expanding')
port_ret[port_ret['Decile']==0]['rcum2'].plot(label='Lasso Expanding')
port_ret[port_ret['Decile']==0]['rcum3'].plot(label='Regression Rolling')
port_ret[port_ret['Decile']==0]['rcum4'].plot(label='Lasso Rolling')
plt.title('Cumulative Strategy Return')
plt.legend()
plt.show()

#The top return portfolio will be Decile 0
winner = port_ret[port_ret['Decile']==0]

rf = pd.read_csv('c:/Users/qiany/OneDrive/桌面/Senior/Fin 427/Final Project/rf.csv',parse_dates=['DATE'])
winner = winner.merge(rf,how='inner',on='DATE')
winner['rfp1']=1+winner['rf']
winner['cumrf']=winner['rfp1'].transform('cumprod')


# In[16]:


#f. Beta
import statsmodels.formula.api as sm

#Calculate the beta with respect to the benchmark and retrieve the parameter
newreg1 = sm.ols('erret1~benchmark',data=winner).fit()
beta1 = newreg1.params[1]

newreg2 = sm.ols('erret2~benchmark',data=winner).fit()
beta2 = newreg2.params[1]

newreg3 = sm.ols('erret3~benchmark',data=winner).fit()
beta3 = newreg3.params[1]

newreg4 = sm.ols('erret4~benchmark',data=winner).fit()
beta4 = newreg4.params[1]


# In[17]:


#d. Geometric average return
geom_avg1 = winner['rcum1'][len(winner)-1]**(1/(2021-iyear))-1
geom_avg2 = winner['rcum2'][len(winner)-1]**(1/(2021-iyear))-1
geom_avg3 = winner['rcum3'][len(winner)-1]**(1/(2021-iyear))-1
geom_avg4 = winner['rcum4'][len(winner)-1]**(1/(2021-iyear))-1
geom_avgb = winner['rcumb'][len(winner)-1]**(1/(2021-iyear))-1
geom_avgf = winner['cumrf'][len(winner)-1]**(1/(2021-iyear))-1

#The alpha is the difference in the geometric average excess return -- 
#Average return on the winner portfolio minus the risk-free rate -- minus
#the beta times the excess return on the benchmark

#e. Alpha 
alpha1 = geom_avg1 - geom_avgf - beta1*(geom_avgb-geom_avgf)
alpha2 = geom_avg2 - geom_avgf - beta2*(geom_avgb-geom_avgf)
alpha3 = geom_avg3 - geom_avgf - beta3*(geom_avgb-geom_avgf)
alpha4 = geom_avg4 - geom_avgf - beta4*(geom_avgb-geom_avgf)


# In[18]:


#g. Maximum one-, three-, and 12-month drawdowns (i used average here)

#linear regression expanding window
lewdrawdown1 = []
lewdrawdown3 = []
lewdrawdown12 = []
for i in range(12,len(winner)):
    lewretm = winner['rcum1'][i]/winner['rcum1'][i-1]-1
    lewretq = winner['rcum1'][i]/winner['rcum1'][i-3]-1
    lewreta = winner['rcum1'][i]/winner['rcum1'][i-12]-1
    lewdrawdown1 = np.append(lewdrawdown1,lewretm)
    lewdrawdown3 = np.append(lewdrawdown3,lewretq)
    lewdrawdown12 = np.append(lewdrawdown12,lewreta)
    
print("Maximum 1-Month Drawdown: {}".format(min(lewdrawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(lewdrawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(lewdrawdown12)))

#lasso regression expanding window
laewdrawdown1 = []
laewdrawdown3 = []
laewdrawdown12 = []
for i in range(12,len(winner)):
    laewretm = winner['rcum2'][i]/winner['rcum2'][i-1]-1
    laewretq = winner['rcum2'][i]/winner['rcum2'][i-3]-1
    laewreta = winner['rcum2'][i]/winner['rcum2'][i-12]-1
    laewdrawdown1 = np.append(laewdrawdown1,laewretm)
    laewdrawdown3 = np.append(laewdrawdown3,laewretq)
    laewdrawdown12 = np.append(laewdrawdown12,laewreta)
    
print("Maximum 1-Month Drawdown: {}".format(min(laewdrawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(laewdrawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(laewdrawdown12)))

#linear regression rolling window
lrwdrawdown1 = []
lrwdrawdown3 = []
lrwdrawdown12 = []
for i in range(12,len(winner)):
    lrwretm = winner['rcum3'][i]/winner['rcum3'][i-1]-1
    lrwretq = winner['rcum3'][i]/winner['rcum3'][i-3]-1
    lrwreta = winner['rcum3'][i]/winner['rcum3'][i-12]-1
    lrwdrawdown1 = np.append(lrwdrawdown1,lrwretm)
    lrwdrawdown3 = np.append(lrwdrawdown3,lrwretq)
    lrwdrawdown12 = np.append(lrwdrawdown12,lrwreta)
    
print("Maximum 1-Month Drawdown: {}".format(min(lrwdrawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(lrwdrawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(lrwdrawdown12)))

#lasso regression rolling window
larwdrawdown1 = []
larwdrawdown3 = []
larwdrawdown12 = []
for i in range(12,len(winner)):
    larwretm = winner['rcum4'][i]/winner['rcum4'][i-1]-1
    larwretq = winner['rcum4'][i]/winner['rcum4'][i-3]-1
    larwreta = winner['rcum4'][i]/winner['rcum4'][i-12]-1
    larwdrawdown1 = np.append(larwdrawdown1,larwretm)
    larwdrawdown3 = np.append(larwdrawdown3,larwretq)
    larwdrawdown12 = np.append(larwdrawdown12,larwreta)
    
print("Maximum 1-Month Drawdown: {}".format(min(larwdrawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(larwdrawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(larwdrawdown12)))


# In[19]:


#h. Sharpe ratio
from statistics import stdev
import math 

#sharpe ratio = (Annualized Average Geometric Return - Annualized Average Geometric Return on a Risk-Free Asset)/(Annualized Portfolio Standard Deviation)

#not sure about the monthly portfolio return!

std1 = stdev(lewdrawdown1)*math.sqrt(12)
std2 = stdev(laewdrawdown1)*math.sqrt(12)
std3 = stdev(lrwdrawdown1)*math.sqrt(12)
std4 = stdev(larwdrawdown1)*math.sqrt(12)


sharpe1 = (geom_avg1-geom_avgf)/std1
sharpe2 = (geom_avg2-geom_avgf)/std2
sharpe3 = (geom_avg3-geom_avgf)/std3
sharpe4 = (geom_avg1-geom_avgf)/std4


print(sharpe1)
print(sharpe2)
print(sharpe3)
print(sharpe4)


# In[20]:


#a. Total return on $1 invested


# In[21]:


#b. Comparable return on a value-weighted benchmark of stocks in your industry


# In[22]:


df2 = pd.DataFrame(np.array([['Linear Regression expanding window',1,2,r2linew,geom_avg1,alpha1,beta1,min(lewdrawdown1),min(lewdrawdown3),min(lewdrawdown12),sharpe1],                   ['Lasso Regression expanding window',1,2,r2lasew,geom_avg2,alpha2,beta2,min(laewdrawdown1),min(laewdrawdown3),min(laewdrawdown12),sharpe2],                   ['Linear Regression rolling window',1,2,r2linroll,geom_avg3,alpha3,beta3,min(lrwdrawdown1),min(lrwdrawdown3),min(lrwdrawdown12),sharpe3],                   ['Lasso Regression rolling window',1,2,r2lasroll,geom_avg4,alpha4,beta4,min(larwdrawdown1),min(larwdrawdown3),min(larwdrawdown12),sharpe4]]),                   columns=['Regression Method','a.Total return on $1 invested','b.Comparable return on a value-weighted benchmark of stocks in your industry',                     'c.Test-sample R-squared','d.Geometric average return','e.Alpha','f.Beta','g.Maximum 1-month drawdowns',                     'g.Maximum 3-month drawdown','g.Maximum 12-month drawdowns','h.Sharpe ratio'])
df2.style


# In[24]:


#a.	Total return on $1 invested
#b.	Comparable return on a value-weighted benchmark of stocks in your industry
#/c.	Test-sample R-squared 
#/d.	Geometric average return
#/e.	Alpha
#/f.	Beta
#/g.	Maximum one-, three-, and 12-month drawdowns (i used average here)
#/h.	Sharpe rati

#things to discuss here: Use which regression method (expand/roll), pilot test

