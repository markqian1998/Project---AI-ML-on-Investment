import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('c:/Users/qiany/OneDrive/桌面/Senior/Fin 427/Final Project/rank_signals7.csv', parse_dates=['DATE'])
df['RET']=df['RET']
df['year']=pd.DatetimeIndex(df['DATE']).year
#To avoid "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')""
df=df.dropna()

#Move headers which are not signals
column_name_list=df.columns.tolist()
column_name_list.remove('year')
column_name_list.remove('mvlag')
column_name_list.remove('permno')
column_name_list.remove('DATE')
column_name_list.remove('RET')

#Train, validation, and test samples.  Train:  2001-2012, Valid:  2013-2016, 
#Test:  2017-2020 
#Since RFR has a lot of hyperparameters to tune, I select the split percetage 60% 20% 20%   
X_train = df[df['year']<2013][column_name_list]
y_train = df[df['year']<2013]['RET']
X_valid = df[(df['year']<2017) & (df['year']>=2013)][column_name_list]
y_valid = df[(df['year']<2017) & (df['year']>=2013)]['RET']
X_test = df[(df['year']<2021) & (df['year']>=2017)][column_name_list]
y_test = df[(df['year']<2021) & (df['year']>=2017)]['RET']
test_data = df[df['year']>=2017]

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor

#There are 36 features(signals) in the ranks_signal7 table
grid = {'n_estimators': [300,500,1000], 'max_depth': [1,3,5], 
       'max_features': [3,5,10,15,20,36], 'random_state': [7]}

rfr = RandomForestRegressor()
test_scores = []
obj = -1

#Loop over combinations of hyperparameters
for g in ParameterGrid(grid):
    rfr.set_params(**g)  
    rfr.fit(X_train, y_train)
    test = rfr.score(X_valid, y_valid)
    test_scores.append(test)
    print(g)
    print(test)
    if test > obj:
        obj = test
        g_keep = g
#Hyperparameters configuration maximizing our object
print(g_keep)

#Prepare for the Value-weighted portfolio
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum()/ w.sum()
    except ZeroDivisionError:
        return np.nan

#Create the RFR with the best g that maximizes our objective
rfr = RandomForestRegressor(n_estimators=g_keep['n_estimators'],
                            max_depth = g_keep['max_depth'],
                            max_features = g_keep['max_features'],
                            random_state=7)
rfr.fit(X_train, y_train)
test_data['fitted']=rfr.predict(X_test)
print(rfr.score(X_test,y_test))

#Produce portfolios
test = test_data
test['ERDecile_before_abs']=test.groupby(['DATE'])['fitted'].\
transform(lambda x: pd.qcut(x,10, labels=False, duplicates='drop'))
test['ERDecile']=np.abs(test['ERDecile_before_abs']-9)

benchmark = test.groupby(['DATE'],as_index=False).apply(wavg,'RET','mvlag')
benchmark = benchmark.rename(columns={None:'benchmark'}) 

erret_out = test.groupby(['DATE','ERDecile'],as_index=False).apply(wavg, 'RET','mvlag')
erret_out = erret_out.rename(columns={None:'erret_out','ERDecile':'Decile'})

erret_out = erret_out.merge(benchmark,how='inner',on='DATE')
erret_out['rbp1'] = 1 + erret_out['benchmark']
erret_out['erp1'] = 1 + erret_out['erret_out']

erret_out['cumer'] = erret_out.groupby(['Decile'])['erp1'].transform('cumprod')
erret_out['cumb'] = erret_out.groupby(['Decile'])['rbp1'].transform('cumprod')

erret_out=erret_out.set_index('DATE')

from numpy import mean
#Total return on $1 invested
erret_out[erret_out['Decile']==0]['cumer'].plot(label='Random Forest')
erret_out[erret_out['Decile']==0]['cumb'].plot(label='Benchmark')
#Get the return on RFR and Benchmark
winner_return = mean(erret_out[erret_out['Decile']==0]['cumer'])
bench_return = mean(erret_out[erret_out['Decile']==0]['cumb'])
plt.title('Cumulative Strategy Return')
plt.legend()
plt.show()

#The decile with the highest rank is the winner
winner = erret_out[erret_out['Decile']==0]
geom_avg = winner['cumer'][len(winner)-1]**(1/20)-1
print("Geometric Average Return: {}".format(geom_avg))

rf = pd.read_csv('c:/Users/qiany/OneDrive/桌面/Senior/Fin 427/Final Project/rf.csv',parse_dates=['DATE'])
#Merge the data
winner = winner.merge(rf,how='inner',on='DATE')
winner['rfp1']=1+winner['rf']
winner['cumrf']=winner['rfp1'].transform('cumprod')
winner['exret']=winner['erret_out']-winner['rf']
winner['exretb']=winner['benchmark']-winner['rf']
# Get geometric average return for risk free rate n(use later for calculating the sharpe ratio)
geom_avgf = winner['cumrf'][len(winner)-1]**(1/(20))-1

import statsmodels.formula.api as sm
#Get alpha and beta through retrieving the parameter
reg = sm.ols('exret~exretb',data=winner).fit()
alpha = reg.params[0]
beta = reg.params[1]
print("Alpha: {}".format(alpha))
print("Beta: {}".format(beta))

#Get sharpe ratio
sharpe = (geom_avg-geom_avgf)/(np.sqrt(12*winner['erret_out'].std()))
print("Sharpe: {}".format(sharpe))

#Get drawdown
drawdown1 = []
drawdown3 = []
drawdown12 = []
for i in range(12,len(winner)):
    retm = winner['cumer'][i]/winner['cumer'][i-1]-1
    retq = winner['cumer'][i]/winner['cumer'][i-3]-1
    reta = winner['cumer'][i]/winner['cumer'][i-12]-1
    drawdown1 = np.append(drawdown1,retm)
    drawdown3 = np.append(drawdown3,retq)
    drawdown12 = np.append(drawdown12, reta)
    
print("Maximum 1-Month Drawdown: {}".format(min(drawdown1)))
print("Maximum 3-Month Drawdown: {}".format(min(drawdown3)))
print("Maximum 12-Month Drawdown: {}".format(min(drawdown12)))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, rfr.predict(X_test))
print("Test-sample R-squared: {}".format(r2))

#Sort the importance of signals 
importances = rfr.feature_importances_
sorted_index = np.argsort(importances)[::-1]
print(sorted_index)
print(importances)

#Create a bar graph
x = range(len(importances))
labels = np.array(column_name_list)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()

#Summary
df2 = pd.DataFrame({ 'Total return' : winner_return, 'Benchmark return' : bench_return, 'R-squared' : r2, \
    'Geometric average' : geom_avg, 'Alpha' : alpha, 'Beta' : beta, '1-month drawdowns' : min(drawdown1), \
        '3-month drawdowns' : min(drawdown3), '12-month drawdowns' : min(drawdown12), 'Sharpe ratio' : sharpe}, index=[0])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df2)


