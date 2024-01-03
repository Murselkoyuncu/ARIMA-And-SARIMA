# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:47:38 2024

@author: mrslk
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("ARIMA.csv", index_col='DATE',parse_dates= True) 
df.index.freq = 'D'
df.dropna(inplace=True) 

df = pd.DataFrame(df['Temp']) 
train = df.iloc[:510,0]
test = df.iloc[510:,0]
 
from statsmodels.tsa.seasonal import seasonal_decompose
decomp_results = seasonal_decompose(df)
decomp_results.plot() 

# finding paramters 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)
plot_pacf(train, lags=50)

from pmdarima import auto_arima 

auto_arima_model = auto_arima(df, trace=True)

#developing ARIMA model 

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd



# Specify the order (p, d, q) for the ARIMA model
order = (1, 1, 2)

# Create and fit the ARIMA model
A_Model = ARIMA(train, order=order)
A_Model_fit = A_Model.fit()

# Print a summary of the model
print(A_Model_fit.summary())

predicted_results = A_Model_fit.predict(len(train),end=len(train)+len(test)-1,type= 'levels') 

plt.plot(test, color='red',label='actual Temp')

plt.plot(predicted_results, color='blue',label='predicted Temp')
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend() 
plt.show()

print(test.mean())
print(predicted_results.mean())







