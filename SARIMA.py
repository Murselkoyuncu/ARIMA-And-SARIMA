# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:02:32 2024

@author: mrslk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read the CSV file
df = pd.read_csv("ARIMA.csv", index_col='DATE', parse_dates=True)
df.index.freq = 'D'
df.dropna(inplace=True)

# Separate the data into training and testing sets
train = df.iloc[:510, 0]
test = df.iloc[510:, 0]

# Extract exogenous variables
exo = df.iloc[:, 1:4]
exo_train = exo.iloc[:510]
exo_test = exo.iloc[510:]

# Decompose the time series
decompose_results = seasonal_decompose(df['Temp'])
decompose_results.plot()
plt.show()

# Use auto_arima for model selection
auto_arima_model = auto_arima(train, exogenous=exo_train, m=7, trace=True, seasonal=True, D=1)
order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order

# Fit SARIMAX model
sarimax_model = SARIMAX(train, exog=exo_train, order=order, seasonal_order=seasonal_order)
sarimax_results = sarimax_model.fit()

# Predict using the fitted model
prediction = sarimax_results.get_forecast(steps=len(test), exog=exo_test)
predicted_mean = prediction.predicted_mean
confidence_interval = prediction.conf_int()

# Plot the results
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predicted_mean, label='Forecast', color='red')
plt.fill_between(test.index, confidence_interval.iloc[:, 0], confidence_interval.iloc[:, 1], color='red', alpha=0.2)
plt.legend()
plt.show()
