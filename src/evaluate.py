import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
data_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\data\AirPassengers.csv"
df = pd.read_csv(data_path)

df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')

y = df['#Passengers']

# Split dataset (train up to 1958, test 1959–1960)
train = y[:'1958']
test = y['1959':]

# Fit model
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()

# Forecast the test period
forecast = results.get_forecast(steps=len(test))
predicted = forecast.predicted_mean

# Metrics
mae = np.mean(np.abs(predicted - test))
rmse = np.sqrt(np.mean((predicted - test)**2))
mape = np.mean(np.abs((test - predicted) / test)) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
