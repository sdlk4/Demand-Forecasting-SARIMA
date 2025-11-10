import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load cleaned data
data_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\data\AirPassengers.csv"
df = pd.read_csv(data_path)

# Prepare data
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.dropna(subset=['Month'])
df = df.sort_values('Month')
df = df.set_index('Month')

y = df['#Passengers']

# 1st differencing
y_diff1 = y.diff().dropna()

# Seasonal differencing: lag = 12
y_diff12 = y.diff(12).dropna()

# Combined differencing (most stationary)
y_diff2 = y.diff().diff(12).dropna()

# Plot the differenced series
plt.figure(figsize=(12,5))
plt.plot(y_diff2, label="Stationary (d=1, D=1, m=12)", color='purple')
plt.title("Combined Differenced Series")
plt.legend()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(12,5))
plt.clf()
plt.close('all')
plot_acf(y_diff2, lags=40)
plt.title("ACF Plot")
plt.show()

plt.figure(figsize=(12,5))
plt.clf()
plt.close('all')
plot_pacf(y_diff2, lags=40, method='ywm')
plt.title("PACF Plot")
plt.show()
