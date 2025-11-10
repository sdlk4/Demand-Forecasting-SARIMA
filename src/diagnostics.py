import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

# Load dataset
data_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\data\AirPassengers.csv"
df = pd.read_csv(data_path)

df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')

y = df['#Passengers']

# Fit SARIMA model
model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()

# Plot diagnostics
results.plot_diagnostics(figsize=(12, 8))
plt.show()
