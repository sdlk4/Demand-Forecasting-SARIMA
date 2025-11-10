import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---- Path to dataset ----
data_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\data\AirPassengers.csv"

# ---- Load and preprocess data ----
df = pd.read_csv(data_path)

df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.dropna(subset=['Month'])
df = df.sort_values('Month')
df = df.set_index('Month')

y = df['#Passengers']

# ---- Fit the recommended SARIMA model ----
model = SARIMAX(
    y,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()

print("\nSARIMA model fitted successfully!")
print(results.summary())

# ---- Forecast next 24 months ----
steps = 24
forecast_obj = results.get_forecast(steps=steps)
forecast = forecast_obj.predicted_mean
forecast_ci = forecast_obj.conf_int()

# ---- Plot observed vs forecast ----
plt.figure(figsize=(12,6))
plt.plot(y, label='Observed', color='blue')
plt.plot(forecast, label='Forecast', color='purple')
plt.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='pink',
    alpha=0.3
)
plt.title("AirPassengers Forecast (SARIMA)")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.show()

# ---- Save forecast to a file ----
output_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\forecast_output.csv"
forecast.to_frame("Forecast").to_csv(output_path)

print(f"\nForecast results saved to: {output_path}")
