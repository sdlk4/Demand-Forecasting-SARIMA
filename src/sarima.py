import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
data_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\data\AirPassengers.csv"
df = pd.read_csv(data_path)

# Prepare dataset
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df = df.dropna(subset=['Month'])
df = df.sort_values('Month')
df = df.set_index('Month')

y = df['#Passengers']

# Fit SARIMA model
model = SARIMAX(
    y,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()

print("\nModel fitted successfully!")
print(results.summary())

# Forecast next 24 months
forecast_steps = 24
forecast = results.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Plot
plt.figure(figsize=(12,5))
plt.plot(y, label='Observed')
plt.plot(forecast.predicted_mean, label='Forecast', color='purple')
plt.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='pink', alpha=0.3
)
plt.title("SARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.show()

# Save forecast
forecast_df = forecast.predicted_mean.to_frame(name="Forecast")
forecast_df.to_csv(r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\forecast_output.csv")
print("\nForecast saved to forecast_output.csv")
