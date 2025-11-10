import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ---- Correct absolute path to your dataset ----
data_path = r"C:\Users\srika\OneDrive\Desktop\demand_forecasting\data\AirPassengers.csv"

# ---- Load the CSV ----
df = pd.read_csv(data_path)

# ---- Print available columns and first few rows ----
print("Columns in dataset:", df.columns)
print("\nData preview:")
print(df.head())

# ---- Convert the correct date column ----
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')

# ---- Remove invalid rows ----
df = df.dropna(subset=['Month'])

# ---- Sort by date just in case ----
df = df.sort_values('Month')

# ---- Set Month as index ----
df = df.set_index('Month')

# ---- Select numeric column (#Passengers) ----
y = df['#Passengers']

print("\nUsing numeric column:", y.name)
print("Data index set to Month")

# ---- Plot original time series ----
plt.figure(figsize=(12,5))
plt.plot(y, color='blue', label='Original Series')
plt.title("Original Time Series (#Passengers)")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.legend()
plt.show()

# ---- ADF Test function ----
def adf_test(series, title=""):
    print(f"\nADF Test for {title}")
    result = adfuller(series.dropna())
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations Used']
    for value, label in zip(result[:4], labels):
        print(f"{label}: {value}")

    print("Critical Values:")
    for key, val in result[4].items():
        print(f"   {key}: {val}")

    if result[1] <= 0.05:
        print("STATIONARY — Reject H0")
    else:
        print("NON-STATIONARY — Fail to reject H0")

# ---- Run ADF test on original series ----
adf_test(y, title="Original Series")

# ---- Apply differencing ----
y_diff = y.diff().dropna()

# ---- Plot differenced series ----
plt.figure(figsize=(12,5))
plt.plot(y_diff, color='green', label='1st Difference')
plt.title("1st Order Differenced Series")
plt.xlabel("Date")
plt.ylabel("Difference")
plt.legend()
plt.show()

# ---- ADF test after differencing ----
adf_test(y_diff, title="1st Difference")
