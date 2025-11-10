import pandas as pd
import matplotlib.pyplot as plt
import os

# Build path to dataset using OS-safe join
data_path = os.path.join("..", "data", "AirPassengers.csv")

print("Loading dataset from:", data_path)

# Read CSV
df = pd.read_csv(data_path)

print("\nLoaded Dataset Preview:")
print(df.head())

print("\nShape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Convert date column
# Change 'Date' if your dataset has different date column name
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print("\nNull dates after conversion:", df['Date'].isna().sum())

# Remove invalid dates
df = df.dropna(subset=['Date'])

# Sort
df = df.sort_values('Date')

# Set index
df = df.set_index('Date')

print("\nFinal Date Index Preview:")
print(df.index[:10])

# Missing values
print("\nMissing values in columns:")
print(df.isna().sum())

# Check duplicates
dupes = df.index.duplicated().sum()
print("\nDuplicate date entries:", dupes)

# Merge duplicates
if dupes > 0:
    df = df.groupby(df.index).sum()
    print("\nDuplicate dates merged.")

# Plot time series
plt.figure(figsize=(12, 5))
plt.plot(df[df.columns[1]] if len(df.columns) > 1 else df[df.columns[0]], label="Time Series")
plt.title("Time Series Plot")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Summary stats
print("\nSummary Statistics:")
print(df.describe())
