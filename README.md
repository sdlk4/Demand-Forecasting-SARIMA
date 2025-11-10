# Demand Forecasting Using SARIMA
This project builds a statistical time-series forecasting model using Seasonal ARIMA (SARIMA) to predict monthly demand based on historical data. The goal is to provide accurate demand forecasts while modeling trend and seasonal variations in the dataset.

## Demand forecasting helps organizations:
- Reduce stockouts and overstocking
- Optimize inventory and resource allocation
- Improve budgeting and supply planning
- Detect seasonal demand fluctuations
- Make data-driven business decisions

## Project Structure
├── data/

│   └── AirPassengers.csv

│

├── output_images/

│   ├── Acf_Plot.png

│   ├── Combined_Differenced_Series.png

│   ├── Diagnostics.png

│   ├── Original_Time_Series.png

│   ├── Order_Differenced_Series.png

│   ├── Pacf_Plot.png

│   └── Sarima_Forecast.png

│

├── src/

│   ├── dataload.py

│   ├── stationarity.py

│   ├── acf_pacf.py

│   ├── sarima.py

│   ├── forecast.py

│   ├── diagnostics.py

│   └── evaluate.py

│

├── forecast_output.csv

├── requirements.txt

└── README.md

## Getting Started
## Prerequisites
- Ensure you have the following installed: Python 3.10 or above
- Packages listed in requirements.txt including:
  - pandas
  - numpy
  - matplotlib
  - statsmodels
  - scikit-learn

## Installation
- Clone the repository:
git clone https://github.com/<your-username>/Demand-Forecasting-SARIMA.git
- cd Demand-Forecasting-SARIMA

## Install dependencies:
- pip install -r requirements.txt
- Run the analysis scripts:
 - python src/stationarity.py
 - python src/acf_pacf.py
 - python src/forecast.py
- How the Project Works
1. Data Loading (dataload.py)
- Reads the dataset
- Converts date column to datetime
- Sets the date as index
- Displays initial preview
2. Stationarity Testing (stationarity.py)
- Performs the Augmented Dickey-Fuller (ADF) test
- Applies differencing and seasonal differencing
- Checks for stationarity
- Plots original and transformed series
3. ACF/PACF Analysis (acf_pacf.py)
- Plots autocorrelation (ACF) and partial autocorrelation (PACF)
- Helps determine ARIMA parameters
- Identifies seasonal behavior at lag 12
4. SARIMA Model Training (sarima.py)
- Fits SARIMA(1,1,1)(1,1,1,12) model
- Handles trend and seasonality
- Displays statistical summary
5. Forecasting (forecast.py)
- Generates 24-month forecast
- Plots actual vs forecasted demand
- Saves forecast to CSV
6. Diagnostics (diagnostics.py)
- Analyzes residuals using:
- Standardized residual plot
- Histogram and KDE
- Q-Q plot
- Correlogram
- Ensures residuals resemble white noise.
7. Evaluation (evaluate.py)
- Computes performance metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Sample output:
MAE: 67.58
RMSE: 73.45
MAPE: 14.69%

## Key Features
- Complete end-to-end SARIMA forecast pipeline
- Statistical testing for model assumptions
- Seasonal differencing and transformation
- Forecast visualization with confidence intervals
- Residual diagnostics for model reliability
- Error metric evaluation for performance assessment

## Sample Outputs
The repository includes the following plots:
- Original time-series plot
- First and seasonal differenced series
- ACF and PACF plots
- SARIMA forecast plot
- Residual diagnostics
- Images are stored inside the output_images/ directory.

## Model Performance Summary
The SARIMA model delivers:
- Accurate 24-month demand prediction
- Correct handling of trend and seasonality
- Acceptable error metrics (MAPE under 15 percent)
- Residuals that approximate white noise

## Data Requirements
The dataset should contain:
- A monthly or periodic date column
- A numeric variable representing demand

## Example:
- Month, #Passengers
- 1949-01, 112
- 1949-02, 118

## Contributing
Contributions are welcome. You can:
- Fork this repository
- Add enhancements or additional forecasting models
- Create a feature branch
- Submit a pull request

## Technical Notes
- SARIMA captures both seasonal and non-seasonal components
- Differencing is applied to achieve stationarity
- ACF/PACF guides parameter selection
- Residual diagnostics verify that the model generalizes well
- The pipeline is modular and extensible

