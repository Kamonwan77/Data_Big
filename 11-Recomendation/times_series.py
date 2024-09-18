import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
# Load the data
df = pd.read_csv('year_sales.csv')

# Inspect the data
print(df.head())

# Convert 'Year' to datetime with the correct format
df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m')  # Correct format for 'YYYY-MM'

# Set 'Year' as the index
df.set_index('Year', inplace=True)

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(df, label='Original Data')
plt.title('Yearly Sales')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Perform the ADF test
result = adfuller(df['Sales'].dropna())  # Drop NA values if any
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into train and test sets
train_size = int(len(df) * 0.8)  # Adjust as needed
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(1, 1, 1))  # Adjust order as needed
model_fit = model.fit()

# Print the summary of the ARIMA model
print(model_fit.summary())

# Create predictions
# Create predictions using get_forecast for more control over confidence intervals
forecast = model_fit.get_forecast(steps=len(test))
predictions = forecast.predicted_mean

# Ensure predictions have the same index as test
predictions.index = test.index

print(forecast)
# Plot the train, test, and predicted data
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predicted', color='red')
plt.title('Train, Test, and Predicted Data')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.show()
