import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
import os

# Disable unnecessary TensorFlow optimizations for the system
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fetch Tesla stock data from Yahoo Finance (last 6 years)
ticker = "TSLA"
df = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Display the first few rows to check the data
df.head()

# Replace any null values with NaN and convert necessary columns to float
df.replace("null", np.nan, inplace=True)
df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].astype(float)

# Create rolling means for visualization (10, 20, 30, 40 day windows)
df_10 = pd.DataFrame()
df_10['Close'] = df['Close'].rolling(window=10).mean()
df_20 = pd.DataFrame()
df_20['Close'] = df['Close'].rolling(window=20).mean()
df_30 = pd.DataFrame()
df_30['Close'] = df['Close'].rolling(window=30).mean()
df_40 = pd.DataFrame()
df_40['Close'] = df['Close'].rolling(window=40).mean()

# Visualize the data
plt.figure(figsize=(20,10))
plt.plot(df['Close'].tail(200), label='Close Price')
plt.plot(df_10['Close'].tail(200), label='10-day MA')
plt.plot(df_20['Close'].tail(200), label='20-day MA')
plt.plot(df_30['Close'].tail(200), label='30-day MA')
plt.plot(df_40['Close'].tail(200), label='40-day MA')
plt.title('Tesla (TSLA) Close Price History (Past 6 Years)')
plt.xlabel('Date')
plt.ylabel('Close Price USD($)')
plt.legend(loc='upper left')
plt.show()

# Prepare data for LSTM
data = df.filter(['Close'])
dataset = data.values

# Define the length of training data (80% of the dataset)
import math
training_data_len = math.ceil(len(dataset) * 0.8)

# Scale the data for LSTM input
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create training data set
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train for LSTM (samples, time_steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(x_train.shape[1], 1)))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)

# Prepare test data set
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert x_test to numpy array
x_test = np.array(x_test)

# Reshape for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get model predictions
lstm_predictions = model.predict(x_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Evaluate the model
print(lstm_predictions[:5], y_test[:5])

mse = np.mean(lstm_predictions - y_test) ** 2
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")

# Plotting the results
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = lstm_predictions

plt.figure(figsize=(16, 8))
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.savefig("TSLA.png")
plt.show()

