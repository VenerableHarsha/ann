import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate synthetic time series data (replace with real dataset)
np.random.seed(42)
data = np.sin(np.arange(0, 100, 0.1)) + np.random.normal(0, 0.1, 1000)  # Sine wave with noise
data = data.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences from data
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# Hyperparameters
seq_length = 50  # Length of the input sequence
train_size = int(len(scaled_data) * 0.8)

# Create training and test datasets
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Make predictions
predicted = model.predict(x_test)
predicted = scaler.inverse_transform(predicted)  # Inverse scaling

# Plot results
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
plt.plot(actual, label='Actual Data')
plt.plot(predicted, label='Predicted Data')
plt.legend()
plt.show()