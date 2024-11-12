import numpy as np

# Data and labels
X = np.array([[2, 3], [1, 1], [2, 1], [3, 3], [2, 2]])
y = np.array([1, -1, -1, 1, -1])

# LMS algorithm
w, b, lr = np.zeros(2), 0, 0.01
for _ in range(1000):
    for i in range(len(X)):
        y_pred = np.dot(X[i], w) + b
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

# Prediction
pred = np.sign(np.dot(X, w) + b)
print(f"Final Weights: {w}, Bias: {b}, Predictions: {pred}")