import numpy as np

# Define the inputs and expected outputs for the AND gate
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
expected_outputs = np.array([0, 0, 0, 1])

# Define weights and threshold
weights = np.array([1, 1])  # Initial weights
threshold = 1.5

# McCulloch-Pitts Perceptron Function
def perceptron(x, weights, threshold):
    # Calculate weighted sum
    weighted_sum = np.dot(x, weights)
    # Apply threshold
    return 1 if weighted_sum >= threshold else 0

# Test the Perceptron on each input and print results
print("AND Gate using McCulloch-Pitts Model:")
for i in range(len(inputs)):
    output = perceptron(inputs[i], weights, threshold)
    print(f"Input: {inputs[i]} -> Output: {output} (Expected: {expected_outputs[i]})")