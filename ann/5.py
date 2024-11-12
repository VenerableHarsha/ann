import torch

# Define the function f(x, y) = x^2 + y^2
def func(x):
    return x[0]**2 + x[1]**2

# Initialize input tensor with gradients enabled
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Compute the function's output
output = func(x)

# Compute the Jacobian (first derivatives) of the function
jacobian = torch.autograd.grad(output, x, create_graph=True)[0]
print("Jacobian:", jacobian)

# Compute the Hessian (second derivatives) matrix
hessian = torch.zeros(2, 2)  # Initialize a 2x2 Hessian matrix
for i in range(2):
    # Compute the gradient of each element of the Jacobian
    hessian_row = torch.autograd.grad(jacobian[i], x, retain_graph=True)[0]
    hessian[i] = hessian_row

print("Hessian:\n", hessian)