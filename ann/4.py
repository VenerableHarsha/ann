import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data
X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

# Training
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for _ in range(1000):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

# Evaluation
X_test = torch.tensor(X_test, dtype=torch.float32)
_, predicted = torch.max(model(X_test), 1)
accuracy = (predicted == torch.tensor(y_test)).float().mean().item()

print(f'Accuracy: {accuracy * 100:.2f}%')