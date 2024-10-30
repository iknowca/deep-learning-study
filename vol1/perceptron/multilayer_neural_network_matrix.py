import numpy as np
from sigmoid import sigmoid

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

x = np.array([1.0, 0.5])

a1 = x@W1 + b1
z1 = sigmoid(a1)
a2 = z1@W2 + b2
z2 = sigmoid(a2)
a3 = z2@W3 + b3
y = a3

print(f"y: {y}")
# y: [0.31682708 0.69627909]