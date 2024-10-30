import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A.shape: {A.shape}")
print(f"B.shape: {B.shape}")
print(f"np.dot(A, B): {np.dot(A, B)}")
print(f"A@B: {A@B}")