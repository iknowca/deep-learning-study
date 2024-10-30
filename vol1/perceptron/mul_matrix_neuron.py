import numpy as np
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = X@W

print(f"X.shape: {X.shape}")
print(f"W.shape: {W.shape}")
print(f"Y.shape: {Y.shape}")

print(f"Y: {Y}")