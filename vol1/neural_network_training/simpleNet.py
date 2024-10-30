import numpy as np
import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'perceptron'))

from gradient import numerical_gradient2d
from cee import cross_entropy_error
from softmax import softmax

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
    
net = SimpleNet()
print("W: " , net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print("p: " , p)
t = np.array([0, 0, 1])

loss = net.loss(x, t)
print("loss: " , loss)

f = lambda w: net.loss(x, t)
dW = numerical_gradient2d(f, net.W)
print("dW: ",dW)