import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sigmoid import sigmoid
from step_function_perceptron import step_function
x = np.arange(-5.0, 5.0, 0.1)
y_sigmoid = sigmoid(x)
y_step = step_function(x)
plt.plot(x, y_sigmoid, label='sigmoid')
plt.plot(x, y_step, linestyle='--', label='step')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()