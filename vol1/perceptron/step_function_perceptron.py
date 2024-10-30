from functools import singledispatch
import numpy as np

@singledispatch
def step_function(x):
    raise NotImplementedError("Unsupported type")

@step_function.register(float)
def _(x):
    return 1 if x > 0 else 0

@step_function.register(np.ndarray)
def _(x):
    y = x > 0
    return y.astype(int)

