
import numpy as np
from and_perceptron import AND
from nand_perceptron import NAND
from or_perceptron import OR


def XOR(x):
    s1 = NAND(x)
    s2 = OR(x)
    y = AND(np.array([s1, s2]))
    return y