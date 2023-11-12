import numpy as np
from main import newton

def function_b(x):
    return (10 * x[0] ** 2 + x[1] ** 2) / 2

def gradient_b(x):
    return np.array([10 * x[0], x[1]])

def hessian_b(x):
    return np.array([[10, 0], [0, 1]])

starting_point = np.array([0, 0])
newton(starting_point, function_b, gradient_b, hessian_b)
