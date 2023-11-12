import numpy as np
from main import newton

def function_d(x):
    return -np.log(x[0]) + x[0]

def gradient_d(x):
    return np.array([-1.0 / x[0] + 1])

def hessian_d(x):
    return np.array([[1.0 / x[0] ** 2]])

starting_point = np.array([3])
newton(starting_point, function_d, gradient_d, hessian_d)