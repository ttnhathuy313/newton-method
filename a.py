import numpy as np
from main import newton

def function_a(x):
    return np.power(x[0], 4) - 1

def gradient_a(x):
    return np.array([4 * np.power(x[0], 3)])

def hessian_a(x):
    return np.array([[12 * np.power(x[0], 2)]])

starting_point = np.array([4])
newton(starting_point, function_a, gradient_a, hessian_a)