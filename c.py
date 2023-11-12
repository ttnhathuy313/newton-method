import numpy as np
from main import newton

def function_c(x):
    return np.log(np.exp(x[0]) + np.exp(-x[0]))

def gradient_c(x):
    return np.array([np.exp(x[0]) - np.exp(-x[0])]) / (np.exp(x[0]) + np.exp(-x[0]))

def hessian_c(x):
    return np.array([[4.0 / (np.exp(x[0]) + np.exp(-x[0])) ** 2]])

starting_point = np.array([1.0])
newton(starting_point, function_c, gradient_c, hessian_c)
