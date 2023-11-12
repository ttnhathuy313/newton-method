import numpy as np

def function_a(x):
    return np.power(x[0], 4) - 1

def function_b(x):
    return (10 * x[0] ** 2 + x[1] ** 2) / 2

def function_c(x):
    return np.log(np.exp(x[0]) + np.exp(-x[0]))

def function_d(x):
    return -np.log(x[0]) + x[0]

def newton(starting_point, function, gradient, hessian, max_iteration=500):
    x = starting_point
    for i in range(max_iteration):
        x = x - np.dot(np.linalg.inv(hessian(x)), gradient(x))
        print("Iteration: ", i, " x: ", x, " f(x): ", function(x))
        print("Gradient: ", gradient(x))
        if np.linalg.norm(gradient(x)) < 1e-8:
            print("CONVERGED!!!")
            break
    return x


