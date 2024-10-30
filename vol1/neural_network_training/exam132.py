from gradient_descent import gradient_descent

def f(x):
    return x[0]**2 + x[1]**2

init_x = [-3.0, 4.0]
result = gradient_descent(f, init_x, 0.000001, 100000000)
print(result)