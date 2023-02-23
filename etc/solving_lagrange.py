# import numpy as np
# from scipy.optimize import minimize

# def new_func(x):
#     return - (x[0]**2 * x[1] - x[0] **2 - x[1]**2)

# def lagr_func(x, b):
#     return - (x[0]**2 * x[1] + x[2] * (b - x[0] **2 + x[1]**2))

# x0 = np.random.rand(3)
# res1 = minimize(new_func, x0, method = 'nelder-mead')
# res2 = minimize(lagr_func, x0, method = 'nelder-mead', args=(1,))

# print(res1)
# print(res2)

import sympy as sp
sp.init_printing()
x, y = sp.var('x, y')

# f = pow(x, 2) * y
f = 100 * pow(x, 2/3) * pow(y, 1/3)
# g = pow(x, 2) + pow(y, 2) - 1
g = 20 * x + 2000 * y - 20000

lamda = sp.symbols('lambda')
L = f- lamda * g

gradL = [sp.diff(L, var) for var in [x, y]]
eqs = gradL + [g]
print(eqs)

solution = sp.solve(eqs, [x, y, lamda], dict=True)
print(solution)

vals = [f.subs(p) for p in solution]
print(vals)