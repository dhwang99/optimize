import numpy as np
from scipy.optimize import linprog
import pdb 

'''
c = [-1., 4]
minimize c.T * x
s.t. [-3., 1].T * x <= 6
     [1,2.].T * x <= 4
     x[1] + x[2] = 10
     x[1] >= -3
'''
c = np.array([-1, 4])
A = np.array([[-3, 1], [1,2]])
b = np.array([6,4])

A_eq = np.array([[1,1]])
b_eq = 6
x0_bounds = (None, None)
x1_bounds = (-3, None)

res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=(x0_bounds, x1_bounds),options={"disp": True})

pdb.set_trace()
print res
