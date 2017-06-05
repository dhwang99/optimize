from cvxopt import matrix, solvers

import pdb

########################################################################

## mimimize 2 x1 + x2

##subject to

## -x1 +x2 <= 1

## x1 + x2 >= 2

## x2 >= 0

## x1 - 2 x2 <= 4

########################################################################

c = matrix([2.0, 1.0])

b = matrix([1.0, -2.0, 0.0, 4.0])

A = matrix([[-1.0, -1.0, 0.0, 1.0],[1.0, -1.0, 1.0, -2.0]])

sol = solvers.lp(c,A,b)

pdb.set_trace()
print sol['x']
