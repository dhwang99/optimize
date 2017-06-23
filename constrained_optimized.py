# encoding: utf8

import numpy as np
import pdb

'''
包括等式、不等式优化。
1. lagrange 等式、不等式优化
2. kkt 条件
3. 梯度下降
'''

# equal constrained
'''
示例1：
必要条件： lagrange_fun对应的梯度分量为0.
注： 并不是原问题有最优解，lagrange函数就能求出该最优解
比如这个函数：
min sqrt(x^2 + y^2)
s.t y^2 - (x-1)^3 = 0 
这个最优解为 (0, 1), 但这个解并不是lagrange函数的解

min x**2 + y**2
   s.t x + y =1
这个解是x + y = 1直线和上述出线等高线的切点
观察可得: 
    x_star = [1/2, 1/2]

   L(x,lambd) = x^2 + y^2 + lambd * (x + y - 1) 
              = x^2 + y^2 + x*lambd + y*lambd - lambd

   x = [x,y,lambd].T
   A = [2,0,1; 0,2,1; 1,1,0]
   b = [0,0,-1]
   c = 0
'''

def f1():
    A = np.matrix('2.,0.,1.; 0,2,1; 1,1,0')
    b = np.matrix('0.;0.;-1.')
    c = 0

    return c,b,A

'''
min x1^2 + 4x2^2
   s.t x1 + 2x2 = 6

x_star = [3., 3./2]

L(x, lamb) = x1^2 + 4x2^2 + lamb * (x1 + 2x2 - 6)
           = x1^2 + 4x2^2 + x1*lamb + 2x2 * lamb - 6*lamb

'''
def f2():
    A = np.matrix('2.,0,1;0,8,2;1,2,0')
    b = np.matrix('0.; 0.; -6.')
    c = 0

    return c,b,A

def equal_constrained_by_lagrange(f):
    c,b,A = f()
    #梯度方程： Ax = -b
    x = np.linalg.solve(A, -b) 

    return x, 1/2. * x.T * A * x + b.T * x


# inequality_constrained
'''

针对 lamb, seta 等于0分别讨论.
lamb_i == 0: seta_i^2 > 0, h_i(x) > 0, 即最优解在可行域里  
seta_i == 0: h_i(x) = 0, 最优解在可行域边界上, 即和等式优化问题一样
lamb_i,seta_i == 0: f_i' = 0,  h_i = 0, 最优解在 h_i边界上，由h_i的边界穿过f_i的最小点

L = f + sum(lamb_i * (h_i(x) - seta^2))

min (x-a)^2 
s.t x>=c

l = (x-a)^2 - lamb*(x-c - seta^2)
  = x^2 - 2ax + a^2 - lamb*x + lamb*c + lamb * seta^2

l是一个三次方程，不太好解。以下为简化解法：约束了 lamb == 0 (l1) or seta == 0 (l2)

l_x = 2(x-a) - lamb 
l_lambda = x - c - seta^2
l_seta = 2 * lamb * seta

'''

def inequality_constrained_by_lagrange(a, c):
    # lamb == 0
    x = a
    if x >= c:   #x > 0, l_lambda < 0
        return x, (x-a)**2

    #seta == 0
    x = c
    lamb = 2*(x - a)

    return x, (x-a)**2

if __name__ == "__main__":
    x_star = [1./2, 1./2]
    rs = equal_constrained_by_lagrange(f1)
    
    print "\nequality_constrained_by_lagrange:"
    print "\nexpect:", x_star
    print "\nreal:", rs

    x_star = [3., 3./2]
    rs = equal_constrained_by_lagrange(f2)

    print "\ninequality_constrained_by_lagrange:"
    print "\nexpect:", x_star
    print "\nreal:", rs

    a = 2
    c = 0
    rs = inequality_constrained_by_lagrange(a, c)
    print "\ninequality_constrained_by_lagrange:"
    print "a,c:", a,c
    print "real:", rs

    a = 2
    c = 2 
    rs = inequality_constrained_by_lagrange(a, c)
    print "\ninequality_constrained_by_lagrange:"
    print "a,c:", a,c
    print "real:", rs

    a = 2
    c = 4 
    rs = inequality_constrained_by_lagrange(a, c)
    print "\ninequality_constrained_by_lagrange:"
    print "a,c:", a,c
    print "real:", rs
