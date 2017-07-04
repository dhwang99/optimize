#encoding: utf8

import numpy as np
import pdb

'''
1. 直接求解：对二次函数，直接求解
    x_star = A.inv * b

2. 最优梯度下降法
 2.1. 给定起始点
 2.2. 找到该方向的梯度
 2.3. 沿负梯度方向，找到该方向最优值 (lambda 参数。)
   a. 一维搜索方法
   b. 对二次函数，如果A为正定矩阵，可以直接求解
      f = c + b.T * x + 1/2 * x.T * A * x

      对纯二次函数：
      f = 1/2 * x.T * Q * x
      其k_star = -1 * x'QQx/x'QQQx

      df(xp + kAxp)/dk = A(xp + kAxp)*Axp = AxpAxp + kAAxpAxp 
                       = xp'AAxp + kxp'AAAxp 

'''

'''
f = 1/2(x1 * x1 + 2x2 * x2)
x0 = [4,4]
k* = -5/9
x1 = (16/9, -4/9)
'''
def f1():
    c = 0
    b = np.matrix('0;0')
    A = np.matrix('1, 0; 0,2')

    return (c, b, A)

from newton_method import f_value
from newton_method import solve_direct
   
def optimal_grandient_for_f1(f,x0, epsilon):  
    c,b,A = f()
    x = x0
    AA = A * A 
    AAA = A * A * A
    f_deriv = A * x
    while True:
        #对纯二次函数，直接求解k的值，而不是用线性搜索方法。推导公式见上
        k = -1. * (x.T * AA * x) / (x.T * AAA * x)
        k = np.sum(k)
        x_n = x + k * f_deriv

        f_deriv_n = A * x_n

        if np.sum(np.abs(f_deriv_n)) < epsilon:
            break
        
        x = x_n
        f_deriv = f_deriv_n

    return x,f_value(f, x)


from line_search import newton_search_for_quad
from newton_method import newton_search_for_quad 
from newton_method import solve_direct 

if __name__ == "__main__":
    rs = solve_direct(f1)
    print "\nsolve_direct:", rs

    x0 = np.matrix('4;4')
    ors = optimal_grandient_for_f1(f1, x0, 0.01)
    print "\noptimal_grandient_for_f1:", ors
   
    nrs = newton_search_for_quad(f1, x0, 0.01)
    print "\nnewton_search_for_quad(f1):", nrs
