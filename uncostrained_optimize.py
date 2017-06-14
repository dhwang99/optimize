#encoding: utf8

import numpy as np

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

3. 共轭梯度
  (x, Ay) = 0, 称x,y为共轭向量 (正交是共轭的特殊形式，A为单位阵)
  3.1 找到n个共轭向量 ui
  3.2 沿共轭向量方向进行最优搜索, 得到每次搜索的最优步长列表 lbi
  3.3 得到最优点：
       x_star = x0 + sum(lbi * ui)

  对非二次函数，可以这样逼近最优解


'''



def f_value(f, x):
    c,b,A = f
    val = c + b.T * x + x.T * A * x

    return val

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

   return [c, b, A] 

def solve_direct(f):
    c,b,A = f
    eigs = np.ling.eigvals(A)
    less_zero = np.take(eigs,np.where(eigs < 0))
    if less_zero.shape[1] > 0:
        #非正定，不能求解。这个还需要确认下？
        return None

    x_star = np.linalg.inv(A) * b
    val = f_value(f, x_star)

    return val, x_star
    
def optimal_grandient_for_f1(f,x0, epsilon):  
    c,b,A = f
    x = x0
    AA = A * A 
    AAA = A * A * A
    f_deriv = A * x
    while True:
        k = -1. * (x.T * AA * x) / (x.T * AAA * x)
        x_n = x + k * f_deriv

        f_deriv_n = A * x_n

        if np.sum(np.abs(f_deriv_n)) < epsilon:
            break
        
        x = x_n
        f_deriv = f_deriv_n

    return x,f_value(f, x)

    
'''
f = 1 + x1 -x2 + x1**2 + 2x2**2
u1 = [1,0]
u2 = [0,1]
x0 = [0,0]
'''
def f2():
    c = 1.
    b = np.matrix('1.; -1.')
    A = np.matrix('2, 0; 0,4')

   return [c, b, A] 

from line_search import newton_search_for_quad

def conj_grandient_method_for_f2():
    u1 = np.matrix('1.;0.')
    u2 = np.matrix('0.;1.')
    x0 = np.matrix('0.;0.')

    c,b, A = f2


    x1 = newton_search_for_quad()

