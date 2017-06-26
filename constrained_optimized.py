# encoding: utf8

import numpy as np
import pdb

'''
本文件主要包括一些算法的示例，并不是完整的实现。仅用来做为加强学习用

包括等式、不等式优化。
1. lagrange 等式、不等式优化
2. kkt 条件(不等式)
3. 梯度下降
4. 罚函数法
5. 外点法
6. 内点法(未实现)
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
l_lambda = x - c - theta^2
l_theta = 2 * lamb * theta

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

'''
相比于lagrange不等式约束， 
KT约束不需要 theta参数，但有四个约束条件（本质上和 lagrange是一致的）

min f(x)
   s.t. gi(x) <= 0

解法:
L(x, lamb) = f(x) + sum(lamb_i * g_i(x))
满足如下四个条件:
partial_L/x = 0
partial_L/lamb <= 0 (g_i(x) = 0)
lamb_i >= 0
lamb * g = 0

min (x-a)^2 
s.t x>=c

L(x, lamb) = (x-a)^2 - lamb * (c-x)

partial_L/x = 2(x-a) - lamb * c
partial_L/lamb = c-x

'''
def inequality_constrained_by_KT(a, c):
    lamb = 0  #条件4,3
    x = a  #条件1
    g = c - x 
    if g < 0: #条件2
        return x, (x-a) ** 2

    x = c #g==0, 条件2,4
    lamb = 2. * (x - a) / c  #条件1

    if lamb >= 0:
        return x, (x-a) ** 2

    return None


'''
lagrange, kkt条件一般理论分析用。以下介绍工业界常用算法

min f(x)
s.t. g(x) >= 0  #注意是 >= 0
'''

'''
min x1^2 + 2*x2^2
s.t. x1 + x2 >= 4
'''
def cons_f1(x, lt=False):
    coef = np.matrix('1.;2')

    f = np.sum(coef.T *  np.multiply(x, x))
    deriv_f = 2 * np.multiply(coef, x)
    deriv_f = deriv_f / np.sqrt(deriv_f.T * deriv_f)  #归一化

    g = np.sum(x) - 4.
    deriv_g = np.matrix('1.;1.') 
    if lt:
        g = -g
        deriv_g = -deriv_g

    deriv_g = deriv_g / np.sqrt(deriv_g.T * deriv_g)  #归一化

    return f,deriv_f, g, deriv_g


'''
hemstitching方法(绣花算法)
1. 如果在可行域里，走负梯度方向
2. 如果不在，则走约束函数梯度方向, 回走(注：约束函数 gi >= 0)

算法：
1. 求 grad(f), grad(gi)
2. 如果 x_p在可行域里，走f负梯度方向；如果在外，走 g梯度方向，拉回到可行域里
3. 如果两次差值很少，则结束 
'''

def hemstitching_method(f_fun, x0, epsilon):
    x = x0
    f_pre = None
    x_pre = None
    max_iter = 1000
    cur_iter = 0
    k=1  #这个步长太大，问题多多
    k=0.1

    while True:
        f,deriv_f, g, deriv_g = f_fun(x)
        #pdb.set_trace()
        cur_iter += 1

        if f_pre != None and np.abs(f - f_pre) < epsilon:
            break

        d = None 
        if g >= 0:  #可行域里
            d = -deriv_f
            #不能停。只好加一个这个限制。加一个步长应该会好一些
            if cur_iter > max_iter:
                print "max itered"
                break

            if x_pre != None:
                dis = np.abs(np.sum((x_pre - x).T * (x_pre - x)))
                if np.sqrt(dis) < epsilon:
                    break

            x_pre = x
        else:
            d = deriv_g

        x = x + k * d 

        f_pre = f

    return x, f

'''
合成方向法
如果在可行点外，d = -deriv_f + sum(deriv_gi)
'''
def combined_method(f_fun, x0, epsilon):
    x = x0
    f_pre = None
    x_pre = None
    max_iter = 1000
    cur_iter = 0
    k=1  #这个步长太大，问题多多
    k=0.1

    while True:
        f,deriv_f, g, deriv_g = f_fun(x)
        #pdb.set_trace()
        cur_iter += 1

        d = None 
        if g >= 0:  #可行域里
            d = -deriv_f
            #不能停。只好加一个这个限制。加一个步长应该会好一些
            if cur_iter > max_iter:
                print "max itered"
                break

            if x_pre != None:
                dis = np.abs(np.sum((x_pre - x).T * (x_pre - x)))
                if np.sqrt(dis) < epsilon:
                    break

            if f_pre != None and np.abs(f - f_pre) < epsilon:
                break

            x_pre = x
            f_pre = f
        else:
            #pdb.set_trace()
            # 用 d = -deriv_f + deriv_g, 出不来
            d = -deriv_f + 2 * deriv_g

        x = x + k * d 

    return x, f



'''
上述两个方法，最好算一下步长。要不然一直收敛不了

可行方向法
保证方向是下降的

max x0
s.t. x0 + deriv_f.T * dp <= 0    #目标值下降
     x0 + deriv_g.T * dp - g(x0) <= 0   #在可行域内
     |dj| <= 1, x0 >= 0
'''

from scipy.optimize import linprog

def fliable_direction_method(f_fun, x0, esplison):
    x = x0
    f_pre = None
    x_pre = None
    max_iter = 1000
    cur_iter = 0
    k=1  #这个步长太大，问题多多
    k = 0.1
    k = 0.01  #这儿给的是固定步长。应该动态算一下：即在可行域里的最大k

    while True:
        f,deriv_f, g, deriv_g = f_fun(x, lt=True)

        if x_pre != None and np.sqrt(np.abs(np.sum((x_pre - x).T * (x_pre - x)))) < epsilon:
            break

        if f_pre != None and np.abs(f - f_pre) < epsilon:
            break

        c=np.array([-1., 0, 0])
        '''
        A = np.array([[1, deriv_f[0], deriv_f[1]], 
                      [1, deriv_g[0], deriv_g[1]]])
        b = np.array([0., -g])
        '''
        A = np.array([[1, deriv_f[0], deriv_f[1]], 
                      [0, deriv_g[0], deriv_g[1]]])
        b = np.array([0., -g])

        x0_bounds = (0, None) 
        d0_bounds = (-1., 1.)
        d1_bounds = (-1., 1.)

        #res = linprog(c, A_ub = A, b_ub = b, bounds=(x0_bounds, d0_bounds, d1_bounds), options={"disp":True})
        res = linprog(c, A_ub = A, b_ub = b, bounds=(x0_bounds, d0_bounds, d1_bounds))
        
        if res.success:
            x0, d0, d1 = res.x
            x[0] += d0 * k
            x[1] += d1 * k

            if x0 < esplison:
                f, deriv_f, g, deriv_g = f_fun(x)
                return x, f
        else:
            return None

    return None

'''
罚函数法

内点法
'''

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
    print "a,c, expect:", a, c, a
    print "real:", rs

    a = 2
    c = 2 
    rs = inequality_constrained_by_lagrange(a, c)
    print "\ninequality_constrained_by_lagrange:"
    print "a,c, expect:", a, c, a
    print "real:", rs

    a = 2
    c = 4
    rs = inequality_constrained_by_lagrange(a, c)
    print "\ninequality_constrained_by_lagrange:"
    print "a,c, expect:", a, c, c
    print "real:", rs

    a = 2
    c = 0
    rs = inequality_constrained_by_KT(a, c)
    print "\ninequality_constrained_by_KT:"
    print "a,c, expect:", a, c, a
    print "real:", rs

    a = 2
    c = 2 
    rs = inequality_constrained_by_KT(a, c)
    print "\ninequality_constrained_by_KT:"
    print "a,c, expect:", a, c, a
    print "real:", rs

    a = 2
    c = 4
    rs = inequality_constrained_by_KT(a, c)
    print "\ninequality_constrained_by_KT:"
    print "a,c, expect:", a, c, c
    print "real:", rs

    x_star = [2.667,1.333]
    x_bar = np.matrix('1; 4.5')
    rst = hemstitching_method(cons_f1, x_bar, 0.001)
    print "\ninequality_constrained_ hemstitching_method:"
    print "\nexpect:", x_star
    print "\nreal:", rst

    rst = combined_method(cons_f1, x_bar, 0.001)
    print "\ninequality_constrained_ combined_method:"
    print "\nexpect:", x_star
    print "\nreal:", rst

    x_star = [2.667,1.333]
    x_bar = np.matrix('0.85; 3.15')
    rst = fliable_direction_method(cons_f1, x_bar, 0.01)
    print "\ninequality_constrained_ fliable_direction_method:"
    print "\nexpect:", x_star
    print "\nreal:", rst

