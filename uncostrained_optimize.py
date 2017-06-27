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

3. 共轭梯度方向下降法
  共轭方向法，开始用来求解 Ax=b方程组. 而在求解二次函数最优解时，其梯度函数为  Ax+b=0的解, 
  于是也用这个方法来求解二次函数的最优解。其它非二次的凸函数，则可以用二次函数来近似

  (x, Ay) = 0, 称x,y为共轭向量 (正交是共轭的特殊形式，A为单位阵)
  3.1 找到n个共轭向量 ui
  3.2 沿共轭向量方向进行最优搜索, 得到每次搜索的最优步长列表 lbi
  3.3 得到最优点：
       x_star = x0 + sum(lbi * ui)

  对非二次函数，可以这样逼近最优解
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

    
'''
f = 1 + x1 -x2 + x1**2 + 2x2**2
u1 = [1,0]
u2 = [0,1]
x0 = [0,0]

lambda1 = -1/2, lambda2 = 1/4
'''
def f2():
    c = 1.
    b = np.matrix('1.; -1.')
    A = np.matrix('2, 0; 0,4')

    return c, b, A 

from line_search import golden_section_search 

'''
共轭方向法：
1. 使用黄金搜索找lambda
'''
def conj_grandient_method_for_f2():
    u1 = np.matrix('1.;0.')
    u2 = np.matrix('0.;1.')
    x0 = np.matrix('0.;0.')

    def_field = [-1,1]
    esplison = 0.005
    c,b, A = f2()
    
    '''
    线性搜索用的一次函数, 参数为k
    f = f(xi + kui)
    '''
    k1 = golden_section_search(lambda k:f_value(f2, x0 + k*u1), def_field, esplison)
    x1 = x0 + k1[0] * x0

    k2 = golden_section_search(lambda k:f_value(f2, x1 + k*u2), def_field, esplison)
    x2 = x0 + k1[0] * u1 + k2[0] * u2 
    
    return x2, f_value(f2, x2)

'''
共轭方向法：
2. 对二次函数直接求解lambda
lamb_i+1 = (u(i+1), Ax0 + b) / (u(i+1), Au(u+1))
'''
def conj_grandient_method_for_f2_direct():
    u1 = np.matrix('1.;0.')
    u2 = np.matrix('0.;1.')
    x0 = np.matrix('0.;0.')

    c,b, A = f2()
    lamb1 = -1. * (u1.T * (A*x0 + b))/(u1.T * (A*u1))
    lamb2 = -1. * (u2.T * (A*x0 + b))/(u2.T * (A*u2))
    
    x2 = x0 + lamb1[0,0] * u1 + lamb2[0,0] * u2 
    
    return x2, f_value(f2, x2)


def conj_f3():
    A = np.matrix('1,1;1,2')
    c = 0
    b = np.matrix('0;0')

    return c,b,A

'''
Fletcher_Reeves_conj
关于v0,v1,...vn共轭，最好推导一次
x0, x1, ...
v0, v1,....
xi = xi_1 + lambda * vi_1
vi = -gi + ||gi||/||gi_1|| * vi_1  
沿共轭方向求极小值：
gi, 第xi点的梯度值

example:

f=1/2x.TAx
A=[1,1;1,2]
x0 = [10.;-5.]
v0 = g0

lamb0 = 0.75
x1 = [1.25,-3.75]
v1=[-4.36,3.75]
lamb1 = 1.34

x2 = [0.4, 0.01]
再迭代一次?
讲义上似乎算错了。有空算一下
'''
def Fletcher_Reeves_conj():
    f = conj_f3
    c,b,A = f()
    x0 = np.matrix('10.;-5.')
    g0 = A * x0 + b 
    v0 = -g0

    #pdb.set_trace() 
    lamb0, f_x0 = golden_section_search(lambda k:f_value(f, x0 + k*v0), [0,2], 0.001) 

    x1 = x0 + lamb0 * v0
    g1 = A*x1 + b 
    v1 = -g1 + np.dot(g1.T, g1)[0,0]/np.dot(g0.T, g0)[0,0] * v0
    lamb1, f_x1 = golden_section_search(lambda k:f_value(f, x1 + k*v1), [0,2], 0.001) 
    x2 = x1 + lamb1 * v1
    return x2, f_x1 

'''
'''
def f_powell():
    c = 0
    b = np.matrix('0.;0.')
    A = np.matrix('2,0;0,4')

    return c,b,A

'''
相比于fletcher算法，powell算不需要计算梯度。但需要有n个线性无关的初始向量, 

1. 每一步的过程 
   xi = xi_1 + lambda * vi_1, lambda线性搜索后的最小值
   vi --> vi_1
   xn-x0 --> vn
   u0 = xn-x0
   x0 = xn + lambda * (xn - x0), 为新的x0值

   算法本身所需要的步数，并不比fletcher小

2. 重复上述步骤，直到收敛 
  留意下收敛条件：如 ||xi - xi_1|| < espilon, |fi - fi_1| < esplilon, max_steps


这种产生共轭向量的方法，并没有推导，只有实现。最好去推一下


这两个算法，和DFP/BFGS关系密切。具体参考 newton算法相关内容

'''
def powell_conj():
    '''
    u1=[-11.14, -24.46]
    u2=[-1.8, -0.28]
    '''
    x0 = np.matrix('20.;20.')
    #v1,v2线性无关
    v = np.matrix('1.,1.;-1.,1.')

    c,b,A = f_powell()
    u = np.matrix('0.,0.;0.,0.')
    lamb = np.matrix('0.;0.')
   
    id = 0
    total = 0
    while total < 3:
        k,min_fk = golden_section_search(lambda k:f_value(f_powell, x0 + k*v[:,0]), [-100., 100], 0.001)
        x1 = x0 + k * v[:,0]

        k,min_fk = golden_section_search(lambda k:f_value(f_powell, x1 + k*v[:,1]), [-100., 100], 0.001)
        x2 = x1 + k * v[:,1]

        #找到u向量
        u[:,id] = x2 - x0
        k,min_fk = golden_section_search(lambda k:f_value(f_powell, x2 + k*u[:, id]), [-100., 100], 0.001)

        x0 = x2 + k*u[:,id]
        v[:,0] = v[:,1]
        v[:,1] = u[:,id]

        id = (id+1) % len(lamb)
        total += 1

        conj = u[:,0].T * A * u[:,1]
        #print "conj: ", conj

    x_star = x0

    return x_star, f_value(f_powell, x_star)

from line_search import newton_search_for_quad

from newton_method import newton_search_for_quad 

if __name__ == "__main__":
    rs = solve_direct(f1)
    print "\nsolve_direct:", rs

    x0 = np.matrix('4;4')
    ors = optimal_grandient_for_f1(f1, x0, 0.01)
    print "\noptimal_grandient_for_f1:", ors
   
    nrs = newton_search_for_quad(f1, x0, 0.01)
    print "\nnewton_search_for_quad(f1):", nrs

    conj_rst = conj_grandient_method_for_f2()
    print "\nconj_grandient_method_for_f2:", conj_rst 

    conj_rst = conj_grandient_method_for_f2_direct()

    print "\nconj_grandient_method_for_f2_direct:", conj_rst 

    nrs = newton_search_for_quad(f2, x0, 0.01)
    print "\nnewton_search_for_quad(f2):", nrs

    rs = solve_direct(f2)
    print "\nsolve_direct:", rs

    frs = Fletcher_Reeves_conj()
    print "Fletcher_Reeves_conj.\nexpect: x2 = [0.4, 0.01]. \nReal:", frs

    frs = powell_conj()
    print "Fletcher_Reeves_conj.\nexpect: x2 = [0.4, 0.01]. \nReal:", frs
