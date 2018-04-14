#encoding: utf8

import numpy as np
import pdb

'''
求二次型函数在指定点的值
其梯度值为： Gx + b 
'''
def f_value(f, x):
    c,b,A = f()
    val = c + b.T * x + 1./2.*x.T * A * x

    return np.sum(val)

def f4():
    # f = 4*x1**2 + 2*x1*x2 + 2 * x2**2 + x1 + x2
    # normal: f = c + b.T * x + 1/2 * x.T * A * x
    # dst_x = np.array([-1.0/14, -3.0/14])
    c = 2
    b = np.matrix('1;1')
    A = np.matrix('8,2;2,4') 

    return c,b,A

'''
牛顿法、拟牛顿法用在非线性优化上
不过因为牛顿法、拟牛顿法在非线性优化上用得比较广，单独写一个文件
包括: 直接求解、牛顿法、dsp、bfgs, l-bfgs(稍晚点实现)
'''

'''
对正定型二次型函数，直接求解最优点
Gx + b = 0
x = -G.inv * b
Ax = -b, 求解x
'''
def solve_direct(f):
    c,b,A = f()

    eigs = np.linalg.eigvals(A)
    less_zero = np.take(eigs, np.where(eigs < 0))
    if less_zero.shape[1] > 0:
        #非正定，不能求解。
        return None

    #x_star = -1. * np.linalg.inv(A) * b
    # 不用求逆. 用求解线性方程线的方法求解： x=A.inv * b ==> Ax = -b 
    x_star = np.linalg.solve(A, -b)

    return x_star, f_value(f, x_star) 


'''
牛顿法. 要求f有二阶导数
牛顿法非区间搜索法。只要给到起始点，就可以下降(对凸函数是这样子)

算法大致如下：
对原函数二阶tayler展开，然后用二次多项式近似：
f(x) = f(x0) + g(x)(x - x0) + 1/2(x-x0).T * H * (x - x0) + O((x-x0).T(x-x0))
f_sim(x) = f(x0) + g(x)(x - x0) + 1/2(x-x0).T * H * (x - x0)
f_sim'(x) = g(x) + H(x-x0) = 0
x = x0 - H.inv * g(x)

dk = -H.inv * f', 称为牛顿方向

牛顿法求解，计算量比较大，还要求H正定
H为海森矩阵, 要求是非奇异的

阻尼牛顿法
 由于牛顿法并不能保证收敛.于是改为沿dk方向进行一维搜索求，xk+1 = xk + lambk * dk, 

实际上并不直接求H.inv, 而改为解线性方程：
H*dk = f'

'''
def newton_search_for_quad(f, x0, epsilon):
    c,b,A = f()
    x_n_1 = x0
    x = x0
    f_n_1 = c + b.T * x + 1/2.0 * x.T * A * x 

    while True:
        H_f = A
        deriv_f = A * x_n_1 + b

        x_n = x_n_1 - np.dot(np.linalg.inv(H_f), deriv_f)
        x = x_n
        f_n = c + b.T * x + 1/2.0 * x.T * A * x 

        if np.abs(f_n - f_n_1) < epsilon:
            return x_n, f_n

        x_n_1 = x_n
        f_n_1 = f_n

    return None

'''
拟牛顿条件
如上式：
~=: sim_eq
f(x)在x_k+1点处理展开
f(x) ~= f(x_k1) + f'(x_k1)(x-xk) + 1/2*(x_k1-x).T * H * (x_k1-x)
两边同时作用一个梯度算子，则：
f'(x) ~= f'(x_k1) + H(x_k1)*(x_k1 - x)
记 x=x_k
   f'(x_k) ~= f'(x_k1) + H(x_k1) * (x_k1 - x_k)
   f'(x_k1) - f'(x_k) ~= H(x_k1) * (x_k1 - x_k)

记 B=H, D=H.inv, y_k = f'(x_k1) - f'(x_k), s_k = (x_k1 - x_k),则
y_k ~= H*s_k 
s_k ~= H.inv * y_k
上述即为拟牛顿条件，
对 H(k+1)或H(k+1).inv做近似  
y_k = B(k+1)*s_k
或：
s_k = D(k+1) * y_k

即用梯度近似计算H或H的逆.
'''

'''
Dk_1=Dk + deltaD
dletaD = alpha * u * u.T + beta * v * v.T
u = sk
v = Dk*yk
alpha = 1/(u.T * yk)
beta = -1/(v.T * yk)
deltaD = 1/(sk.T * yk) * sk * sk.T - 1/((Dk * yk).T * yk) * (Dk * yk).T * (Dk * yk) 
'''

from line_search import golden_section_search 

def DFP(f, f_deriv, x0, epsilon):
    x_k = x0
    D_k = np.eye(x0.shape[0])
    g_k = f_deriv(x0)

    while True:
        def_field = [-100., 100.]
        d = -1. * D_k * g_k
        k,min_fk = golden_section_search(lambda k:f(x_k + k*d), def_field, epsilon)
        x_k1 = x_k + k*d
        g_k1 = f_deriv(x_k1)
        
        #pdb.set_trace()
        g_sum = np.sum((g_k1.T * g_k1))
        g_sum = np.sqrt(g_sum)
        
        if g_sum < epsilon:
            break

        sk = x_k1 - x_k
        yk = g_k1 - g_k

        v =  sk
        u = D_k * yk

        alpha = 1. / (v.T * yk)[0,0] 
        beta = -1. / (u.T * yk)[0,0]

        delta_D = alpha * v * v.T + beta * u * u.T 
        
        #下一个点的D
        D_k = D_k + delta_D
        x_k = x_k1
        g_k = g_k1

    return x_k1, f(x_k1) 

'''
BFGS: 近似求H, 这样在计算下降方向时，需要求H的逆。求逆时会有优化方法. 而不是直接调用
方法1: 求解线性方程组：dk = -Bk * gk ---> Bk * dk = -gk, 
方法2: B_k1.inv = (I - sk*yk.T/(yk.T *sk) Bk.inv (I - yk*sk.T/(yk.T*sk)) + sk*sk.T/(yk.T*sk)
   np.linalg.inv(B_k)
'''
def BFGS_simple(f, f_deriv, x0, epsilon):
    x_k = x0
    B_k = np.eye(x0.shape[0])
    g_k = f_deriv(x0)

    while True:
        def_field = [-100., 100.]
        #D_k = np.linalg.inv(B_k)
        #d = -1. * D_k * g_k
        d = np.linalg.solve(B_k, g_k)

        k,min_fk = golden_section_search(lambda k:f(x_k + k*d), def_field, epsilon)
        x_k1 = x_k + k*d
        g_k1 = f_deriv(x_k1)
        
        #pdb.set_trace()
        g_sum = np.sum((g_k1.T * g_k1))
        g_sum = np.sqrt(g_sum)
        
        if g_sum < epsilon:
            break

        sk = x_k1 - x_k
        yk = g_k1 - g_k

        v = yk
        u = B_k * sk

        alpha = 1. / (v.T * sk)[0,0] 
        beta = -1. / (u.T * sk)[0,0]

        delta_B = alpha * v * v.T + beta * u * u.T 
        
        #下一个点的D
        B_k = B_k + delta_B
        x_k = x_k1
        g_k = g_k1

    return x_k1, f(x_k1) 

'''
方法2: B_k1.inv = (I - sk*yk.T/(yk.T *sk) Bk.inv (I - yk*sk.T/(yk.T*sk)) + sk*sk.T/(yk.T*sk)
该实现方法和原版算法书上的介绍不太一致，需要对一下
'''
def BFGS(f, f_deriv, x0, epsilon):
    x_k = x0
    B_k = np.eye(x0.shape[0])
    D_k = np.linalg.inv(B_k)
    g_k = f_deriv(x0)

    while True:
        def_field = [-100., 100.]
        d = -1. * D_k * g_k

        k,min_fk = golden_section_search(lambda k:f(x_k + k*d), def_field, epsilon)
        sk = k * d 
        x_k1 = x_k + sk 
        g_k1 = f_deriv(x_k1)
        yk = g_k1 - g_k

        g_sum = np.sqrt(np.sum((g_k1.T * g_k1)))
        if g_sum < epsilon:
            break

        #下一个点的D
        I = np.eye(x0.shape[0])
        rho = 1./(yk.T * sk)[0,0] 
        V = I - sk * yk.T * rho
        #D_k1 = (I - rho * sk * yk.T) * D_k * (I - rho* yk*sk.T) + rho * sk * sk.T
        D_k1 = V * D_k * V.T + rho * sk * sk.T #与上式等价

        D_k = D_k1
        x_k = x_k1
        g_k = g_k1

    return x_k1, f(x_k1) 

'''

'''
def L_BFGS(f, f_deriv, x0, epsilon):
    return None

if __name__ == "__main__":
    x0 = np.matrix('0.0;0.0') 
    esplison = 0.005
    c,b,A = f4()
    dst_x = np.matrix(np.array([-1.0/14, -3.0/14]))
    print "\nnr: dst:", dst_x

    dr = solve_direct(f4)
    print "\ndr: rst:", dr

    rst = newton_search_for_quad(f4, x0, 0.01)

    print "\nnr: rst:", rst 

    f = lambda x:f_value(f4, x)
    f_deriv = lambda x:(A*x + b)

    dfp_rs = DFP(f, f_deriv, x0, esplison)
    print "\ndfp rst:", dfp_rs 

    bfgs_rs = BFGS_simple(f, f_deriv, x0, esplison)
    print "\nbfgs rst:", bfgs_rs 

    bfgs2_rs = BFGS(f, f_deriv, x0, esplison)
    print "\nbfgs rst:", bfgs2_rs 
    
