#encoding: utf8

import numpy as np
import pdb

'''
因为牛顿法、拟牛顿法在非线性优化上用得比较广，单独写一个文件
包括牛顿法、
'''

def f4():
    # f = 4*x1**2 + 2*x1*x2 + 2 * x2**2 + x1 + x2
    # normal: f = c + b.T * x + 1/2 * x.T * A * x
    # dst_x = np.array([-1.0/14, -3.0/14])
    c = 2
    b = np.matrix('1;1')
    A = np.matrix('8,2;2,4') 

    return c,b,A

'''
牛顿法. 要求有二阶导数
非区间搜索法。只要给到起始点，就可以下降(对凸函数是这样子)
'''
def newton_search_for_quad(f, x0, espilon):
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

        if np.abs(f_n - f_n_1) < espilon:
            return x_n, f_n

        x_n_1 = x_n
        f_n_1 = f_n

    return None


if __name__ == "__main__":
    x0 = np.matrix('0.0;0.0') 
    rst = newton_search_for_quad(f4, x0, 0.01)
    dst_x = np.array([-1.0/14, -3.0/14])
    
    print "expect x:0.63"
    print "nr: dst:", dst_x
    print "nr: rst:", rst 
