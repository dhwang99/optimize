# encoding: utf8

'''
线性规划的单纯型方法实现

z = c.T * x_b
min z
s.t.  D*x_b <= b,
      x_b >= 0

引入松驰变量(向量)xp，  x_n = b - A*x_b  >= 0,
   D * x_b + I * x_n = b
   [D  I][x_b;x_n] = b

简写为：
   min z = c.T * x
   s.t. Ax = b,
   x >= 0

   A = [D I],
   x = [x_b; x_n]


其中对A要求行满秩(如果非行满秩，则约束条件行相关，Ax = b可能无解，需要把线性相关的线束去掉)

单纯形法，上式转为如下线性方程组：
 [1 -c.T 0; 0 D I]  * [z;x_b;x_n] = [0;b] 
 x >= 0
或
 [1 -c.T 0; 0 A]  * [z;x] = [0;b] 

 x >= 0

求满足上式z的最小值


以下解法中，只考虑 约束条件 <= b， 不考虑等式情况
引入松驰变量后的系数矩阵，其秩为m (m个松驰变量构成了m * m单位阵)

等式情况, 可以通过类似增加松驰变量的方式解决

选主元算法，两种：
1. dantzig规则 
   最大正判别数对应的列，正判别数最小标为列标
   最小比值行标出基
   对退化问题，可能会死循环，无最优解

2. Bland规则  
   正判别数最小下标进基， 即
     l = min{j|seta_j > 0, 1<=j <=n}

'''

import numpy as np
import sys
import pdb

class Simplex:
    def __init__(self, C, max_mode=False):
        self.mat = np.array([0] + C) 
        if max_mode:
            self.mat *= -1

    '''
    ax <= b
    x >= 0
    '''
    def add_constraint(self, a, b):
        self.mat = np.vstack((self.mat, [b] + a))

    def solve(self):
        '''
        松驰后的增广矩阵 [z C;b A] 
        '''
        cm,cn = self.mat.shape
        B = np.array(range(cm-1, cm+cn-1))  #B0, 初始基的列编号
        temp = np.vstack((np.zeros(cm-1), np.eye(cm-1)))  #第一行：z=c.T * x; 2 ..m:  m-1 维限制条件
        self.mat = np.hstack((self.mat, temp))
        m,n = self.mat.shape
        #判别数: C.T - Cb.T * B.inv * A, 
        #Z.new - Z = (CN.T - Cb.T * B.inv * CN) * XN
        #theta0: B.inv = B = np.eye(m - 1)
        theta = self.mat[0]
        while theta[1:].min() < 0:
            col = theta[1:].argmin() + 1
            #出基判别
            out_row = np.zeros(m)
            out_row[0] =  sys.maxint
            for i in range(1,m):
                out_row[i] = sys.maxint
                if self.mat[i, col] > 1e-9:   #如果全小于0, 需要处理
                    out_row[i] = self.mat[i, 0] / self.mat[i, col]

            row = out_row.argmin()

            self.mat[row,:] = self.mat[row,:] / self.mat[row, col] 
            for i in range(m):
                if np.abs(self.mat[i, col]) > 1e-9 and i != row:
                    self.mat[i,:] -= self.mat[row,:] * self.mat[i, col] 
            
            B[row-1] = col

        z = -1 * self.mat[0,0]

        sol=np.zeros(cn-1)
        for i in range(len(B)):
            if B[i] < cn:
                sol[B[i]-1] = self.mat[i+1,0]

         
        return z, sol


'''
min z=-x1 - 14x2 - 6x3
   s.t.  x1 + x2 + x3 <= 4
         x1 <= 2
         x3 <= 3
         3x2 + x3 <= 6

 answer: -32.0
'''

if __name__ == "__main__":
    z = [-1, -14, -6]
    t1 = Simplex(z)
    t1.add_constraint([1,1,1], 4)
    t1.add_constraint([1,0,0], 2)
    t1.add_constraint([0,0,1], 3)
    t1.add_constraint([0,3,1], 6)

    result,sol = t1.solve()
    
    print "expect result -32; real result:", result

