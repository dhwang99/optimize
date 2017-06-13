#encoding: utf8

'''
f = 8 * x**3 - 2*x**2 - 7*x + 3, x* = 0.63, epsilon = 0.1, 0=< 0 <= 1

主要方法：
1. 二分法
2. 等分法（四等分）
3. 

'''

import numpy as np
import pdb

#二分法

def f1(x):
    f = 8 * (x ** 3) - 2 * (x ** 2) - 7 * x + 3.0
    return f

# def_field2 = [0, 10.0]
# for fibonacci_search
# x* = 2.98
def f2(x):
    f = x ** 2 - 6 * x + 2
    return f

# def_field3 = [1, 2.0]
# for golden_section_search
# x* = 1.609
def f3(x):
    f = np.exp(x) - 5x
    return f

#二分
def bisection_search(f, def_field, epsilon):
    half_epsilon = epsilon / 2.0
    lf = list(def_field)
    f_vals = [f(lf[0]), f(lf[1])]

    while True:
        mid_x = (lf[1] + lf[0])/2.0
        mid_esp = [mid_x - half_epsilon, mid_x + half_epsilon]
        tmp_f = [f(mid_esp[0]), f(mid_esp[1])]
        if tmp_f[0] > tmp_f[1]:
            f_vals[0] = tmp_f[0]
            lf[0] = mid_esp[0]
        else:
            f_vals[1] = tmp_f[1]
            lf[1] = mid_esp[1]

        if (lf[1] - lf[0]) <= epsilon * 1.000001:
            x_star = (lf[0] + lf[1]) / 2
            return x_star, f(x_star)


#等分搜索法
#四等分
#取中间3个点的最小值点。该点两边两个点为新的区间，最小值落在该区间里
#如果有两个相同的最小值，则最小值就在这两个最小值点之间。算法适用
def equal_interval_search(f, def_field, epsilon):
    steps = np.linspace(def_field[0], def_field[1], 5)
    values = np.apply_along_axis(f, 0, steps)

    while True:
        min_id = values[1:4].argmin() + 1
        steps = np.linspace(steps[min_id-1], steps[min_id+1], 5)
        values = np.array([values[min_id-1], f(steps[1]), values[min_id], f(steps[3]), values[min_id+1]])

        if (steps[4] - steps[0]) < epsilon * 0.1:
            return steps[2], values[2]

    return None


fib_const_list = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, \
                  987, 1597, 2584, 4181, 6765,  10946, 17711, 28657, 46368, \
                  75025, 121393, 196418, 317811, 514229, 832040, 1346269,   \
                  2178309, 3524578, 5702887, 9227465, 14930352, 24157817,   \
                  39088169, 63245986, 102334155, 165580141,267914296,       \
                  433494437, 701408733, 1134903170, 1836311903, 2971215073, \
                  4807526976, 7778742049]

#fibonacci搜索
def fibonacci_search(f, def_field, espilon):
    fib_values = np.array(fib_const_list)  
    num = (def_field[1] - def_field[0]) / espilon
    fidx_n = np.where(fib_values >= num)[0][0]
    
    step = (def_field[1] - def_field[0]) * fib_values[fidx_n - 1] * 1.0/fib_values[fidx_n]
    points = [def_field[0], def_field[1] - step, def_field[0] + step, def_field[1]]
    f_values = np.apply_along_axis(f, 0, points)

    for i in range(fibid_n - 1, 2, -1):
        step = points[2] - points[1]
        if f_values[1] < f_values[2]):
            min_point = 1
            points[3] = points[2]
            f_values[3] = f_values[2]
            points[2] = points[1]
            f_values[2] = f_values[1]
            points[1] = points[2] - step
            f_values[1] = f(points[1])
        else:
            min_point = 2
            points[0] = points[1]
            f_values[0] = f_values[1]
            points[1] = points[2]
            f_values[1] = f_values[2]
            points[2] = points[1] + step
            f_values[2] = f(points[2])

    dst_p = (points[2] + points[1]) / 2

    return (dst_p, f(dst_p))

#黄金分割法
def golden_section_search(f, def_field, espilon):
    golden_const = 0.618

    step = (def_field[1] - def_field[0]) * 0.618 
    points = [def_field[0], def_field[1] - step, def_field[0] + step, def_field[1]]
    f_values = np.apply_along_axis(f, 0, points)

    while (step[2] - step[1]) > espilon:
        #step = (points[2] - points[1])
        step = (points[2] - points[0]) * (1 - golden_const)
        if f_values[1] < f_values[2]):
            min_point = 1
            points[3] = points[2]
            f_values[3] = f_values[2]
            points[2] = points[1]
            f_values[2] = f_values[1]
            points[1] = points[0] + step
            f_values[1] = f(points[1])
        else:
            min_point = 2
            points[0] = points[1]
            f_values[0] = f_values[1]
            points[1] = points[2]
            f_values[1] = f_values[2]
            points[2] = points[3] - step
            f_values[2] = f(points[2])

    dst_p = (points[2] + points[1]) / 2

    return dst_p, f(dst_p)

#牛顿法. 要求有二阶导数
#非区间搜索法。只要给到起始点，就可以下降(对凸函数是这样子)
def newton_method_search(x0, espilon):
    dst_x = np.array([-1.0/14, -3.0/14])
    A = np.matrix('8,2;2,4')
    b = np.matrix('1,1')
    x_n_1 = x0
    x = x0
    f_n_1 = b.T * x + 1/2.0 * x.T * A * x 

    while True:
        H_f = A
        deriv_f = A * x

        x_n = x_n_1 - np.dot(np.linalg.inv(H_f), deriv_f)
        x = x_n
        f_n = b.T * x + 1/2.0 * x.T * A * x 

        if np.abs(f_n - f_n_1) < espilon:
            return f_n, x_n

        x_n_1 = x_n
        f_n_1 = f_n

    return None


#非精确搜索
#多项式拟合法

#另一个方法()

if __name__ == "__main__":
    print "test"

    # x* = 2.98
    def_field2 = [0, 10.0]
    fr = fibonacci_search(f1, def_field2, 0.3)
    
    # x* = 1.609
    def_field3 = [1, 2.0]
    gr = golden_section_search(f1, def_fd, 0.01)

    def_fd = [0.,1.]
    br = bisection_search(f1, def_fd, 0.1)
    er = equal_interval_search(f1, def_fd, 0.1)

    print "br:", br
    print "er:", er
