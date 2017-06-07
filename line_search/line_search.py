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

epsilon = 0.1

def f1(x):
    f = 8 * (x ** 3) - 2 * (x ** 2) - 7 * x + 3.0
    return f

#二分
def bisection_search(f, def_field):
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


#四等分
def equal_interval_search(f, def_field):
    steps = np.linspace(def_field[0], def_field[1], 5).tolist()
    values = map(f, steps) 

    while True:
        sid = 0 
        if values[1] > values[3]:
            sid = 1
        
        ssid = sid
        if values[sid+1] > values[sid+2]:
            ssid = ssid + 1

        steps = np.linspace(steps[ssid], steps[ssid + 2], 5).tolist()
        values = [values[ssid], f(steps[1]), values[ssid+1], f(steps[3]), values[ssid+2]]

        if (steps[4] - steps[0]) < epsilon * 0.1:
            return steps[2], values[2]


if __name__ == "__main__":
    print "test"
    def_fd = [0.,1.]
    br = bisection_search(f1, def_fd)
    er = equal_interval_search(f1, def_fd)

    print "br:", br
    print "er:", er
