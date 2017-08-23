### optimize
>  用python写了一些最优化相关的算法。包括线性规划、线搜索、无约束优化、约束优化等. 主要是为了加深对算法的理解，细节没考虑

#### 1. 线性规划(line program)
>    采用单纯形法解线性规划。在写的时候，部分实现有些小差异，这儿也保留下来了

>    相比非线性约束的优化问题，线性规划研究得比较透，这也是单独放到一个目录下的原因

>    [line_program](line_program)

#### 2. 线搜索法(line search)
>    实现了二等分、四等分、fibonacci搜索、黄金分割等精确搜索法
>
>    实现了二次多项式拟合等非精确搜索
>
>    [line_search](line_search)

#### 3. 牛顿法、拟牛顿法(newton_method)
>    实现了几个无约束优化的算法。包括 newton_method, DFP, bfgs, l-bfgs(没有实现)
>    正定二次型直接求解也丢到这里了(solve_direct)

>    [newton_method.py](newton_method.py)

#### 4. 最优梯度法(optimal_grandient)
>    实现了最优梯度, 二次多项式直接求解等

>    [optimal_grandient.py](optimal_grandient.py)

#### 5. 共轭方向、共轭梯度法(newton_method)
>    包括 共轭方向法(Fletcher_Reeves_conj, powell_conj, 已知共轭方向的二次多项式直接求解)

>    [conj_method.py](conj_method.py)

#### 6. 约束优化(constrained optimized)
>    实现了几个约束优化的算法。

>    包括几个原理性的程序，如 lagrange 等式、不等式优化, kkt 条件(不等式)

>    基于梯度的算法：hemstitching, combined_direction,  可行方向法

>    罚函数法(外点法)

>    内点法(未实现)
    
>    [costrained_optimize.py](costrained_optimize.py)

#### 7. 外部库写的小例子
>    调用cvxopt,scipy实现的线性规划示例和非线性优化示例

>    [sample](sample)

#### 8. 待补充内容
>    1. 线性搜索wolf条件
>    2. lagrange对偶问题
>    3. 共轭梯度&共轭方向的区别？
>    4. 在线梯度下降法
>    5. 随机算法：模拟退火、遗传算法
