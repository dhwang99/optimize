### optimize
>  用python写了一些最优化相关的算法。包括线性规划、线搜索、无约束优化、约束优化等. 主要是为了加深对算法的理解，细节没考虑

#### 1. 线性规划(line program)
>    采用单纯形法解线性规划。在写的时候，部分实现有些小差异，这儿也保留下来了

>    相比非线性约束的优化问题，线性规划研究得比较透，这也是单独放到一个目录下的原因

>    [line_program](line_program)

#### 2. 线搜索法(line search)
>    [line_search](line_search)

#### 3. 无约束优化(uncostrained optimize)
>    实现了几个无约束优化的算法。包括

>    [uncostrained_optimize.py](uncostrained_optimize.py)

#### 4. 牛顿法、拟牛顿法(newton_method)
>    实现了几个无约束优化的算法。包括 newton_method, DFP, bfgs, l-bfgs(没有实现)

>    [newton_method.py](newton_method.py)

#### 5. 约束优化(constrained optimized)
>    实现了几个无约束优化的算法。包括

>    [costrained_optimize.py](costrained_optimize.py)
    

#### 6. 外部库写的小例子
>    调用cvxopt,scipy实现的线性规划示例和非线性优化示例

>    [sample](sample)
