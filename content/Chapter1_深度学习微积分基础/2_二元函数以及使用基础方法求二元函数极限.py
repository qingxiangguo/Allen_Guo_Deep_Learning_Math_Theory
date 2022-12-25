# _*_ coding=utf-8 _*_
"""
n元函数要在n+1维空间里面研究
z = x + y 是一个斜面；z = x^2 + y^2是一个桶状

二元函数求极限有多种方法，比如，可以化为一元函数求极限
例一：求极限lim(x->0, y->0) (x^2 + y^2)*sin(1/(x^2 + y^2))
令x^2 + y^2 = u, 可知 u ->0，原函数变为lim(->0) (u)*sin(1/(u))，变为无穷小*有界函数，所以结果为0

例二：使用分子有理化，求极限lim(x->0, y->0)(√(xy+1) - 1)/xy   # √是根号下的意思
分子分母同乘(√(xy+1) + 1)，可得结果为1/2

例三：使用重要极限，第一重要极限：lim(x->0)(sinx)/x = 1  第二重要极限：lim(n->无穷)(1+1/n)^n = e
求lim(x->0, y->2)  (1+x*(y)^2)^(1/sin2x)
解：把幂乘以 (x*y^2)*(1/xy^2)   # 这里是x乘以y的平方，不是x和y的平方

变成求lim(x->0, y->2)  [(1+xy^2)^(1/xy^2)]^(xy^2/sin(2x))

[(1+xy^2)^(1/xy^2)] 这个部分刚好是第二重要极限，等于e

由于第二重要极限，原式变为 e^(limit xy^2/sin2x)，再指数加一个2x/2x，进一步变形为e^lim (xy^2/2x * 2x/sin2x)
由于第一重要极限，lim(2x/sin2x) =1 ，代回原式，变为e^(y^2/2) = e^2

反正求极限中心思想就是各种变形，把无知变为已知
"""