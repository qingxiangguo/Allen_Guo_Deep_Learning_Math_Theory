# _*_ coding=utf-8 _*_
"""
n元函数要在n+1维空间里面研究
z = x + y 是一个斜面；z = x^2 + y^2是一个桶状

二元函数求极限有多种方法，比如，可以化为一元函数求极限
例一：求极限lim(x->0, y->0) (x^2 + y^2)*sin(1/(x^2 + y^2))
令x^2 + y^2 = u, 可知 u ->0，原函数变为lim(->0) (u)*sin(1/(u))，变为无穷小*有界函数，所以结果为0

例二：使用分子有理化
"""