# _*_ coding=utf-8 _*_
"""
在一元微分学里面，可导完全等价于可微，导数的几何意义是该函数曲线在某点上的切线斜率
求导的等价形式，limit(x->x0)(f(x)-f(x0)/(x-x0))
微分的数学定义，微分是一种改变量，我们先观察一个例子。y=f(x)=x^3
微分Δy = (x0+Δx)^3 - (x0)^3 = 3*(x0)^2*Δx + 3*x0*(Δx)^2 + (Δx)^3
微分Δy的满足要求是，Δy = A*Δx + o(Δx), o(Δx)是高阶无穷小的意思，指的是上面式子中3*x0*(Δx)^2 + (Δx)^3部分，因为都是3次方
所以趋近为零，因此进一步可得， Δy = A*Δx，也就是微分与x的改变量呈类似线性关系，我们就把这个函数称为在这点可微
那么这个A等于多少呢？上面式子可以进一步变成  f(x0+Δx) - f(x0) = A*Δx
两边都除以Δx，可得A = (f(x0+Δx) - f(x0))/Δx，也就是A就等于x0处的导数

所以微分的定义是，当自变量的变化很小时，自变量的变化量与函数导数的乘积
导数:是指函数在某一点处变化的快慢,是一种变化率，类似于速度
微分：是指函数在某一点处（趋近于无穷小）的变化量，是一种变化的量，类似于距离
导数就是横坐标的微分，除以纵坐标的微分,所以导数又叫微商

一个复杂的函数，自变量有微小的变化量，比如0.003，求函数的变化量是多少，也就是求函数的微分，微分就是微小的部分，微小的变化量；
复杂函数意味着里面可能有x^2,lnx,...x^2,lnx,...x^2,lnx,...等很难计算的函数，所以通过导数求出微分，也就是微分公式：dy = f(x)' * Δx
将x的改变量乘以导数，获得的y的微小改变距离，就是微分
"""