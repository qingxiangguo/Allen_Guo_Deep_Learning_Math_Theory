# _*_ coding=utf-8 _*_

"""
使用一阶泰勒展开证明，梯度下降确实能让目标函数（损失函数）越来越小

假设w是你要优化的参数, w(n+1) = w(n) - α* J(wn)'，这是第一个公式 （1）

J(w)是代价函数，我们的目的就是让他变小

J(w)函数，在w(n)处的泰勒一阶展开式是，J(w) = J(wn) + J(wn)'(w-wn)  （2）

可以用这个公式求任意w的值，只要w不要离wn太远

那么我们令w = w(n+1)，并带入公式2

J(w(n+1)) = J(wn) + J(wn)'(w(n+1)-wn)

变形， J(w(n+1)) - J(wn) = J(wn)'(w(n+1)-wn)

代入公式(1) (w(n+1)-wn) = -α* J(wn)'

所以，J(w(n+1)) - J(wn) = -α* [J(wn)']^2

又因为学习率肯定是个正数，所以可得，J(w(n+1)) - J(wn) < 0

由此证明了梯度下降算法
"""