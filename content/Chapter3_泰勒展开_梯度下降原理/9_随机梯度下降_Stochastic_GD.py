# _*_ coding=utf-8 _*_

"""
随机梯度下降（Stochastic Gradient Descent，SGD）是一种梯度下降算法。它在每次迭代中，只使用一个样本来计算梯度，
并用该样本的梯度来更新参数。因为每次只使用一个样本来计算梯度，所以SGD称为随机梯度下降。

由于每次只使用一个样本来计算梯度，所以SGD比批量梯度下降更加敏捷，并且可以在大规模
数据集上进行在线学习。但是由于每次只使用一个样本来计算梯度，所以SGD的每一步更新
可能都是不准确的，所以SGD的收敛速度较慢

对于第一次迭代，我们随机选择一个样本(4,3)，计算该样本的损失函数，并更新参数b。

真实值 y1 = 3, 预测值 y^1 = 3*x1 + b = 3(4) + 0 = 12
损失函数L1 = (y真实值 - y预测值)^2 = (3 - 12)^2 = 81
梯度Δ1 = -2(y真实值 - y预测值) = 18
更新参数 b = b - η * 梯度 = 0 - η * (18) = -0.01 * (18) = -0.18

对于第二次迭代，我们随机选择一个样本(3,7)，计算该样本的损失函数，并更新参数b。

真实值 y3 = 7, 预测值 y^3 = 3*x3 + b = 3(3) + (-0.18) = 8.82
损失函数L3 = (y真实值 - y预测值)^2 = (7 - 8.82)^2 = 0.9924
梯度Δ3 = -2(y真实值 - y预测值) = -2(7 - 8.82) = 3.64
更新参数 b = b - η * 梯度 = -0.18 - η * (3.64) = -0.18 - 0.01 * (3.64) = -0.2044

这就是随机梯度下降算法（SGD）的前两次迭代的具体过程，
每次迭代都随机选择一个样本来更新参数，这种方法有一个显著的特点，
就是每次迭代都可能会跳到一个新的局部最优解上面，所以不像批量梯度下降算法（BGD）
和小批量样本梯度下降算法（MGD）那样会被卡在一个局部最优解上面。
"""