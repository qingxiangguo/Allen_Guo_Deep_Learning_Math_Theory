# _*_ coding=utf-8 _*_

"""
绝大多数的机器学习模型都会有一个损失函数，用来衡量机器学习模型的精确度
我们一般采用梯度下降这个方法。所以，梯度下降的目的，就是为了最小化损失函数
寻找损失函数的最低点，就像我们在山谷里行走，希望找到山谷里最低的地方。那么如何寻找损失函数的最低点呢？
在这里，我们使用了微积分里导数，通过求出函数导数的值，从而找到函数下降的方向或者是最低点（极值点）。

对于二元函数，导数就是梯度，函数值沿着梯度的负方向走，可以慢慢找到函数的全局极小值，这就是梯度下降的
核心思想。

梯度下降算法是一种优化算法，用于优化机器学习中的模型参数。它的基本思想是，
对于给定的训练数据和模型，通过不断地更新模型参数来最小化损失函数

我们要找到一个曲面上的最低点。这个曲面就是我们的损失函数。我们可以用一个点来代表当前的参数值，然后让这个点沿着梯度的反方向移动。
梯度是这个点周围的斜率。每次移动后，我们会发现损失函数值会变小。我们重复这个过程，直到点落在曲面上最低点。

举个例子，假设你要找到一个线条来拟合五个点（即线性回归问题）。你可以随便找一条线条作为初始线条
然后，我们需要定义一个损失函数来衡量我们选择的直线与所有点之间的差距。
一种常用的损失函数是均方误差（Mean Squared Error，MSE）。

接下来，我们要计算损失函数对于参数的梯度，这里的参数是斜率和截距。
对于所有样本点来说，我们都会计算出它们对应的损失函数，然后我们求所有样本点对应的损失函数对于参数的偏导，获得梯度，然后再求这些梯度的平均值。
这就是梯度下降算法中使用所有样本点来计算损失函数和梯度的思想。
"""