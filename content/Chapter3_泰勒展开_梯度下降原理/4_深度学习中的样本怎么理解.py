# _*_ coding=utf-8 _*_

"""
在深度学习中，样本通常指的是输入数据和对应的标签。这些数据可以是图像、音频、文本等。
 例如，在图像分类中，一个样本可能包括一张图像和它的标签（比如“狗”，“猫”，“汽车”等）。
在自然语言处理中，一个样本可能包括一段文本和它的标签（比如“正面”，“负面”，“中立”等）。

在线性回归中，一个已知的点通常表示为一个样本，包含输入值和对应的输出值。
例如，在训练集中，有多个样本点，每个样本点都有一个x值和一个y值

可以表示为(x1,y1),(x2,y2),(x3,y3)....(xn,yn)，这些样本点将用来训练线性模型来预测未知的y值。

每个样本都有自己的损失函数和梯度。在计算损失函数时，会使用每个样本的输入值和对应的输出值，
来计算该样本对应的损失值。在计算梯度时，则会使用该样本对应的损失函数来计算梯度

批量梯度下降会使用所有样本的梯度的平均值来优化参数
"""