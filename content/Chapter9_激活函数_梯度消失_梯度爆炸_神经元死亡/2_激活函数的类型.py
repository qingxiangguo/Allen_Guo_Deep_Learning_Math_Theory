# _*_ coding=utf-8 _*_

"""
在深度学习中，常见的激活函数包括：

Sigmoid函数（Logistic函数）
Tanh函数（双曲正切函数）
ReLU函数（修正线性单元）
Leaky ReLU函数（泄漏修正线性单元）
ELU函数（指数线性单元）
Softplus函数
Swish函数
GELU函数
Maxout函数
softmax函数

每种激活函数都有其独特的优缺点，适用于不同的神经网络架构和任务。例如，Sigmoid函数和Tanh函数在某些场景下可能会导致梯度消失问题，
ReLU函数可能存在神经元死亡问题，而Leaky ReLU和ELU函数则是针对这些问题提出的改进版本。

通常情况下，sigmoid函数也可以用于二分类问题的输出层，因为其可以将输出限制在0-1之间，可以看作是概率值的近似。
而softmax函数则常用于多分类问题的输出层。其他激活函数则通常用于隐藏层。
"""