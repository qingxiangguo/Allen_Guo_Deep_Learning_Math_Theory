# _*_ coding=utf-8 _*_

"""
Softmax 激活函数是一种常用的分类模型中的激活函数。它可以将任意数字转化为概率分布，每个输入数字对应一个类别，概率表示该数字对应类别的概率。

假设你有一个分类问题，有三个类别：猫，狗和鸟。训练好的神经网络给出了三个数字，分别表示这三个类别的得分。
Softmax 激活函数可以将这三个数字转化为三个概率，分别表示属于猫，狗和鸟三个类别的概率。
最后，我们可以选择概率最大的那个类别作为模型的预测。

Softmax激活函数通常用于分类问题的多项分类（多分类）任务中。它的输入值是隐藏层的输出值，

如果输出层有三个节点z1, z2, z3（分类为三个，一般来说，在隐藏层通常会使用其他的激活函数，例如sigmoid、ReLU等，来计算隐藏层的输出）
如果有三个隐藏层节点，h1, h2, h3，三个隐藏层的真正输出值为sigmoid(h1), sigmoid(h2), sigmoid(h3)

输出层权重分别为w11, w12, w13, b1; w21, w22, w23, b2; w31, w32, w33, b3

z1 = w11 * sigmoid(h1) + w12 * sigmoid(h2) + w13 * sigmoid(h3) + b1
z2 = w21 * sigmoid(h1) + w22 * sigmoid(h2) + w23 * sigmoid(h3) + b2
z3 = w31 * sigmoid(h1) + w32 * sigmoid(h2) + w33 * sigmoid(h3) + b3

接下来，softmax激活函数将该线性组合作为输入，
并对其进行归一化处理，将输入值转换为概率值，以代表每个类别的概率，即：

p1 = exp(z1) / (exp(z1) + exp(z2) + exp(z3))
p2 = exp(z2) / (exp(z1) + exp(z2) + exp(z3))
p3 = exp(z3) / (exp(z1) + exp(z2) + exp(z3))

其中，p1，p2，p3是三个类别的概率，exp(z)是自然常数e的幂。

计算z1, z2, z3时，对于同一个隐藏层节点h的计算结果（如sigmoid(h)），可以先提出来，避免重复计算。例如：

sigmoid_h1 = sigmoid(h1)
sigmoid_h2 = sigmoid(h2)
sigmoid_h3 = sigmoid(h3)

这样，对于一个隐藏层节点h，只需要计算一次sigmoid(h)，再分别使用该计算结果计算z1, z2, z3，可以节约计算时间。

计算概率p1, p2, p3时，确保每个值都是非负的，并且所有值的总和为1
通常情况下，在隐藏层上使用其他的激活函数，如sigmoid或ReLU，而不是softmax，
因为softmax只适用于输出层

"""