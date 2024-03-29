# _*_ coding=utf-8 _*_

"""
在深度学习中，神经网络的最后一层通常被称为输出层。输出层的节点数量取决于问题的类型，
其中对于分类问题通常有输出节点数量等于类别的数量，对于回归问题通常只有一个输出节点。
输出层通常使用特殊的激活函数，如softmax或sigmoid来将输出转换为概率或预测值。

对于分类问题，输出层通常使用softmax激活函数。它将输出转换为概率分布，使得每个输出节点的值都在0到1之间，
并且所有输出节点的值的总和为1。这样可以很容易地确定输出类别。

对于回归问题，输出层通常使用线性激活函数(identity)或者叫做无激活函数，它不对输出值进行转换，直接输出预测值。

对于二分类问题，输出层通常使用Sigmoid激活函数，它将输出转换为概率值，在0到1之间。

对于非线性问题，隐藏层通常使用ReLU(Rectified Linear Unit)或者tanh(hyperbolic tangent)激活函数，
它们能够解决非线性问题。

我们知道输入层是没有激活函数的，隐藏层是肯定有激活函数的，而输出层【不一定要有激活函数】

在回归任务中，通常不使用激活函数，也可以使用线性激活函数，如恒等函数。这是因为回归任务的目标是预测一个连续的输出值，
不需要进行非线性变换，所以输出层通常不使用激活函数。

假设你要预测一个房屋的价格，这是一个回归任务。你的模型输入是房屋的面积、年龄、地理位置等信息，最后预测出一个价格，也就是一个连续的数字。
这时，如果你在输出层使用了恒等函数作为激活函数，这个激活函数不会对结果进行非线性的变换，因此输出的价格仍然是一个连续的数字，符合了回归任务的要求。

与此相反，如果你在输出层使用了非线性的激活函数，例如sigmoid，则输出的价格会经过非线性变换，被限制在0~1之间，这就不符合回归任务的要求了。
因此，在回归任务中，输出层通常不使用非线性的激活函数，而是使用恒等函数。

这种恒等函数就不是我们传统意义上的激活函数,相当于"没有激活函数"。

"""