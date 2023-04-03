# _*_ coding=utf-8 _*_

"""
均方误差（Mean Squared Error，简称MSE）损失函数是一种常用的损失函数，用于衡量预测值与真实值之间的差距。MSE损失函数的计算公式为：

MSE = (1 / n) * Σ(y_i - ŷ_i)²

其中，n 表示样本数量，y_i 表示第 i 个样本的真实值，ŷ_i 表示第 i 个样本的预测值。

MSE损失函数的特点：

MSE损失函数值始终为非负数。当预测值与真实值完全相等时，MSE的值为0。
MSE损失函数对于较大的误差会给予更大的惩罚（误差的平方），因此它对于异常值较为敏感。
MSE损失函数是连续可导的，这使得它在优化过程中更容易处理。

"""

# 这是一个使用MSE损失函数的简单神经网络实例代码：

import torch
import torch.nn as nn
import torch.optim as optim

# 生成一些模拟数据
x = torch.randn(100, 1) # torch.randn(100, 1)生成了一个100行1列的张量，张量中的值是从标准正态分布中随机采样得到的。
# 可以理解为生成了100个随机数，这些随机数可以用来作为模型的输入。
y = 3 * x + 2 + torch.randn(100, 1) * 0.3
# x 是一个100行1列的张量，而 3x+2 表示将 x 中每个元素乘以 3 然后加上 2，得到一个100行1列的张量 y
# torch.randn(100, 1) * 0.3 的意思是，将上述生成的 100 个随机数乘以 0.3 得到一个标准差为 0.3 的正态分布
# 生成的随机数被用作噪声

# 定义一个简单的线性回归模型
"""
让我们从nn.Linear(1, 1)开始。nn.Linear是PyTorch中的一个类，它表示一个线性层。当我们调用nn.Linear(1, 1)时，
我们实际上是在创建一个输入维度为1，输出维度为1的线性层。这意味着该层只有一个输入特征和一个输出特征。在这个例子中，我们将这个线性层实例化为self.linear。

接下来，我们来看forward方法。在PyTorch中，我们需要为自定义的神经网络模型重载forward方法。这是因为forward方法定义了如何在给定输入数据时
处理模型的正向传播过程。在我们的例子中，正向传播很简单，只需将输入数据（x）传递给我们刚刚创建的线性层（self.linear），然后返回线性层的输出。

总之，我们创建了一个简单的线性模型，只包含一个线性层（nn.Linear(1, 1)）。我们重载forward方法，以便在给定输入数据时，该方法会将数据
传递给线性层并返回输出。这就是为什么我们在forward方法中返回self.linear(x)的原因。

self.linear 是 nn.Linear 类的一个实例，当你将这个实例当作一个函数来调用，如 self.linear(x)，实际上是在调用
 nn.Linear 类的 __call__ 方法。在 Python 中，当你定义一个类并实现了 __call__ 方法，该类的实例就可以像函数一样被调用。
【所以多亏nn.Linear 类的实现了__call__ 方法，我们可以像调用函数一样使用它的实例】

在 nn.Linear 类中，__call__ 方法实现了正向传播的计算逻辑。当你调用 self.linear(x) 时，你实际上是在执行这个计算逻辑。具体来说，
nn.Linear 类会对输入数据 x 应用线性变换（即矩阵乘法与偏置项的加法），并返回变换后的结果。

因此，self.linear(x) 表示将输入数据 x 传递给 nn.Linear 类的实例 self.linear，并计算正向传播的结果。这使得我们能够在 
forward 方法中方便地使用线性层的计算功能。

当你将输入数据 x 传递给 self.linear(x) 时，输出的 y 会是一个具有相同形状的张量。在我们的示例中，x 的形状是 (100, 1)，
因此输出张量 y 的形状也将是 (100, 1)。这是因为 nn.Linear 类的实例 self.linear 是一个输入维度为1，
输出维度为1的线性层。在执行矩阵乘法与偏置项的加法后，输出张量的形状将与输入张量的形状相同。

"""


class LinearModel(nn.Module):  # 继承了PyTorch中的nn.Module类， 并重载了它的两个方法：__init__和forward
    def __init__(self):
        super(LinearModel, self).__init__()  # 调用父类的初始化函数来初始化模型，以确保我们的自定义类包含了nn.Module类的所有方法和属性。
        # self 是指当前实例本身
        self.linear = nn.Linear(1, 1)  # nn.Linear(1, 1)的意思是构建一个输入维度为1，输出维度也为1的线性层，相当于实例化了一个对象，
        # 而且这个对象实现了__call__方法，可以像函数一样调用。这个对象被实例化出来后，赋予给了self.linear属性。

    def forward(self, x):  # 然后又重载了forward方法
        return self.linear(x)

model = LinearModel()  # 当你对一个继承自 nn.Module 的类进行实例化之后，就可以使用这个实例来构建神经网络模型。

# 使用MSE损失函数
loss_function = nn.MSELoss()  # 这行代码实例化了一个 MSELoss 对象，并将其赋值给了 loss_function 变量

# 使用随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    # 前向传播
    predictions = model(x) # 将张量x输入到模型中时，实际上是调用了模型的forward方法，forward方法接收输入的数据张量x作为参数，
    # 然后将其通过模型的线性层计算得到预测值predictions。

    # 计算损失
    loss = loss_function(predictions, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
