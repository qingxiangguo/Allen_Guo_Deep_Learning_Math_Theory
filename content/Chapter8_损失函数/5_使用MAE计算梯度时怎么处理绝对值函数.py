# _*_ coding=utf-8 _*_

"""
在使用 MAE 损失函数时，我们确实需要处理绝对值。为了计算梯度，我们需要求损失函数相对于参数的偏导数。

假设我们的 MAE 损失函数是：

MAE = (1/n) * ∑|y_i - f(x_i; w, b)|，其中 y_i 是真实值，f(x_i; w, b) 是预测值，n 是样本数量。

为了求解关于 w 和 b 的偏导数，我们需要考虑绝对值函数的导数。绝对值函数 |x| 在 x > 0 时导数为 1，在 x < 0 时导数为 -1。对于 x = 0，绝对值函数导数不存在。

由于我们需要计算关于 w 和 b 的偏导数，我们可以将绝对值函数分为两部分处理：

当 y_i - f(x_i; w, b) > 0 时，|y_i - f(x_i; w, b)| = y_i - f(x_i; w, b)。
当 y_i - f(x_i; w, b) < 0 时，|y_i - f(x_i; w, b)| = -(y_i - f(x_i; w, b))。

现在我们可以分别计算两种情况下损失函数关于参数 w 和 b 的偏导数。

当 y_i - f(x_i; w, b) > 0 时：

dMAE/dw = -(1/n) * ∑(y_i - f(x_i; w, b)) * df/dw
dMAE/db = -(1/n) * ∑(y_i - f(x_i; w, b)) * df/db

当 y_i - f(x_i; w, b) < 0 时：

dMAE/dw = (1/n) * ∑(y_i - f(x_i; w, b)) * df/dw
dMAE/db = (1/n) * ∑(y_i - f(x_i; w, b)) * df/db

注意，我们需要计算神经网络输出 f(x_i; w, b) 关于参数 w 和 b 的导数，这取决于具体的神经网络结构和激活函数。

对于 MAE 损失函数，我们需要分情况讨论。在实际计算中，我们可以根据每个样本的 y_i 和 f(x_i; w, b) 的值来确定具体是哪种情况。
实际上，我们在计算梯度时，会针对每个样本分别计算梯度，并将它们加起来。所以，每个样本在累积梯度时，都会根据自己的 y_i 和 f(x_i; w, b) 的
值选择相应的情况。

对于每个样本 i，我们可以按照以下步骤计算梯度：

计算 y_i 和 f(x_i; w, b) 的差值。
根据差值的符号确定是大于0还是小于0的情况。
对于大于0的情况，计算 dMAE/dw 和 dMAE/db 的梯度；对于小于0的情况，计算 -dMAE/dw 和 -dMAE/db 的梯度。
将每个样本的梯度累加起来得到总梯度。
在计算总梯度后，我们可以使用梯度下降法或其他优化算法来更新

在实际应用中，我们可以使用自动微分（automatic differentiation）工具，如 TensorFlow 或 PyTorch，来自动计算损失函数关于参数的梯度。
这些工具可以处理绝对值函数和其他非光滑函数的梯度计算，使得实现神经网络优化更加简便。

计算出损失函数关于参数 w 和 b 的梯度后，我们可以使用梯度下降法或其他优化算法来更新参数，从而降低损失函数的值。具体的更新公式如下：

w = w - α * dMAE/dw
b = b - α * dMAE/db

在 PyTorch 中，使用 MAE 作为损失函数时，可以方便地使用自动求导功能。我们不需要显式地考虑 y_i 和 f(x_i; w, b) 之间差值的符号。
这是因为 PyTorch 的自动微分系统会自动处理这些问题。

下面是一个使用 PyTorch 实现 MAE 损失函数的简单示例：

"""

import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性回归模型
model = nn.Linear(1, 1)

# 定义 MAE 损失函数
mae_loss = nn.L1Loss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 随机生成一些输入数据 x 和目标值 y
x = torch.randn(10, 1)
y = 2 * x + 1

# 训练模型
for epoch in range(100):
    # 计算模型的预测值
    predictions = model(x)

    # 计算 MAE 损失
    loss = mae_loss(predictions, y)

    # 使用自动求导计算梯度
    loss.backward()

    # 使用优化器更新模型参数
    optimizer.step()

    # 清除优化器的梯度缓存
    optimizer.zero_grad()

    # 打印损失值，查看训练过程
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

"""
loss.backward() 是在计算损失函数关于所有模型参数（即可训练的权重和偏差）的梯度。在 PyTorch 中，当你调用 loss.backward() 时，
它会自动遍历模型的所有参数（即所有需要梯度的张量），并计算它们关于损失函数的梯度。这些梯度会存储在每个参数的 .grad 属性中。

例如，在我们之前的线性回归模型示例中，模型有两个参数：权重 w 和偏差 b。当我们调用 loss.backward() 时，PyTorch 会计算损失函数关于 w 和 b 
的梯度，并将它们存储在对应的 .grad 属性中。这样，在使用优化器更新参数时，它就可以利用这些梯度进行更新。

如果你想在调用 loss.backward() 之后查看某个参数的梯度，可以直接访问其 .grad 属性。例如，要查看线性模型中权重的梯度，可以使用：
"""

print(model.weight.grad)

# 同样，要查看偏差的梯度，可以
print(model.bias.grad)


# 需要注意的是，在每次执行反向传播之前，你需要将梯度清零，以防止梯度累加。这可以通过调用优化器的 zero_grad() 方法实现：
optimizer.zero_grad()

# 总之，loss.backward() 会自动计算损失函数关于所有模型参数的梯度，并将它们存储在各自的 .grad 属性中。
# 然后，你可以使用优化器根据这些梯度更新参数。在 PyTorch 中，这个过程非常简单，只需几行代码即可完成。
