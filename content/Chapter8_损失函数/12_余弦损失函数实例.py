# _*_ coding=utf-8 _*_

"""
余弦损失函数简介
余弦损失函数是一种度量向量之间相似性的损失函数，用于学习任务中需要度量相似性的场景，例如文本相似度计算、词向量学习、图像检索等。
它的主要目标是使得相似的向量在空间中靠近，而不相似的向量在空间中远离。

余弦损失函数的定义
对于两个向量 a 和 b，其余弦相似度计算公式为：

cos(a, b) = (a · b) / (||a|| ||b||)

余弦损失函数的计算方式根据样本标签 y 的取值不同而有所变化：

如果 y = 1（表示向量 a 和 b 相似），损失函数为：loss = 1 - cos(a, b)
如果 y = -1（表示向量 a 和 b 不相似），损失函数为：loss = max(0, cos(a, b) - margin)
其中，margin 是一个超参数，用于控制相似度的阈值。

实际的简单神经网络例子
假设我们要构建一个神经网络来度量文本之间的相似性。我们将文本表示为固定长度的向量。以下是一个简单的神经网络结构：

输入层：接收一个固定长度的文本向量（例如 300 个元素的一维向量）。
隐藏层：包含 128 个神经元，使用 ReLU 激活函数。
输出层：包含 300 个神经元，使用线性激活函数（即没有激活函数）。

输入数据长这样：   apple_vector = [0.12, -0.56, 0.32, ..., 0.07, 0.21, -0.19]  # 300个元素

这里，apple_vector是一个长度为300的一维数组，每个元素是一个浮点数。这个向量捕捉了单词"apple"的语义信息，
使得与"apple"语义相似的单词，如"fruit"、"pear"等，它们的向量在向量空间中距离较近。
"""

import torch

# 构建神经网络

import torch
import torch.nn as nn

"""
可以将这里的 300 和 128 理解为神经元的数量。在全连接层 self.fc1 = nn.Linear(300, 128) 中，输入层有 300 个神经元，与隐藏层的 
128 个神经元全连接。这意味着每一个输入层神经元都连接到隐藏层的每一个神经元，共计 300 × 128 个连接。这些连接通过权重矩阵（
尺寸为 300 x 128）来表示。

从a层，到b层，如果a有X个神经元，b有Y个神经元，那么b的权重矩阵就是(X, Y)的二维矩阵
"""

class TextSimilarityModel(nn.Module):
    def __init__(self):
        super(TextSimilarityModel, self).__init__()  # 调用父类构造函数
        self.fc1 = nn.Linear(300, 128) # 第一个全连接层，其实也是隐藏层，连接输入的300神经元与隐藏层的128神经元
        # self.fc1的输出是一个128元素的向量
        self.fc2 = nn.Linear(128, 300) # 第二个全连接层，也是输出层，连接隐藏层的128神经元与输出层的300神经元
        # self.fc2的输出是一个300元素的向量

    def forward(self, x):  # model(x)的时候，x会传进来，进行前向传播
        x = torch.relu(self.fc1(x))  # nn.Linear和TextSimilarityModel一样，也实现了__call__()方法，所以可以self.fc1(x)调用
        # 将输入数据x传递给第一个全连接层self.fc1，然后应用ReLU激活函数。ReLU激活函数可以增加模型的非线性能力，
        # 有助于学习复杂的输入-输出映射关系。
        x = self.fc2(x)
        # 将ReLU激活函数的输出传递给第二个全连接层self.fc2，得到模型的最终输出
        return x

model = TextSimilarityModel() # 实例的类继承自nn.Module，拥有__call__方法，__call__方法里面还会调用forward()方法

"""
当你使用self.fc1 = nn.Linear(300, 128) 的时候，其实是连接了输入层的300神经元和隐藏层的128神经元，然后权重矩阵的形状是(300, 128)。
输入是向量形状维度为（10,300）因为是10个样本一次性输入的，也就是a_batch和b_batch，就会触发矩阵乘法操作中的：一维向量与二维矩阵相乘，
会将(300)变成(1, 300)，然后与(300, 128)相乘，然后变成(1, 128)，再删除掉预置位，变成(128)的输出。

self.fc1 = nn.Linear(128, 300)则是承接了隐藏层与输出层，输入是刚刚输出的(128)，又会触发矩阵乘法操作中的：一维向量与二维矩阵相乘，
会将(128)变成(1, 128)，然后与(128, 300)相乘，然后变成(1, 300)，再删除掉预置位，变成(300)的输出。

这就是一个三层的神经网络，完成了输入层的300神经元，隐藏层128神经元，输出层300神经元
"""

"""
。当你创建自己的模型时，需要继承 nn.Module 类，这样你的模型就可以享受 nn.Module 提供的一些功能和方法。在你自定义的模型类中，
你需要实现一个名为 forward() 的方法。这个方法用于定义模型的前向传播逻辑。

当你实例化模型并像函数一样调用它，比如 model(x)，这实际上是在调用 nn.Module 基类的 __call__() 方法。
__call__() 方法会处理一些额外的操作，比如将输入数据移动到适当的设备（CPU 或 GPU），然后调用你自定义的 forward() 方法。
"""

# 定义余弦损失函数

class CosineLoss(nn.Module):
    def __init__(self, margin):
        super(CosineLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1) # 定义余弦相似度的计算，输入是两个向量

    def forward(self, a, b, y):  # 输入a,b都是二维数组，y是标签，是一维向量
        cos_sim = self.cosine_similarity(a, b)  # cos_sim 的形状为 10，表示计算了 a 和 b 中每一对相应向量之间的余弦相似度
        # 包含了 a 和 b 中每对相应行向量之间的余弦相似度。
        loss = torch.where(y == 1, 1 - cos_sim, torch.clamp(cos_sim - self.margin, min=0))
        # loss也是10个元素的一维张量，记录了每个样本的余弦损失值
        return loss.mean()  # 求平均值，也就是真正的损失值

"""
对于两个二维数组，dim=0 表示沿着列方向（竖着）对应求余弦相似度，dim=1 表示沿着行方向（横着）对应求余弦相似度。

对于两个一维向量，由于它们只有一个维度，所以 dim 只能为 0。在这种情况下，会计算这两个一维向量之间的余弦相似度。

举例：

a = torch.tensor([[1, 7],
                  [3, 4]])
b = torch.tensor([[1, 0],
                  [4, 6]])
                  
当 dim=0 时，沿着列方向（竖着）计算余弦相似度：

第一列：[1, 3] 和 [1, 4]，余弦相似度为：(1*1 + 3*4) / (sqrt(1^2 + 3^2) * sqrt(1^2 + 4^2)) = 
13 / (sqrt(10) * sqrt(17)) ≈ 0.9978。

第二列：[7, 4] 和 [0, 6]，余弦相似度为：(7*0 + 4*6) / (sqrt(7^2 + 4^2) * sqrt(0^2 + 6^2)) = 
24 / (sqrt(65) * sqrt(36)) ≈ 0.6196。

所以，cos_sim_dim0 = [0.9978, 0.6196]。

当 dim=1 时，沿着行方向（横着）计算余弦相似度：

第一行：[1, 7] 和 [1, 0]，余弦相似度为：(1*1 + 7*0) / (sqrt(1^2 + 7^2) * sqrt(1^2 + 0^2)) = 
1 / (sqrt(50) * sqrt(1)) ≈ 0.1414。

第二行：[3, 4] 和 [4, 6]，余弦相似度为：(3*4 + 4*6) / (sqrt(3^2 + 4^2) * sqrt(4^2 + 6^2)) = 
34 / (sqrt(25) * sqrt(52)) ≈ 0.9407。

所以，cos_sim_dim1 = [0.1414, 0.9407]。

对于相似度任务，一般dim = 1居多

loss = torch.where(y == 1, 1 - cos_sim, torch.clamp(cos_sim - self.margin, min=0))
根据标签 y 计算损失。

y == 1：这是一个逐元素比较操作，将 y 中的每个元素与 1 进行比较。如果相等，则返回 True，否则返回 False。结果是一个与 y 形状相同的布尔张量。

当 y 中的某个元素等于 1 时，表示 a 和 b 中对应的行向量应该相似，损失为 1 - cos_sim。
当 y 中的某个元素不等于 1 时，表示 a 和 b 中对应的行向量不应该相似，损失为 cos_sim - self.margin。

为了避免负数损失，使用 torch.clamp 将损失限制在 0 以上。torch.clamp是限制值范围的，最小就是0.

使用 torch.where() 函数根据 y 的值选择适当的损失值。对于 y == 1 的元素，选择 1 - cos_sim；对于 y != 1 的元素，
选择 torch.clamp(cos_sim - self.margin, min=0)。这将为每一对输入向量 a 和 b 生成一个损失值。
"""

cosine_loss = CosineLoss(margin=0.5)   # 实例化一个余弦损失计算器

# 训练神经网络

import torch.optim as optim

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 创建一个 Adam 优化器，用于更新模型的参数。学习率（lr）设置为 0.001。
# 将模型参数传递给优化器

# 训练数据样例（这里使用随机数据作为示例，实际应用中需要使用实际文本向量）
a_batch = torch.randn(10, 300)  # 10 个文本向量 a，10个样本，每个300个元素，也就是10行，300列
b_batch = torch.randn(10, 300)  # 10 个文本向量 b，10个样本，每个300个元素，也就是10行，300列
y_batch = torch.tensor([1, -1, 1, 1, -1, -1, 1, 1, 1, -1])  # 10 个相似性标签，就是我们知道a, b中对应样本的相似度如何，作为训练

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    # 梯度清零，将优化器中的梯度清零，以避免在每次迭代中累积梯度
    optimizer.zero_grad()

    # 前向传播
    a_output = model(a_batch) # 其实就是model返回的x，形状都是 (10, 300)
    b_output = model(b_batch) # 其实就是model返回的x，形状都是 (10, 300)

    # 计算损失
    loss = cosine_loss(a_output, b_output, y_batch) # 两个(10, 300)形状的二维数组，计算余弦损失函数，得一个10元素向量
    # loss 的形状也是 10，表示根据相似性标签 y 计算出的每个样本的余弦损失值

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
