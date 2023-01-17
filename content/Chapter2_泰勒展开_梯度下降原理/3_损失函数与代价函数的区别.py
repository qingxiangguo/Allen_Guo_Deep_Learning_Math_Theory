# _*_ coding=utf-8 _*_

"""
损失函数和代价函数是深度学习中经常提到的两个概念。
损失函数（loss function）是指对于给定的输入数据和模型参数，来度量预测值与真实值之间的差距。
它是一个非负实值函数，其值越小，说明预测值与真实值之间的差距越小。
代价函数（cost function）是指在训练模型时，需要最小化的函数。在许多情况下，它就是损失函数的平均值。
总的来说，损失函数度量的是单个样本的误差，而代价函数度量的是所有样本的误差的平均值。

比如，在2_损失函数_SSE_MSE中提到的误差函数，其实是代价函数，

MSE(Mean Squared Error)，均方误差，L = 1/2n * Σ(y_pred - y)²，这个是针对所有训练样本的

在深度学习中，单个训练样本是指一个数据样本，它包含了输入特征和对应的输出标签。
在训练过程中，算法会使用大量的训练样本来学习模型的参数。

比如在图像分类问题中，一个训练样本就是一张图片和对应的标签（类别）。在训练过程中，
算法会使用大量的图片和对应的标签来学习如何对图片进行分类。

对于单个样本的误差，可以用均方误差(Mean Squared Error, MSE)来度量预测值和真实值之间的差距，
公式为 L(y, y^) = (y - y^)^2, 其中y 是真实值， y^ 是预测值。

对于整个训练集的误差，通常使用均方误差(Mean Squared Error, MSE)来表示，其公式为:
MSE = 1/n * Σ(y_i - y^_i)^2

其中，n是样本数量，y_i是真实值，y^_i是预测值，Σ表示对所有样本求和。
这个MSE值表示整个训练集的预测值与真实值之间的差距的平均值
"""