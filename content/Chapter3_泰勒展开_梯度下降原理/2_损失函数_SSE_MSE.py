# _*_ coding=utf-8 _*_
"""
关于 Loss Function、Cost Function 和 Objective Function 的区别和联系。
在机器学习的语境下这三个术语经常被交叉使用

损失函数 Loss Function 通常是针对单个训练样本而言，给定一个模型输出和一个真实值，
损失函数输出一个真实值损失

代价函数 Cost Function 通常是针对整个训练集（或者在使用 mini-batch gradient descent 时一个 mini-batch）
的总损失

目标函数 Objective Function 是一个更通用的术语，表示任意希望被优化的函数，
用于机器学习领域和非机器学习领域

下面的损失函数，在公式其实指的是【代价函数】，是反应整个训练集，而不是针对某一个样本的

损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数
平方形式的损失函数一般为, L = 1/2 * Σ(y_pred - y)²，这称为 SSE（The sum of squares due to error）
， 称为误差平方和，该统计参数计算的是拟合数据和原始数据对应点的误差的平方和

还有一种称为MSE(Mean Squared Error)，均方误差，L = 1/2n * Σ所有样本的(y_pred - y)²
该统计参数是预测数据和原始数据对应点误差的平方和的均值，也就是SSE/n，和SSE没有太大的区别

这两种形式本质上是等价的。只是MSE计算得到的值比SSE计算得到的值要小，因为除了一个n。误差平方和以及均方差的公式中有系数1/2，
是为了求导后，系数被约去。

它们都是平方形式，一个重要原因是：误差的平方形式是正的，是正数。
这样正的误差和负的误差不会相互抵消。这就是为什么不用一次方，三次方的原因。

但是，误差的绝对值也是正的，为什么不用绝对值呢。所有还有第二个重要原因是：
平方形式对大误差的惩罚大于小误差。此外，还有第三个重要原因：平方形式对数学运算也更友好。我们经常要求损失函数的导数，
平方形式求导后变成一次函数；而绝对值形式对求导数学运算很不友好，需要分段求导。
"""