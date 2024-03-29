# _*_ coding=utf-8 _*_

"""
在深度学习中，神经网络是一种非常强大的模型，它能够实现高级的数据处理能力。在文中，我们将深入解析神经网络的基本运行机制，特别是对于多个训练样本的处理。
首先，我们需要对神经网络进行初始化参数。这些参数是网络中各个层之间连接的权重和偏置。权重和偏置是通过随机初始化得到的，这是为了避免网络陷入局部最优解。

接下来，我们进行正向传播计算结果。正向传播是指从输入层传递到输出层的过程。在这个过程中，网络对每个样本都进行了一次计算，并得到了对应的输出结果。
在这个过程中，我们需要使用激活函数来提高网络的非线性能力。常用的激活函数有sigmoid, ReLU, tanh等。

接下来，我们计算样本各自的损失函数。损失函数是用来衡量预测结果与真实结果之间差距的函数。常用的损失函数有平方误差，交叉熵损失等。
然后，我们利用反向传播机制（链式法则），计算每个样本的损失函数对各个参数的偏导（也就是梯度）。这个过程是通过求导来实现的。

在这一步完成后，我们需要对所有样本都进行这个操作，计算出所有样本的平均梯度。这样可以避免对某些样本的偏差过大的影响。

接下来，我们通过梯度下降原理，优化迭代参数。梯度下降是一种优化算法，它通过不断减小损失函数的值来更新参数。
常用的优化算法有SGD, Adam, Adagrad等。我们需要不断重复这个过程，直到总体代价函数收敛。

在这一次正向传播的过程中，n个训练样本就相当于整张神经网络进行n次操作。最终，我们可以得到一个经过训练的神经网络模型，它可以对新的样本进行预测。

正向传播和反向传播之间相互依赖，它们是训练深度学习模型的基础。正向传播用于计算模型的预测结果，
而反向传播则用于计算损失函数的梯度并更新参数。模型的参数在反向传播中进行更新，然后再用于正向传播。这个过程会不断重复，直到损失函数收敛为止。

在训练过程中，需要注意的是，由于正向传播和反向传播之间相互依赖，所以中间变量的存储占用了大量内存。在训练深度学习模型时，需要注
意内存的使用，并且采用一些优化方法来避免过多的内存使用，比如采用小批量训练和梯度消失等。

总之，深度学习中神经网络的基本运行机制包括：多个训练样本，初始化参数，正向传播计算结果，计算样本各自的损失函数，
利用反向传播机制（链式法则），计算每个样本的损失函数对各个参数的偏导（也就是梯度,指代了这个参数下降最快的方向），
然后对所有样本都进行这个操作，计算出所有样本的【平均梯度】。

然后通过梯度下降原理，每个参数都减去【平均梯度】，优化迭代参数，查看总体代价函数，然后再正向传播计算结果，开始循环，直至总体代价函数收敛。

通过这个过程，我们可以在训练样本上得到一个较好的神经网络模型，并且可以用来对新样本进行预测。
更易懂的理解是，初始化参数是为了正向传播，正向传播是为了计算每个样本的损失函数，计算每个样本的损失函数是为了求得损失函数

对每个样本的梯度，求梯度是为了求平均梯度，求平均梯度是为了更新优化参数，更新优化参数后又正向传播是为了计算每个样本的损失函数，
计算每个样本的损失函数是为了取平均值获得全局代价函数，获得全局代价函数是为了评估模型是否在被不断优化，直到收敛。
"""


