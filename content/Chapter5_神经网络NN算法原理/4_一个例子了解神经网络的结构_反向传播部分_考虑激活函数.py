# _*_ coding=utf-8 _*_

"""
对于我给定的神经网络，我在求梯度的时候，为了便于理解简化，是省略了激活函数的，不考虑复合函数求导
但在实际情况中，进入下一层神经元的值，实际上是sigmoid的输出值，包括最后h8的输出值，也是要经过sigmoid输出sigmoid(h8)的。

现在我们将每个神经元的raw输出，设为h1,h2, ... h8，
然后再加入激活函数sigmoid(h1), sigmoid(h5)等等，才是真正输入下一层的值

那么考虑到激活函数sigmoid的时候，链式法则如何使用？

比如和上面一模一样的例子，损失函数L 就变成了 = (y真实值-sigmoid(h8))^2

已知sigmoid函数的导数  sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

这个时候就相当于复合函数里面，多加了一层激活函数，本质是一样的

【损失函数 L 对 w11 的偏导数】
根据链式法则，∂L/∂w11 = ∂L/∂h8 * ∂h8/∂h5 * ∂h5/∂h1 * ∂h1/∂w11
其中，
L = (y真实值-sigmoid(h8))^2

h8 = w81*sigmoid(h5) + w82*sigmoid(h6) + w83*sigmoid(h7) + b8

h5 = w51*sigmoid(h1) + w52*sigmoid(h2) + w53*sigmoid(h3) + w54*sigmoid(h4) + b5

h1 = w11 * x1 + w12 * x2 + w13 * x3 + b1  # 输入层是没有激活函数的，直接进来，所以并不是 w11* sigmoid(x1)，这里要注意

所以，∂L/∂h8 = -2 * （y真实值 - sigmoid'(h8)） = -2(y真实值 - sigmoid(h8)) * sigmoid(h8) * (1- sigmoid(h8))
∂h8/∂h5 = w81 * sigmoid(h5) * (1- sigmoid(h5))
∂h1/∂w11 = x1

代入即可得损失函数 L 对 w11 的偏导数

【损失函数 L 对 w51 的偏导数】
∂L/∂w51 = ∂L/∂h8 * ∂h8/∂h5 * ∂h5/∂w51

其中，
L = (y真实值-sigmoid(h8))^2

h8 = w81*sigmoid(h5) + w82*sigmoid(h6) + w83*sigmoid(h7) + b8

h5 = w51*sigmoid(h1) + w52*sigmoid(h2) + w53*sigmoid(h3) + w54*sigmoid(h4) + b5

∂L/∂h8 = -2 * (y真实值 - sigmoid(h8)) * sigmoid(h8) * （1- sigmoid(h8)）

∂h8/∂h5 = w81 * sigmoid(h5) * (1- sigmoid(h5))

∂h5/∂w51 = sigmoid(h1)

因此，代入即可得损失函数 L 对 w51 的偏导数

【损失函数 L 对 w81 的偏导数】

根据链式法则，损失函数 L 对 w81 的偏导数为：

∂L/∂w81 = ∂L/∂h8 * ∂h8/∂w81

其中，
L = (y真实值-sigmoid(h8))^2

h8 = w81 * sigmoid(h5) + w82 * sigmoid(h6) + w83 * sigmoid(h7) + b8

∂L/∂h8 = -2 * (y真实值 - sigmoid(h8)) * sigmoid(h8) * (1 - sigmoid(h8))

∂h8/∂w81 = sigmoid(h5)

最终，
∂L/∂w81 = -2 * (y真实值 - sigmoid(h8)) * sigmoid(h8) * (1 - sigmoid(h8)) * sigmoid(h5)

"""