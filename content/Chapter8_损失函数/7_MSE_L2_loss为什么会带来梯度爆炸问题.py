# _*_ coding=utf-8 _*_

"""
当我们计算MSE损失函数关于某个参数的梯度时，需要使用链式法则。在这个过程中，损失函数中的平方项确实会通过求导消失。
然而，损失函数中的平方项在求导之前已经放大了误差，因此在求导过程中，梯度仍然可能较大。我将通过一个简单的例子来说明这一点。

假设我们的损失函数是：

L(w) = (y_true - y_pred)^2

其中 y_pred = w * x。

对于损失函数关于参数 w 的梯度，我们需要计算：

dL/dw = d((y_true - y_pred)^2) / dw

使用链式法则，我们得到：

dL/dw = 2 * (y_true - y_pred) * (-dx/dw)

这里，dx/dw = x。因此：

dL/dw = 2 * (y_true - y_pred) * (-x)

从这个结果中，我们可以看到损失函数的梯度与误差（y_true - y_pred）成正比。虽然求导过程中消除了平方项，但梯度仍然受到误差的影响。
当误差较大时，梯度值也可能较大，这可能导致模型训练过程中的不稳定。

。对于MAE损失函数，当我们计算梯度时，误差项没有被平方，因此损失函数对参数的梯度不会被误差放大。假设我们的损失函数为：

L(w) = |y_true - y_pred|

其中 y_pred = w * x。

对于损失函数关于参数 w 的梯度，我们需要计算：

dL/dw = d(|y_true - y_pred|) / dw

dL/dw = (d|y_true - y_pred| / d(y_pred)) * (dy_pred/dw)

其中 (d|y_true - y_pred| / d(y_pred)) 是 1 或 -1，取决于 y_true - y_pred 的符号。所以，我们最终得到：

dL/dw = sign(y_true - y_pred) * x

综上， MSE函数dL/dw = 2 * (y_true - y_pred) * (-x)， MAE函数dL/dw = sign(y_true - y_pred) * x

这里，sign(y_true - y_pred) 表示 1（当 y_true > y_pred）或 -1（当 y_true < y_pred），所以MSE函数在误差较大时容易梯度爆炸

在 MAE 损失函数中，梯度不会因为误差的绝对值变大而显著增大，所以相对于 MSE 损失函数，它对异常值不敏感。

"""