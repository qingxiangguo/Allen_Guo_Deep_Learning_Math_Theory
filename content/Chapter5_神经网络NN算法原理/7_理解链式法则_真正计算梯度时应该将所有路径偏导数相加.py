# _*_ coding=utf-8 _*_

"""
我们常常说反向传播会用到链式法则，而链式法则有多种表示，比如：（1）dz/dt = ∂z/∂u * du/dt + ∂z/∂v * dv/dt（2）dy/dz = dy/du * du/dz。
如果想优化参数w21，就需要求损失函数对于参数的梯度，而这个梯度不仅仅是一个路径的结果，而是多个路径的结果的和，

比如∂L/∂w21 = (∂L/∂h8 * ∂h8/∂h5 * ∂h5/∂h2 * ∂h2/∂w21) + (∂L/∂h8 * ∂h8/∂h6 * ∂h6/∂h2 * ∂h2/∂w21) +
(∂L/∂h8 * ∂h8/∂h7 * ∂h7/∂h2 * ∂h2/∂w21).

深度学习真正运用的是链式法则的思想，就是《要把每一个贡献路径的偏导都加起来》

对于一个复杂的神经网络，计算梯度确实会变得非常复杂，然而，这正是反向传播算法的优势所在。反向传播算法可以有效地计算损失函数对每个权重的梯度，
即使在复杂的神经网络中。

反向传播的基本思想是从输出层开始，逐层向前计算梯度。在计算过程中，每一层的梯度可以利用后一层的梯度计算得出。
这样一来，我们不需要显式地为每个权重计算所有可能的路径，而是可以逐层计算梯度，从而避免了大量的重复计算。这种逐层计算梯度的方法能够显著地提高计算效率。

比如，∂L/∂w21 = (∂L/∂h8 * ∂h8/∂h5 * ∂h5/∂h2 * ∂h2/∂w21) + (∂L/∂h8 * ∂h8/∂h6 * ∂h6/∂h2 * ∂h2/∂w21) +
(∂L/∂h8 * ∂h8/∂h7 * ∂h7/∂h2 * ∂h2/∂w21)

尽管我们需要计算所有可能路径的导数，但反向传播的一个优势是我们可以将已经计算过的中间导数值存储起来，并在之后的计算中重复使用它们。
这可以减少重复计算，从而提高计算效率。

在你的例子中，我们可以首先计算损失函数对于 h8 的偏导数：∂L/∂h8。接下来，我们可以计算 h8 关于 h5、h6 和 h7 的偏导数：∂h8/∂h5、∂h8/∂h6 和 ∂h8/∂h7。然后我们可以继续计算 h5、h6 和 h7 关于 h2 的偏导数：∂h5/∂h2、∂h6/∂h2 和 ∂h7/∂h2。最后，我们计算 h2 关于 w21 的偏导数：∂h2/∂w21。

在计算过程中，我们可以将这些中间导数值存储起来，并在需要时重复使用它们。这样，我们可以避免在计算复杂网络时的重复计算，从而提高计算效率。
而这正是反向传播算法的优势所在。

在实际应用中，尤其是在深度学习框架中，这些中间导数值的存储和重复使用是自动完成的，这使得在训练复杂神经网络时能够更加高效地进行梯度计算。
"""