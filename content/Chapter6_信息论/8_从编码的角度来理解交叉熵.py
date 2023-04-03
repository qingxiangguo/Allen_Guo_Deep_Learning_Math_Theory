# _*_ coding=utf-8 _*_

"""
从编码的角度理解交叉熵，我们可以将其视为一种衡量使用基于估计概率分布Q的编码方案来对真实概率分布P进行编码的平均编码长度。
换句话说，交叉熵描述了基于估计分布Q对实际分布P的事件进行编码时的信息损失。

假设我们有一个离散概率分布P，表示真实事件发生的概率。我们还有另一个离散概率分布Q，用于估计P。
我们希望基于Q创建一个编码方案来对P中的事件进行编码。在理想情况下，我们希望编码长度尽可能短，即最接近熵（理论最小编码长度）。

交叉熵定义为：

H(P, Q) = -∑ P(x) * log(Q(x))

其中，x表示P和Q中的事件，P(x)表示事件x在P中发生的概率，log(Q(x))表示基于Q的编码方案对事件x的编码长度。

交叉熵H(P, Q)可以理解为：当我们使用Q作为编码方案对真实分布P的事件进行编码时，所得到的平均编码长度。从这个角度来看，
交叉熵表达了我们的编码方案与理想编码方案（基于真实概率分布P的熵）之间的差距。换句话说，交叉熵衡量了我们的编码方案有多有效，或者说有多接近理想编码。

当估计概率分布Q与真实概率分布P完全相同时，交叉熵等于P的熵，这意味着我们的编码方案达到了理论上的最小编码长度。然而，在实际应用中，
Q通常不会完全等于P，因此交叉熵会大于熵。

我们希望找到一个编码方案，使得编码长度尽可能短。理论上的最小编码长度是由真实概率分布P的熵给出的，即 -∑ p(x) * log2(p(x))。
而我们实际使用的编码方案是基于估计概率分布Q，其编码长度为 -∑ Q(x) * log2(Q(x))。

为了比较这两个编码长度，我们将它们都放在真实概率分布P的基础上。我们将实际编码方案的编码长度改为 -∑ P(x) * log2(Q(x))。
这样，我们得到了交叉熵 H(P, Q) = -∑ P(x) * log2(Q(x))，它表示我们使用估计概率分布Q的编码方案在真实概率分布P下的平均编码长度。

这样一来，我们就可以将交叉熵 H(P, Q) 与真实概率分布P的熵进行比较，从而评估我们的编码方案相对于理论最优编码。

。交叉熵H(P, Q)表示在基于估计概率分布Q制定了一套编码方案（密码本）的情况下，对真实概率分布P进行编码的平均编码长度。这个编码方案可能不是最优的，
因为它是基于估计分布Q的。

而理论最优的编码方案是基于真实分布P的，对应的编码长度是熵 -∑ p(x) * log2(p(x))。为了比较这两个编码方案的效率，
我们需要在同一个真实分布P下进行比较。这样，我们就可以通过最小化交叉熵来优化我们基于估计分布Q的编码方案，使其尽可能接近理论最优编码方案。

在实际应用中，我们通常希望找到一个编码方案，使得基于真实分布P的熵最小，也就是平均编码长度最短。通过最小化交叉熵H(P, Q)，
我们可以优化我们的估计分布Q，使其尽可能地接近真实分布P。

H(P, Q)（交叉熵）表示在真实概率分布P下，使用基于估计概率分布Q进行编码所需的平均编码长度（以比特为单位）。
而H(P)（熵）表示对真实概率分布P所需的最小编码长度（以比特为单位）。

熵是理论上的最佳编码长度，当我们使用真实概率分布P进行编码时，可以达到最短的平均编码长度。
而交叉熵则表示我们在使用基于估计概率分布Q的编码方案时，实际所需的平均编码长度。通过最小化交叉熵，
我们可以使估计概率分布Q尽可能接近真实概率分布P，从而使得实际编码方案的效率尽可能接近理论最佳编码方案。
"""