# _*_ coding=utf-8 _*_

"""
上面介绍了交叉熵H(P, Q)可以理解为：当我们使用Q作为编码方案对真实分布P的事件进行编码时，所得到的平均编码长度(换了个假设的密码本，不一定是最优)。

那么交叉熵H(P, Q)减去真正的熵，H(P)，就可以作为衡量预测分布和真实分布之间差异的一种度量，这就是相对熵，也就是KL散度，记为DKL(P||Q)

DKL(P||Q) = H(P, Q) - H(P) = -∑ P(x) * log(Q(x)) + ∑ P(x) * log(P(x))  # 因为有个负号，负负得正

DKL(P||Q)表示在真实分布为p的前提下，使用q分布进行编码相对于使用真实分布p进行编码（即最优编码）所多出来的bit数，也就是多走的弯路

具体地说，它表示在使用估计概率分布Q来近似真实概率分布P时，相对于直接使用P进行编码，所导致的平均编码长度的额外增加。

-∑ P(x) * log(Q(x)) + ∑ P(x) * log(P(x)) 公式进一步变形

D(P || Q) = ∑ P(x) * log2(P(x) / Q(x))

其中，P和Q分别表示真实概率分布和估计概率分布，x表示事件。

让我们用一个简单的例子来解释KL散度。

假设有两个概率分布，P = {0.7, 0.3} 和 Q = {0.5, 0.5}，表示某地两种天气状况（晴天和阴天）的概率。P是真实概率分布，Q是估计概率分布。
我们希望比较在使用Q来近似P时，编码效率的损失。

首先，计算P的熵（理论上的最小编码长度）：

H(P) = -[0.7 * log2(0.7) + 0.3 * log2(0.3)] ≈ 0.881

接下来，计算交叉熵 H(P, Q)（使用Q进行编码时，基于P分布的平均编码长度）：

H(P, Q) = -[0.7 * log2(0.5) + 0.3 * log2(0.5)] = 1

现在，我们可以计算KL散度，表示在使用Q来近似P时的编码效率损失：

D(P || Q) = H(P, Q) - H(P) = 1 - 0.881 ≈ 0.119

这个结果表明，在使用估计概率分布Q进行编码时，相对于直接使用真实概率分布P，平均每个事件的编码长度额外增加了大约0.119比特。
KL散度的值越大，表示两个概率分布之间的差异越大

需要注意的是，KL散度是非对称的，即 D(P || Q) 不等于 D(Q || P)。在某些应用场景中，这两个方向的散度可能具有不同的意义。
"""