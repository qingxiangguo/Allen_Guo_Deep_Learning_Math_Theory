# _*_ coding=utf-8 _*_
"""
首先可微，由于10中的内容，那么可微一定连续，因此有邻域，因此可以在任意方向上趋近，Δx ->0, Δy -> 0

而全微分的公式，limit Δx ->0, Δy -> 0, f(x0 + Δx, y0 + Δy) - f(x0, y0) = AΔx + BΔy + o(√(Δx²+Δy²))

由于x0， y0往任意方向跑，这个式子都成立，所以我们先让x0+Δx， y0不变，也就是在平行x轴的方向上跑
这个结论也是成立的，由于Δy = 0，上面式子变成

f(x0 + Δx, y0) - f(x0, y0) = AΔx + o(√(Δx²)
进一步变成，f(x0 + Δx, y0) - f(x0, y0) = AΔx + o(Δx)， 其中Δx ->0

两边取极限，再除以Δx
limit Δx ->0，[f(x0 + Δx, y0) - f(x0, y0)]/Δx = A

很巧的是，这个式子刚好就是，函数f(x, y)，在x0, y0处，关于x的偏导数

因此不光证明了，多元函数可微一定可偏导， 并且还证明了，全微分公式中，Δz = AΔx + BΔy + o(√(Δx²+Δy²))
A就等于函数f(x, y)，在x0, y0处，关于x的偏导数，同理，B等于函数f(x, y)，在x0, y0处，关于y的偏导数

可微的几何意义是这个地方存在切平面
"""