# _*_ coding=utf-8 _*_
"""
在二维平面上，这个点，总是在平行于x轴，或平行于y轴的方向上运动的。比如，x的偏导数，y不动，那这个点就是在平行于x轴上运动的
但实际过程中，这个点是可以自由运动的，因此引入了方向向量的概念。

为了描述方向导数，继续回忆第二个知识点，方向余弦。假设在二维x,y平面坐标系上，有一个l向量
l向量坐标系为(a, b)
下面将这个向量进行单位化，得到一个单位向量，l0 = (a/√(a²+b²), b/√(a²+b²))
回到坐标系中，设角度α = XOL, β = YOL，那么单位向量就可以表示为（cosα, cosβ）
也就是方向向量，可以表示为方向余弦的形式，留着备用

现在继续，f(x, y, z)，平面上的一个点x0, y0是可以向不同方向发生变化的，z也会相应的变化
这里就会有一个变化率的概念，这个概念实际上是一个斜率，就是【函数值在，这个点在这个方向上形成的截面与原函数形成的截曲线的切线的斜率】，也就是导数
随着运动方向不同，切线斜率也不同，而且是可以360度变化的，这就是方向导数：某个点在二维平面往某个方向移动时，函数值在这个方向上的变化率

不同方向，导数不一样，这个就叫方向导数。偏导数是方向导数的特例，是沿正x轴方向的和正y轴方向的方向导数。

"""
